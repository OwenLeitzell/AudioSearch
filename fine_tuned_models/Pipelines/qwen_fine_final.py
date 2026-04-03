
"""
qwen_fine_final.py
Two-stage pipeline for math formula transcription using Qwen models.

Stage 1 (ASR): Qwen2-Audio-7B-Instruct transcribes audio to spoken text using voice
               chat mode (ASR mode). In this mode the model only transcribes speech
               into text and does not use LLM reasoning capabilities.
Stage 2 (LLM): Qwen2.5-Math-1.5B extracts mathematical content from the transcription
               and parses it as LaTeX. The LLM is fine-tuned on spoken English -> LaTeX
               pairs, then evaluated using ASR transcriptions as input.

Evaluation compares predicted LaTeX (pipeline output) against ground truth equations
from the dataset, measuring WER, CER, and hallucination.
"""

#IMPORTS
import os
import re
import gc

# /home is full (1TB used) — redirect all HuggingFace cache to /tmp (65GB free)
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache/datasets"
os.environ["TMPDIR"] = "/tmp"

# Resolve HF auth token: check env var, then common token file locations.
# Needed because HF_HOME redirect above moves where libraries look for the token.
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    for _path in [
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
    ]:
        if os.path.exists(_path):
            with open(_path) as _f:
                HF_TOKEN = _f.read().strip()
            break
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    print("WARNING: No HuggingFace token found. Gated datasets will fail.")
    print("  Run: huggingface-cli login")

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load
os.environ["WANDB_DISABLED"] = "true"

# CONFIG
#Qwen has two models, one for ASR and one for LLM
# Stage 1: ASR Model (voice chat mode — transcription only, no reasoning)
ASR_MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
# Stage 2: LLM Model (math extraction and LaTeX parsing)
LLM_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

#Load dataset1: mathbridge-audio
DATASET_NAME = "abby1492/mathbridge-audio"
SPLIT = "train"
MAX_SAMPLES = None
OUTPUT_XLSX = "qwen_pipeline_outputs2.xlsx"
OUTPUT_MODEL_DIR = "./qwen_math_best"

#Load dataset2: FormulaSearch
DATASET2_NAME = "OwenLeitzell/FormulaSearch"
DATASET2_SPLIT = "train"
DATASET2_AUDIO_COLUMN = "audio"
DATASET2_MAX_SAMPLES = None

# Stage 2 LLM fine-tuning hyperparameters
PER_DEVICE_BATCH = 32
GRAD_ACCUM = 2
fp16 = True
NUM_EPOCHS = 20
LR = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

SYSTEM_PROMPT = (
    "You are a LaTeX transcription assistant. "
    "Given a sentence that mixes spoken math with English, extract ONLY the mathematical "
    "formula and output it as LaTeX. Output nothing except the LaTeX formula itself — "
    "no explanation, no reasoning, no surrounding text."
)

# -------------------- DATASET -------------------- #
# Both datasets share: audio, equation, spoken_English, context_before, context_after
# Only difference is the index column name (source_row vs row_number) which we drop
AUDIO_COLUMN = "audio"
REQUIRED_COLUMNS = [AUDIO_COLUMN, "equation", "spoken_English", "context_before", "context_after"]

print(f"Loading datasets: {DATASET_NAME} ({SPLIT}) and {DATASET2_NAME} ({DATASET2_SPLIT})")
dataset1 = load_dataset(DATASET_NAME, split=SPLIT, token=HF_TOKEN)
dataset2 = load_dataset(DATASET2_NAME, split=DATASET2_SPLIT, token=HF_TOKEN)

def keep_required_columns(ds, required):
    to_remove = [c for c in ds.column_names if c not in required]
    return ds.remove_columns(to_remove) if to_remove else ds
#Stage 1 requires the audio for transcription (ASR input)
#Stage 2 requires spoken English, context before, and context after for the prompt and the equation for evaluation ground truth
dataset1 = keep_required_columns(dataset1, REQUIRED_COLUMNS)
dataset2 = keep_required_columns(dataset2, REQUIRED_COLUMNS)
dataset1 = dataset1.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
dataset2 = dataset2.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
dataset = concatenate_datasets([dataset1, dataset2])

# Split into train and validation (80/20 split)
#seed used for reproducibility
dataset = dataset.train_test_split(test_size=0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

# -------------------- NORMALIZATION -------------------- #
#Same normalization as in the whisper fine-tuned model
def normalize_latex(tex: str) -> str:
    if tex is None:
        return ""
    s = str(tex).replace("$", "").replace("\\,", ",").replace("\\;", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"_\{\s*([^\}]+)\s*\}", r"_\1", s)
    s = re.sub(r"\^\{\s*([^\}]+)\s*\}", r"^\1", s)
    s = re.sub(r"(?<!\\){\s*([A-Za-z0-9_\\]+)\s*}", r"\1", s)
    return s

train_dataset = train_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})
eval_dataset = eval_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# STAGE 2: FINE-TUNE LLM (Qwen2.5-Math-1.5B) on spoken text → LaTeX
# The LLM is trained on the golden spoken_English text from the
# dataset. At evaluation time it receives ASR transcriptions instead.
print("\n" + "="*50)
print("STAGE 2 MODEL: Loading LLM for fine-tuning...")
print("="*50)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token #set padding token to the end of sequence token
tokenizer.padding_side = "left" #inputs must be left padded because LLMs are trained to coninue from pad tokens

# Load in FP32 so the Trainer's AMP (fp16=True) can keep FP32 master weights.
# Loading directly in FP16 breaks GradScaler which needs FP32 to unscale gradients.
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME, device_map="auto", trust_remote_code=True,
    dtype=torch.float32,
)
print(f"LLM loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def build_user_content(context_before, spoken, context_after):
    """Build the user message content from the dataset fields."""
    return f"{context_before.strip()} {spoken.strip()} {context_after.strip()}"

def build_chat_prompt(context_before, spoken, context_after):
    """Build a ChatML-formatted prompt (everything up to the assistant's response)."""
    user_text = build_user_content(context_before, spoken, context_after)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

def prepare_example(ex):
    prompt = build_chat_prompt(ex["context_before"], ex["spoken_English"], ex["context_after"])
    label = ex["text_norm"]
    # Full training sequence: prompt + LaTeX answer + EOS
    full_text = prompt + label + tokenizer.eos_token

    tokenized = tokenizer(
        full_text, truncation=True, max_length=512, padding=False, return_tensors=None,
    )
    input_ids = tokenized["input_ids"]

    # Mask the prompt tokens in labels so loss is ONLY computed on the LaTeX output.
    prompt_ids = tokenizer(
        prompt, truncation=True, max_length=512, padding=False, return_tensors=None,
    )["input_ids"]
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

print("Preprocessing datasets for LLM fine-tuning...")
train_ft = train_dataset.map(prepare_example, remove_columns=train_dataset.column_names)
eval_ft = eval_dataset.map(prepare_example, remove_columns=eval_dataset.column_names)
print(f"Preprocessing complete. Train: {len(train_ft)}, Eval: {len(eval_ft)}")

# Custom collator that pads input_ids, attention_mask, and labels separately.
# DataCollatorForLanguageModeling can't handle pre-set labels with -100 masking.
def collate_fn(features):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(f["attention_mask"] + [0] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }

wer_metric = load("wer")
cer_metric = load("cer")

def extract_latex(text):
    match = re.search(r'\$(.+?)\$', text)
    if match:
        return f"${match.group(1)}$"
    latex_match = re.search(r'\\([a-zA-Z]+(?:\{[^}]*\})*)', text)
    if latex_match:
        return f"${latex_match.group(0)}$"
    return text.strip()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {}

steps_per_epoch = len(train_ft) // (PER_DEVICE_BATCH * GRAD_ACCUM)
EVAL_STEPS = max(100, steps_per_epoch // 4) #evaluate every 100 steps
SAVE_STEPS = EVAL_STEPS #save every 100 steps

training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR, #output model directory
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_steps=max(1, int(len(train_ft) // (PER_DEVICE_BATCH * GRAD_ACCUM) * NUM_EPOCHS * WARMUP_RATIO)),
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=fp16 and torch.cuda.is_available(),
    gradient_checkpointing=True,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=50,
    report_to="tensorboard",

    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ft,
    eval_dataset=eval_ft,
    data_collator=collate_fn,
    processing_class=tokenizer,
    callbacks=[early_stopping],
)

print("\n" + "="*50)
print("Starting Stage 2 LLM fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Total training steps: ~{len(train_ft) // (PER_DEVICE_BATCH * GRAD_ACCUM) * NUM_EPOCHS}")
print(f"Eval steps: {EVAL_STEPS}, Save steps: {SAVE_STEPS}")
print("="*50 + "\n")

trainer.train()
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"\nFine-tuned LLM saved to {OUTPUT_MODEL_DIR}")

del trainer
model.cpu()
del model
torch.cuda.empty_cache()
gc.collect()
print("Training resources freed.\n")

# ================================================================
# PIPELINE EVALUATION
# Stage 1 (ASR): Qwen2-Audio voice chat mode → transcription
# Stage 2 (LLM): Qwen2.5-Math with transcription → LaTeX
# Metrics: WER, CER, Hallucination against ground truth equations
# ================================================================
print("\n" + "="*50)
print("PIPELINE EVALUATION")
print("Stage 1: Qwen2-Audio ASR  →  Stage 2: Qwen2.5-Math LLM  →  LaTeX")
print("="*50 + "\n")

def detect_hallucination(pred: str, gt: str):
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred.lower())
    g_toks = tok_pat.findall(gt.lower())
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"

# ---------- STAGE 1: ASR with Qwen2-Audio (voice chat mode) ----------
print(f"Loading ASR model: {ASR_MODEL_NAME}")
asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME, trust_remote_code=True)
asr_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    ASR_MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16 if fp16 and torch.cuda.is_available() else torch.float32,
)
asr_model.eval()

# Cast audio column to the sampling rate expected by the ASR feature extractor
target_sr = asr_processor.feature_extractor.sampling_rate
eval_data_full = dataset["test"]
eval_data_full = eval_data_full.cast_column("audio", Audio(sampling_rate=target_sr))
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

print(f"Transcribing {len(eval_data_full)} audio samples, Stage 1 ASR")
transcriptions = []
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"  ASR Progress: {i}/{len(eval_data_full)}")

    ex = eval_data_full[i]
    audio_array = np.array(ex["audio"]["array"], dtype=np.float32)

    # Audio analysis mode: provide audio + text instruction for transcription.
    # Voice chat mode (audio-only) causes the model to RESPOND to speech.
    # The instruction is kept terse to minimize wrapper text in the output.
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "audio.mp3"},
            {"type": "text", "text": "What does the speaker say? Reply with only their words."},
        ]},
    ]
    text_prompt = asr_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )

    try:
        with torch.no_grad():
            inputs = asr_processor(
                text=text_prompt, audios=[audio_array],
                return_tensors="pt", padding=True,
            )
            inputs = inputs.to(device)

            generated_ids = asr_model.generate(**inputs, max_new_tokens=256)
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            transcription = asr_processor.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        raw = transcription.strip()
        # Strip common wrapper patterns from the ASR output:
        # "The speaker says: '...'" or "The exact transcription is: '...'"
        cleaned = re.sub(
            r"^(?:The\s+)?(?:speaker|person|exact\s+transcription|audio)\s+"
            r"(?:says?|said|is|reads?)[:\s]+['\"]?",
            "", raw, flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip("'\"").strip()
        transcriptions.append(cleaned if cleaned else raw)
    except Exception as e:
        print(f"  ASR failed for sample {i}: {e}")
        transcriptions.append("")

print(f"ASR complete. Transcribed {len(transcriptions)} samples.\n")

asr_model.cpu()
del asr_model, asr_processor
torch.cuda.empty_cache()
gc.collect()
print("ASR model freed from GPU.\n")

# ---------- STAGE 2: LLM Inference (base vs fine-tuned) ----------
print("Loading base LLM for comparison...")
base_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME, device_map="auto", trust_remote_code=True,
    dtype=torch.float16 if fp16 else torch.float32,
)
base_model.eval()

print("Loading fine-tuned LLM...")
ft_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_MODEL_DIR, device_map="auto", trust_remote_code=True,
    dtype=torch.float16 if fp16 else torch.float32,
)
ft_model.eval()

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def generate_prediction(llm_model, tokenizer, chat_prompt, device):
    """Generate LaTeX from a ChatML-formatted prompt."""
    llm_model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            chat_prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)

        outputs = llm_model.generate(
            **encoded,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Slice off the prompt tokens to get only the generated response
        generated_ids = outputs[0][encoded["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return normalize_latex(response)

print(f"\nRunning Stage 2 (LLM) on {len(eval_data_full)} examples...")
rows = []
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"  LLM Progress: {i}/{len(eval_data_full)}")

    ex = eval_data_full[i]
    gt = ex["text_norm"]
    asr_text = transcriptions[i]
    spoken_english = ex.get("spoken_English", "")
    context_before = ex.get("context_before", "")
    context_after = ex.get("context_after", "")

    # Build ChatML prompt with ASR transcription replacing spoken_English
    chat_prompt = build_chat_prompt(context_before, asr_text, context_after)

    pred_base = generate_prediction(base_model, tokenizer, chat_prompt, device)
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt])
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt])
    hall_base = detect_hallucination(pred_base, gt)

    pred_ft = generate_prediction(ft_model, tokenizer, chat_prompt, device)
    wer_ft = wer_metric.compute(predictions=[pred_ft], references=[gt])
    cer_ft = cer_metric.compute(predictions=[pred_ft], references=[gt])
    hall_ft = detect_hallucination(pred_ft, gt)

    rows.append({
        "id": i,
        "spoken_English": spoken_english,
        "ASR_Transcription": asr_text,
        "GT_LaTeX": gt,
        "Base_Pred_LaTeX": pred_base,
        "FineTuned_Pred_LaTeX": pred_ft,
        "Base_WER": wer_base,
        "Base_CER": cer_base,
        "Base_Hallucination": hall_base,
        "FineTuned_WER": wer_ft,
        "FineTuned_CER": cer_ft,
        "FineTuned_Hallucination": hall_ft,
        "WER_Improvement": wer_base - wer_ft,
        "CER_Improvement": cer_base - cer_ft,
    })

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

# -------------------- SUMMARY -------------------- #
print("\n" + "="*50)
print("PIPELINE RESULTS SUMMARY")
print("Stage 1: Qwen2-Audio ASR  →  Stage 2: Qwen2.5-Math LLM  →  LaTeX")
print("="*50)

print(f"\nBase LLM Pipeline Performance:")
print(f"  Average WER: {df['Base_WER'].mean():.4f}")
print(f"  Average CER: {df['Base_CER'].mean():.4f}")

print(f"\nFine-Tuned LLM Pipeline Performance:")
print(f"  Average WER: {df['FineTuned_WER'].mean():.4f}")
print(f"  Average CER: {df['FineTuned_CER'].mean():.4f}")

base_wer_mean = max(df['Base_WER'].mean(), 1e-8)
base_cer_mean = max(df['Base_CER'].mean(), 1e-8)
print(f"\nImprovement:")
print(f"  WER Reduction: {df['WER_Improvement'].mean():.4f} ({df['WER_Improvement'].mean() / base_wer_mean * 100:.1f}%)")
print(f"  CER Reduction: {df['CER_Improvement'].mean():.4f} ({df['CER_Improvement'].mean() / base_cer_mean * 100:.1f}%)")
print(f"\n% of samples improved: {(df['WER_Improvement'] > 0).sum() / len(df) * 100:.1f}%")
print("="*50 + "\n")
