"""
whisper_mathbert_pipeline.py
Two-stage pipeline for math audio transcription:
1) Base Whisper ASR converts audio -> spoken text
2) MATHBert (encoder-decoder) converts spoken text -> LaTeX

The script combines both datasets:
- abby1492/mathbridge-audio
- OwenLeitzell/FormulaSearch

It fine-tunes MATHBert on spoken text -> LaTeX pairs, then evaluates
pipeline output (Whisper transcription -> MATHBert LaTeX) against ground truth.
"""

import os
import re
import gc
import inspect
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from datasets import Audio, concatenate_datasets, load_dataset
from evaluate import load
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
# Stage 1: ASR (base Whisper only, no fine-tuning)
ASR_MODEL_NAME = "openai/whisper-base"

# Stage 2: text -> LaTeX (MATHBert)
# Common public checkpoint used as MATHBert.
MATHBERT_MODEL_NAME = "tbs17/MathBERT"

DATASET1_NAME = "abby1492/mathbridge-audio"
DATASET2_NAME = "OwenLeitzell/FormulaSearch"
SPLIT = "train"
DATASET2_SPLIT = "train"

AUDIO_COLUMN = "audio"
OUTPUT_MODEL_DIR = "./mathbert_latex_best"
OUTPUT_XLSX = "whisper_mathbert_pipeline_outputs.xlsx"
MAX_SAMPLES = None

# MATHBert fine-tuning args
PER_DEVICE_BATCH = 16
GRAD_ACCUM = 2
NUM_EPOCHS = 12
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -------------------- HELPERS -------------------- #
def normalize_latex(tex: str) -> str:
    if tex is None:
        return ""
    s = str(tex).replace("$", "").replace("\\,", ",").replace("\\;", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"_\{\s*([^\}]+)\s*\}", r"_\1", s)
    s = re.sub(r"\^\{\s*([^\}]+)\s*\}", r"^\1", s)
    s = re.sub(r"(?<!\\){\s*([A-Za-z0-9_\\]+)\s*}", r"\1", s)
    return s.strip()


def keep_required_columns(ds, required: List[str]):
    remove_cols = [c for c in ds.column_names if c not in required]
    return ds.remove_columns(remove_cols) if remove_cols else ds


def build_spoken_input(example: Dict) -> str:
    """
    Build the source text for MATHBert training/inference.
    Prefer spoken_English when available; otherwise reconstruct from context fields.
    """
    spoken = str(example.get("spoken_English", "") or "").strip()
    if spoken:
        return spoken

    before = str(example.get("context_before", "") or "").strip()
    after = str(example.get("context_after", "") or "").strip()
    merged = f"{before} {after}".strip()
    return re.sub(r"\s+", " ", merged)


def detect_hallucination(pred: str, gt: str) -> str:
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall((pred or "").lower())
    g_toks = tok_pat.findall((gt or "").lower())
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"


# -------------------- DATASET -------------------- #
required_columns = [AUDIO_COLUMN, "equation", "spoken_English", "context_before", "context_after"]

print(f"Loading datasets: {DATASET1_NAME} ({SPLIT}) and {DATASET2_NAME} ({DATASET2_SPLIT})")
dataset1 = load_dataset(DATASET1_NAME, split=SPLIT)
dataset2 = load_dataset(DATASET2_NAME, split=DATASET2_SPLIT)

dataset1 = keep_required_columns(dataset1, required_columns)
dataset2 = keep_required_columns(dataset2, required_columns)

dataset1 = dataset1.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
dataset2 = dataset2.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
dataset = concatenate_datasets([dataset1, dataset2])

dataset = dataset.train_test_split(test_size=0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES, len(eval_dataset))))

train_dataset = train_dataset.map(
    lambda ex: {
        "text_norm": normalize_latex(ex.get("equation", "")),
        "spoken_input": build_spoken_input(ex),
    }
)
eval_dataset = eval_dataset.map(
    lambda ex: {
        "text_norm": normalize_latex(ex.get("equation", "")),
        "spoken_input": build_spoken_input(ex),
    }
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")


# -------------------- STAGE 2 TRAINING (MATHBERT) -------------------- #
print("\n" + "=" * 50)
print("STAGE 2: Fine-tuning MATHBert (spoken text -> LaTeX)")
print("=" * 50)

tokenizer = BertTokenizer.from_pretrained(MATHBERT_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.sep_token
# BERT tokenizers usually do not define EOS; use SEP as EOS for seq2seq generation.
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    MATHBERT_MODEL_NAME,
    MATHBERT_MODEL_NAME,
)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
# Keep generation-only settings in generation_config (required by newer transformers).
model.generation_config.max_length = MAX_TARGET_LEN
model.generation_config.min_length = 1
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.early_stopping = True
model.generation_config.length_penalty = 1.0
model.generation_config.num_beams = 4
model.generation_config.eos_token_id = tokenizer.sep_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id


def preprocess_text_pair(batch):
    inputs = tokenizer(
        batch["spoken_input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SOURCE_LEN,
    )
    targets = tokenizer(
        text_target=batch["text_norm"],
        truncation=True,
        padding="max_length",
        max_length=MAX_TARGET_LEN,
    )
    labels = []
    for seq in targets["input_ids"]:
        labels.append([tok if tok != tokenizer.pad_token_id else -100 for tok in seq])
    inputs["labels"] = labels
    return inputs


train_ft = train_dataset.map(
    preprocess_text_pair,
    batched=True,
    remove_columns=train_dataset.column_names,
)
eval_ft = eval_dataset.map(
    preprocess_text_pair,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

steps_per_epoch = max(1, len(train_ft) // (PER_DEVICE_BATCH * GRAD_ACCUM))
eval_steps = max(50, steps_per_epoch // 2)
total_train_steps = max(1, steps_per_epoch * NUM_EPOCHS)
warmup_steps = max(1, int(total_train_steps * WARMUP_RATIO))

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=25,
    logging_first_step=True,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    report_to="tensorboard",
)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_ft,
    "eval_dataset": eval_ft,
}
trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
if "processing_class" in trainer_init_params:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_init_params:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Seq2SeqTrainer(**trainer_kwargs)

trainer.train()
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"Fine-tuned MATHBert model saved to {OUTPUT_MODEL_DIR}")

del trainer
model.cpu()
del model
torch.cuda.empty_cache()
gc.collect()


# -------------------- STAGE 1 INFERENCE (BASE WHISPER) -------------------- #
print("\n" + "=" * 50)
print("STAGE 1: Whisper base ASR")
print("=" * 50)

asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME).to(device)
asr_model.generation_config.forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="en", task="transcribe")
asr_model.generation_config.suppress_tokens = []
asr_model.eval()

eval_audio = eval_dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
transcriptions = []
print(f"Transcribing {len(eval_audio)} samples with Whisper base...")
for i in range(len(eval_audio)):
    if i % 20 == 0:
        print(f"  ASR progress: {i}/{len(eval_audio)}")
    ex = eval_audio[i]
    audio_array = np.array(ex[AUDIO_COLUMN]["array"], dtype=np.float32)
    inputs = asr_processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        pred_ids = asr_model.generate(
            input_features=input_features,
            max_length=128,
            num_beams=4,
        )
    text = asr_processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    transcriptions.append(re.sub(r"\s+", " ", text))

asr_model.cpu()
del asr_model, asr_processor
torch.cuda.empty_cache()
gc.collect()
print("Whisper ASR complete.")


# -------------------- STAGE 2 INFERENCE (MATHBERT BASE VS FT) -------------------- #
print("\n" + "=" * 50)
print("STAGE 2: MATHBert text -> LaTeX inference")
print("=" * 50)

wer_metric = load("wer")
cer_metric = load("cer")

base_tokenizer = BertTokenizer.from_pretrained(MATHBERT_MODEL_NAME)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.sep_token
if base_tokenizer.eos_token is None:
    base_tokenizer.eos_token = base_tokenizer.sep_token
base_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    MATHBERT_MODEL_NAME,
    MATHBERT_MODEL_NAME,
).to(device)
base_model.config.decoder_start_token_id = base_tokenizer.cls_token_id
base_model.config.eos_token_id = base_tokenizer.sep_token_id
base_model.config.pad_token_id = base_tokenizer.pad_token_id
base_model.generation_config.decoder_start_token_id = base_tokenizer.cls_token_id
base_model.generation_config.bos_token_id = base_tokenizer.cls_token_id
base_model.generation_config.eos_token_id = base_tokenizer.sep_token_id
base_model.generation_config.pad_token_id = base_tokenizer.pad_token_id
base_model.eval()

ft_tokenizer = BertTokenizer.from_pretrained(OUTPUT_MODEL_DIR)
if ft_tokenizer.pad_token is None:
    ft_tokenizer.pad_token = ft_tokenizer.sep_token
if ft_tokenizer.eos_token is None:
    ft_tokenizer.eos_token = ft_tokenizer.sep_token
ft_model = EncoderDecoderModel.from_pretrained(OUTPUT_MODEL_DIR).to(device)
ft_model.config.decoder_start_token_id = ft_tokenizer.cls_token_id
ft_model.config.eos_token_id = ft_tokenizer.sep_token_id
ft_model.config.pad_token_id = ft_tokenizer.pad_token_id
ft_model.generation_config.decoder_start_token_id = ft_tokenizer.cls_token_id
ft_model.generation_config.bos_token_id = ft_tokenizer.cls_token_id
ft_model.generation_config.eos_token_id = ft_tokenizer.sep_token_id
ft_model.generation_config.pad_token_id = ft_tokenizer.pad_token_id
ft_model.eval()


def generate_latex(seq2seq_model, seq2seq_tokenizer, src_text: str) -> str:
    if not src_text:
        return ""
    encoded = seq2seq_tokenizer(
        src_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_SOURCE_LEN,
    ).to(device)
    with torch.no_grad():
        out_ids = seq2seq_model.generate(
            **encoded,
            max_length=MAX_TARGET_LEN,
            num_beams=4,
            early_stopping=True,
        )
    pred = seq2seq_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return normalize_latex(pred)


rows = []
print(f"Evaluating pipeline on {len(eval_audio)} validation samples...")
for i in range(len(eval_audio)):
    if i % 20 == 0:
        print(f"  Pipeline progress: {i}/{len(eval_audio)}")

    ex = eval_audio[i]
    gt = ex["text_norm"]
    asr_text = transcriptions[i]
    audio_path = ex[AUDIO_COLUMN].get("path", "unknown")
    spoken_ref = ex.get("spoken_input", "")

    pred_base = generate_latex(base_model, base_tokenizer, asr_text)
    pred_ft = generate_latex(ft_model, ft_tokenizer, asr_text)

    base_wer = wer_metric.compute(predictions=[pred_base], references=[gt])
    base_cer = cer_metric.compute(predictions=[pred_base], references=[gt])
    ft_wer = wer_metric.compute(predictions=[pred_ft], references=[gt])
    ft_cer = cer_metric.compute(predictions=[pred_ft], references=[gt])

    rows.append(
        {
            "id": i,
            "audio_file": audio_path,
            "spoken_English_ref": spoken_ref,
            "ASR_Transcription": asr_text,
            "GT_LaTeX": gt,
            "BaseMathBert_Pred_LaTeX": pred_base,
            "FineTunedMathBert_Pred_LaTeX": pred_ft,
            "Base_WER": base_wer,
            "Base_CER": base_cer,
            "Base_Hallucination": detect_hallucination(pred_base, gt),
            "FineTuned_WER": ft_wer,
            "FineTuned_CER": ft_cer,
            "FineTuned_Hallucination": detect_hallucination(pred_ft, gt),
            "WER_Improvement": base_wer - ft_wer,
            "CER_Improvement": base_cer - ft_cer,
        }
    )

base_model.cpu()
ft_model.cpu()
del base_model, ft_model
torch.cuda.empty_cache()
gc.collect()

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed pipeline outputs saved to {OUTPUT_XLSX}")

print("\n" + "=" * 50)
print("PIPELINE RESULTS SUMMARY")
print("=" * 50)
print(f"Base MATHBert WER: {df['Base_WER'].mean():.4f}")
print(f"Base MATHBert CER: {df['Base_CER'].mean():.4f}")
print(f"Fine-tuned MATHBert WER: {df['FineTuned_WER'].mean():.4f}")
print(f"Fine-tuned MATHBert CER: {df['FineTuned_CER'].mean():.4f}")
print(f"Average WER improvement: {df['WER_Improvement'].mean():.4f}")
print(f"Average CER improvement: {df['CER_Improvement'].mean():.4f}")
print("=" * 50 + "\n")
