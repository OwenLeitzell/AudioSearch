#!/usr/bin/env python3
"""
whisper_math_finetune_optimized.py
Optimized fine-tuning of Whisper for math formula transcription.
Includes validation, early stopping, and best practices for maximum performance.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from evaluate import load
from dataclasses import dataclass
from typing import Dict, List, Union
import librosa

# Disable W&B if you don't want it
os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "openai/whisper-large-v3"  # or "openai/whisper-medium" for faster training
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
MAX_SAMPLES = None  # Use full dataset for best results
OUTPUT_XLSX = "whisper_math_turing.xlsx"
OUTPUT_MODEL_DIR = "./whisper_math_best"

# Optimized training hyperparameters
PER_DEVICE_BATCH = 1  # Increase if you have more memory
GRAD_ACCUM = 16  # Effective batch size = 2 * 8 = 16
NUM_EPOCHS = 5  # More epochs for better learning
LR = 3e-5  # Higher learning rate for faster convergence
WARMUP_RATIO = 0.1  # Warm up for 10% of training
WEIGHT_DECAY = 0.01  # Regularization to prevent overfitting
MAX_GRAD_NORM = 1.0  # Gradient clipping

# Evaluation settings
EVAL_STEPS = 500  # Evaluate every 500 steps
SAVE_STEPS = 500  # Save checkpoint every 500 steps
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 evaluations

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------- LOAD DATASET -------------------- #
print(f"Loading dataset {DATASET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)

# Split into train and validation (90/10 split)
dataset = load_dataset(DATASET_NAME, split=SPLIT)
dataset = dataset.train_test_split(test_size = 0.1, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

#train_dataset = train_dataset.cast_column(AUDIO_COLUMN, None)
#eval_dataset = eval_dataset.cast_column(AUDIO_COLUMN, None)

# Normalize LaTeX
def normalize_latex(tex: str) -> str:
    if tex is None: return ""
    s = str(tex).replace("$", "").replace("\\,", ",").replace("\\;", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"_\{\s*([^\}]+)\s*\}", r"_\1", s)
    s = re.sub(r"\^\{\s*([^\}]+)\s*\}", r"^\1", s)
    s = re.sub(r"(?<!\\){\s*([A-Za-z0-9_\\]+)\s*}", r"\1", s)
    return s

train_dataset = train_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})
eval_dataset = eval_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# -------------------- MODEL & TOKENIZER -------------------- #
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tk = processor.tokenizer

# Extra tokens for math + Greek
extra_tokens = ["{","}","^","_","/","*","[","]","(",")","+/-","\\pm","\\in","\\Sigma","\\Gamma",
                "\\alpha","\\beta","\\gamma","\\delta","\\epsilon","\\zeta","\\eta","\\theta",
                "\\lambda","\\mu","\\pi","\\rho","\\sigma","\\tau","\\phi","\\chi","\\psi","\\omega",
                "\\mathcal","\\mathbf","\\boldsymbol","\\bar","\\prime","\\sqrt","\\frac","\\sum",
                "\\int","\\partial","\\nabla","\\infty","\\leq","\\geq","\\neq","\\approx","\\Phi","\\Pi"]
extra_tokens = [t for t in dict.fromkeys(extra_tokens) if t not in tk.get_vocab()]
if extra_tokens:
    tk.add_tokens(extra_tokens)
    print(f"Added {len(extra_tokens)} special tokens")

# Load Whisper model
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tk))
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
model.config.suppress_tokens = []
model.config.use_cache = False  # Disable for training

print(f"Model loaded. Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -------------------- PREPROCESS FUNCTION -------------------- #
def prepare_example(ex):
    # Use torchaudio instead of librosa
    audio_path = ex[AUDIO_COLUMN]["path"] if isinstance(ex[AUDIO_COLUMN], dict) else ex[AUDIO_COLUMN]
    arr, sr = torchaudio.load(audio_path)  # Works for MP3s natively
    arr = arr.mean(dim=0).numpy()  # Convert to mono + numpy array
    if sr != 16000:
        arr = torchaudio.functional.resample(torch.tensor(arr), sr, 16000).numpy()
    inputs = processor(arr, sampling_rate=16000, return_tensors="pt")
    labels = tk(ex["text_norm"]).input_ids
    return {
        "input_features": inputs.input_features[0],
        "labels": labels
    }

# Map datasets without casting to Audio
remove_cols = [col for col in train_dataset.column_names if col not in ["input_features", "labels", "text_norm"]]
train_dataset = train_dataset.map(prepare_example, remove_columns=remove_cols)
eval_dataset = eval_dataset.map(prepare_example, remove_columns=remove_cols)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Stack input features
        input_features = torch.stack([torch.tensor(f["input_features"]) for f in features])
        
        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Remove BOS token if present (model adds it)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        return {
            "input_features": input_features,
            "labels": labels
        }

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# -------------------- METRICS -------------------- #
wer_metric = load("wer")
cer_metric = load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = tk.pad_token_id
    
    # Decode predictions and labels
    pred_str = tk.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tk.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize
    pred_str = [normalize_latex(p) for p in pred_str]
    label_str = [normalize_latex(l) for l in label_str]
    
    # Compute metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "cer": cer}

# -------------------- TRAINING -------------------- #
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    
    # Evaluation and saving
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,  # Keep only best 3 checkpoints
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="wer",  # Use WER to determine best model
    greater_is_better=False,  # Lower WER is better
    
    # Logging
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",  # Use tensorboard for monitoring
    
    # Generation settings for evaluation
    predict_with_generate=True,
    generation_max_length=225,
    
    # Other settings
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",  # Use PyTorch AdamW
    lr_scheduler_type="cosine",  # Cosine learning rate schedule
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001  # Minimum improvement threshold
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("\n" + "="*50)
print("Starting optimized fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Total training steps: {len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM) * NUM_EPOCHS}")
print("="*50 + "\n")

trainer.train()
trainer.save_model(OUTPUT_MODEL_DIR)
processor.save_pretrained(OUTPUT_MODEL_DIR)
print(f"\nBest model saved to {OUTPUT_MODEL_DIR}")

# -------------------- FINAL EVALUATION -------------------- #
print("\n" + "="*50)
print("Running final evaluation on validation set...")
print("="*50 + "\n")

def detect_hallucination(pred: str, gt: str):
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"

def generate_prediction(model, input_feat):
    model.eval()
    with torch.no_grad():
        ids = model.generate(
            input_features=torch.tensor(input_feat).unsqueeze(0).to(device),
            max_length=225,
            num_beams=5,  # Use beam search for better results
        )
    return normalize_latex(tk.decode(ids[0], skip_special_tokens=True))

# Load base model for comparison
print("Loading base model for comparison...")
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
base_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

# Load evaluation data with original fields
eval_data_full = load_dataset(DATASET_NAME, split=SPLIT)
eval_data_full = eval_data_full.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

if MAX_SAMPLES:
    eval_data_full = eval_data_full.select(range(min(MAX_SAMPLES // 10, len(eval_data_full))))

# Prepare for prediction
eval_data_processed = eval_data_full.map(
    lambda ex: {"input_features": processor(ex[AUDIO_COLUMN]["array"], sampling_rate=16000).input_features[0]}
)

model.to(device)
rows = []

print(f"Evaluating {len(eval_data_processed)} examples...")
for i in range(len(eval_data_processed)):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_processed)}")
    
    ex_full = eval_data_full[i]
    ex_proc = eval_data_processed[i]
    
    gt = ex_full["text_norm"]
    input_feat = ex_proc["input_features"]
    audio_path = ex_full[AUDIO_COLUMN].get("path", "unknown")

    # Base model prediction
    pred_base = generate_prediction(base_model, input_feat)
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt])
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt])
    hall_base = detect_hallucination(pred_base, gt)

    # Fine-tuned model prediction
    pred_ft = generate_prediction(model, input_feat)
    wer_ft = wer_metric.compute(predictions=[pred_ft], references=[gt])
    cer_ft = cer_metric.compute(predictions=[pred_ft], references=[gt])
    hall_ft = detect_hallucination(pred_ft, gt)

    rows.append({
        "id": i,
        "audio_file": audio_path,
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

# Save detailed results
df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

# Print summary statistics
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"\nBase Model Performance:")
print(f"  Average WER: {df['Base_WER'].mean():.4f}")
print(f"  Average CER: {df['Base_CER'].mean():.4f}")
print(f"\nFine-Tuned Model Performance:")
print(f"  Average WER: {df['FineTuned_WER'].mean():.4f}")
print(f"  Average CER: {df['FineTuned_CER'].mean():.4f}")
print(f"\nImprovement:")
print(f"  WER Reduction: {df['WER_Improvement'].mean():.4f} ({df['WER_Improvement'].mean() / df['Base_WER'].mean() * 100:.1f}%)")
print(f"  CER Reduction: {df['CER_Improvement'].mean():.4f} ({df['CER_Improvement'].mean() / df['Base_CER'].mean() * 100:.1f}%)")
print(f"\n% of samples improved: {(df['WER_Improvement'] > 0).sum() / len(df) * 100:.1f}%")
print("="*50 + "\n")