"""
granite_finetune.py
Fine-tuning of IBM Granite Speech model for math formula transcription.
Includes validation, early stopping, and best practices for maximum performance.
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from evaluate import load
from dataclasses import dataclass
from typing import Dict, List, Union

os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "ibm-granite/granite-speech-3.3-8b"
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
MAX_SAMPLES = None
OUTPUT_XLSX = "../results/granite_math_outputs.xlsx"
OUTPUT_MODEL_DIR = "../models/granite_math_best"

# Training hyperparameters
# Granite is a large 8B model, need smaller batch size
PER_DEVICE_BATCH = 4  # Smaller batch for 8B model
GRAD_ACCUM = 8  # Effective batch size = 4 * 8 = 32
fp16 = True
NUM_EPOCHS = 20
LR = 5e-6  # Lower LR for large model
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1.0

# Evaluation settings
steps_per_epoch = 100  # Approximate
EVAL_STEPS = max(50, steps_per_epoch // 4)
SAVE_STEPS = EVAL_STEPS
EARLY_STOPPING_PATIENCE = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------- LOAD DATASET -------------------- #
print(f"Loading dataset {DATASET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)
dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

# Split into train and validation (80/20 split)
dataset = dataset.train_test_split(test_size=0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

# -------------------- NORMALIZE LATEX -------------------- #
def normalize_latex(tex: str) -> str:
    """Normalize LaTeX so x^2, x^{2}, and x^{ 2 } are all the same."""
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

# -------------------- MODEL & PROCESSOR -------------------- #
print(f"Loading Granite Speech model: {MODEL_NAME}")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if fp16 and torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# Disable cache for training
model.config.use_cache = False

print(f"Model loaded. Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -------------------- PREPROCESS FUNCTION -------------------- #
def prepare_example(ex):
    """Prepare audio and labels for training."""
    audio_array = ex[AUDIO_COLUMN]["array"]
    sampling_rate = ex[AUDIO_COLUMN]["sampling_rate"]
    
    # Process audio
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )
    
    # Tokenize labels
    labels = processor.tokenizer(ex["text_norm"], return_tensors="pt").input_ids
    
    return {
        "input_features": inputs.input_features[0] if hasattr(inputs, 'input_features') else inputs.input_values[0],
        "labels": labels[0].tolist()
    }

# Map datasets
print("Preprocessing datasets...")
remove_cols = [col for col in train_dataset.column_names if col not in ["input_features", "labels"]]
train_dataset = train_dataset.map(prepare_example, remove_columns=remove_cols)
eval_dataset = eval_dataset.map(prepare_example, remove_columns=remove_cols)

print(f"Preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# -------------------- DATA COLLATOR -------------------- #
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Stack input features
        input_features = torch.stack([torch.tensor(f["input_features"]) for f in features])
        
        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        return {
            "input_features": input_features,
            "labels": labels
        }

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# -------------------- METRICS -------------------- #
wer_metric = load("wer")
cer_metric = load("cer")

def compute_metrics(pred):
    """Compute WER and CER metrics."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize
    pred_str = [normalize_latex(p) for p in pred_str]
    label_str = [normalize_latex(l) for l in label_str]
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "cer": cer}

# -------------------- TRAINING -------------------- #
# Recalculate steps based on actual dataset size
steps_per_epoch = len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM)
EVAL_STEPS = max(50, steps_per_epoch // 4)
SAVE_STEPS = EVAL_STEPS

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
    fp16=fp16 and torch.cuda.is_available(),
    gradient_checkpointing=True,
    
    # Evaluation and saving
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    # Logging
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",
    
    # Generation settings
    predict_with_generate=True,
    generation_max_length=225,
    
    # Other settings
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("\n" + "=" * 50)
print("Starting optimized fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Learning rate: {LR}")
print(f"Eval steps: {EVAL_STEPS}")
print("=" * 50 + "\n")

trainer.train()
trainer.save_model(OUTPUT_MODEL_DIR)
processor.save_pretrained(OUTPUT_MODEL_DIR)
print(f"\nBest model saved to {OUTPUT_MODEL_DIR}")

# -------------------- FINAL EVALUATION -------------------- #
print("\n" + "=" * 50)
print("Running final evaluation on validation set...")
print("=" * 50 + "\n")

def detect_hallucination(pred: str, gt: str):
    """Find tokens in prediction that weren't in ground truth."""
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"

def generate_prediction(model, processor, audio_array, sampling_rate, device):
    """Generate prediction from model."""
    model.eval()
    with torch.no_grad():
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    return normalize_latex(transcription)

# Load base model for comparison
print("Loading base model for comparison...")
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if fp16 and torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()

# Get device
model_device = next(model.parameters()).device

# Reload evaluation data with original fields
eval_data_full = dataset["test"]
eval_data_full = eval_data_full.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

model.eval()
rows = []

print(f"Evaluating {len(eval_data_full)} examples")
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_full)}")
    
    ex = eval_data_full[i]
    gt = ex["text_norm"]
    audio_array = ex[AUDIO_COLUMN]["array"]
    sampling_rate = ex[AUDIO_COLUMN]["sampling_rate"]
    audio_path = ex[AUDIO_COLUMN].get("path", "unknown")
    
    # Base model prediction
    pred_base = generate_prediction(base_model, processor, audio_array, sampling_rate, model_device)
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt])
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt])
    hall_base = detect_hallucination(pred_base, gt)
    
    # Fine-tuned model prediction
    pred_ft = generate_prediction(model, processor, audio_array, sampling_rate, model_device)
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
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
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
print("=" * 50 + "\n")
