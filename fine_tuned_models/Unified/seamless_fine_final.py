"""
seamless_fine_final.py
Unified SeamlessM4T fine-tuning pipeline for math audio -> LaTeX extraction.
Mirrors whisper_fine_final.py flow:
1) Combine two datasets into one schema
2) Fine-tune on normalized LaTeX targets
3) Evaluate fine-tuned vs base model in one pass
4) Export per-sample comparison dataframe
"""

import os
import re
import gc
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio, concatenate_datasets
from evaluate import load
from transformers import (
    AutoProcessor,
    SeamlessM4TForSpeechToText,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "facebook/hf-seamless-m4t-medium" #SeamlessM4T-Medium
TARGET_LANG = "eng" #English

DATASET_NAME = "abby1492/mathbridge-audio" #Dataset 1
DATASET2_NAME = "OwenLeitzell/FormulaSearch" #Dataset 2
AUDIO_COLUMN = "audio" #Cast column name for audio
SPLIT = "train" #Split name for dataset 1
DATASET2_SPLIT = "train" #Split name for dataset 2
MAX_SAMPLES = None #Max samples to train on

OUTPUT_XLSX = "seamless_full_outputs.xlsx" #Output Excel file name
OUTPUT_MODEL_DIR = "./seamless_full_best" #Output model directory

PER_DEVICE_BATCH = 8 #Batch size per device 8 samples per batch per GPU (for 50-100GB VRAM on Turing)
PER_DEVICE_EVAL_BATCH = 2 #Smaller eval batch to reduce memory pressure from long sequences
GRAD_ACCUM = 2 #effecctive batch size = 2 * 8 = 16, more GPUs = more effective batch size
NUM_EPOCHS = 20 #epochs
LR = 1e-5 #learning rate
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005 
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5 #Stop training if there is no improvement for 5 epochs
#will use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


#DATASET 
REQUIRED_COLUMNS = [AUDIO_COLUMN, "equation"]

#Keep only the required columns, audio and ground truth LaTeX
def keep_required_columns(ds, required):
    to_remove = [c for c in ds.column_names if c not in required]
    return ds.remove_columns(to_remove) if to_remove else ds


print(f"Loading datasets: {DATASET_NAME} ({SPLIT}) and {DATASET2_NAME} ({DATASET2_SPLIT})")
dataset1 = load_dataset(DATASET_NAME, split=SPLIT)
dataset2 = load_dataset(DATASET2_NAME, split=DATASET2_SPLIT)

dataset1 = keep_required_columns(dataset1, REQUIRED_COLUMNS)
dataset2 = keep_required_columns(dataset2, REQUIRED_COLUMNS)
dataset1 = dataset1.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
dataset2 = dataset2.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.train_test_split(test_size=0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))


# -------------------- NORMALIZATION -------------------- #
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


# -------------------- MODEL / PROCESSOR -------------------- #
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
model = SeamlessM4TForSpeechToText.from_pretrained(MODEL_NAME)
model.config.use_cache = False
print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def prepare_example(ex):
    audio_array = ex[AUDIO_COLUMN]["array"]
    inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")
    labels = tokenizer(ex["text_norm"]).input_ids
    return {
        "input_features": inputs.input_features[0],
        "labels": labels,
    }

remove_cols = [col for col in train_dataset.column_names if col not in ["input_features", "labels"]]
train_dataset = train_dataset.map(prepare_example, remove_columns=remove_cols)
eval_dataset = eval_dataset.map(prepare_example, remove_columns=remove_cols)
print(f"Preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad variable-length acoustic features to batch max length.
        # Input padding should be normal feature padding (zeros), while labels use -100 for loss masking.
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        out = {
            "input_features": batch["input_features"],
            "labels": labels,
        }
        if "attention_mask" in batch:
            out["attention_mask"] = batch["attention_mask"]
        return out


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# -------------------- METRICS -------------------- #
wer_metric = load("wer")
cer_metric = load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    # When predict_with_generate=False, predictions are logits [B, T, V].
    if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    pred_ids[(pred_ids < 0) | (pred_ids >= tokenizer.vocab_size)] = tokenizer.pad_token_id

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [normalize_latex(p) for p in pred_str]
    label_str = [normalize_latex(l) for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


# -------------------- TRAINING -------------------- #
steps_per_epoch = max(1, len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM))
total_training_steps = max(1, steps_per_epoch * NUM_EPOCHS)
TARGET_EVAL_RUNS = 5
# Use ceil division so eval runs are <= TARGET_EVAL_RUNS across training.
EVAL_STEPS = max(1, (total_training_steps + TARGET_EVAL_RUNS - 1) // TARGET_EVAL_RUNS)
SAVE_STEPS = EVAL_STEPS

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    eval_accumulation_steps=1,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",
    predict_with_generate=False,
    prediction_loss_only=True,
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("\n" + "=" * 50)
print("Starting SeamlessM4T fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Total training steps: {len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM) * NUM_EPOCHS}")
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
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"


def generate_prediction(m, audio_array):
    m.eval()
    with torch.no_grad():
        inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt").to(device)
        ids = m.generate(
            **inputs,
            tgt_lang=TARGET_LANG,
            max_length=225,
            num_beams=5,
        )
    return normalize_latex(tokenizer.decode(ids[0], skip_special_tokens=True))


eval_data_full = dataset["test"]
eval_data_full = eval_data_full.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# ---------- PASS 1: Fine-tuned model ----------
model.to(device)
model.eval()
ft_results = []
print(f"[Fine-tuned] Evaluating {len(eval_data_full)} examples")
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(eval_data_full)}")
    ex_full = eval_data_full[i]
    gt = ex_full["text_norm"]
    audio_path = ex_full[AUDIO_COLUMN].get("path", "unknown")
    audio_array = ex_full[AUDIO_COLUMN]["array"]

    pred_ft = generate_prediction(model, audio_array)
    ft_results.append(
        {
            "id": i,
            "audio_file": audio_path,
            "GT_LaTeX": gt,
            "FineTuned_Pred_LaTeX": pred_ft,
            "FineTuned_WER": wer_metric.compute(predictions=[pred_ft], references=[gt]),
            "FineTuned_CER": cer_metric.compute(predictions=[pred_ft], references=[gt]),
            "FineTuned_Hallucination": detect_hallucination(pred_ft, gt),
        }
    )
print("[Fine-tuned] Done.")

# Free fine-tuned model from GPU before loading base
model.cpu()
del model
del trainer
torch.cuda.empty_cache()
gc.collect()
print("Fine-tuned model freed from GPU.\n")

# ---------- PASS 2: Base model ----------
print(f"Loading base model {MODEL_NAME} for comparison...")
base_model = SeamlessM4TForSpeechToText.from_pretrained(MODEL_NAME).to(device)
base_model.eval()
base_results = []
print(f"[Base] Evaluating {len(eval_data_full)} examples")
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(eval_data_full)}")
    ex_full = eval_data_full[i]
    gt = ex_full["text_norm"]
    audio_array = ex_full[AUDIO_COLUMN]["array"]

    pred_base = generate_prediction(base_model, audio_array)
    base_results.append(
        {
            "Base_Pred_LaTeX": pred_base,
            "Base_WER": wer_metric.compute(predictions=[pred_base], references=[gt]),
            "Base_CER": cer_metric.compute(predictions=[pred_base], references=[gt]),
            "Base_Hallucination": detect_hallucination(pred_base, gt),
        }
    )
print("[Base] Done.")

base_model.cpu()
del base_model
torch.cuda.empty_cache()
gc.collect()

# ---------- Merge results ----------
rows = []
for ft, base in zip(ft_results, base_results):
    row = {**ft, **base}
    row["WER_Improvement"] = row["Base_WER"] - row["FineTuned_WER"]
    row["CER_Improvement"] = row["Base_CER"] - row["FineTuned_CER"]
    rows.append(row)

col_order = [
    "id",
    "audio_file",
    "GT_LaTeX",
    "Base_Pred_LaTeX",
    "Base_WER",
    "Base_CER",
    "Base_Hallucination",
    "FineTuned_Pred_LaTeX",
    "FineTuned_WER",
    "FineTuned_CER",
    "FineTuned_Hallucination",
    "WER_Improvement",
    "CER_Improvement",
]

df = pd.DataFrame(rows)[col_order]
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"\nBase Model Performance:")
print(f"  Average WER: {df['Base_WER'].mean():.4f}")
print(f"  Average CER: {df['Base_CER'].mean():.4f}")
print(f"\nFine-Tuned Model Performance:")
print(f"  Average WER: {df['FineTuned_WER'].mean():.4f}")
print(f"  Average CER: {df['FineTuned_CER'].mean():.4f}")

base_wer_mean = max(df["Base_WER"].mean(), 1e-8)
base_cer_mean = max(df["Base_CER"].mean(), 1e-8)
print(f"\nImprovement:")
print(f"  WER Reduction: {df['WER_Improvement'].mean():.4f} ({df['WER_Improvement'].mean() / base_wer_mean * 100:.1f}%)")
print(f"  CER Reduction: {df['CER_Improvement'].mean():.4f} ({df['CER_Improvement'].mean() / base_cer_mean * 100:.1f}%)")
print(f"\n% of samples improved: {(df['WER_Improvement'] > 0).sum() / len(df) * 100:.1f}%")
print("=" * 50 + "\n")

