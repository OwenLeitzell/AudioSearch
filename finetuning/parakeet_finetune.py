"""
parakeet_fine_final.py
Fine-tuning of NVIDIA's Parakeet CTC model for math formula transcription.
Includes validation, early stopping, and best practices for maximum performance.
Similar structure to whisper_fine_final.py, but adapted for NeMo's ASR training framework.
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from datasets import load_dataset, Audio
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from evaluate import load
import json
import tempfile
import soundfile as sf

# Disable wandb for account log in and verification
os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "nvidia/parakeet-ctc-1.1b"  # Options: parakeet-ctc-1.1b, parakeet-ctc-0.6b
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
MAX_SAMPLES = None
OUTPUT_XLSX = "../results/parakeet_math_outputs.xlsx"
OUTPUT_MODEL_DIR = "../models/parakeet_math_best"

# Training hyperparameters
# Parakeet uses audio features similar to Whisper, so similar batch size constraints
PER_DEVICE_BATCH = 16  # 16 samples per batch per GPU (for 50-100GB VRAM)
GRAD_ACCUM = 2  # Effective batch size = 2 * 16 = 32
NUM_EPOCHS = 20  # More epochs for better learning
LR = 1e-5  # Conservative learning rate for fine-tuning
WARMUP_RATIO = 0.1  # Warm up for 10% of training
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1.0

# Evaluation settings
EARLY_STOPPING_PATIENCE = 5

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------- LOAD DATASET -------------------- #
print(f"Loading dataset {DATASET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)
# 16000 is Parakeet's required sample rate
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

# Apply normalization to dataset
train_dataset = train_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})
eval_dataset = eval_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# -------------------- PREPARE NeMo MANIFEST FILES -------------------- #
def create_nemo_manifest(dataset_split, manifest_path, temp_audio_dir):
    """
    Create NeMo-compatible manifest file from HuggingFace dataset.
    NeMo requires a JSON manifest with audio_filepath, text, and duration.
    """
    os.makedirs(temp_audio_dir, exist_ok=True)
    
    manifest_entries = []
    for idx, example in enumerate(dataset_split):
        # Save audio to temporary file (NeMo needs file paths)
        audio_array = example[AUDIO_COLUMN]["array"]
        sample_rate = example[AUDIO_COLUMN]["sampling_rate"]
        
        audio_path = os.path.join(temp_audio_dir, f"audio_{idx}.wav")
        sf.write(audio_path, audio_array, sample_rate)
        
        # Get duration
        duration = len(audio_array) / sample_rate
        
        # Get transcription (using spoken_english for ASR task)
        text = example.get("spoken_english", example.get("text_norm", ""))
        
        manifest_entries.append({
            "audio_filepath": audio_path,
            "text": text,
            "duration": duration
        })
    
    # Write manifest file
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
    
    return manifest_path

# Create temporary directories for audio files and manifests
TEMP_DIR = tempfile.mkdtemp(prefix="parakeet_finetune_")
TRAIN_MANIFEST = os.path.join(TEMP_DIR, "train_manifest.json")
EVAL_MANIFEST = os.path.join(TEMP_DIR, "eval_manifest.json")
TRAIN_AUDIO_DIR = os.path.join(TEMP_DIR, "train_audio")
EVAL_AUDIO_DIR = os.path.join(TEMP_DIR, "eval_audio")

print("Creating NeMo manifest files...")
create_nemo_manifest(train_dataset, TRAIN_MANIFEST, TRAIN_AUDIO_DIR)
create_nemo_manifest(eval_dataset, EVAL_MANIFEST, EVAL_AUDIO_DIR)
print(f"Manifests created at {TEMP_DIR}")

# -------------------- LOAD MODEL -------------------- #
print(f"Loading Parakeet model: {MODEL_NAME}")
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

# Update model config for fine-tuning
# Configure training data
model.cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
model.cfg.train_ds.batch_size = PER_DEVICE_BATCH
model.cfg.train_ds.num_workers = 4
model.cfg.train_ds.pin_memory = True

# Configure validation data
model.cfg.validation_ds.manifest_filepath = EVAL_MANIFEST
model.cfg.validation_ds.batch_size = PER_DEVICE_BATCH
model.cfg.validation_ds.num_workers = 4

# Setup the data loaders
model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

print(f"Model loaded. Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -------------------- TRAINING SETUP -------------------- #
# Configure optimizer
model.cfg.optim.lr = LR
model.cfg.optim.weight_decay = WEIGHT_DECAY

# Create output directory
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_MODEL_DIR,
    filename="parakeet-{epoch:02d}-{val_wer:.4f}",
    monitor="val_wer",
    mode="min",
    save_top_k=3,
    verbose=True,
)

early_stopping_callback = EarlyStopping(
    monitor="val_wer",
    patience=EARLY_STOPPING_PATIENCE,
    mode="min",
    verbose=True,
    min_delta=0.001,
)

# Configure trainer
trainer = pl.Trainer(
    devices=1 if torch.cuda.is_available() else "auto",
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=NUM_EPOCHS,
    accumulate_grad_batches=GRAD_ACCUM,
    gradient_clip_val=MAX_GRAD_NORM,
    precision=16 if torch.cuda.is_available() else 32,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=50,
    val_check_interval=0.25,  # Validate 4 times per epoch
    default_root_dir=OUTPUT_MODEL_DIR,
    enable_progress_bar=True,
)

# -------------------- TRAINING -------------------- #
print("\n" + "=" * 50)
print("Starting optimized fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Learning rate: {LR}")
print(f"Epochs: {NUM_EPOCHS}")
print("=" * 50 + "\n")

# Train the model
trainer.fit(model)

# Save the best model
model.save_to(os.path.join(OUTPUT_MODEL_DIR, "parakeet_math_finetuned.nemo"))
print(f"\nBest model saved to {OUTPUT_MODEL_DIR}")

# -------------------- FINAL EVALUATION -------------------- #
print("\n" + "=" * 50)
print("Running final evaluation on validation set...")
print("=" * 50 + "\n")

# Load metrics
wer_metric = load("wer")
cer_metric = load("cer")

def detect_hallucination(pred: str, gt: str):
    """Find tokens in prediction that weren't in ground truth."""
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"

def generate_prediction(model, audio_path):
    """Generate prediction from model."""
    model.eval()
    with torch.no_grad():
        transcriptions = model.transcribe([audio_path])
        if isinstance(transcriptions, list) and len(transcriptions) > 0:
            result = transcriptions[0]
            if isinstance(result, (list, tuple)):
                return str(result[0])
            return str(result)
        return str(transcriptions)

# Load base model for comparison
print("Loading base model for comparison...")
base_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    base_model = base_model.cuda()
base_model.eval()

# Reload evaluation data with original fields
eval_data_full = dataset["test"]
eval_data_full = eval_data_full.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

model.eval()
if torch.cuda.is_available():
    model = model.cuda()

rows = []

print(f"Evaluating {len(eval_data_full)} examples")
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_full)}")
    
    ex_full = eval_data_full[i]
    gt = ex_full.get("spoken_english", ex_full.get("text_norm", ""))
    
    # Get audio file path from eval manifest
    audio_path = os.path.join(EVAL_AUDIO_DIR, f"audio_{i}.wav")
    
    # Base model prediction
    pred_base = generate_prediction(base_model, audio_path)
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt])
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt])
    hall_base = detect_hallucination(pred_base, gt)
    
    # Fine-tuned model prediction
    pred_ft = generate_prediction(model, audio_path)
    wer_ft = wer_metric.compute(predictions=[pred_ft], references=[gt])
    cer_ft = cer_metric.compute(predictions=[pred_ft], references=[gt])
    hall_ft = detect_hallucination(pred_ft, gt)
    
    rows.append({
        "id": i,
        "audio_file": audio_path,
        "GT_Text": gt,
        "Base_Pred": pred_base,
        "FineTuned_Pred": pred_ft,
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

# Cleanup temp directory notice
print(f"\nNote: Temporary audio files are in {TEMP_DIR}")
print("You can delete this directory after verification if needed.")
