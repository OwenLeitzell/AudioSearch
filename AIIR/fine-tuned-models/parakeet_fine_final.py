"""
parakeet_fine_final.py
Two-stage fine-tuning of NVIDIA's Parakeet CTC model for math formula transcription.
Stage 1: Audio → Spoken English (Parakeet model)
Stage 2: Spoken English → LaTeX (rule-based conversion)
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

# Configure NeMo logging to reduce verbose WER output during validation
# Set to WARNING to suppress INFO-level WER comparison logs
logging.setLevel(logging.WARNING)

# NeMo 24.12+ uses lightning.pytorch instead of pytorch_lightning
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
except ImportError:
    # Fallback for older NeMo versions
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
from evaluate import load
import json
import tempfile
import soundfile as sf
import librosa  # For audio resampling if needed
import glob
import shutil

# Disable wandb for account log in and verification
os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "nvidia/parakeet-ctc-1.1b"  # Options: parakeet-ctc-1.1b, parakeet-ctc-0.6b
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
MAX_SAMPLES = None
OUTPUT_XLSX = "parakeet_math_outputs2.xlsx"
OUTPUT_MODEL_DIR = "./parakeet_math_best"

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
        
        # Get transcription (using spoken_english for Stage 1: Audio→English)
        # Stage 2 (English→LaTeX) will be handled separately during evaluation
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
# Reduce verbose WER logging during validation
if hasattr(model.cfg, 'decoding'):
    if hasattr(model.cfg.decoding, 'log_prediction'):
        model.cfg.decoding.log_prediction = False

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

# Clean up any old checkpoints from previous runs (we don't save checkpoints anymore) improves memory usage
old_checkpoints = glob.glob(os.path.join(OUTPUT_MODEL_DIR, "*.ckpt"))
if old_checkpoints:
    print(f"Found {len(old_checkpoints)} old checkpoint(s) from previous runs. Cleaning up...")
    for ckpt in old_checkpoints:
        try:
            os.remove(ckpt)
            print(f"  Removed: {os.path.basename(ckpt)}")
        except Exception as e:
            print(f"  Warning: Could not remove {os.path.basename(ckpt)}: {e}")

# Check available disk space
try:
    total, used, free = shutil.disk_usage(OUTPUT_MODEL_DIR)
    free_gb = free / (1024**3)
    print(f"\nAvailable disk space: {free_gb:.2f} GB")
    if free_gb < 10:
        print(f"WARNING: Low disk space ({free_gb:.2f} GB). Consider freeing up space or changing OUTPUT_MODEL_DIR.")
except Exception:
    pass  # Skip disk space check if not available

# Callbacks
# Early stopping (doesn't require checkpoints - just monitors validation metric)
early_stopping_callback = EarlyStopping(
    monitor="val_wer",
    patience=EARLY_STOPPING_PATIENCE,
    mode="min",
    verbose=True,
    min_delta=0.001,
)

# Custom callback to track best model in memory (without saving checkpoints)
class BestModelTracker(pl.Callback):
    """Track the best model during training without saving checkpoints."""
    def __init__(self, monitor="val_wer", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.best_model_state = None
        
    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
            
        is_better = (current_score < self.best_score) if self.mode == "min" else (current_score > self.best_score)
        if is_better:
            self.best_score = current_score
            # Save model state dict (much smaller than full checkpoint)
            self.best_model_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            print(f"New best {self.monitor}: {self.best_score:.4f}")

best_model_tracker = BestModelTracker(monitor="val_wer", mode="min")

# Configure trainer
trainer = pl.Trainer(
    devices=1 if torch.cuda.is_available() else "auto",
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=NUM_EPOCHS,
    accumulate_grad_batches=GRAD_ACCUM,
    gradient_clip_val=MAX_GRAD_NORM,
    precision=16 if torch.cuda.is_available() else 32,
    callbacks=[early_stopping_callback, best_model_tracker],
    log_every_n_steps=50,
    val_check_interval=0.5,  # Validate 2 times per epoch (reduced from 4 to save disk space)
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

# Load the best model state (tracked in memory) and save it
if best_model_tracker.best_model_state is not None:
    print(f"\nLoading best model state (val_wer: {best_model_tracker.best_score:.4f})...")
    model.load_state_dict(best_model_tracker.best_model_state)
    if torch.cuda.is_available():
        model = model.cuda()
else:
    print("\nNo best model state tracked, saving current model state...")

# Save the best model
model.save_to(os.path.join(OUTPUT_MODEL_DIR, "parakeet_math_finetuned.nemo"))
print(f"Best model saved to {OUTPUT_MODEL_DIR}")

# -------------------- FINAL EVALUATION -------------------- #
print("\n" + "=" * 50)
print("Running final evaluation on validation set...")
print("=" * 50 + "\n")

# Load metrics
wer_metric = load("wer")
cer_metric = load("cer")

def extract_math_formula(english_text: str) -> str:
    """
    Extract the math formula portion from a spoken English sentence.
    Examples:
    - "for n greater than or equal to two" → "n greater than or equal to two"
    - "x sub j equals five" → "x sub j equals five"
    - "we have sigma sub f" → "sigma sub f"
    """
    if not english_text:
        return ""
    
    text = english_text.lower().strip()
    
    # Common math-related phrases that indicate a formula
    math_indicators = [
        "sub", "superscript", "to the power", "squared", "cubed",
        "equals", "equal to", "greater than", "less than",
        "plus", "minus", "times", "divided by", "over",
        "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
        "mu", "pi", "sigma", "phi", "psi", "omega",
        "sum", "integral", "partial", "nabla", "infinity"
    ]
    
    # Find sentences/phrases containing math indicators
    words = text.split()
    math_start = None
    math_end = None
    
    # Find the start of math content
    for i, word in enumerate(words):
        if any(indicator in word for indicator in math_indicators):
            # Look backwards for context (e.g., "for", "let", "where")
            start = max(0, i - 2)
            math_start = start
            break
    
    # Find the end of math content
    if math_start is not None:
        # Look for end of sentence or next non-math word
        for i in range(math_start, len(words)):
            word = words[i]
            # Stop at common sentence endings or non-math words
            if word in [".", ",", "and", "or", "the", "a", "an", "is", "are", "was", "were"]:
                if i > math_start + 2:  # Make sure we have at least a few words
                    math_end = i
                    break
        
        if math_end is None:
            math_end = len(words)
        
        extracted = " ".join(words[math_start:math_end])
        return extracted.strip()
    
    # If no math indicators found, return the whole text
    return text

def english_to_latex(english_text: str) -> str:
    """
    Stage 2: Convert spoken English math to LaTeX.
    Handles common math phrases and converts them to LaTeX notation.
    
    Examples:
    - "x sub j" → "x_j"
    - "sigma sub f" → "\\sigma_f"
    - "n greater than or equal to two" → "n \\geq 2"
    - "x squared" → "x^2"
    """
    if not english_text:
        return ""
    
    text = english_text.lower().strip()
    
    # Common math symbol mappings
    symbol_map = {
        "alpha": "\\alpha", "beta": "\\beta", "gamma": "\\gamma", "delta": "\\delta",
        "epsilon": "\\epsilon", "theta": "\\theta", "lambda": "\\lambda", "mu": "\\mu",
        "pi": "\\pi", "sigma": "\\sigma", "phi": "\\phi", "psi": "\\psi", "omega": "\\omega",
        "ket": "\\ket", "bra": "\\bra", "sum": "\\sum", "int": "\\int",
        "partial": "\\partial", "nabla": "\\nabla", "infty": "\\infty", "infinity": "\\infty"
    }
    
    # Replace Greek letters and math symbols
    for eng, latex in symbol_map.items():
        pattern = rf'\b{re.escape(eng)}\b'
        text = re.sub(pattern, lambda m: latex, text)
    
    # Handle comparisons: "greater than or equal to" → "\\geq"
    # Use lambda functions to avoid regex interpreting backslashes in replacement strings
    text = re.sub(r'\bgreater\s+than\s+or\s+equal\s+to\b', lambda m: '\\geq', text)
    text = re.sub(r'\bless\s+than\s+or\s+equal\s+to\b', lambda m: '\\leq', text)
    text = re.sub(r'\bgreater\s+than\b', '>', text)
    text = re.sub(r'\bless\s+than\b', '<', text)
    text = re.sub(r'\bnot\s+equal\s+to\b', lambda m: '\\neq', text)
    text = re.sub(r'\bapproximately\s+equal\s+to\b', lambda m: '\\approx', text)
    
    # Handle subscripts: "x sub j" → "x_j"
    text = re.sub(r'(\w+)\s+sub\s+(\w+)', r'\1_\2', text)
    
    # Handle superscripts: "x to the power of 2" → "x^2" or "x squared" → "x^2"
    text = re.sub(r'(\w+)\s+to\s+the\s+power\s+of\s+(\w+)', r'\1^\2', text)
    text = re.sub(r'(\w+)\s+squared', r'\1^2', text)
    text = re.sub(r'(\w+)\s+cubed', r'\1^3', text)
    
    # Handle fractions: "a over b" → "\\frac{a}{b}"
    # Use lambda to avoid regex backslash interpretation
    text = re.sub(r'(\w+)\s+over\s+(\w+)', lambda m: f'\\frac{{{m.group(1)}}}{{{m.group(2)}}}', text)
    
    # Handle equals: "equals" → "="
    text = re.sub(r'\bequals\b', '=', text)
    text = re.sub(r'\bequal\s+to\b', '=', text)
    
    # Handle numbers: "two" → "2", "three" → "3", etc.
    number_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20"
    }
    for word, num in number_map.items():
        pattern = rf'\b{re.escape(word)}\b'
        text = re.sub(pattern, num, text)
    
    # Handle operations: "plus" → "+", "minus" → "-", "times" → "*"
    text = re.sub(r'\bplus\b', '+', text)
    text = re.sub(r'\bminus\b', '-', text)
    text = re.sub(r'\btimes\b', '*', text)
    text = re.sub(r'\bdivided\s+by\b', '/', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_hallucination(pred: str, gt: str):
    """Find tokens in prediction that weren't in ground truth."""
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"

def generate_prediction(model, audio_input):
    """
    Generate prediction from model.
    Accepts either audio file path (str) or audio array (numpy array).
    NeMo's transcribe() returns nested lists of Hypothesis objects.
    """
    model.eval()
    with torch.no_grad():
        # NeMo can transcribe directly from audio arrays, which avoids dataloader setup issues
        # If audio_input is a string (file path), read it first
        if isinstance(audio_input, str):
            # Read audio file
            audio_array, sample_rate = sf.read(audio_input)
            # Ensure correct sample rate (NeMo expects 16kHz)
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            transcriptions = model.transcribe([audio_array])
        else:
            # Assume it's already an audio array
            transcriptions = model.transcribe([audio_input])
        
        # NeMo's transcribe returns: [[Hypothesis(...)]]
        # We need to extract the text from the Hypothesis object
        def extract_text_from_hypothesis(obj):
            """Recursively extract text from Hypothesis object or nested structures."""
            # If it's a Hypothesis object (or similar), try to get text attribute
            if hasattr(obj, 'text'):
                text = obj.text
                if text is not None and text != '':
                    return str(text).strip()
            
            # If it's a list/tuple, recurse into it
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    result = extract_text_from_hypothesis(item)
                    if result and result != '':
                        return result
            
            # If it's a dict-like object, try common keys
            if hasattr(obj, '__dict__'):
                for key in ['text', 'transcript', 'prediction', 'y_sequence']:
                    if hasattr(obj, key):
                        val = getattr(obj, key)
                        if val is not None:
                            if isinstance(val, str) and val.strip():
                                return val.strip()
                            # If y_sequence is a tensor, we can't use it directly
                            # Skip tensor attributes
            
            return None
        
        # Try to extract text
        text = extract_text_from_hypothesis(transcriptions)
        if text:
            return text
        
        # Final fallback: convert to string and try to extract meaningful text
        # This handles the case where Hypothesis object is converted to string
        result_str = str(transcriptions)
        # Try to find text='...' pattern in the string representation
        import re
        match = re.search(r"text=['\"]([^'\"]+)['\"]", result_str)
        if match:
            return match.group(1).strip()
        
        # Last resort: return the string representation (shouldn't happen)
        return result_str.strip()

# Load base model for comparison (FRESH - not the fine-tuned model)
print("Loading fresh base model for comparison...")
print("Note: This is a completely separate model instance, not the fine-tuned model.")
base_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    base_model = base_model.cuda()
base_model.eval()
print("Base model loaded successfully.")

# Load evaluation data with original fields (use test set from train_test_split)
eval_data_full = dataset["test"]
eval_data_full = eval_data_full.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
# Normalize the LaTeX
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

model.eval()
if torch.cuda.is_available():
    model = model.cuda()

rows = []

print(f"Evaluating {len(eval_data_full)} examples")
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_full)}")  # Print the progress so I know it is working
    
    ex_full = eval_data_full[i]
    # Use normalized LaTeX as ground truth for final comparison
    gt_latex = ex_full.get("text_norm", ex_full.get("equation", ""))
    
    # Construct full English ground truth from context_before, spoken_english, and context_after
    context_before = str(ex_full.get("context_before", "")).strip()
    spoken_english = str(ex_full.get("spoken_english", "")).strip()
    context_after = str(ex_full.get("context_after", "")).strip()
    # Combine all parts, filtering out empty strings
    english_parts = [p for p in [context_before, spoken_english, context_after] if p]
    gt_english = " ".join(english_parts)
    
    # Get audio array directly from dataset (avoids dataloader setup issues)
    audio_array = ex_full[AUDIO_COLUMN]["array"]
    sample_rate = ex_full[AUDIO_COLUMN]["sampling_rate"]
    
    # Ensure correct sample rate
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    
    # Stage 1: Base model prediction (Audio → English)
    try:
        pred_base_english = generate_prediction(base_model, audio_array)
        if i == 0:
            print(f"\nDEBUG - Base model English output: '{pred_base_english[:150]}'")
    except Exception as e:
        print(f"Error generating base prediction for sample {i}: {e}")
        pred_base_english = "[ERROR]"
    
    # Stage 2: Extract math formula and convert to LaTeX
    math_formula_base = extract_math_formula(pred_base_english)
    pred_base_latex = english_to_latex(math_formula_base)
    
    # Stage 1: Fine-tuned model prediction (Audio → English)
    try:
        pred_ft_english = generate_prediction(model, audio_array)
        if i == 0:
            print(f"DEBUG - Fine-tuned model English output: '{pred_ft_english[:150]}'")
    except Exception as e:
        print(f"Error generating fine-tuned prediction for sample {i}: {e}")
        pred_ft_english = "[ERROR]"
    
    # Stage 2: Extract math formula and convert to LaTeX
    math_formula_ft = extract_math_formula(pred_ft_english)
    pred_ft_latex = english_to_latex(math_formula_ft)
    
    if i == 0:
        print(f"DEBUG - Extracted math (base): '{math_formula_base}'")
        print(f"DEBUG - Converted to LaTeX (base): '{pred_base_latex}'")
        print(f"DEBUG - Extracted math (fine-tuned): '{math_formula_ft}'")
        print(f"DEBUG - Converted to LaTeX (fine-tuned): '{pred_ft_latex}'")
        print(f"DEBUG - Ground truth English: '{gt_english[:150]}'")
        print(f"DEBUG - Ground truth LaTeX: '{gt_latex[:150]}'")
    
    # Compare base model English prediction with ground truth English
    wer_base_english = wer_metric.compute(predictions=[pred_base_english], references=[gt_english])
    cer_base_english = cer_metric.compute(predictions=[pred_base_english], references=[gt_english])
    hall_base_english = detect_hallucination(pred_base_english, gt_english)
    
    # Compare extracted LaTeX predictions with LaTeX ground truth (for both models)
    wer_base_latex = wer_metric.compute(predictions=[pred_base_latex], references=[gt_latex])
    cer_base_latex = cer_metric.compute(predictions=[pred_base_latex], references=[gt_latex])
    hall_base_latex = detect_hallucination(pred_base_latex, gt_latex)
    
    # Fine-tuned model: Compare English prediction with ground truth English
    wer_ft_english = wer_metric.compute(predictions=[pred_ft_english], references=[gt_english])
    cer_ft_english = cer_metric.compute(predictions=[pred_ft_english], references=[gt_english])
    hall_ft_english = detect_hallucination(pred_ft_english, gt_english)
    
    # Fine-tuned model: Compare extracted LaTeX with ground truth LaTeX
    wer_ft_latex = wer_metric.compute(predictions=[pred_ft_latex], references=[gt_latex])
    cer_ft_latex = cer_metric.compute(predictions=[pred_ft_latex], references=[gt_latex])
    hall_ft_latex = detect_hallucination(pred_ft_latex, gt_latex)
    
    # Get audio file path for reference
    audio_path = ex_full[AUDIO_COLUMN].get("path", "unknown")
    
    rows.append({
        "id": i,
        "audio_file": audio_path,
        "GT_LaTeX": gt_latex,
        "GT_English": gt_english,
        "Base_Pred_English": pred_base_english,
        "Base_Extracted_Math": math_formula_base,
        "Base_Pred_LaTeX": pred_base_latex,
        "FineTuned_Pred_English": pred_ft_english,
        "FineTuned_Extracted_Math": math_formula_ft,
        "FineTuned_Pred_LaTeX": pred_ft_latex,
        # Base model: English comparison
        "Base_WER_English": wer_base_english,
        "Base_CER_English": cer_base_english,
        "Base_Hallucination_English": hall_base_english,
        # Base model: LaTeX comparison (extracted math)
        "Base_WER_LaTeX": wer_base_latex,
        "Base_CER_LaTeX": cer_base_latex,
        "Base_Hallucination_LaTeX": hall_base_latex,
        # Fine-tuned model: English comparison
        "FineTuned_WER_English": wer_ft_english,
        "FineTuned_CER_English": cer_ft_english,
        "FineTuned_Hallucination_English": hall_ft_english,
        # Fine-tuned model: LaTeX comparison (extracted math)
        "FineTuned_WER_LaTeX": wer_ft_latex,
        "FineTuned_CER_LaTeX": cer_ft_latex,
        "FineTuned_Hallucination_LaTeX": hall_ft_latex,
        # Improvements
        "WER_Improvement_English": wer_base_english - wer_ft_english,
        "CER_Improvement_English": cer_base_english - cer_ft_english,
        "WER_Improvement_LaTeX": wer_base_latex - wer_ft_latex,
        "CER_Improvement_LaTeX": cer_base_latex - cer_ft_latex,
    })

# Save detailed results
df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

# Print summary statistics
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)

# English comparison (base model English vs GT English)
print(f"\nBase Model Performance (English):")
print(f"  Average WER: {df['Base_WER_English'].mean():.4f}")
print(f"  Average CER: {df['Base_CER_English'].mean():.4f}")

# LaTeX comparison (extracted math vs GT LaTeX)
print(f"\nBase Model Performance (LaTeX - Extracted Math):")
print(f"  Average WER: {df['Base_WER_LaTeX'].mean():.4f}")
print(f"  Average CER: {df['Base_CER_LaTeX'].mean():.4f}")

print(f"\nFine-Tuned Model Performance (English):")
print(f"  Average WER: {df['FineTuned_WER_English'].mean():.4f}")
print(f"  Average CER: {df['FineTuned_CER_English'].mean():.4f}")

print(f"\nFine-Tuned Model Performance (LaTeX - Extracted Math):")
print(f"  Average WER: {df['FineTuned_WER_LaTeX'].mean():.4f}")
print(f"  Average CER: {df['FineTuned_CER_LaTeX'].mean():.4f}")

print(f"\nImprovement (English):")
print(f"  WER Reduction: {df['WER_Improvement_English'].mean():.4f} ({df['WER_Improvement_English'].mean() / df['Base_WER_English'].mean() * 100:.1f}%)")
print(f"  CER Reduction: {df['CER_Improvement_English'].mean():.4f} ({df['CER_Improvement_English'].mean() / df['Base_CER_English'].mean() * 100:.1f}%)")
print(f"  % of samples improved: {(df['WER_Improvement_English'] > 0).sum() / len(df) * 100:.1f}%")

print(f"\nImprovement (LaTeX):")
print(f"  WER Reduction: {df['WER_Improvement_LaTeX'].mean():.4f} ({df['WER_Improvement_LaTeX'].mean() / df['Base_WER_LaTeX'].mean() * 100:.1f}%)")
print(f"  CER Reduction: {df['CER_Improvement_LaTeX'].mean():.4f} ({df['CER_Improvement_LaTeX'].mean() / df['Base_CER_LaTeX'].mean() * 100:.1f}%)")
print(f"  % of samples improved: {(df['WER_Improvement_LaTeX'] > 0).sum() / len(df) * 100:.1f}%")

print("=" * 50 + "\n")

# Cleanup temp directory notice
print(f"\nNote: Temporary audio files are in {TEMP_DIR}")
print("You can delete this directory after verification if needed.")