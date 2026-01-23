"""
parakeet_evaluate.py
Basic evaluation of NVIDIA's Parakeet ASR model on the MathBridge dataset.
"""

import os
import torch
import evaluate
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
from datasets import load_dataset, Audio
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "nvidia/parakeet-ctc-1.1b"  # Options: parakeet-ctc-1.1b, parakeet-rnnt-1.1b, parakeet-tdt-1.1b
HF_DATASET_PATH = "abby1492/mathbridge-audio"
OUTPUT_DIR = "../results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load evaluation metrics
try:
    bleu = evaluate.load("bleu")
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    logger.info("Successfully loaded all metrics")
except Exception as e:
    logger.error(f"Failed to load metrics: {e}")
    raise

# ============================================================
# METRIC COMPUTATION
# ============================================================
def compute_metrics(preds, refs):
    """Compute evaluation metrics with error handling."""
    try:
        preds = [p.strip() for p in preds]
        refs = [r.strip() for r in refs]

        results = {
            "bleu": bleu.compute(predictions=preds, references=refs)["bleu"],
            "sacrebleu": sacrebleu.compute(predictions=preds, references=refs)["score"],
            "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
            "wer": wer.compute(predictions=preds, references=refs),
            "cer": cer.compute(predictions=preds, references=refs)
        }
        return results
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================
# MODEL LOADING - PARAKEET
# ============================================================
def load_parakeet_model():
    """Load NVIDIA Parakeet model with error handling."""
    logger.info(f"Loading Parakeet model: {MODEL_NAME}")
    
    try:
        logger.info(f"  Downloading/loading {MODEL_NAME}...")
        logger.info(f"  This may take a while on first run...")
        
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("  Model moved to GPU")
        
        logger.info(f"Successfully loaded Parakeet")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Parakeet: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================
# GENERATION LOGIC - PARAKEET
# ============================================================
def generate_parakeet_output(model, row):
    """Generate output from Parakeet with error handling."""
    try:
        audio_path = row["audio"]["path"]
        logger.debug(f"  Processing audio: {audio_path}")
        
        # Parakeet uses NeMo's transcribe method
        transcriptions = model.transcribe([audio_path])
        
        # Handle different return types from NeMo models
        if isinstance(transcriptions, list):
            if len(transcriptions) > 0:
                result = transcriptions[0]
                # Some models return tuples or lists
                if isinstance(result, (list, tuple)):
                    return str(result[0])
                return str(result)
        return str(transcriptions)
        
    except Exception as e:
        logger.error(f"Error generating output: {e}")
        logger.error(traceback.format_exc())
        return "[ERROR]"

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def evaluate_parakeet():
    """Evaluate Parakeet model with comprehensive error handling."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting evaluation for: PARAKEET")
    logger.info(f"{'='*60}")
    
    # Load model
    model = load_parakeet_model()
    if model is None:
        logger.error(f"Cannot proceed - model loading failed")
        return None
    
    # Load dataset
    try:
        logger.info(f"Loading dataset: {HF_DATASET_PATH}")
        dataset = load_dataset(HF_DATASET_PATH, split="train")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        logger.info(f"Loaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return None
    
    preds, refs = [], []
    errors = 0

    logger.info(f"Starting inference on {len(dataset)} samples...")
    
    for idx, row in enumerate(tqdm(dataset, desc="Parakeet evaluation")):
        try:
            ref = row["spoken_english"]
            pred = generate_parakeet_output(model, row)
            
            if pred == "[ERROR]":
                errors += 1
                continue
                
            preds.append(pred)
            refs.append(ref)
            
            # Log first few examples
            if idx < 3:
                logger.info(f"\n  Example {idx + 1}:")
                logger.info(f"  Reference: {ref}")
                logger.info(f"  Prediction: {pred}")
            
        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            logger.error(traceback.format_exc())
            errors += 1
            continue

    logger.info(f"\nCompleted inference: {len(preds)} successful, {errors} errors")
    
    if len(preds) == 0:
        logger.error(f"No successful predictions for Parakeet")
        return None

    # Compute metrics
    logger.info(f"Computing metrics...")
    metrics = compute_metrics(preds, refs)
    
    if metrics is None:
        logger.error(f"Failed to compute metrics for Parakeet")
        return None

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "parakeet_summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Model: Parakeet ({MODEL_NAME})\n")
            f.write(f"Total samples: {len(dataset)}\n")
            f.write(f"Successful predictions: {len(preds)}\n")
            f.write(f"Errors: {errors}\n")
            f.write(f"\nMetrics:\n")
            f.write("-" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        logger.info(f"Saved results to: {summary_path}")
        logger.info(f"\nFINAL RESULTS FOR PARAKEET:")
        logger.info("-" * 40)
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("-" * 40)
            
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        logger.error(traceback.format_exc())
        return None
    
    return metrics

# ============================================================
# RUN PARAKEET EVALUATION
# ============================================================
if __name__ == "__main__":
    logger.info("Starting Parakeet model evaluation")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        metrics = evaluate_parakeet()
        if metrics:
            logger.info("\nEVALUATION COMPLETED SUCCESSFULLY")
        else:
            logger.info("\nEVALUATION FAILED")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"\nCheck results in: {OUTPUT_DIR}")
