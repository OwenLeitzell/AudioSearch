"""
granite_evaluate.py
Evaluation of IBM Granite Speech model on the MathBridge dataset.
"""

import os
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "ibm-granite/granite-speech-3.3-8b"
HF_DATASET_PATH = "abby1492/mathbridge-audio"
OUTPUT_DIR = "../results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Evaluation metrics
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
# MODEL LOADING - GRANITE
# ============================================================
def load_granite_model():
    """Load IBM Granite Speech model with error handling."""
    logger.info(f"Loading Granite Speech model: {MODEL_NAME}")
    
    try:
        logger.info(f"  Downloading/loading {MODEL_NAME}...")
        logger.info(f"  This may take a while on first run...")
        
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"Successfully loaded Granite Speech")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load Granite: {e}")
        logger.error(traceback.format_exc())
        return None, None

# ============================================================
# GENERATION LOGIC - GRANITE
# ============================================================
def generate_granite_output(model, processor, row, device):
    """Generate output from Granite Speech with error handling."""
    try:
        # Get audio array
        audio_array = row["audio"]["array"]
        sampling_rate = row["audio"]["sampling_rate"]
        
        # Process audio
        inputs = processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
            )
        
        # Decode output
        transcription = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"Error generating output: {e}")
        logger.error(traceback.format_exc())
        return "[ERROR]"

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def evaluate_granite():
    """Evaluate Granite Speech model with comprehensive error handling."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting evaluation for: GRANITE SPEECH")
    logger.info(f"{'='*60}")
    
    # Load model
    model, processor = load_granite_model()
    if model is None or processor is None:
        logger.error(f"Cannot proceed - model loading failed")
        return None
    
    device = next(model.parameters()).device
    logger.info(f"Model device: {device}")
    
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
    
    for idx, row in enumerate(tqdm(dataset, desc="Granite evaluation")):
        try:
            ref = row["spoken_english"]
            pred = generate_granite_output(model, processor, row, device)
            
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
        logger.error(f"No successful predictions for Granite")
        return None

    # Compute metrics
    logger.info(f"Computing metrics...")
    metrics = compute_metrics(preds, refs)
    
    if metrics is None:
        logger.error(f"Failed to compute metrics for Granite")
        return None

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "granite_summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Model: Granite Speech ({MODEL_NAME})\n")
            f.write(f"Total samples: {len(dataset)}\n")
            f.write(f"Successful predictions: {len(preds)}\n")
            f.write(f"Errors: {errors}\n")
            f.write(f"\nMetrics:\n")
            f.write("-" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        logger.info(f"Saved results to: {summary_path}")
        logger.info(f"\nFINAL RESULTS FOR GRANITE:")
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
# RUN GRANITE EVALUATION
# ============================================================
if __name__ == "__main__":
    logger.info("Starting Granite Speech model evaluation")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        metrics = evaluate_granite()
        if metrics:
            logger.info("\nEVALUATION COMPLETED SUCCESSFULLY")
        else:
            logger.info("\nEVALUATION FAILED")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"\nCheck results in: {OUTPUT_DIR}")
