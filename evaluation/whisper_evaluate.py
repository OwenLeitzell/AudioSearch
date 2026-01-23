import os
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline
)
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
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
    logger.info(" Successfully loaded all metrics")
except Exception as e:
    logger.error(f" Failed to load metrics: {e}")
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
        logger.error(f" Error computing metrics: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================
# MODEL LOADING - WHISPER ONLY
# ============================================================
def load_whisper_model():
    """Load Whisper model with error handling."""
    logger.info(f" Loading Whisper model...")
    
    try:
        model_id = "openai/whisper-large-v3"
        logger.info(f"   Downloading/loading {model_id}...")
        logger.info(f"   This may take a while on first run...")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info(f" Successfully loaded Whisper")
        return pipe
        
    except Exception as e:
        logger.error(f" Failed to load Whisper: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================
# GENERATION LOGIC - WHISPER ONLY
# ============================================================
def generate_whisper_output(pipe, row):
    """Generate output from Whisper with error handling."""
    try:
        audio_path = row["audio"]["path"]
        logger.debug(f"   Processing audio: {audio_path}")
        result = pipe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f" Error generating output: {e}")
        logger.error(traceback.format_exc())
        return "[ERROR]"

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def evaluate_whisper():
    """Evaluate Whisper model with comprehensive error handling."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting evaluation for: WHISPER")
    logger.info(f"{'='*60}")
    
    # Load model
    pipe = load_whisper_model()
    if pipe is None:
        logger.error(f" Cannot proceed - model loading failed")
        return None
    
    # Load dataset
    try:
        logger.info(f" Loading dataset: {HF_DATASET_PATH}")
        dataset = load_dataset(HF_DATASET_PATH, split="train")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        logger.info(f" Loaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f" Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return None
    
    preds, refs = [], []
    errors = 0

    logger.info(f"🔄 Starting inference on {len(dataset)} samples...")
    
    for idx, row in enumerate(tqdm(dataset, desc="Whisper evaluation")):
        try:
            ref = row["spoken_english"]
            pred = generate_whisper_output(pipe, row)
            
            if pred == "[ERROR]":
                errors += 1
                continue
                
            preds.append(pred)
            refs.append(ref)
            
            # Log first few examples
            if idx < 3:
                logger.info(f"\n   Example {idx + 1}:")
                logger.info(f"   Reference: {ref}")
                logger.info(f"   Prediction: {pred}")
            
        except Exception as e:
            logger.error(f" Error on sample {idx}: {e}")
            logger.error(traceback.format_exc())
            errors += 1
            continue

    logger.info(f"\n Completed inference: {len(preds)} successful, {errors} errors")
    
    if len(preds) == 0:
        logger.error(f" No successful predictions for Whisper")
        return None

    # Compute metrics
    logger.info(f" Computing metrics...")
    metrics = compute_metrics(preds, refs)
    
    if metrics is None:
        logger.error(f" Failed to compute metrics for Whisper")
        return None

    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "whisper_summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Model: Whisper (openai/whisper-large-v3)\n")
            f.write(f"Total samples: {len(dataset)}\n")
            f.write(f"Successful predictions: {len(preds)}\n")
            f.write(f"Errors: {errors}\n")
            f.write(f"\nMetrics:\n")
            f.write("-" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        logger.info(f" Saved results to: {summary_path}")
        logger.info(f"\n FINAL RESULTS FOR WHISPER:")
        logger.info("-" * 40)
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("-" * 40)
            
    except Exception as e:
        logger.error(f" Failed to save results: {e}")
        logger.error(traceback.format_exc())
        return None
    
    return metrics

# ============================================================
# RUN WHISPER EVALUATION
# ============================================================
if __name__ == "__main__":
    logger.info(" Starting Whisper model evaluation")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        metrics = evaluate_whisper()
        if metrics:
            logger.info("\n EVALUATION COMPLETED SUCCESSFULLY")
        else:
            logger.info("\n EVALUATION FAILED")
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"\n Check results in: {OUTPUT_DIR}")