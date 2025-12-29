"""
Qwen_evaluate.py
Basic evaluation of the Qwen model on the MathBridge dataset.
"""

import os
import torch
import evaluate
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
from datasets import load_dataset, Audio

HF_DATASET_PATH = "abby1492/mathbridge-audio"
OUTPUT_DIR = "./eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metrics
bleu = evaluate.load("bleu")
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
wer = evaluate.load("wer")
cer = evaluate.load("cer")

def compute_metrics(preds, refs):
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

def load_qwen_model():
    print("Loading Qwen model...")
    model_id = "nvidia/canary-qwen-2.5b"
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    print("Model loaded")
    return model

def generate_qwen_output(model, row):
    audio_path = row["audio"]["path"]
    transcriptions = model.transcribe([audio_path])
    return transcriptions[0]

def evaluate_qwen():
    print("=" * 60)
    print("Starting Qwen evaluation")
    print("=" * 60)
    
    model = load_qwen_model()
    
    print(f"Loading dataset: {HF_DATASET_PATH}")
    dataset = load_dataset(HF_DATASET_PATH, split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"Loaded {len(dataset)} samples")
    
    preds = []
    refs = []
    errors = 0
    
    print(f"Running inference on {len(dataset)} samples...")
    for idx, row in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            ref = row["spoken_english"]
            pred = generate_qwen_output(model, row)
            
            preds.append(pred)
            refs.append(ref)
            
            # Show first few examples
            if idx < 3:
                print(f"\nExample {idx + 1}:")
                print(f"  Reference: {ref}")
                print(f"  Prediction: {pred}")
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            errors += 1
            continue
    
    print(f"\nDone: {len(preds)} successful, {errors} errors")
    
    if len(preds) == 0:
        print("No successful predictions!")
        return None
    
    print("Computing metrics...")
    metrics = compute_metrics(preds, refs)
    
    # Save results
    summary_path = os.path.join(OUTPUT_DIR, "qwen_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: Qwen (nvidia/canary-qwen-2.5b)\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Successful predictions: {len(preds)}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"\nMetrics:\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    print(f"\nResults saved to: {summary_path}")
    print("\nFinal Results:")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 40)
    
    return metrics

if __name__ == "__main__":
    print("Qwen model evaluation")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    metrics = evaluate_qwen()
    if metrics:
        print("\nEvaluation completed!")
    else:
        print("\nEvaluation failed")
    
    print(f"\nCheck results in: {OUTPUT_DIR}")
