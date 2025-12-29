import os
import torch
import evaluate
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
from datasets import load_from_disk
from datasets import load_dataset
from datasets import Audio
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline
)

# ============================================================
# CONFIG
# ============================================================
HF_DATASET_PATH = "abby1492/mathbridge-audio"
OUTPUT_DIR = "./eval_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Evaluation metrics
bleu = evaluate.load("bleu")
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
wer = evaluate.load("wer")
cer = evaluate.load("cer")

# ============================================================
# METRIC COMPUTATION
# ============================================================
def compute_metrics(preds, refs):
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in refs]

    return {
        "bleu": bleu.compute(predictions=preds, references=refs)["bleu"],
        "sacrebleu": sacrebleu.compute(predictions=preds, references=refs)["score"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "wer": wer.compute(predictions=preds, references=refs),
        "cer": cer.compute(predictions=preds, references=refs)
    }

# ============================================================
# MODEL LOADING
# ============================================================
def load_model(name):
    """Load a model by its short name."""
    if name.lower() == "phi":
        model_id = "microsoft/phi-4-multimodal-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return ("phi", (model, tokenizer))

    elif name.lower() == "whisper":
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            device=0 if torch.cuda.is_available() else -1
        )
        return ("whisper", pipe)

    elif name.lower() == "qwen":
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")
        return ("qwen", model)

    else:
        raise ValueError(f"Unknown model: {name}")

# ============================================================
# GENERATION LOGIC
# ============================================================
def generate_output(model_tuple, row):
    model_type, model_obj = model_tuple
    context = f"{row['context_before']} {row['spoken_english']} {row['context_after']}".strip()

    # Phi-4 text generation
    if model_type == "phi":
        model, tokenizer = model_obj
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Whisper ASR
    elif model_type == "whisper":
        audio_path = row["audio"]["path"]
        result = model_obj(audio_path)
        return result["text"]

    # Qwen ASR
    elif model_type == "qwen":
        audio_path = row["audio"]["path"]
        transcriptions = model_obj.transcribe([audio_path])
        return transcriptions[0]

    else:
        raise ValueError("Unsupported model type")

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def evaluate_model(model_name):
    model_tuple = load_model(model_name)
    dataset = load_dataset("abby1492/mathbridge-audio", split = "train")
    dataset=dataset.cast_column("audio", Audio(sampling_rate=16000))
    preds, refs = [], []


    print(f"\n🔹 Evaluating {model_name} on {len(dataset)} samples...")

    for row in tqdm(dataset, desc=f"{model_name} evaluation"):
        ref = row["spoken_english"]
        pred = generate_output(model_tuple, row)
        preds.append(pred)
        refs.append(ref)

    metrics = compute_metrics(preds, refs)

    # Save summary only (no per-example outputs)
    summary_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"✅ Saved summary to {summary_path}")
    return metrics

# ============================================================
# RUN ALL MODELS
# ============================================================
if __name__ == "__main__":
    all_models = ["phi", "whisper", "qwen"]
    for m in all_models:
        evaluate_model(m)
