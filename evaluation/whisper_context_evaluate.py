#!/usr/bin/env python3
import torch
import torchaudio
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load

# -----------------------
# CONFIGURATION
# -----------------------
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
LANGUAGE = "en"
MAX_SAMPLES = None
OUTPUT_FILE = "../results/whisper_v3_context_eval.txt"

# -----------------------
# LOAD MODEL & PROCESSOR
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")

# -----------------------
# LOAD DATASET
# -----------------------
print(f"Loading dataset: {DATASET_NAME} ({SPLIT} split)")
dataset = load_dataset(DATASET_NAME, split=SPLIT)
if MAX_SAMPLES:
    dataset = dataset.select(range(MAX_SAMPLES))
print(f"Loaded {len(dataset)} samples.")

# Ensure audio column is Audio feature (decodes to waveform)
dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

# -----------------------
# BUILD FULL REFERENCE SENTENCE
# -----------------------
def combine_text(batch):
    before = str(batch.get("context_before", "")).strip()
    spoken = str(batch.get("spoken_english", "")).strip()
    after = str(batch.get("context_after", "")).strip()
    parts = [p for p in [before, spoken, after] if p]
    batch["reference"] = " ".join(parts)
    return batch

dataset = dataset.map(combine_text)

# -----------------------
# TRANSCRIPTION FUNCTION (robust)
# -----------------------
def transcribe(batch):
    audio_entry = batch[AUDIO_COLUMN]

    # If already decoded by datasets.Audio
    if isinstance(audio_entry, dict) and "array" in audio_entry:
        waveform = audio_entry["array"]
    # Otherwise, load manually
    elif isinstance(audio_entry, dict) and "path" in audio_entry:
        waveform, sr = torchaudio.load(audio_entry["path"])
        waveform = waveform.mean(dim=0).numpy()  # convert to mono
    else:
        raise ValueError(f"Unrecognized audio format: {audio_entry}")

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)
    batch["prediction"] = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return batch

# -----------------------
# RUN TRANSCRIPTION
# -----------------------
print("Running transcription...")
dataset = dataset.map(transcribe)

# -----------------------
# EVALUATE
# -----------------------
preds = dataset["prediction"]
refs = dataset["reference"]

print("Computing metrics...")
metrics = {}

bleu = load("bleu")
metrics["BLEU"] = bleu.compute(predictions=preds, references=refs)["bleu"]

sacrebleu = load("sacrebleu")
metrics["SacreBLEU"] = sacrebleu.compute(predictions=preds, references=refs)["score"]

rouge = load("rouge")
rouge_scores = rouge.compute(predictions=preds, references=refs)
metrics.update({
    "ROUGE-1": rouge_scores["rouge1"],
    "ROUGE-2": rouge_scores["rouge2"],
    "ROUGE-L": rouge_scores["rougeL"],
})

wer = load("wer")
metrics["WER"] = wer.compute(predictions=preds, references=refs)

cer = load("cer")
metrics["CER"] = cer.compute(predictions=preds, references=refs)

# -----------------------
# OUTPUT RESULTS
# -----------------------
print("\n=== Evaluation Results ===")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=== Whisper Large V3 Evaluation (with context) ===\n\n")
    f.write(f"Dataset: {DATASET_NAME}\nSplit: {SPLIT}\n\n")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        f.write(f"{key}: {value:.4f}\n")

    f.write("\n--- Individual Predictions ---\n")
    for ref, pred in zip(refs, preds):
        f.write(f"\nReference: {ref}\nPrediction: {pred}\n")

print(f"\nResults saved to {OUTPUT_FILE}")
