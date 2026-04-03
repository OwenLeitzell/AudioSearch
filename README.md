# AudioSearch
# Math Formula Transcription Models

This repository contains model evaluation scripts for transcribing spoken math phrases and forumlas into LaTeX. This repository contains both base models and fine-tuned models for Whisper (OpenAI) and Qwen (Nvidia)

# AudioSearch

## Project Structure
```
AudioSearch/
│
├── base_models/
│   ├── Qwen_evaluate.py           # First attempt at Qwen evaluation (crude, no tuning)
│   │                               # Evaluates base Qwen ASR model (nvidia/canary-qwen-2.5b)
│   │                               # on MathBridge dataset with BLEU, ROUGE, WER, and CER metrics
│   │
│   ├── Real_Whisper_evaluate.py   # Improved evaluation script
│   │                               # Evaluates Whisper-large-v3 on MathBridge dataset
│   │                               # with before/after context
│   │
│   ├── Whisper_evaluate.py        # Base Whisper-large-v3 evaluation
│   │                               # Metrics: BLEU, SacreBLEU, WER, CER, and ROUGE-L
│   │
│   └── evaluate_models.py         # Early multi-model runner
│                                   # Includes Phi-4 (Microsoft)
│
├── fine_tuned_models/
│   ├── Pipelines/                  # Multi-model pipeline scripts
│   │   ├── parakeet_fine_final.py  # Fine-tuned Parakeet pipeline for math transcription and extraction
│   │   ├── qwen_fine_final.py      # Fine-tuned Qwen2.5-Math-1.5B for formula transcription and extration using multiple Qwen modes
│   │   │                           # Compares base vs fine-tuned with detailed Excel output
│   │   └── whisper_mathbert_pipeline.py  # Whisper (transcription) + MathBERT(extraction) combined pipeline
│   │
│   └── Unified/                    # Single-model fine-tuned scripts
│       ├── granite_fine_final.py   # Fine-tuned Granite model (IBM)
│       ├── seamless_fine_final.py  # Fine-tuned SeamlessM4T (Meta)
│       └── whisper_fine_final.py   # Fine-tuned Whisper-medium (OpenAI)
│                                   # Side-by-side base vs fine-tuned analysis
│                                   # Note: Can use Whisper-Large-v3 (line 25) but memory-intensive
│                                   # Change: MODEL_NAME = "openai/whisper-large-v3"
│
├── requirements/                   # Per-model dependency files 
│   ├── granite_requirments.txt
│   ├── parakeet_requirements.txt
│   ├── qwen_fine_final_requirements.md
│   ├── seamless_fine_final_requirements.md
│   ├── whisper_fine_final_requirements.md
│   └── whisper_mathbert_pipeline_requirements.md
│
└── results/
├── qwen_pipeline_outputs2.xlsx         # Qwen pipeline evaluation results
├── seamless_full_outputs.xlsx          # SeamlessM4T evaluation results
├── whisper_full_outputs.xlsx           # Whisper full evaluation results
├── whisper_mathbert_pipeline_outputs.xlsx  # Whisper + MathBERT pipeline results
└──                                         # Includes: predictions, hallucinations, WER, CER, improvements

```

## Quick Reference

### Base Models
Scripts for evaluating pre-trained models without fine-tuning.

### Fine-Tuned Models
Scripts for training and evaluating customized models for mathematical formula transcription.

### Results
Performance metrics and benchmark data from model evaluations.'''

## Overview
This project aims to create working fine-tuned models to transcribe spoken mathematical formulas into LaTeX. The base Whisper model aims to convert the recorded audio into text output, the fine-tuned model goes directly from audio to LaTeX. The fine-tuning trains Whisper to transcriber spoken math directly into LaTeX, skipping an additional transcription step into English text first. The Qwen base model does not process audio directly but instead builds a prompt from the spoken math columns in the dataset. Qwen then generates a LaTeX formula from the text prompt. The base model goes from audio to text (spoken english) and the fine-tuned model processing text into LaTeX.

## Dataset
For this project and all training and testing I used 'abby1492/mathbridge-audio' and "OwenLeitzell/FormulaSearch merged into a single dataframe using HuggingFace method concatenate_datasets. FormulaSearch contains 1055 audio recording of mathematical formulas and includes context before and context after, as well as ground truth LaTeX equations. Mathbridge-audio contains 1000 samples of identical form. The samples are taken from Kyudan/MathBridge.






