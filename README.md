# AudioSearch
# Math Formula Transcription Models

This repository contains model evaluation scripts for transcribing spoken math phrases and forumlas into LaTeX. This repository contains both base models and fine-tuned models for Whisper (OpenAI) and Qwen (Nvidia)

# AudioSearch

## Project Structure
```
AudioSearch/
│
├── base_models/                    (ChatGPT/Cursor assisted)
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
│   ├── qwen_fine_final.py         # Fine-tuned Qwen2.5-Math-1.5B for formula transcription
│   │                               # Compares base vs fine-tuned with detailed Excel output
│   │
│   ├── whisper_fine_final.py      # Fine-tuned Whisper-medium for math transcription
│   │                               # Side-by-side base vs fine-tuned analysis
│   │                               # Note: Can use Whisper-Large-v3 (line 25) but memory-intensive
│   │                               # Change: MODEL_NAME = "openai/whisper-large-v3"
│   │
│   └── whisper_fine_lora.py       # Early LoRA fine-tuning attempt
│                                   # Parameter-efficient fine-tuning for math transcription
│
└── results/
    ├── math-qwen_speed_times.json # Qwen model timing benchmarks
    └── whisper_math_outputs.xlsx  # Whisper evaluation results
                                    # Includes: predictions, hallucinations, WER, CER, improvements
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
For this project and all training and testing I used 'abby1492/mathbridge-audio'. This contains 1000 audio recording of mathematical formulas and includes context before and context after, as well as ground truth LaTeX equations. The sampels are taken from Kyudan/MathBridge.
I am constructing my own dataset mirrored after abby1492 dataset with context_before, context_after, ground truth equations, and spoken_english. The goal is that this dataset can be interchanged with the abby1492 dataset in the models without adapting the models. I am collecting new recording from MathBridge without overlapping with abby1492 samples.






