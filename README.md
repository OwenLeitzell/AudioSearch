# Math Speech Direct

Transcription models for converting spoken mathematical formulas into LaTeX. Includes evaluation scripts for base models and fine-tuning pipelines for Whisper, Qwen, and Parakeet.

## Quick Start

```bash
# Install dependencies
./bin/install              # Basic install
./bin/install --with-nemo  # Include NeMo for Qwen/Parakeet

# Activate environment
conda activate math-speech

# Run evaluation
./bin/evaluate whisper

# Run fine-tuning
./bin/finetune whisper
```

## Project Structure

```
math-speech-direct/
├── bin/
│   ├── install            # Create conda env and install dependencies
│   ├── evaluate           # Run evaluation scripts
│   └── finetune           # Run fine-tuning scripts
│
├── evaluation/
│   ├── whisper_evaluate.py         # Whisper large-v3 evaluation
│   ├── whisper_context_evaluate.py # Whisper with context combination
│   ├── qwen_evaluate.py            # NVIDIA Canary Qwen 2.5B evaluation
│   ├── parakeet_evaluate.py        # NVIDIA Parakeet CTC 1.1B evaluation
│   └── granite_evaluate.py         # IBM Granite Speech 3.3 8B evaluation
│
├── finetuning/
│   ├── whisper_finetune.py         # Whisper fine-tuning (medium)
│   ├── whisper_finetune_lora.py    # Whisper LoRA fine-tuning (large-v3)
│   ├── qwen_finetune.py            # Qwen 2.5 Math fine-tuning
│   ├── parakeet_finetune.py        # Parakeet CTC fine-tuning
│   └── granite_finetune.py         # Granite Speech fine-tuning
│
├── models/                # Fine-tuned model outputs (git-ignored)
├── results/               # Evaluation results (git-ignored)
├── requirements.txt
└── README.md
```

## Models

| Model | Type | Base Model | Task |
|-------|------|------------|------|
| Whisper | ASR | openai/whisper-medium | Audio → LaTeX |
| Qwen | ASR | nvidia/canary-qwen-2.5b | Audio → Text |
| Parakeet | ASR | nvidia/parakeet-ctc-1.1b | Audio → Text |
| Granite | ASR | ibm-granite/granite-speech-3.3-8b | Audio → Text |

## Usage

### Evaluation

Evaluate pre-trained models on the MathBridge dataset:

```bash
./bin/evaluate whisper    # Run Whisper evaluation
./bin/evaluate qwen       # Run Qwen evaluation
./bin/evaluate parakeet   # Run Parakeet evaluation
./bin/evaluate all        # Run all evaluations
```

Results are saved to `results/` with metrics including BLEU, SacreBLEU, ROUGE-L, WER, and CER.

### Fine-tuning

Fine-tune models for math formula transcription:

```bash
./bin/finetune whisper    # Fine-tune Whisper
./bin/finetune qwen       # Fine-tune Qwen
./bin/finetune parakeet   # Fine-tune Parakeet
./bin/finetune all        # Fine-tune all models
```

Fine-tuned models are saved to `models/`. Each script compares base vs fine-tuned performance and outputs detailed results to `results/`.

## Dataset

Training and evaluation uses the [abby1492/mathbridge-audio](https://huggingface.co/datasets/abby1492/mathbridge-audio) dataset, which contains 1000 audio recordings of mathematical formulas with:

- `audio` - Audio recording of spoken formula
- `spoken_english` - Transcription of spoken math
- `equation` - Ground truth LaTeX
- `context_before` / `context_after` - Surrounding context

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- conda (for environment management)

### Dependencies

Core dependencies are installed via `./bin/install`:

- PyTorch with CUDA support
- Transformers, Datasets, Accelerate
- NeMo toolkit (optional, for Qwen/Parakeet)
- Evaluation metrics (BLEU, ROUGE, WER, CER)

See `requirements.txt` for full list.

## Overview

This project explores different approaches to transcribing spoken math:

- **Whisper**: Fine-tuned to go directly from audio to LaTeX, skipping the intermediate English transcription step
- **Qwen**: NVIDIA's Canary Qwen ASR model fine-tuned for math speech recognition
- **Parakeet**: NVIDIA's Parakeet ASR model fine-tuned for math speech recognition
- **Granite**: IBM's speech-language model with strong ASR capabilities

The fine-tuning process adds special tokens for LaTeX symbols (Greek letters, math operators, etc.) and trains on the MathBridge dataset to improve math-specific transcription accuracy.
