# AudioSearch
# Math Formula Transcription Models

This repository contains model evaluation scripts for transcribing spoken math phrases and forumlas into LaTeX. This repository contains both base models and fine-tuned models for Whisper (OpenAI) and Qwen (Nvidia)

## Repository Structure
'''
AudioSearch/
|--base_models/ (ChatGPT/Cursor assisted)
  |--Qwen_evaluate.py #My first attempt at Qwen, very crude and without tuning. Evaluates the base Qwen ASR model (nvidia/canary-qwen-2.5b) on MathBridge audio dataset with BLEU, ROUGE, WER, and CER evaluation metrics.
  |--Real_Whisper_evaluate.py #Built off the first attempt to get a working output. Evaluates the base Whisper-large-v3 model on MathBridge audio dataset with before and after context.
  |--Whisper_evaluate.py #Evaluation of Whisper-large-v3 base model with Bleu, Sacrebleu, wer, cer, and rougeL
  |--evaluate_models.py #Early attempt at running all models, includes Phi-4 (Microsoft)
  
|--fine_tuned_models
  |--qwen_fine_final.py #The fine-tuned Qwen2.5-Math-1.5B model for formula transcription, runs both base and fine-tuned model and compares with a detailed Excel file output.
  |--whisper_fine_final.py #The fine-tuned Whisper-medium model for math formula transcription with side by side analysis of base vs fine-tuned model. This program can be changed to use Whisper-Large-v3, though, it costs heavily in memory. Change line 25 MODEL_NAME = "openai/whisper-medium" to "open
  |--whisper_fine_lora.py #Early attempt at fine-tuning Whisper using LoRA (parameter-efficient fine-tuning) for math transcription.

|--results
  |--math-qwen_speed_times.json #Qwen model timing for benchmarking performance.
  |--whisper_math_outputs.xlsx #Excel file with evaluation results for comparision between base and fine-tuned whisper model. Including predictions, hallucinations, WER, CER, and improvements.
'''

## Overview
This project aims to create working fine-tuned models to transcribe spoken mathematical formulas into LaTeX. The base Whisper model aims to convert the recorded audio into text output, the fine-tuned model goes directly from audio to LaTeX. The fine-tuning trains Whisper to transcriber spoken math directly into LaTeX, skipping an additional transcription step into English text first. The Qwen base model does not process audio directly but instead builds a prompt from the spoken math columns in the dataset. Qwen then generates a LaTeX formula from the text prompt. The base model goes from audio to text (spoken english) and the fine-tuned model processing text into LaTeX.

## Dataset
For this project and all training and testing I used 'abby1492/mathbridge-audio'. This contains 1000 audio recording of mathematical formulas and includes context before and context after, as well as ground truth LaTeX equations. The sampels are taken from Kyudan/MathBridge.
I am constructing my own dataset mirrored after abby1492 dataset with context_before, context_after, ground truth equations, and spoken_english. The goal is that this dataset can be interchanged with the abby1492 dataset in the models without adapting the models. I am collecting new recording from MathBridge without overlapping with abby1492 samples.






