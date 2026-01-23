"""
whisper_fine_final.py
Optimized fine-tuning of Whisper for math formula transcription.
Includes validation, early stopping, and best practices for maximum performance.
"""

import os
import re
import torch
import torchaudio
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from evaluate import load
from dataclasses import dataclass
from typing import Dict, List, Union

#No account log in required for this model to work
os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "openai/whisper-medium"  # or "openai/whisper-large-v3" on more GPUs
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
MAX_SAMPLES = None  
OUTPUT_XLSX = "../results/whisper_math_outputs2.xlsx"
OUTPUT_MODEL_DIR = "../models/whisper_math_best"


# Optimized training hyperparameters
#Whisper uses audio features which use more memory than text, need a smaller batch size
PER_DEVICE_BATCH = 16  #16 samples per batch per GPU (for 50-100GB VRAM on Turing)
GRAD_ACCUM = 2 #effective batch size = 2 * 16 = 32, more GPUs = more effective batch size
fp16 = True #use True to save memory if required, False for more accuracy
NUM_EPOCHS = 20  # More epochs for better learning
LR = 1e-5  #also checked 2e-5 for full fine tuning
WARMUP_RATIO = 0.1  # Warm up for 10% of training
WEIGHT_DECAY = 0.005 #also checked 0.01 for full fine tuning
MAX_GRAD_NORM = 1.0  

# Evaluation settings
steps_per_epoch = len(SPLIT) // (PER_DEVICE_BATCH * GRAD_ACCUM) #calculate eval/save steps dynamically for specific dataset (This will help when using my custom dataset)
EVAL_STEPS = max(100, steps_per_epoch // 4)  # 4 evaluations per epoch
SAVE_STEPS = EVAL_STEPS  # should match EVAL_STEPS
EARLY_STOPPING_PATIENCE = 5  # increased from 3 for more tolerance to overfitting

#relativly unneccessary but good to have, will use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------- LOAD DATASET -------------------- #
print(f"Loading dataset {DATASET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)
#16000 is whispers required rate
dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

# Split into train and validation (80/20 split)
#seed used for reproducibility
dataset = dataset.train_test_split(test_size = 0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

#This will also help when using my custom dataset or larger datasets, you can use this to limit the dataset to a certain number of samples
if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

# Normalize LaTeX
#So x^2 x^{2} and x^{ 2 } are all the same
def normalize_latex(tex: str) -> str:
    if tex is None: return ""
    #Remove dollar signs and spacing 
    s = str(tex).replace("$", "").replace("\\,", ",").replace("\\;", " ").strip()
    #Remove multiple spaces
    s = re.sub(r"\s+", " ", s)
    #Remove left and right braces and replace \cdot and \times with *
    s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\times", "*")
    #normalize sub and superscripts
    s = re.sub(r"_\{\s*([^\}]+)\s*\}", r"_\1", s)
    s = re.sub(r"\^\{\s*([^\}]+)\s*\}", r"^\1", s)
    s = re.sub(r"(?<!\\){\s*([A-Za-z0-9_\\]+)\s*}", r"\1", s)
    return s

#make sure the normalize function is applied to the dataset
train_dataset = train_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})
eval_dataset = eval_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# -------------------- MODEL & TOKENIZER -------------------- #
#processor used to process the audio and tokenize the text (using whisper)
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tk = processor.tokenizer

# Extra tokens for math + Greek (not in whispers vocab)
extra_tokens = ["{","}","^","_","/","*","[","]","(",")","+/-","\\pm","\\in","\\Sigma","\\Gamma",
                "\\alpha","\\beta","\\gamma","\\delta","\\epsilon","\\zeta","\\eta","\\theta",
                "\\lambda","\\mu","\\pi","\\rho","\\sigma","\\tau","\\phi","\\chi","\\psi","\\omega",
                "\\mathcal","\\mathbf","\\boldsymbol","\\bar","\\prime","\\sqrt","\\frac","\\sum",
                "\\int","\\partial","\\nabla","\\infty","\\leq","\\geq","\\neq","\\approx","\\Phi","\\Pi"]
extra_tokens = [t for t in dict.fromkeys(extra_tokens) if t not in tk.get_vocab()]#Ensure no duplicates
if extra_tokens:
    tk.add_tokens(extra_tokens)
    print(f"Added {len(extra_tokens)} special tokens")#Print number of extra tokens added

# Load Whisper model 
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
#have to accomidate new tokens
model.resize_token_embeddings(len(tk))
#choose the english language and transcribe task, whisper can also translate
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
#use an empty list to suppress no tokens
model.config.suppress_tokens = []
#disable caching for training, will save memroy but hurt inference speed
model.config.use_cache = False  # Disable for training

print(f"Model loaded. Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -------------------- PREPROCESS FUNCTION -------------------- #
def prepare_example(ex):
    #Preloaded audio list of numbers representing audio waves
    audio_array = ex[AUDIO_COLUMN]["array"]
    #Process audio mel-spectrogram
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    #Tokenize ground truth labels
    labels = tk(ex["text_norm"]).input_ids

    return {
        "input_features": inputs.input_features[0],
        #Target output
        "labels": labels

    }

# Map datasets without casting to Audio, only keep the input features and labels
remove_cols = [col for col in train_dataset.column_names if col not in ["input_features", "labels"]]
train_dataset = train_dataset.map(prepare_example, remove_columns=remove_cols)
eval_dataset = eval_dataset.map(prepare_example, remove_columns=remove_cols)

print(f"Preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Stack input features
        input_features = torch.stack([torch.tensor(f["input_features"]) for f in features])
        
        # Pad labels to the same length for batching
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        #Model wont be penalized for padding being wrong
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Remove BOS (beginning of sequence) token if present (model adds it)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        return {
            "input_features": input_features,
            "labels": labels
        }
#create a data collator instance to handle the padding and batching
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# -------------------- METRICS -------------------- #
wer_metric = load("wer") #word error rate
cer_metric = load("cer") #character error rate
#Used in validation and evaluation to monitor progress and performance
def compute_metrics(pred): #compute the metrics for the model
    pred_ids = pred.predictions #predictions from the model (ids)
    label_ids = pred.label_ids #ground truth labels (ids)
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = tk.pad_token_id #replace padding with -100 to ignore in loss
    
    # Decode predictions and labels
    pred_str = tk.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tk.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize
    pred_str = [normalize_latex(p) for p in pred_str]
    label_str = [normalize_latex(l) for l in label_str]
    
    # Compute metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "cer": cer}

# -------------------- TRAINING -------------------- #
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR, #output directory for the model (save results)
    per_device_train_batch_size=PER_DEVICE_BATCH, #batch size for training
    per_device_eval_batch_size=PER_DEVICE_BATCH, #batch size for evaluation
    gradient_accumulation_steps=GRAD_ACCUM, #gradients
    num_train_epochs=NUM_EPOCHS, #number of epochs to train for
    learning_rate=LR, #learning rate for the model
    warmup_ratio=WARMUP_RATIO, #warmup ratio for the model
    weight_decay=WEIGHT_DECAY, #weight decay for the model
    max_grad_norm=MAX_GRAD_NORM, #max gradient norm for the model
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True, #use gradient checkpointing to save memory
    
    # Evaluation and saving
    evaluation_strategy="steps", #evaluate every n steps
    eval_steps=EVAL_STEPS,
    save_strategy="steps", #save every n steps
    save_steps=SAVE_STEPS, #save every n steps  
    save_total_limit=3,  # Keep only best 3 checkpoints
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="wer",  # Use WER to determine best model
    greater_is_better=False,  # Lower WER is better
    
    # Logging
    logging_steps=50, #log every 50 steps
    logging_first_step=True,#and the first step
    report_to="tensorboard",  # Use tensorboard for monitoring
    
    # Generation settings for evaluation
    predict_with_generate=True,
    generation_max_length=225,
    
    # Other settings
    remove_unused_columns=False,
    label_names=["labels"], #make sure labels match the model's input
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",  # Use PyTorch AdamW
    lr_scheduler_type="cosine",  # Cosine learning rate schedule
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001  # Minimum improvement threshold
)
#this handels the training loop using the training arguments and the dataset
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)
#exectute the training loop
print("\n" + "="*50)
print("Starting optimized fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Total training steps: {len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM) * NUM_EPOCHS}")
print("="*50 + "\n")

trainer.train() #fine tuned model training (DO THE TRAINING!)
trainer.save_model(OUTPUT_MODEL_DIR)#save the model for future use 
processor.save_pretrained(OUTPUT_MODEL_DIR)#save the processor for future use
print(f"\nBest model saved to {OUTPUT_MODEL_DIR}")#print the path so I can find it later

# -------------------- FINAL EVALUATION -------------------- #
#train and save 
print("\n" + "="*50)
print("Running final evaluation on validation set...")
print("="*50 + "\n")
#Finds all tokens in the prediction that werent in the ground truth
#very important step for finding how the fine tuned model deviates from the ground truth
def detect_hallucination(pred: str, gt: str):
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]#any extra tokens (not in ground truth)
    return ",".join(halluc) if halluc else "none"

def generate_prediction(model, input_feat):
    model.eval() #DO THE EVALUATION!
    with torch.no_grad(): #freeze from updating
        ids = model.generate(
            input_features=torch.tensor(input_feat).unsqueeze(0).to(device),
            max_length=225,
            num_beams=5,  # Use beam search for better results(5 beams or best paths through the tree)
            #Takes five best predictions for better quality
        )
    return normalize_latex(tk.decode(ids[0], skip_special_tokens=True))

# Load base model for comparison
#uses the same config as the fine tuned model and dataset
print("Loading base model for comparison...")
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
base_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
base_model.eval() #no training just evaluation
# Load evaluation data with original fields
eval_data_full = dataset["test"] #on the test set (200 samples)
eval_data_full = eval_data_full.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
#normalize the LaTeX
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# Prepare for prediction
eval_data_processed = eval_data_full.map(
    lambda ex: {"input_features": processor(ex[AUDIO_COLUMN]["array"], sampling_rate=16000).input_features[0]}
)

model.to(device)
model.eval()
rows = []

#Unfine tuned model
print(f"Evaluating {len(eval_data_processed)} examples")
for i in range(len(eval_data_processed)):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_processed)}")#Print the progress so I know it is working
    #original vs preprocessed
    ex_full = eval_data_full[i]
    ex_proc = eval_data_processed[i]
    
    gt = ex_full["text_norm"]#save the ground truth
    input_feat = ex_proc["input_features"]
    audio_path = ex_full[AUDIO_COLUMN].get("path", "unknown")

    # Base model prediction
    pred_base = generate_prediction(base_model, input_feat)
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt])
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt])
    hall_base = detect_hallucination(pred_base, gt)

    # Fine-tuned model prediction
    pred_ft = generate_prediction(model, input_feat)
    wer_ft = wer_metric.compute(predictions=[pred_ft], references=[gt])
    cer_ft = cer_metric.compute(predictions=[pred_ft], references=[gt])
    hall_ft = detect_hallucination(pred_ft, gt)
#save results to the output excel file
    rows.append({
        "id": i,
        "audio_file": audio_path,
        "GT_LaTeX": gt,
        "Base_Pred_LaTeX": pred_base,
        "FineTuned_Pred_LaTeX": pred_ft,
        "Base_WER": wer_base,
        "Base_CER": cer_base,
        "Base_Hallucination": hall_base,
        "FineTuned_WER": wer_ft,
        "FineTuned_CER": cer_ft,
        "FineTuned_Hallucination": hall_ft,
        "WER_Improvement": wer_base - wer_ft,
        "CER_Improvement": cer_base - cer_ft,
    })

# Save detailed results
df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

# Print summary statistics
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"\nBase Model Performance:")
print(f"  Average WER: {df['Base_WER'].mean():.4f}")
print(f"  Average CER: {df['Base_CER'].mean():.4f}")
print(f"\nFine-Tuned Model Performance:")
print(f"  Average WER: {df['FineTuned_WER'].mean():.4f}")
print(f"  Average CER: {df['FineTuned_CER'].mean():.4f}")
print(f"\nImprovement:")
print(f"  WER Reduction: {df['WER_Improvement'].mean():.4f} ({df['WER_Improvement'].mean() / df['Base_WER'].mean() * 100:.1f}%)")
print(f"  CER Reduction: {df['CER_Improvement'].mean():.4f} ({df['CER_Improvement'].mean() / df['Base_CER'].mean() * 100:.1f}%)")
print(f"\n% of samples improved: {(df['WER_Improvement'] > 0).sum() / len(df) * 100:.1f}%")
print("="*50 + "\n")