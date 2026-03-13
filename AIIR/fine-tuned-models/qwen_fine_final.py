
"""
qwen_fine_final.py
Optimized fine-tuning of Qwen2.5-Math-1.5B for math formula transcription.
Includes validation, early stopping, and best practices for maximum performance.
Similar structure to whisper_fine_final.py, but adapted for Qwen's causal LM architecture.
There is a lot of carry over from the whisper_fine_final.py file.
"""

#-------------IMPORTS-------------------
import os
import re
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from evaluate import load
#disable wandb for account log in an verification for the model
#wandb is Weights and Biases
os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"#from HuggingFace
DATASET_NAME = "abby1492/mathbridge-audio"
SPLIT = "train"
MAX_SAMPLES = None
OUTPUT_XLSX = "qwen_math_outputs.xlsx"
OUTPUT_MODEL_DIR = "./qwen_math_best"
#Rule of thumb, batch size as big as hardware can handle
#For 50-100GB VRAM: Start with 32-64, increase until it cant handle it
#Qwen uses text features so we can increate the batch size without running out of memory
PER_DEVICE_BATCH = 32  #32 samples per batch per GPU (for 50-100GB VRAM on Turing)
GRAD_ACCUM = 2  #gradient accumulation steps (effective batch = 32 * 2 = 64) very good
fp16 = True #half precision 
NUM_EPOCHS = 20 #more epochs for better learning
LR = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1.0
# Evaluation settings
steps_per_epoch = len(SPLIT) // (PER_DEVICE_BATCH * GRAD_ACCUM) #calculate eval/save steps dynamically for specific dataset (This will help when using my custom dataset)
EVAL_STEPS = max(100, steps_per_epoch // 4)  # 4 evaluations per epoch
SAVE_STEPS = EVAL_STEPS  # should match EVAL_STEPS
EARLY_STOPPING_PATIENCE = 5 #stop training if there is no improvement for 5 epochs

#unneccassary but good to have, will use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
#prompt telling the model what to do
INSTRUCTION = (
    "The following sentence mixes spoken parts of formulas with English. "
    "Translate the part of the sentence that represents a formula into LaTeX."
)

print(f"Loading dataset {DATASET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)
#split dataset with seed for reproducibility
dataset = dataset.train_test_split(test_size=0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

#limit samples for larger datasets like my custom dataset
if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

#makes latex consistent, (x^2 = x^{2} = x^{ 2 } = x^{2} = x^{2})
def normalize_latex(tex: str) -> str:
    if tex is None:
        return ""
    s = str(tex).replace("$", "").replace("\\,", ",").replace("\\;", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"_\{\s*([^\}]+)\s*\}", r"_\1", s)
    s = re.sub(r"\^\{\s*([^\}]+)\s*\}", r"^\1", s)
    s = re.sub(r"(?<!\\){\s*([A-Za-z0-9_\\]+)\s*}", r"\1", s)
    return s
#this will:
#handle null values
#remove dollar signs and make spacing consistent
#remove left and right braces adn replace \cdot and \times with *
#normalize sub and superscripts
#remove any extra braces

#apply the normalize function to the dataset
train_dataset = train_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})
eval_dataset = eval_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

print("Loading tokenizer and model...")
#tokenize the text into token ids and loads Qwen2.5-Math-1.5B model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#set padding so tokens are the same length for batching
tokenizer.pad_token = tokenizer.eos_token
#inputs must be left padded because LLMs are trained to coninue from pad tokens
tokenizer.padding_side = "left"
#This load the Qwen model from HuggingFace and places it on the GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16 if fp16 else torch.float32
)
#Print status and the number of trainable parameters
print(f"Model loaded. Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#The model needs a prompt for instructions, this is built from the context before, spoken English, and context after columns from the dataset
def build_prompt(context_before, spoken, context_after):
    return f"{INSTRUCTION}\n\n{context_before.strip()}\n{spoken.strip()}\n{context_after.strip()}\n"
#Prepare the example for training
def prepare_example(ex):
    prompt = build_prompt(ex["context_before"], ex["spoken_English"], ex["context_after"])
    label = ex["text_norm"]
    
    full_text = prompt + label + tokenizer.eos_token
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized
#Apply the prepare_example function to the entire dataset
print("Preprocessing datasets...")
train_dataset = train_dataset.map(prepare_example, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_example, remove_columns=eval_dataset.column_names)

print(f"Preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
#Collate the data into a batch for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
#Load the word error rate and character error rate metrics
wer_metric = load("wer")
cer_metric = load("cer")
#Extract the latex from the text
def extract_latex(text):
    match = re.search(r'\$(.+?)\$', text)
    if match:
        return f"${match.group(1)}$"
    latex_match = re.search(r'\\([a-zA-Z]+(?:\{[^}]*\})*)', text)
    if latex_match:
        return f"${latex_match.group(0)}$"
    return text.strip()
#compute the metrics for the model 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {}

#calculate the number of steps per epoch for the training
steps_per_epoch = len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM)
#calculate the number of steps per epoch for the evaluation
EVAL_STEPS = max(100, steps_per_epoch // 4)
SAVE_STEPS = EVAL_STEPS

training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR, #where to save the outputs and checkpoints ./qwen_math_best
    per_device_train_batch_size=PER_DEVICE_BATCH,#batch size for training per GPU 32
    per_device_eval_batch_size=PER_DEVICE_BATCH,#batch size for evaluation per GPU 32
    gradient_accumulation_steps=GRAD_ACCUM,#effective batch size 2
    num_train_epochs=NUM_EPOCHS,#number of epochs to train for, 20
    learning_rate=LR,#learning rate for the model 1e-5
    warmup_ratio=WARMUP_RATIO,#warmup ratio for the model 0.1 = 10% of training steps
    weight_decay=WEIGHT_DECAY,#weight decay for the model 0.005 = 0.5% of the learning rate
    max_grad_norm=MAX_GRAD_NORM,#max gradient norm for the model 1.0
    fp16=fp16 and torch.cuda.is_available(),#use half precision if available
    gradient_checkpointing=True,#use gradient checkpointing to save memory
    
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS, 
    save_strategy="steps",
    save_steps=SAVE_STEPS,#save every 100 steps
    save_total_limit=3,#save only the best 3 checkpoints
    load_best_model_at_end=True,#load the best model at the end
    metric_for_best_model="eval_loss",#use the evaluation loss to determine the best model
    greater_is_better=False,
    
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",
    
    remove_unused_columns=False,
    dataloader_num_workers=4, #how many subprocesses are used to load data in parallel
    dataloader_pin_memory=True,
    optim="adamw_torch",#using AdamW optimizer
    lr_scheduler_type="cosine", #how to adjust the learning rate during excecution
)
#stop training if there is no improvement for 5 epochs
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001 #0.1% improvement is the minimum to be considered an improvement
)
#Trainer object for the training loop
trainer = Trainer(
    model=model,#the model to train
    args=training_args,#configuration for the training
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,#how to batch the data to be loaded into the model
    tokenizer=tokenizer,#tokenizer for saving the model
    callbacks=[early_stopping],
)
#print the training information so I can track the progess
print("\n" + "="*50)
print("Starting optimized fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Total training steps: {len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM) * NUM_EPOCHS}")
print(f"Eval steps: {EVAL_STEPS}, Save steps: {SAVE_STEPS}")
print("="*50 + "\n")
#start the training loop and train the model 
trainer.train()
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"\nBest model saved to {OUTPUT_MODEL_DIR}")

print("\n" + "="*50)
print("Running final evaluation on validation set...")
print("="*50 + "\n")
#find the hallucinations in the predictions, anything in the prediction that is not in the ground truth
def detect_hallucination(pred: str, gt: str):
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred.lower())
    g_toks = tok_pat.findall(gt.lower())
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"
#generate the prediction from the model as the prompt is fed in
def generate_prediction(model, tokenizer, prompt_text, device):
    model.eval()#switch to evaluation mode
    with torch.no_grad():#disabling the gradient computation saves memory
        encoded = tokenizer(
            prompt_text,#what text is inputted to the model (context before, spoken English, context after)
            return_tensors="pt",#return PyTorch tensors
            padding=True,#pad the input to the same length because the model needs consistent input length
            truncation=True,#truncate the input to the same length because the model needs consistent input length
            max_length=512,#the maximum length of the input (truncate if longer)
        ).to(device)#move the input to GPU
        
        outputs = model.generate(
            **encoded,#get the encoded input
            max_new_tokens=64,#only generate 64 new tokens
            num_beams=4,#use 4 beams to generate the prediction for finding the best path through the tree
            do_sample=False,#this ensures there is no randomness (not multinomial sampling https://huggingface.co/docs/transformers/en/generation_strategies)
            early_stopping=True,#use early stopping to prevent overfitting
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)#decode the tokens
        continuation = generated[len(prompt_text):].strip()#remove the prompt from the prediction so we only get what the model generated
        latex = extract_latex(continuation)#extract the prediciton LaTeX
        return normalize_latex(latex)#normalize the LaTeX after it is extracted
#load the base model for comparison, this has same config as the fine tuned model but not fine tuned
print("Loading base model for comparison...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16 if fp16 else torch.float32
)
base_model.eval() #use just the evaluation mode

eval_data_full = dataset["test"]#use the test set (200 samples)
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})#normalize the inputs

model.eval()#fine tuned model evaluation mode
rows = []#a list to store the results

print(f"Evaluating {len(eval_data_full)} examples")#print the number of samples (should be 200 if using the Abby dataset)
for i in range(len(eval_data_full)):#loop through all samples
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_full)}")#print the progress so I know it is working every 20 samples
    
    ex = eval_data_full[i]#get current sample
    prompt_text = build_prompt(ex["context_before"], ex["spoken_English"], ex["context_after"])#build the prompt for the the specific sample
    gt = ex["text_norm"]#get the ground truth
    raw_input = f'{ex["context_before"]} {ex["spoken_English"]} {ex["context_after"]}'#create input string
    
    pred_base = generate_prediction(base_model, tokenizer, prompt_text, device)#make the model generate the prediction for the base model
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt])
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt])
    hall_base = detect_hallucination(pred_base, gt)
    
    pred_ft = generate_prediction(model, tokenizer, prompt_text, device)#make the model generate the prediction for the fine tuned model
    wer_ft = wer_metric.compute(predictions=[pred_ft], references=[gt])
    cer_ft = cer_metric.compute(predictions=[pred_ft], references=[gt])
    hall_ft = detect_hallucination(pred_ft, gt)
    #use the rows list to store the results
    rows.append({
        "id": i,
        "input": raw_input,
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

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)#output the results to an organized excel file
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

#print the summary statistics for the base and fine tuned models to the console
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
