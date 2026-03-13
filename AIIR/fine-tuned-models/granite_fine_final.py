"""
granite_finetune.py
Fine-tuning of IBM Granite Speech model for math formula transcription.

Granite Speech is an LLM-based speech model (CTC Encoder + Q-Former Projector + Granite LLM).
Unlike Whisper (encoder-decoder), it uses a decoder-only LLM with audio embeddings prepended.
Training requires: encoding audio -> prepending to decoder inputs -> causal LM loss.
"""

import os
import re
import io
import math
import torch
import torchaudio
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from evaluate import load
from dataclasses import dataclass
from typing import Dict, List, Union, Any

os.environ["WANDB_DISABLED"] = "true"

# -------------------- CONFIG -------------------- #
MODEL_NAME = "ibm-granite/granite-speech-3.3-8b"
DATASET_NAME = "abby1492/mathbridge-audio"
AUDIO_COLUMN = "audio"
SPLIT = "train"
MAX_SAMPLES = None
OUTPUT_XLSX = "../results/granite_math_outputs.xlsx"
OUTPUT_MODEL_DIR = "../models/granite_math_best"

# Training hyperparameters
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 8  # Effective batch size = 4 * 8 = 32
fp16 = True
NUM_EPOCHS = 1
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if torch.cuda.is_available():
    print(f"CUDA is available - GPU training enabled")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

# -------------------- LOAD DATASET -------------------- #
print(f"Loading dataset {DATASET_NAME}, split {SPLIT}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)

# Inspect audio column structure
sample = dataset[0]
if AUDIO_COLUMN in sample:
    audio_info = sample[AUDIO_COLUMN]
    if isinstance(audio_info, dict):
        print(f"Audio column keys: {list(audio_info.keys())}")
        if "bytes" in audio_info and audio_info["bytes"] is not None:
            print("Dataset provides audio bytes - will use bytes for loading")
        elif "path" in audio_info:
            print(f"Dataset provides audio path: {audio_info['path']}")

# Split into train and validation
dataset = dataset.train_test_split(test_size=0.2, seed=41)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

if MAX_SAMPLES:
    train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

# -------------------- NORMALIZE LATEX -------------------- #
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

train_dataset = train_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})
eval_dataset = eval_dataset.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

# -------------------- MODEL & PROCESSOR -------------------- #
print(f"Loading Granite Speech model: {MODEL_NAME}")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = processor.tokenizer

# Debug processor structure
print(f"Processor type: {type(processor)}")
print(f"Processor attributes: {[a for a in dir(processor) if not a.startswith('_') and not callable(getattr(processor, a, None))]}")
for attr_name in ['feature_extractor', 'audio_processor', 'audio_feature_extractor']:
    val = getattr(processor, attr_name, None)
    if val is not None:
        print(f"  processor.{attr_name} = {type(val)}")

# Add LaTeX-specific tokens
extra_tokens = [
    "{", "}", "^", "_", "/", "*", "[", "]", "(", ")", "+/-", "\\pm", "\\in",
    "\\Sigma", "\\Gamma", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon",
    "\\zeta", "\\eta", "\\theta", "\\lambda", "\\mu", "\\pi", "\\rho", "\\sigma",
    "\\tau", "\\phi", "\\chi", "\\psi", "\\omega", "\\mathcal", "\\mathbf",
    "\\boldsymbol", "\\bar", "\\prime", "\\sqrt", "\\frac", "\\sum", "\\int",
    "\\partial", "\\nabla", "\\infty", "\\leq", "\\geq", "\\neq", "\\approx",
    "\\Phi", "\\Pi", "\\Delta", "\\Omega", "\\Lambda", "\\Theta", "\\Xi",
    "\\prod", "\\lim", "\\log", "\\ln", "\\exp", "\\sin", "\\cos", "\\tan",
    "\\arcsin", "\\arccos", "\\arctan", "\\sinh", "\\cosh", "\\tanh",
    "\\det", "\\dim", "\\ker", "\\im", "\\rank", "\\tr", "\\span",
    "\\cup", "\\cap", "\\setminus", "\\subset", "\\subseteq", "\\supset", "\\supseteq",
    "\\in", "\\notin", "\\exists", "\\forall", "\\land", "\\lor", "\\neg",
    "\\rightarrow", "\\leftarrow", "\\leftrightarrow", "\\Rightarrow", "\\Leftarrow",
    "\\Leftrightarrow", "\\equiv", "\\sim", "\\cong", "\\simeq", "\\propto"
]
extra_tokens = [t for t in dict.fromkeys(extra_tokens) if t not in tokenizer.get_vocab()]
if extra_tokens:
    tokenizer.add_tokens(extra_tokens)
    print(f"Added {len(extra_tokens)} LaTeX special tokens to vocabulary")
    print(f"New vocabulary size: {len(tokenizer)}")

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# Resize token embeddings for new tokens
if extra_tokens:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens")

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added [PAD] token, vocab size now {len(tokenizer)}")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
model.config.pad_token_id = tokenizer.pad_token_id

# Ensure pad != eos to avoid attention mask warnings
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token_id = len(tokenizer) - 2
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Set pad_token_id={tokenizer.pad_token_id} (different from eos_token_id={tokenizer.eos_token_id})")

model.config.use_cache = False

# Debug model structure
print(f"\nModel type: {type(model)}")
print(f"Has encoder: {hasattr(model, 'encoder')} -> {type(getattr(model, 'encoder', None))}")
print(f"Has projector: {hasattr(model, 'projector')} -> {type(getattr(model, 'projector', None))}")
print(f"Has language_model: {hasattr(model, 'language_model')} -> {type(getattr(model, 'language_model', None))}")

# -------------------- MODEL WRAPPER -------------------- #
# Granite Speech is an LLM-based model (not encoder-decoder like Whisper).
# Architecture: CTC Encoder -> Q-Former Projector -> Granite LLM
# The LLM is decoder-only, so audio embeddings are PREPENDED to text token embeddings.
# For training: audio_embeds + decoder_input_embeds -> LLM -> loss on text portion.
# Labels must be expanded to match the full sequence (audio + text) length.

class GraniteSpeechWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        object.__setattr__(self, 'model', model)
        self._modules['model'] = model
        self._warned_empty_supervision = False
    
    def forward(self, input_features=None, labels=None, attention_mask=None,
                input_ids=None, inputs_embeds=None, **kwargs):
        wrapped_model = object.__getattribute__(self, 'model')
        
        # If we have input_features, we need to encode audio and build the full sequence
        if input_features is not None:
            # Step 1: Encode audio through CTC encoder + Q-Former projector
            encoder_outputs = wrapped_model.encoder(input_features)
            if hasattr(encoder_outputs, "last_hidden_state") and encoder_outputs.last_hidden_state is not None:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, (tuple, list)) and len(encoder_outputs) > 0:
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs
            audio_embeds = wrapped_model.projector(encoder_hidden_states)
            # audio_embeds: [batch, N_audio, hidden_size]
            embed_fn = wrapped_model.language_model.get_input_embeddings()
            lm_device = embed_fn.weight.device
            audio_embeds = audio_embeds.to(lm_device)
            
            batch_size = audio_embeds.shape[0]
            audio_len = audio_embeds.shape[1]
            
            if labels is not None:
                # Keep label IDs in a safe range to avoid CUDA device-side asserts in CE loss.
                vocab_size = wrapped_model.language_model.get_input_embeddings().num_embeddings
                labels = labels.clone()
                # Preserve ignore index, invalidate any other negative IDs.
                labels[(labels < 0) & (labels != -100)] = -100
                # IDs outside vocab are mapped to a safe token to preserve supervision.
                safe_id = getattr(wrapped_model.config, "unk_token_id", None)
                if safe_id is None:
                    safe_id = getattr(wrapped_model.config, "eos_token_id", None)
                if safe_id is None:
                    safe_id = getattr(wrapped_model.config, "pad_token_id", 0)
                safe_id = int(min(max(safe_id, 0), vocab_size - 1))
                labels[labels >= vocab_size] = safe_id

                # Step 2: Create decoder input embeddings from labels
                # Replace -100 (ignore index) with pad_token_id for embedding lookup
                decoder_ids = labels.clone()
                decoder_ids[decoder_ids == -100] = wrapped_model.config.pad_token_id or 0
                
                # Clamp to valid vocab range
                decoder_ids = decoder_ids.clamp(0, vocab_size - 1)
                
                # Shift right: prepend BOS token, drop last token
                bos_id = getattr(wrapped_model.config, 'bos_token_id', None)
                if bos_id is None:
                    bos_id = getattr(wrapped_model.config, 'decoder_start_token_id', 0) or 0
                bos_id = min(bos_id, vocab_size - 1)
                
                bos_col = torch.full(
                    (batch_size, 1), bos_id,
                    dtype=decoder_ids.dtype, device=decoder_ids.device
                )
                shifted_decoder_ids = torch.cat([bos_col, decoder_ids[:, :-1]], dim=1)
                
                # Get text embeddings
                decoder_embeds = embed_fn(shifted_decoder_ids.to(lm_device))
                
                # Step 3: Concatenate [audio_embeds, decoder_embeds]
                combined_embeds = torch.cat([audio_embeds, decoder_embeds], dim=1)
                
                # Step 4: Expand labels to match combined sequence length
                # Labels for audio positions are -100 (ignored in loss)
                audio_labels = torch.full(
                    (batch_size, audio_len), -100,
                    dtype=labels.dtype, device=lm_device
                )
                expanded_labels = torch.cat([audio_labels, labels.to(lm_device)], dim=1)
                
                # Step 5: Create attention mask for full sequence
                total_len = combined_embeds.shape[1]
                if attention_mask is not None:
                    audio_mask = torch.ones(
                        batch_size, audio_len,
                        dtype=attention_mask.dtype, device=lm_device
                    )
                    combined_mask = torch.cat([audio_mask, attention_mask.to(lm_device)], dim=1)
                else:
                    combined_mask = torch.ones(
                        batch_size, total_len,
                        dtype=torch.long, device=lm_device
                    )
                
                # Step 6: Run through language model
                lm_outputs = wrapped_model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                )
                logits = lm_outputs.logits
                
                # Step 7: Compute causal LM loss (shifted)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = expanded_labels[..., 1:].to(logits.device).contiguous()
                shift_mask = combined_mask[..., 1:].to(logits.device).contiguous()
                
                # Apply mask to select valid (non-padding) positions
                valid_logits = shift_logits[shift_mask != 0]
                valid_labels = shift_labels[shift_mask != 0]

                # Extra safety: ensure labels are valid for cross entropy.
                valid_labels[(valid_labels < 0) & (valid_labels != -100)] = -100
                valid_labels[valid_labels >= logits.shape[-1]] = safe_id

                # Avoid NaN when a micro-batch has no valid supervised tokens.
                if valid_logits.numel() == 0:
                    if not self._warned_empty_supervision:
                        print("WARNING: Empty supervised token set in batch; returning zero loss for this batch.")
                        self._warned_empty_supervision = True
                    zero_loss = logits.sum() * 0.0
                    return CausalLMOutputWithPast(loss=zero_loss, logits=logits)
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    valid_logits.view(-1, logits.shape[-1]),
                    valid_labels.view(-1)
                )
                
                return CausalLMOutputWithPast(loss=loss, logits=logits)
            
            else:
                # No labels (inference/generation) - just return audio context
                lm_outputs = wrapped_model.language_model(
                    inputs_embeds=audio_embeds,
                    attention_mask=attention_mask.to(lm_device) if attention_mask is not None else None,
                )
                return CausalLMOutputWithPast(logits=lm_outputs.logits)
        
        else:
            # No input_features - pass through to model normally
            return wrapped_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask,
                **kwargs
            )
    
    def generate(self, input_features=None, **kwargs):
        """Generate text from audio features."""
        wrapped_model = object.__getattribute__(self, 'model')
        
        if input_features is not None:
            # Encode audio
            encoder_outputs = wrapped_model.encoder(input_features)
            if hasattr(encoder_outputs, "last_hidden_state") and encoder_outputs.last_hidden_state is not None:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, (tuple, list)) and len(encoder_outputs) > 0:
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs
            audio_embeds = wrapped_model.projector(encoder_hidden_states)
            lm_device = wrapped_model.language_model.get_input_embeddings().weight.device
            audio_embeds = audio_embeds.to(lm_device)
            
            # Generate continuation from audio context
            # Remove input_features from kwargs to avoid double processing
            kwargs.pop('input_features', None)
            # Trainer may pass text-side masks/labels that do not match audio_embeds length.
            # For generation from audio context, build a fresh attention mask aligned to audio_embeds.
            kwargs.pop('labels', None)
            trainer_attention_mask = kwargs.pop('attention_mask', None)
            audio_attention_mask = torch.ones(
                audio_embeds.shape[:2],
                dtype=torch.long,
                device=audio_embeds.device,
            )
            if trainer_attention_mask is not None and trainer_attention_mask.shape == audio_attention_mask.shape:
                # Keep it only if shape is already compatible.
                audio_attention_mask = trainer_attention_mask.to(audio_embeds.device)
            # GraniteSpeechConfig may not expose eos_token_id directly.
            # Resolve special token ids from multiple config locations safely.
            eos_id = getattr(wrapped_model.config, "eos_token_id", None)
            if eos_id is None and hasattr(wrapped_model, "language_model"):
                eos_id = getattr(wrapped_model.language_model.config, "eos_token_id", None)
            if eos_id is None:
                eos_id = getattr(wrapped_model.generation_config, "eos_token_id", None)

            pad_id = getattr(wrapped_model.config, "pad_token_id", None)
            if pad_id is None and hasattr(wrapped_model, "language_model"):
                pad_id = getattr(wrapped_model.language_model.config, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(wrapped_model.generation_config, "pad_token_id", None)
            if pad_id is None:
                pad_id = 0

            return wrapped_model.language_model.generate(
                inputs_embeds=audio_embeds,
                attention_mask=audio_attention_mask,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **kwargs
            )
        else:
            return wrapped_model.generate(**kwargs)
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        try:
            wrapped_model = object.__getattribute__(self, 'model')
            return getattr(wrapped_model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Wrap the model
model = GraniteSpeechWrapper(model)
print(f"Model loaded. Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -------------------- AUDIO LOADING HELPER -------------------- #
def load_audio_from_example(ex):
    """Load audio as a numpy array at 16kHz mono from a dataset example."""
    import librosa
    
    audio_entry = ex[AUDIO_COLUMN]
    
    if isinstance(audio_entry, dict):
        # Priority 1: bytes (most reliable - no file path issues)
        if "bytes" in audio_entry and audio_entry["bytes"] is not None:
            audio_array, sr = librosa.load(io.BytesIO(audio_entry["bytes"]), sr=16000, mono=True)
            return audio_array, 16000
        
        # Priority 2: array (already decoded by datasets)
        if "array" in audio_entry and audio_entry["array"] is not None:
            audio_array = np.array(audio_entry["array"], dtype=np.float32)
            sr = audio_entry.get("sampling_rate", 16000)
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            return audio_array, 16000
        
        # Priority 3: path (try to resolve)
        if "path" in audio_entry and audio_entry["path"] is not None:
            audio_path = audio_entry["path"]
            
            # If path doesn't exist, try to find in cache
            if not os.path.exists(audio_path):
                cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
                for root, _, files in os.walk(cache_dir):
                    if os.path.basename(audio_path) in files:
                        audio_path = os.path.join(root, os.path.basename(audio_path))
                        break
            
            if os.path.exists(audio_path):
                audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
                return audio_array, 16000
            
            # If bytes are available as fallback
            if "bytes" in audio_entry and audio_entry["bytes"] is not None:
                audio_array, sr = librosa.load(io.BytesIO(audio_entry["bytes"]), sr=16000, mono=True)
                return audio_array, 16000
            
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    raise ValueError(f"Unrecognized audio format: {type(audio_entry)}")


# -------------------- FIND AUDIO PROCESSOR -------------------- #
# Granite Speech processor may use 'audio_processor' instead of 'feature_extractor'
audio_feature_extractor = None
for attr_name in ['feature_extractor', 'audio_processor', 'audio_feature_extractor']:
    val = getattr(processor, attr_name, None)
    if val is not None:
        audio_feature_extractor = val
        print(f"Found audio processor: processor.{attr_name} = {type(val)}")
        break

if audio_feature_extractor is None:
    print("WARNING: No audio feature extractor found on processor!")
    print("Will try using processor(text, audio) directly.")

# -------------------- PREPROCESS FUNCTION -------------------- #
def prepare_example(ex):
    """Prepare audio features and labels for training."""
    # Step 1: Load audio
    audio_array, sr = load_audio_from_example(ex)
    
    # Step 2: Extract audio features for the CTC encoder
    if audio_feature_extractor is not None:
        # Use the audio processor component directly
        feature_output = audio_feature_extractor(audio_array)
        
        # Extract tensor from BatchFeature object
        if hasattr(feature_output, 'input_features') and feature_output.input_features is not None:
            feat = feature_output.input_features
        elif hasattr(feature_output, 'input_values') and feature_output.input_values is not None:
            feat = feature_output.input_values
        elif isinstance(feature_output, dict):
            if 'input_features' in feature_output:
                feat = feature_output['input_features']
            elif 'input_values' in feature_output:
                feat = feature_output['input_values']
            else:
                feat = list(feature_output.values())[0]
        else:
            raise ValueError(f"Cannot extract features from {type(feature_output)}")
        
        # Convert to numpy for serialization
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        elif isinstance(feat, list):
            feat = np.array(feat)
        
        # Remove batch dimension if present: [1, T, F] -> [T, F]
        if feat.ndim == 3:
            feat = feat[0]
        
        input_features = feat.tolist()
    else:
        # Fallback: try using processor(text, audio) and extract features
        wav_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        try:
            model_inputs = processor("transcribe", wav_tensor)
            if hasattr(model_inputs, 'input_features') and model_inputs.input_features is not None:
                feat = model_inputs.input_features
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()
                elif isinstance(feat, list):
                    feat = np.array(feat)
                if feat.ndim == 3:
                    feat = feat[0]
                input_features = feat.tolist()
            else:
                raise ValueError("Processor did not return input_features")
        except Exception as e:
            raise ValueError(
                f"Cannot extract audio features. No audio_processor found and "
                f"processor(text, audio) failed: {e}. "
                f"Processor attributes: {[a for a in dir(processor) if not a.startswith('_')]}"
            )
    
    # Step 3: Tokenize target text (the LaTeX equation)
    target_text = ex["text_norm"]
    labels = tokenizer(target_text, add_special_tokens=False).input_ids
    
    # Add EOS token
    if tokenizer.eos_token_id is not None:
        labels.append(tokenizer.eos_token_id)
    
    return {
        "input_features": input_features,
        "labels": labels,
    }

# Map datasets
print("Preprocessing datasets...")
remove_cols = [col for col in train_dataset.column_names if col not in ["input_features", "labels"]]
train_dataset = train_dataset.map(prepare_example, remove_columns=remove_cols)
eval_dataset = eval_dataset.map(prepare_example, remove_columns=remove_cols)
print(f"Preprocessing complete. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# Debug: check a sample
sample = train_dataset[0]
feat = np.array(sample["input_features"])
print(f"Sample input_features shape: {feat.shape}")
print(f"Sample labels length: {len(sample['labels'])}")

# -------------------- DATA COLLATOR -------------------- #
@dataclass
class DataCollatorGraniteSpeech:
    """Pads input_features (variable-length audio) and labels (variable-length text)."""
    pad_token_id: int
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- Pad input_features ---
        feat_list = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        
        # Features could be 1D [T] or 2D [T, F]
        if feat_list[0].dim() == 1:
            # 1D features: pad to max length
            max_len = max(f.shape[0] for f in feat_list)
            padded = []
            for f in feat_list:
                pad_len = max_len - f.shape[0]
                if pad_len > 0:
                    f = torch.cat([f, torch.zeros(pad_len)])
                padded.append(f)
            input_features = torch.stack(padded)
        else:
            # 2D features [T, F]: pad along time dimension
            max_len = max(f.shape[0] for f in feat_list)
            feat_dim = feat_list[0].shape[1]
            padded = []
            for f in feat_list:
                pad_len = max_len - f.shape[0]
                if pad_len > 0:
                    padding = torch.zeros(pad_len, feat_dim)
                    f = torch.cat([f, padding], dim=0)
                padded.append(f)
            input_features = torch.stack(padded)
        
        # --- Pad labels ---
        max_label_len = max(len(f["labels"]) for f in features)
        labels_padded = []
        attention_mask_list = []
        for f in features:
            lab = f["labels"]
            pad_len = max_label_len - len(lab)
            # Pad labels with -100 (ignored in loss)
            labels_padded.append(lab + [-100] * pad_len)
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask_list.append([1] * len(lab) + [0] * pad_len)
        
        return {
            "input_features": input_features,
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }

data_collator = DataCollatorGraniteSpeech(pad_token_id=tokenizer.pad_token_id)

# -------------------- METRICS -------------------- #
wer_metric = load("wer")
cer_metric = load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    vocab_size = len(tokenizer)
    
    # Replace -100 with pad_token_id for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Clamp to valid range
    pred_ids = np.clip(pred_ids, 0, vocab_size - 1)
    label_ids = np.clip(label_ids, 0, vocab_size - 1)
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    pred_str = [normalize_latex(p) for p in pred_str]
    label_str = [normalize_latex(l) for l in label_str]
    
    # Filter out empty strings (WER/CER can't handle them)
    valid = [(p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0]
    if not valid:
        return {"wer": 1.0, "cer": 1.0}
    pred_str, label_str = zip(*valid)
    
    wer = wer_metric.compute(predictions=list(pred_str), references=list(label_str))
    cer = cer_metric.compute(predictions=list(pred_str), references=list(label_str))
    
    return {"wer": wer, "cer": cer}

# -------------------- TRAINING ARGUMENTS -------------------- #
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_steps=max(
        1,
        int(
            math.ceil(len(train_dataset) / (PER_DEVICE_BATCH * GRAD_ACCUM))
            * NUM_EPOCHS
            * WARMUP_RATIO
        ),
    ),
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=fp16 and torch.cuda.is_available(),
    gradient_checkpointing=False,  # Granite Speech doesn't support this
    
    eval_strategy="epoch",
    save_strategy="no",
    save_total_limit=1,
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",
    
    predict_with_generate=True,
    generation_max_length=225,
    
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# -------------------- TRAINING -------------------- #
print("\nStarting fine-tuning...")
print(f"Effective batch size: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"Learning rate: {LR}")
print("=" * 50)

trainer.train()

# Save final model and processor
# Save wrapped base model without safetensors (handles tied/shared weights safely)
wrapped_base_model = object.__getattribute__(model, 'model')
wrapped_base_model.save_pretrained(OUTPUT_MODEL_DIR, safe_serialization=False)
processor.save_pretrained(OUTPUT_MODEL_DIR)
print(f"\nBest model saved to {OUTPUT_MODEL_DIR}")

# -------------------- FINAL EVALUATION -------------------- #
print("\n" + "=" * 50)
print("Running final evaluation on validation set...")
print("=" * 50 + "\n")

def detect_hallucination(pred: str, gt: str):
    tok_pat = re.compile(r"(\\[A-Za-z]+|[A-Za-z0-9]+|[^A-Za-z0-9\s])")
    p_toks = tok_pat.findall(pred)
    g_toks = tok_pat.findall(gt)
    halluc = [t for t in set(p_toks) if t not in g_toks]
    return ",".join(halluc) if halluc else "none"

def generate_prediction(model_wrapper, audio_array, device_hint):
    """Generate a LaTeX prediction from audio."""
    model_wrapper.eval()
    with torch.no_grad():
        # Extract features
        if audio_feature_extractor is not None:
            feature_output = audio_feature_extractor(audio_array)
            if hasattr(feature_output, 'input_features') and feature_output.input_features is not None:
                feat = feature_output.input_features
            elif hasattr(feature_output, 'input_values') and feature_output.input_values is not None:
                feat = feature_output.input_values
            elif isinstance(feature_output, dict):
                feat = feature_output.get('input_features', feature_output.get('input_values', list(feature_output.values())[0]))
            else:
                raise ValueError(f"Cannot extract features from {type(feature_output)}")
            
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat, dtype=torch.float32)
            
            # Ensure 3D: [batch, T, F]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0).unsqueeze(-1)
            elif feat.dim() == 2:
                feat = feat.unsqueeze(0)
            
            input_features = feat.to(device_hint)
        else:
            raise ValueError("No audio feature extractor available")
        
        generated_ids = model_wrapper.generate(
            input_features=input_features,
            max_new_tokens=225,
        )
        
        transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return normalize_latex(transcription)

# Load base model for comparison
print("Loading base model for comparison...")
base_model_raw = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if fp16 and torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
base_model = GraniteSpeechWrapper(base_model_raw)
base_model.eval()

model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reload eval data with original fields
eval_data_full = dataset["test"]
eval_data_full = eval_data_full.map(lambda ex: {"text_norm": normalize_latex(ex.get("equation", ""))})

model.eval()
rows = []

print(f"Evaluating {len(eval_data_full)} examples")
for i in range(len(eval_data_full)):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(eval_data_full)}")
    
    ex = eval_data_full[i]
    gt = ex["text_norm"]
    
    try:
        audio_array, sr = load_audio_from_example(ex)
    except Exception as e:
        print(f"  Skipping example {i}: {e}")
        continue
    
    # Base model prediction
    try:
        pred_base = generate_prediction(base_model, audio_array, model_device)
    except Exception as e:
        pred_base = f"ERROR: {e}"
    
    wer_base = wer_metric.compute(predictions=[pred_base], references=[gt]) if "ERROR" not in pred_base else 1.0
    cer_base = cer_metric.compute(predictions=[pred_base], references=[gt]) if "ERROR" not in pred_base else 1.0
    hall_base = detect_hallucination(pred_base, gt)
    
    # Fine-tuned model prediction
    try:
        pred_ft = generate_prediction(model, audio_array, model_device)
    except Exception as e:
        pred_ft = f"ERROR: {e}"
    
    wer_ft = wer_metric.compute(predictions=[pred_ft], references=[gt]) if "ERROR" not in pred_ft else 1.0
    cer_ft = cer_metric.compute(predictions=[pred_ft], references=[gt]) if "ERROR" not in pred_ft else 1.0
    hall_ft = detect_hallucination(pred_ft, gt)
    
    audio_path = ex[AUDIO_COLUMN].get("path", "unknown") if isinstance(ex[AUDIO_COLUMN], dict) else "unknown"
    
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
        "WER_Improvement": (wer_base if isinstance(wer_base, float) else 1.0) - (wer_ft if isinstance(wer_ft, float) else 1.0),
        "CER_Improvement": (cer_base if isinstance(cer_base, float) else 1.0) - (cer_ft if isinstance(cer_ft, float) else 1.0),
    })

# Save results
df = pd.DataFrame(rows)
df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nDetailed results saved to {OUTPUT_XLSX}")

# Print summary
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
if len(df) > 0:
    print(f"\nBase Model Performance:")
    print(f"  Average WER: {df['Base_WER'].mean():.4f}")
    print(f"  Average CER: {df['Base_CER'].mean():.4f}")
    print(f"\nFine-Tuned Model Performance:")
    print(f"  Average WER: {df['FineTuned_WER'].mean():.4f}")
    print(f"  Average CER: {df['FineTuned_CER'].mean():.4f}")
    print(f"\nImprovement:")
    print(f"  WER Reduction: {df['WER_Improvement'].mean():.4f}")
    print(f"  CER Reduction: {df['CER_Improvement'].mean():.4f}")
    print(f"\n% of samples improved: {(df['WER_Improvement'] > 0).sum() / len(df) * 100:.1f}%")
print("=" * 50 + "\n")
