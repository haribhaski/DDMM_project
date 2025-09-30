#!/bin/bash
set -e

# Fine-tune MatSciBERT for NER with PEFT/LoRA (Mac-friendly, no CUDA deps)
# Expects: training_data.conll (token + tag per line, blank line between examples)

python - << 'PY'
import os, sys, re, json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model
import evaluate

TRAIN_PATH = "training_data.conll"
OUT_DIR   = "matscibert-hea-hydrogen-lora"
MODEL_NAME = "m3rg-iitd/matscibert"
MAX_LEN   = 256
SEED      = 3407

# ---------------------------
# 1) CoNLL parser + sanitizer
# ---------------------------
# Only allow tags from your ontology (O + B-/I-)
VALID_ENTITIES = {
    "Material","ChemicalElement","SynthesisProcess","Microstructure",
    "HydrogenProperty","MechanicalProperty","TestingMethod","Value",
    "Application","Phase",
}
ALLOWED_TAGS = {"O"} | {f"{p}-{e}" for e in VALID_ENTITIES for p in ("B","I")}

def read_conll_clean(path):
    ex_tokens, ex_tags = [], []
    tokens, tags = [], []

    def flush():
        if tokens:
            ex_tokens.append(tokens.copy()); ex_tags.append(tags.copy())
            tokens.clear(); tags.clear()

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            # New example boundary
            if not line:
                flush(); continue
            # skip comment/prompt lines if any leaked in
            if line.startswith(("#",";")) or line.lower().startswith(("input:", "output:", "example:")):
                flush(); continue

            parts = line.split()
            if len(parts) == 1:
                token, tag = parts[0], "O"
            else:
                token = " ".join(parts[:-1])
                tag   = parts[-1]

            if tag not in ALLOWED_TAGS:
                tag = "O"  # sanitize unknown tags (e.g., stray punctuation)

            tokens.append(token); tags.append(tag)

    flush()
    if not ex_tokens:
        raise RuntimeError(f"No usable examples parsed from {path}.")
    return ex_tokens, ex_tags

tokens_batch, tags_batch = read_conll_clean(TRAIN_PATH)

# --------------------------------
# 2) Build label maps from the data
# --------------------------------
labels_seen = sorted({t for seq in tags_batch for t in seq})
if "O" not in labels_seen:
    labels_seen = ["O"] + labels_seen
label2id = {l:i for i,l in enumerate(labels_seen)}
id2label = {i:l for l,i in label2id.items()}
print("Num labels:", len(labels_seen))
print("Labels:", labels_seen)

# Save label maps for later inference
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "label2id.json"), "w") as f: json.dump(label2id, f, indent=2)
with open(os.path.join(OUT_DIR, "id2label.json"), "w") as f: json.dump(id2label, f, indent=2)

# ----------------------------
# 3) Make a HuggingFace Dataset
# ----------------------------
ds = Dataset.from_dict({
    "tokens":   tokens_batch,
    "ner_tags": [[label2id[t] for t in seq] for seq in tags_batch],
})

# -----------------------------
# 4) Tokenizer + base model (HF)
# -----------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
assert getattr(tok, "is_fast", False), "Fast tokenizer required."

base = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_seen),
    id2label=id2label,
    label2id=label2id,
)

# -----------------------
# 5) Wrap with LoRA (PEFT)
# -----------------------
peft_cfg = LoraConfig(
    task_type="TOKEN_CLS",
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query","key","value","dense"],  # BERT-safe targets
    bias="none",
)
model = get_peft_model(base, peft_cfg)
model.print_trainable_parameters()

# --------------------------------------------
# 6) Align word-level labels to subword tokens
# --------------------------------------------
def tokenize_and_align_labels(examples):
    enc = tok(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True, padding="max_length", max_length=MAX_LEN,
        return_attention_mask=True,
    )
    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = enc.word_ids(batch_index=i)
        word_labels = examples["ner_tags"][i]
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(word_labels[wid] if wid < len(word_labels) else -100)
            else:
                label_ids.append(-100)
            prev = wid
        labels.append(label_ids)
    enc["labels"] = labels
    return enc

tok_ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens","ner_tags"])

# -----------------------
# 7) Metrics (seqeval NER)
# -----------------------
metric = evaluate.load("seqeval")
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_preds, true_labels = [], []
    for pr, lb in zip(preds, labels):
        tp, tl = [], []
        for pi, li in zip(pr, lb):
            if li != -100:
                tp.append(id2label[pi]); tl.append(id2label[li])
        true_preds.append(tp); true_labels.append(tl)
    res = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": res.get("overall_precision", 0.0),
        "recall":    res.get("overall_recall", 0.0),
        "f1":        res.get("overall_f1", 0.0),
        "accuracy":  res.get("overall_accuracy", 0.0),
    }

# -----------------------
# 8) Training (Mac-safe)
# -----------------------
torch.manual_seed(SEED)
use_mps = torch.backends.mps.is_available()
print("Using MPS:", use_mps)

# Check Transformers version to handle API changes
import transformers
transformers_version = transformers.__version__
print(f"Transformers version: {transformers_version}")

# Handle different TrainingArguments API versions
training_args_kwargs = {
    "output_dir": OUT_DIR,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "logging_steps": 20,
    "fp16": False,
    "bf16": False,
    "dataloader_num_workers": 0,
    "report_to": "none",
}

# Handle evaluation_strategy/eval_strategy parameter based on version
if transformers_version >= "4.37.0":
    training_args_kwargs["eval_strategy"] = "no"
else:
    training_args_kwargs["evaluation_strategy"] = "no"

args = TrainingArguments(**training_args_kwargs)

collator = DataCollatorForTokenClassification(tokenizer=tok, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,     # training only (no eval set in this script)
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,   # harmless even if not evaluating
)

print("Starting LoRA training…")
train_result = trainer.train()
trainer.save_model()
tok.save_pretrained(OUT_DIR)

# Save basic train metrics/state
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("Saved adapter + tokenizer to:", OUT_DIR)
PY