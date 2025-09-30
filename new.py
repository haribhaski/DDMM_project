import os
import re
import json
import requests
import base64
import subprocess
import sys
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from dotenv import load_dotenv
from pathlib import Path

# PDF processing imports
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
    print("✅ PDF processing libraries available")
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF processing libraries not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "PyMuPDF"])
    import PyPDF2
    import fitz
    PDF_AVAILABLE = True

load_dotenv()

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Get Mistral API key from environment
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("Please set MISTRAL_API_KEY in your .env file")

# Define the ontology for High Entropy Alloys and Hydrogen Storage
ONTOLOGY = {
    "Material": "HEA material names and compositions (e.g., 'TiVZrNbHf', 'Cantor alloy')",
    "ChemicalElement": "Chemical elements (e.g., 'Ti', 'V', 'Zr', 'Nb', 'Hf')",
    "SynthesisProcess": "Material synthesis methods (e.g., 'arc melting', 'mechanical alloying')",
    "Microstructure": "Microstructural features (e.g., 'BCC phase', 'FCC phase', 'nanocrystalline')",
    "HydrogenProperty": "Hydrogen-related properties (e.g., 'hydrogen storage capacity', 'absorption kinetics')",
    "MechanicalProperty": "Mechanical properties (e.g., 'hardness', 'strength', 'ductility')",
    "TestingMethod": "Characterization methods (e.g., 'XRD', 'PCT measurement', 'SEM')",
    "Value": "Numerical values with units (e.g., '2.5 wt%', '300 °C', '100 MPa')",
    "Application": "Applications (e.g., 'solid-state hydrogen storage', 'fuel cells')",
    "Defect": "Material defects (e.g., 'crack', 'void', 'dislocation')"
}

class MistralOCR:
    """OCR service using Mistral API for images and dedicated libraries for PDFs"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
    
    def extract_text_from_pdf_direct(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (faster) with fallback to PyPDF2"""
        text = ""
        
        try:
            # Try PyMuPDF first (better for complex PDFs)
            print("🔍 Extracting text using PyMuPDF...")
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\n"
            
            doc.close()
            
            if text.strip():
                print(f"✅ Successfully extracted {len(text)} characters using PyMuPDF")
                return text
                
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}")
            
        try:
            # Fallback to PyPDF2
            print("🔍 Falling back to PyPDF2...")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
            if text.strip():
                print(f"✅ Successfully extracted {len(text)} characters using PyPDF2")
                return text
                
        except Exception as e:
            print(f"⚠️ PyPDF2 also failed: {e}")
            
        return text
    
    def extract_text_from_pdf_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF by converting to images and using OCR"""
        try:
            print("🖼️ Converting PDF to images for OCR...")
            doc = fitz.open(pdf_path)
            all_text = ""
            
            # Process first 10 pages to avoid overwhelming the API
            max_pages = min(10, len(doc))
            
            for page_num in range(max_pages):
                print(f"📄 Processing page {page_num + 1}/{max_pages}...")
                page = doc.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Use Mistral for OCR
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "pixtral-12b-2409",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this page image. Focus on scientific content and maintain formatting. Return only the extracted text."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_base64}"
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    page_text = result['choices'][0]['message']['content']
                    all_text += page_text + "\n\n"
                    print(f"✅ Page {page_num + 1} processed")
                else:
                    print(f"⚠️ OCR failed for page {page_num + 1}: {response.status_code}")
            
            doc.close()
            return all_text
            
        except Exception as e:
            print(f"❌ PDF OCR extraction failed: {e}")
            return ""
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Main PDF text extraction with multiple fallback methods"""
        print(f"📖 Processing PDF: {pdf_path}")
        
        # Method 1: Direct text extraction (fastest)
        text = self.extract_text_from_pdf_direct(pdf_path)
        
        # If direct extraction yields little text, try OCR
        if len(text.strip()) < 100:  # Threshold for "little text"
            print("📝 Direct extraction yielded minimal text, trying OCR...")
            ocr_text = self.extract_text_from_pdf_with_ocr(pdf_path)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        
        return text
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using Mistral's OCR"""
        try:
            # Read image file as base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Get file extension to determine MIME type
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp'
            }.get(ext, 'image/jpeg')
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Focus on scientific content and maintain formatting."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime_type};base64,{image_base64}"
                            }
                        ]
                    }
                ],
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"Error with Mistral API: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove references/citations
        text = re.sub(r'\[\d+\]', '', text)
        # Remove page numbers and headers/footers (simple approach)
        text = re.sub(r'\n\d+\n', '\n', text)
        return text.strip()
    
    @staticmethod
    def segment_into_sentences(text: str) -> List[str]:
        """Segment text into sentences using NLTK"""
        return sent_tokenize(text)
    
    @staticmethod
    def filter_relevant_sentences(sentences: List[str]) -> List[str]:
        """Filter sentences relevant to HEAs and hydrogen storage"""
        keywords = [
            'high entropy', 'hea', 'hydrogen storage', 'hydride', 
            'absorption', 'desorption', 'capacity', 'alloy',
            'materials', 'synthesis', 'microstructure', 'phase',
            'mechanical properties', 'characterization', 'defect',
            'crack', 'void', 'dislocation'
        ]
        
        relevant_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence)
        
        return relevant_sentences

class AnnotationGenerator:
    """Generate NER annotations using Mistral API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
    
    def create_annotation_prompt(self, sentences: List[str]) -> str:
        """Create a prompt for NER annotation"""
        examples = """
Example annotations using BIO format:

Sentence: "The TiVZrNbHf high-entropy alloy exhibited a hydrogen storage capacity of 2.5 wt% at 300°C."
Annotations:
The O
TiVZrNbHf B-Material
high-entropy I-Material
alloy I-Material
exhibited O
a O
hydrogen B-HydrogenProperty
storage I-HydrogenProperty
capacity I-HydrogenProperty
of O
2.5 B-Value
wt% I-Value
at O
300 B-Value
°C I-Value
. O

Sentence: "XRD analysis confirmed the presence of a single BCC phase in the arc-melted sample."
Annotations:
XRD B-TestingMethod
analysis I-TestingMethod
confirmed O
the O
presence O
of O
a O
single O
BCC B-Microstructure
phase I-Microstructure
in O
the O
arc-melted B-SynthesisProcess
sample O
. O

Sentence: "Microscopic cracks were observed in the material structure."
Annotations:
Microscopic O
cracks B-Defect
were O
observed O
in O
the O
material B-Material
structure O
. O
"""
        
        prompt = f"""
You are a materials science expert specializing in named entity recognition (NER) for high-entropy alloys and hydrogen storage research.

Entity types to identify:
{json.dumps(ONTOLOGY, indent=2)}

Instructions:
1. Analyze each sentence and identify all entities according to the defined types
2. Use BIO tagging format (B- prefix for beginning, I- prefix for inside, O for outside)
3. For multi-word entities, tag each word appropriately
4. Always respect the original text casing
5. Output only the tokenized text with annotations, one token per line
6. Add an empty line between different sentences

{examples}

Now annotate the following sentences:

"""
        
        for i, sentence in enumerate(sentences):
            prompt += f"Sentence {i+1}: \"{sentence}\"\nAnnotations:\n"
        
        return prompt
    
    def generate_annotations(self, sentences: List[str], batch_size: int = 3) -> List[List[Tuple[str, str]]]:
        """Generate NER annotations for sentences"""
        annotated_sentences = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            prompt = self.create_annotation_prompt(batch)
            
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "mistral-large-latest",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content']
                    
                    # Parse the response
                    batch_annotations = self.parse_annotations(response_text, batch)
                    annotated_sentences.extend(batch_annotations)
                else:
                    print(f"Error with Mistral API: {response.status_code}")
                    # Fallback: mark all tokens as O
                    for sentence in batch:
                        tokens = word_tokenize(sentence)
                        annotated_sentences.append([(token, "O") for token in tokens])
                        
            except Exception as e:
                print(f"Error generating annotations: {e}")
                # Fallback: mark all tokens as O
                for sentence in batch:
                    tokens = word_tokenize(sentence)
                    annotated_sentences.append([(token, "O") for token in tokens])
        
        return annotated_sentences
    
    def parse_annotations(self, response_text: str, original_sentences: List[str]) -> List[List[Tuple[str, str]]]:
        """Parse Mistral's annotation response"""
        annotated_sentences = []
        
        # Split by annotation blocks
        blocks = response_text.split("Annotations:")[1:]
        
        for i, block in enumerate(blocks):
            if i >= len(original_sentences):
                break
            
            # Get original tokens for alignment
            original_tokens = word_tokenize(original_sentences[i])
            
            # Parse annotation lines
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            annotated_tokens = []
            
            for line in lines:
                if ' ' in line:
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        token, tag = parts
                        annotated_tokens.append((token, tag))
            
            # Align with original tokens
            aligned = self.align_annotations(original_tokens, annotated_tokens)
            annotated_sentences.append(aligned)
        
        return annotated_sentences
    
    def align_annotations(self, original_tokens: List[str], annotated_tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Align annotations with original tokenization"""
        aligned = []
        ann_idx = 0
        
        for orig_token in original_tokens:
            if ann_idx < len(annotated_tokens):
                ann_token, ann_tag = annotated_tokens[ann_idx]
                
                if orig_token.lower() == ann_token.lower():
                    aligned.append((orig_token, ann_tag))
                    ann_idx += 1
                else:
                    # Try to find matching token nearby
                    found = False
                    for j in range(ann_idx, min(ann_idx + 3, len(annotated_tokens))):
                        j_token, j_tag = annotated_tokens[j]
                        if orig_token.lower() == j_token.lower():
                            aligned.append((orig_token, j_tag))
                            ann_idx = j + 1
                            found = True
                            break
                    
                    if not found:
                        aligned.append((orig_token, "O"))
            else:
                aligned.append((orig_token, "O"))
        
        return aligned

class TrainingManager:
    """Manages the training process and script generation"""
    
    def __init__(self):
        self.training_script_content = self._get_training_script()
    
    def _get_training_script(self) -> str:
        """Get the integrated training script with Unsloth optimization"""
        return '''#!/usr/bin/env python3
"""
MatSciBERT Fine-tuning Script with Unsloth
Optimized for materials science NER tasks with maximum efficiency
"""

import os
import torch
import json
import numpy as np
from datasets import Dataset
from transformers import (
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth detected - using optimized training")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available - falling back to standard PEFT")
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from peft import LoraConfig, get_peft_model, TaskType

class NERDataset:
    def __init__(self, conll_file_path):
        self.tokens = []
        self.labels = []
        self._load_data(conll_file_path)
    
    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence_tokens = []
            sentence_labels = []
            
            for line in f:
                line = line.strip()
                if line == "":
                    if sentence_tokens:
                        self.tokens.append(sentence_tokens)
                        self.labels.append(sentence_labels)
                        sentence_tokens = []
                        sentence_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        sentence_tokens.append(token)
                        sentence_labels.append(label)
            
            # Add last sentence if file doesn't end with empty line
            if sentence_tokens:
                self.tokens.append(sentence_tokens)
                self.labels.append(sentence_labels)

def setup_label_mappings():
    """Setup label to ID mappings"""
    labels = [
        "O",
        "B-Material", "I-Material",
        "B-ChemicalElement", "I-ChemicalElement", 
        "B-SynthesisProcess", "I-SynthesisProcess",
        "B-Microstructure", "I-Microstructure",
        "B-HydrogenProperty", "I-HydrogenProperty",
        "B-MechanicalProperty", "I-MechanicalProperty",
        "B-TestingMethod", "I-TestingMethod", 
        "B-Value", "I-Value",
        "B-Application", "I-Application",
        "B-Defect", "I-Defect"
    ]
    
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    
    return id2label, label2id

def load_model_with_unsloth(model_name: str, num_labels: int, id2label: dict, label2id: dict):
    """Load model using Unsloth if available, otherwise use standard approach"""
    if UNSLOTH_AVAILABLE:
        print("🚀 Loading model with Unsloth optimization...")
        # Load Unsloth-optimized model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization
            trust_remote_code=True,
        )
        
        # Configure for token classification
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Manually set up classification head if needed
        if not hasattr(model, 'classifier') or model.classifier.out_features != num_labels:
            import torch.nn as nn
            model.classifier = nn.Linear(model.config.hidden_size, num_labels)
            model.num_labels = num_labels
            model.config.num_labels = num_labels
            model.config.id2label = id2label
            model.config.label2id = label2id
        
        return model, tokenizer
    else:
        print("📚 Loading model with standard PEFT...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        # Setup LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"]
        )
        
        model = get_peft_model(model, peft_config)
        return model, tokenizer

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=512):
    """Tokenize text and align labels"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=max_length
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word
                label_ids.append(label2id.get(label[word_idx], 0))
            else:
                # Other subwords of the same word
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(eval_pred, id2label):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten for sklearn metrics
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    # Calculate metrics
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"], 
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    """Main training function"""
    
    # Configuration
    MODEL_NAME = "m3rg-iitd/matscibert"
    DATA_FILE = "training_data.conll"
    OUTPUT_DIR = "matscibert-hea-ner-unsloth"
    MAX_LENGTH = 512
    
    print("🔧 Setting up label mappings...")
    id2label, label2id = setup_label_mappings()
    num_labels = len(id2label)
    
    print("📖 Loading dataset...")
    dataset = NERDataset(DATA_FILE)
    
    if len(dataset.tokens) == 0:
        print("❌ No training data found. Please ensure training_data.conll exists and contains data.")
        return
    
    # Create train/validation split
    train_size = int(0.8 * len(dataset.tokens))
    train_tokens = dataset.tokens[:train_size]
    train_labels = dataset.labels[:train_size]
    val_tokens = dataset.tokens[train_size:]
    val_labels = dataset.labels[train_size:]
    
    print(f"📊 Training samples: {len(train_tokens)}")
    print(f"📊 Validation samples: {len(val_tokens)}")
    
    # Load model with Unsloth optimization
    model, tokenizer = load_model_with_unsloth(MODEL_NAME, num_labels, id2label, label2id)
    
    if UNSLOTH_AVAILABLE:
        model.print_trainable_parameters()
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "tokens": train_tokens,
        "labels": train_labels
    })
    
    val_dataset = Dataset.from_dict({
        "tokens": val_tokens, 
        "labels": val_labels
    })
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, MAX_LENGTH),
        batched=True
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, MAX_LENGTH),
        batched=True
    )
    
    # Training arguments optimized for Unsloth
    if UNSLOTH_AVAILABLE:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=8,
            per_device_train_batch_size=4,  # Smaller batch size for memory efficiency
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Compensate for smaller batch size
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=25,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps", 
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,
            learning_rate=2e-4,  # Higher LR for LoRA
            lr_scheduler_type="linear",
            save_total_limit=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            optim="adamw_8bit",  # Use 8-bit Adam
            max_grad_norm=1.0,
            seed=3407,
        )
    else:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            save_total_limit=3,
        )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id2label),
    )
    
    # Start training
    print("🎯 Starting training with Unsloth optimization..." if UNSLOTH_AVAILABLE else "🎯 Starting standard training...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving model...")
    if UNSLOTH_AVAILABLE:
        # Use Unsloth's optimized saving
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    else:
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label mappings
    with open(f"{OUTPUT_DIR}/label_mappings.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
    
    print(f"✅ Training completed! Model saved to {OUTPUT_DIR}")
    
    # Final evaluation
    if val_tokens:
        print("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        print("Final evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Save evaluation results
        with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    main()
'''
    
    def create_training_script(self, output_path: str):
        """Create the training script file"""
        with open(output_path, 'w') as f:
            f.write(self.training_script_content)
        
        os.chmod(output_path, 0o755)
        print(f"✅ Training script created at {output_path}")
    
    def create_shell_script(self, script_path: str):
        """Create shell script for easier training execution"""
        shell_script = f'''#!/bin/bash

# MatSciBERT Fine-tuning with Unsloth
# This script sets up the environment and runs the training

echo "🚀 Starting MatSciBERT fine-tuning with Unsloth optimization..."

# Check if training data exists
if [ ! -f "training_data.conll" ]; then
    echo "❌ Error: training_data.conll not found!"
    echo "Please run the data processing pipeline first."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📥 Installing required packages..."
pip install --upgrade pip

# Install core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft evaluate seqeval scikit-learn accelerate

# Try to install Unsloth for optimization
echo "🔧 Attempting to install Unsloth for optimization..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "⚠️ Unsloth installation failed, will use standard PEFT"

# Create output directory
mkdir -p matscibert-hea-ner-unsloth/logs

# Run the training script
echo "🎯 Starting training..."
python train_matscibert_unsloth.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Model saved to: matscibert-hea-ner-unsloth/"
    echo "📊 Check logs at: matscibert-hea-ner-unsloth/logs/"
else
    echo "❌ Training failed!"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "🎉 All done!"
'''
        
        with open(script_path, 'w') as f:
            f.write(shell_script)
        
        os.chmod(script_path, 0o755)
        print(f"✅ Shell script created at {script_path}")
    
    def run_training(self, script_path: str = "train_matscibert_unsloth.py"):
        """Run the training script"""
        try:
            print("🚀 Starting MatSciBERT training...")
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                print(result.stdout)
            else:
                print("❌ Training failed!")
                print("Error output:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running training: {e}")

def convert_to_conll_format(annotated_sentences: List[List[Tuple[str, str]]], output_path: str):
    """Convert annotated sentences to CoNLL format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in annotated_sentences:
            for token, tag in sentence:
                f.write(f"{token} {tag}\n")
            f.write("\n")  # Empty line between sentences

def process_documents_in_folder(folder_path: str, output_file: str):
    """Process all documents in a folder and create training data"""
    
    # Initialize components
    ocr_service = MistralOCR(MISTRAL_API_KEY)
    text_processor = TextProcessor()
    annotation_generator = AnnotationGenerator(MISTRAL_API_KEY)
    
    all_annotated_sentences = []
    
    # Supported file extensions
    pdf_extensions = ['.pdf']
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    text_extensions = ['.txt', '.md']
    
    print(f"🔍 Scanning folder: {folder_path}")
    
    # Process each file
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = Path(filename).suffix.lower()
        
        print(f"📄 Processing {filename}...")
        
        # Extract text based on file type
        text = ""
        if file_ext in pdf_extensions:
            text = ocr_service.extract_text_from_pdf(file_path)
        elif file_ext in image_extensions:
            text = ocr_service.extract_text_from_image(file_path)
        elif file_ext in text_extensions:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            print(f"⚠️ Unsupported file type: {file_ext}")
            continue
        
        if not text:
            print(f"⚠️ No text extracted from {filename}")
            continue
        
        # Process text
        cleaned_text = text_processor.clean_text(text)
        sentences = text_processor.segment_into_sentences(cleaned_text)
        relevant_sentences = text_processor.filter_relevant_sentences(sentences)
        
        print(f"✅ Found {len(relevant_sentences)} relevant sentences in {filename}")
        
        if relevant_sentences:
            # Generate annotations
            annotated = annotation_generator.generate_annotations(relevant_sentences)
            all_annotated_sentences.extend(annotated)
            print(f"🏷️ Generated annotations for {len(annotated)} sentences")
    
    # Convert to CoNLL format
    convert_to_conll_format(all_annotated_sentences, output_file)
    print(f"💾 Training data saved to {output_file}")
    print(f"📊 Total annotated sentences: {len(all_annotated_sentences)}")
    
    return len(all_annotated_sentences)

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """torch>=2.0.0
torchvision
torchaudio
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
evaluate>=0.4.0
seqeval
scikit-learn>=1.3.0
accelerate>=0.23.0
requests>=2.28.0
nltk>=3.8
python-dotenv>=1.0.0
pathlib
numpy>=1.24.0
PyPDF2>=3.0.0
PyMuPDF>=1.23.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Requirements file created: requirements.txt")

def create_config_file():
    """Create configuration file for the pipeline"""
    config = {
        "model_config": {
            "base_model": "m3rg-iitd/matscibert",
            "max_length": 512,
            "output_dir": "matscibert-hea-ner-unsloth"
        },
        "training_config": {
            "num_epochs": 8,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        },
        "data_config": {
            "train_split": 0.8,
            "batch_size_annotation": 3
        },
        "ontology": ONTOLOGY
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: config.json")

class PipelineManager:
    """Main pipeline manager that coordinates all components"""
    
    def __init__(self, data_folder: str = "Data", output_file: str = "training_data.conll"):
        self.data_folder = data_folder
        self.output_file = output_file
        self.training_manager = TrainingManager()
        
    def setup_environment(self):
        """Set up the environment and create necessary files"""
        print("🔧 Setting up environment...")
        
        # Create data folder if it doesn't exist
        if not os.path.exists(self.data_folder):
            print(f"📁 Creating data folder: {self.data_folder}")
            os.makedirs(self.data_folder)
            print(f"📝 Please place your documents (PDFs, images, text files) in the '{self.data_folder}' folder")
            return False
        
        # Create requirements and config files
        create_requirements_file()
        create_config_file()
        
        return True
    
    def run_full_pipeline(self, auto_train: bool = False):
        """Run the complete pipeline from data processing to model training"""
        
        # Setup environment
        if not self.setup_environment():
            return
        
        # Check if data folder has files
        files = [f for f in os.listdir(self.data_folder) 
                if Path(f).suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.txt', '.md']]
        
        if not files:
            print(f"⚠️ No supported files found in {self.data_folder}")
            print("Supported formats: PDF, JPG, JPEG, PNG, GIF, BMP, TXT, MD")
            return
        
        print(f"📁 Found {len(files)} files to process")
        
        # Process documents and create training data
        total_sentences = process_documents_in_folder(self.data_folder, self.output_file)
        
        if total_sentences == 0:
            print("❌ No training data generated. Please check your documents.")
            return
        
        # Create training scripts
        script_path = "train_matscibert_unsloth.py"
        shell_script_path = "finetune_matscibert_unsloth.sh"
        
        self.training_manager.create_training_script(script_path)
        self.training_manager.create_shell_script(shell_script_path)
        
        print("\n🎉 Pipeline setup completed!")
        print(f"📊 Generated {total_sentences} annotated sentences")
        print(f"📄 Training data: {self.output_file}")
        print(f"🐍 Python training script: {script_path}")
        print(f"🔧 Shell script: {shell_script_path}")
        
        print(f"\n📋 Next steps:")
        print(f"1. Install requirements: pip install -r requirements.txt")
        print(f"2. Run training:")
        print(f"   - Using shell script: ./{shell_script_path}")
        print(f"   - Or Python directly: python {script_path}")
        
        # Auto-train if requested
        if auto_train:
            response = input("\n🚀 Start training now? (y/n): ")
            if response.lower() == 'y':
                self.training_manager.run_training(script_path)

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MatSciBERT NER Pipeline for HEA and Hydrogen Storage")
    parser.add_argument("--data-folder", default="Data", help="Folder containing documents to process")
    parser.add_argument("--output-file", default="training_data.conll", help="Output file for training data")
    parser.add_argument("--auto-train", action="store_true", help="Automatically start training after data processing")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment, don't process data")
    
    args = parser.parse_args()
    
    # Initialize pipeline manager
    pipeline = PipelineManager(args.data_folder, args.output_file)
    
    if args.setup_only:
        pipeline.setup_environment()
        print("✅ Environment setup completed!")
    else:
        # Run the full pipeline
        pipeline.run_full_pipeline(args.auto_train)