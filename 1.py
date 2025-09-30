import os
import re
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from dotenv import load_dotenv
load_dotenv()

# Download NLTK data
nltk.download('punkt')

GEMINI_API_KEY1 = os.getenv("GEMINI_API_KEY")

# Configure Gemini API (replace with your API key)
genai.configure(api_key=GEMINI_API_KEY1)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

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
    "Application": "Applications (e.g., 'solid-state hydrogen storage', 'fuel cells')"
}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Clean and preprocess extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove references/citations (simplified)
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def segment_into_sentences(text):
    """Segment text into sentences using NLTK"""
    return sent_tokenize(text)

def create_annotation_prompt(sentences, num_examples=5):
    """Create a prompt for Gemini to annotate sentences"""
    
    # Create few-shot examples
    examples = """
Example annotations:
Sentence: The TiVZrNbHf high-entropy alloy exhibited a hydrogen storage capacity of 2.5 wt% at 300°C.
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

Sentence: XRD analysis confirmed the presence of a single BCC phase in the arc-melted sample.
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
"""
    
    # Create the main prompt
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

{examples}

Now annotate the following sentences from research papers:

"""
    
    # Add the sentences to annotate
    for i, sentence in enumerate(sentences[:num_examples]):
        prompt += f"Sentence {i+1}: {sentence}\nAnnotations:\n"
    
    return prompt

def extract_entities_with_gemini(sentences, batch_size=5):
    """Use Gemini to extract entities from sentences"""
    annotated_sentences = []
    
    # Process in batches to manage context window
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        prompt = create_annotation_prompt(batch)
        
        try:
            response = model.generate_content(prompt)
            annotated_text = response.text
            
            # Parse the response to extract annotations
            annotations = parse_gemini_response(annotated_text, batch)
            annotated_sentences.extend(annotations)
            
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            # Fallback: mark all tokens as O for failed batches
            for sentence in batch:
                tokens = word_tokenize(sentence)
                annotated_sentences.append([(token, "O") for token in tokens])
    
    return annotated_sentences

def parse_gemini_response(response_text, original_sentences):
    """Parse Gemini's response to extract BIO annotations"""
    annotated_sentences = []
    
    # Split response by sentence annotations
    annotation_blocks = response_text.split("Annotations:")[1:]  # Skip the prompt part
    
    for i, block in enumerate(annotation_blocks):
        if i >= len(original_sentences):
            break
            
        # Tokenize the original sentence for comparison
        original_tokens = word_tokenize(original_sentences[i])
        annotated_tokens = []
        
        # Parse each line of the annotation block
        lines = block.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                token = ' '.join(parts[:-1])  # Handle tokens with spaces
                tag = parts[-1]
                annotated_tokens.append((token, tag))
        
        # Align with original tokens (Gemini might tokenize differently)
        aligned_annotations = align_annotations(original_tokens, annotated_tokens)
        annotated_sentences.append(aligned_annotations)
    
    return annotated_sentences

def align_annotations(original_tokens, annotated_tokens):
    """Align Gemini's annotations with original tokenization"""
    aligned = []
    ann_idx = 0
    
    for orig_token in original_tokens:
        if ann_idx < len(annotated_tokens):
            ann_token, ann_tag = annotated_tokens[ann_idx]
            
            # Check if tokens match (with some flexibility)
            if orig_token == ann_token or orig_token in ann_token or ann_token in orig_token:
                aligned.append((orig_token, ann_tag))
                ann_idx += 1
            else:
                # Try to find a matching token
                found = False
                for j in range(ann_idx, min(ann_idx+3, len(annotated_tokens))):
                    j_token, j_tag = annotated_tokens[j]
                    if orig_token == j_token or orig_token in j_token or j_token in orig_token:
                        aligned.append((orig_token, j_tag))
                        ann_idx = j + 1
                        found = True
                        break
                
                if not found:
                    aligned.append((orig_token, "O"))
        else:
            aligned.append((orig_token, "O"))
    
    return aligned

def convert_to_conll_format(annotated_sentences, output_path):
    """Convert annotated sentences to CONLL format for training"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in annotated_sentences:
            for token, tag in sentence:
                f.write(f"{token} {tag}\n")
            f.write("\n")  # Empty line between sentences

def process_pdfs_in_folder(folder_path, output_file):
    """Process all PDFs in a folder and create training data"""
    all_annotated_sentences = []
    
    # Process each PDF file
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            print(f"Processing {filename}...")
            pdf_path = os.path.join(folder_path, filename)
            
            # Extract and clean text
            text = extract_text_from_pdf(pdf_path)
            if not text:
                continue
                
            cleaned_text = clean_text(text)
            sentences = segment_into_sentences(cleaned_text)
            
            # Filter relevant sentences (those mentioning HEAs or hydrogen)
            relevant_sentences = [
                s for s in sentences 
                if any(keyword in s.lower() for keyword in [
                    'high entropy', 'hea', 'hydrogen storage', 'hydride', 
                    'absorption', 'desorption', 'capacity'
                ])
            ]
            
            print(f"Found {len(relevant_sentences)} relevant sentences in {filename}")
            
            # Annotate sentences with Gemini
            annotated = extract_entities_with_gemini(relevant_sentences)
            all_annotated_sentences.extend(annotated)
    
    # Convert to CONLL format
    convert_to_conll_format(all_annotated_sentences, output_file)
    print(f"Training data saved to {output_file}")

def create_finetuning_script(output_path):
    """Create a script to fine-tune MatSciBERT using PEFT/LoRA and Unsloth"""
    script = """
#!/bin/bashx

# Script to fine-tune MatSciBERT on HEA hydrogen storage data using PEFT/LoRA and Unsloth
# Requires unsloth, transformers, datasets, and torch libraries

python - << EOF
import os
from unsloth import FastLanguageModel
import torch
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, load_dataset
import evaluate
import numpy as np

# Load Unsloth-optimized model
model_name = "m3rg-iitd/matscibert"
max_seq_length = 512
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

# Load the model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Prepare for LoRA fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=["query", "key", "value", "output.dense"],  # Target modules for LoRA
    lora_dropout=0.1,  # Dropout for LoRA
    bias="none",  # No bias for LoRA
    use_gradient_checkpointing=True,  # Use gradient checkpointing
    random_state=3407,
    use_rslora=False,  # Not using RS LoRA
    loftq_config=None,  # No LoftQ
)

# Define label mappings
id2label = {
    0: "O",
    1: "B-Material", 2: "I-Material",
    3: "B-ChemicalElement", 4: "I-ChemicalElement",
    5: "B-SynthesisProcess", 6: "I-SynthesisProcess",
    7: "B-Microstructure", 8: "I-Microstructure",
    9: "B-HydrogenProperty", 10: "I-HydrogenProperty",
    11: "B-MechanicalProperty", 12: "I-MechanicalProperty",
    13: "B-TestingMethod", 14: "I-TestingMethod",
    15: "B-Value", 16: "I-Value",
    17: "B-Application", 18: "I-Application"
}
label2id = {v: k for k, v in id2label.items()}

# Load dataset
dataset = load_dataset('text', data_files={'train': 'training_data.conll'})

# Tokenize function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
        max_length=max_seq_length
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Preprocess dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Training arguments with Unsloth optimizations
training_args = TrainingArguments(
    output_dir="matscibert-hea-hydrogen-unsloth",
    per_device_train_batch_size=4,  # Reduced for 4-bit
    gradient_accumulation_steps=4,  # Increase accumulation steps
    warmup_steps=50,
    max_steps=500,  # Reduced steps for efficiency
    learning_rate=2e-4,  # Higher learning rate for LoRA
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",  # Use 8-bit Adam optimizer
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="no",  # No evaluation during training
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)

# Start training with Unsloth optimizations
trainer.train()

# Save the model
model.save_pretrained("matscibert-hea-hydrogen-unsloth-final")
tokenizer.save_pretrained("matscibert-hea-hydrogen-unsloth-final")
EOF
"""
    
    with open(output_path, 'w') as f:
        f.write(script)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    print(f"Fine-tuning script created at {output_path}")

# Main execution
if __name__ == "__main__":
    # Configuration
    pdf_folder = "Data"  # Folder containing your PDFs
    output_file = "training_data.conll"  # Output file for training data
    script_path = "finetune_matscibert_unsloth.sh"  # Path for fine-tuning script
    
    # Process PDFs and create training data
    process_pdfs_in_folder(pdf_folder, output_file)
    
    # Create fine-tuning script
    create_finetuning_script(script_path)
    
    print("Process completed!")
    print(f"1. Training data saved to: {output_file}")
    print(f"2. Fine-tuning script created at: {script_path}")
    print(f"3. Run '{script_path}' to fine-tune MatSciBERT with Unsloth and PEFT")