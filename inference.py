

# inference_script.py
#!/usr/bin/env python3
"""
Inference script for trained MatSciBERT NER model
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple

class MatSciBERTNER:
    def __init__(self, model_path: str):
        """Initialize the NER model"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.id2label = None
        self.label2id = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        
        # Load label mappings
        try:
            with open(f"{self.model_path}/label_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
                self.label2id = mappings['label2id']
        except FileNotFoundError:
            print("Label mappings not found, using default mappings")
            self._setup_default_labels()
        
        # Try to load with Unsloth first, then fallback to standard approach
        try:
            from unsloth import FastLanguageModel
            print("Loading with Unsloth...")
            
            # Load the Unsloth model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Switch to inference mode
            FastLanguageModel.for_inference(model)
            
            self.model = model
            self.tokenizer = tokenizer
            print("Model loaded with Unsloth optimization!")
            
        except ImportError:
            print("Unsloth not available, loading with standard method...")
            self._load_standard_model()
        except Exception as e:
            print(f"Unsloth loading failed ({e}), falling back to standard method...")
            self._load_standard_model()
    
    def _load_standard_model(self):
        """Load model using standard transformers + PEFT approach"""
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from peft import PeftModel
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load base model
        base_model = AutoModelForTokenClassification.from_pretrained(
            "m3rg-iitd/matscibert",
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        print("Model loaded with standard PEFT!")
    
    def _setup_default_labels(self):
        """Setup default label mappings"""
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
        
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}
    
    def predict(self, text: str) -> List[Dict]:
        """Predict entities in text"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Prepare input
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Align predictions with tokens
        word_ids = inputs.word_ids()
        previous_word_idx = None
        predicted_labels = []
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                predicted_labels.append(self.id2label[predictions[0][i].item()])
            previous_word_idx = word_idx
        
        # Create entity list
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                current_entity = {
                    'text': token,
                    'label': label[2:],  # Remove B- prefix
                    'start': i,
                    'end': i + 1
                }
            elif label.startswith('I-') and current_entity and current_entity['label'] == label[2:]:
                # Continue current entity
                current_entity['text'] += ' ' + token
                current_entity['end'] = i + 1
            else:
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Predict entities for a batch of texts"""
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append(entities)
        return results
    
    def format_output_json(self, text: str, entities: List[Dict]) -> Dict:
        """Format output as structured JSON"""
        return {
            "text": text,
            "entities": entities,
            "entity_count": len(entities),
            "entity_types": list(set([entity['label'] for entity in entities]))
        }
    
    def format_output_csv_ready(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Format output ready for CSV export"""
        csv_rows = []
        for entity in entities:
            csv_rows.append({
                "original_text": text,
                "entity_text": entity['text'],
                "entity_label": entity['label'],
                "start_position": entity['start'],
                "end_position": entity['end']
            })
        return csv_rows

def demo_inference():
    """Demo function showing how to use the model"""
    
    # Example texts
    sample_texts = [
        "The TiVZrNbHf high-entropy alloy exhibited a hydrogen storage capacity of 2.5 wt% at 300°C.",
        "XRD analysis confirmed the presence of a single BCC phase in the arc-melted sample.",
        "Microscopic cracks were observed in the material structure after mechanical testing.",
        "The Cantor alloy showed excellent mechanical properties with a hardness of 250 HV.",
        "SEM imaging revealed nanocrystalline microstructure in the mechanically alloyed powder."
    ]
    
    try:
        # Load model
        model = MatSciBERTNER("matscibert-hea-ner-lora")
        
        print("Running inference on sample texts...\n")
        
        for i, text in enumerate(sample_texts, 1):
            print(f"Text {i}: {text}")
            entities = model.predict(text)
            
            if entities:
                print("Extracted entities:")
                for entity in entities:
                    print(f"  - {entity['text']} [{entity['label']}]")
            else:
                print("  No entities found")
            
            # Show JSON format
            json_output = model.format_output_json(text, entities)
            print(f"JSON format: {json.dumps(json_output, indent=2)}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have trained the model first by running train_matscibert_lora.py")

if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    
    demo_inference()

# pipeline_api.py
#!/usr/bin/env python3
"""
FastAPI server for MatSciBERT NER pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
from pathlib import Path

# Import your processing classes (assuming they're in the main script)
# from your_main_script import MistralOCR, TextProcessor, MatSciBERTNER

app = FastAPI(title="MatSciBERT NER API", version="1.0.0")

class TextInput(BaseModel):
    text: str
    return_format: Optional[str] = "json"  # json, csv, or entities_only

class PredictionResponse(BaseModel):
    text: str
    entities: List[Dict]
    entity_count: int
    entity_types: List[str]

# Global model instance (load once)
ner_model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global ner_model
    try:
        ner_model = MatSciBERTNER("matscibert-hea-ner-lora")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        ner_model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MatSciBERT NER API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict entities from raw text"""
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        entities = ner_model.predict(input_data.text)
        
        if input_data.return_format == "entities_only":
            return {"entities": entities}
        elif input_data.return_format == "csv":
            csv_data = ner_model.format_output_csv_ready(input_data.text, entities)
            return {"csv_data": csv_data}
        else:
            return ner_model.format_output_json(input_data.text, entities)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Predict entities from uploaded file (PDF, image, or text)"""
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file type
    file_ext = Path(file.filename).suffix.lower()
    supported_extensions = ['.pdf', '.txt', '.md', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Supported: {supported_extensions}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract text based on file type
        text = ""
        if file_ext == '.pdf':
            ocr_service = MistralOCR(os.getenv("MISTRAL_API_KEY"))
            text = ocr_service.extract_text_from_pdf(temp_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            ocr_service = MistralOCR(os.getenv("MISTRAL_API_KEY"))
            text = ocr_service.extract_text_from_image(temp_path)
        elif file_ext in ['.txt', '.md']:
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        # Process text
        text_processor = TextProcessor()
        cleaned_text = text_processor.clean_text(text)
        
        # Predict entities
        entities = ner_model.predict(cleaned_text)
        
        return {
            "filename": file.filename,
            "extracted_text": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            "full_text": cleaned_text,
            "entities": entities,
            "entity_count": len(entities),
            "entity_types": list(set([entity['label'] for entity in entities]))
        }
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(texts: List[str]):
    """Predict entities for multiple texts"""
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for text in texts:
            entities = ner_model.predict(text)
            results.append(ner_model.format_output_json(text, entities))
        
        return {"results": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# streamlit_app.py
#!/usr/bin/env python3
"""
Streamlit interface for MatSciBERT NER
"""

import streamlit as st
import pandas as pd
import json
from io import StringIO
import tempfile
import os
from pathlib import Path

# Import your classes here
# from your_main_script import MistralOCR, TextProcessor, MatSciBERTNER

st.set_page_config(
    page_title="MatSciBERT NER Interface",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the NER model (cached)"""
    try:
        return MatSciBERTNER("matscibert-hea-ner-lora")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def main():
    st.title("🔬 MatSciBERT Named Entity Recognition")
    st.markdown("Extract materials science entities from scientific texts using fine-tuned MatSciBERT")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please ensure the trained model exists.")
        return
    
    # Sidebar
    st.sidebar.header("Options")
    input_method = st.sidebar.selectbox(
        "Input Method",
        ["Text Input", "File Upload", "Batch Processing"]
    )
    
    output_format = st.sidebar.selectbox(
        "Output Format",
        ["Structured View", "JSON", "CSV"]
    )
    
    # Main interface
    if input_method == "Text Input":
        st.header("Text Input")
        
        # Sample texts
        sample_texts = {
            "Sample 1": "The TiVZrNbHf high-entropy alloy exhibited a hydrogen storage capacity of 2.5 wt% at 300°C.",
            "Sample 2": "XRD analysis confirmed the presence of a single BCC phase in the arc-melted sample.",
            "Sample 3": "Microscopic cracks were observed in the material structure after mechanical testing.",
            "Custom": ""
        }
        
        selected_sample = st.selectbox("Select a sample or use custom text:", list(sample_texts.keys()))
        
        if selected_sample == "Custom":
            text_input = st.text_area(
                "Enter your text:",
                height=200,
                placeholder="Enter scientific text about materials, high-entropy alloys, hydrogen storage, etc."
            )
        else:
            text_input = st.text_area(
                "Text to analyze:",
                value=sample_texts[selected_sample],
                height=100
            )
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    entities = model.predict(text_input)
                    display_results(text_input, entities, output_format)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif input_method == "File Upload":
        st.header("File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md', 'jpg', 'jpeg', 'png', 'gif', 'bmp'],
            help="Upload PDF, text files, or images"
        )
        
        if uploaded_file is not None:
            if st.button("Process File", type="primary"):
                with st.spinner("Processing file..."):
                    text = extract_text_from_file(uploaded_file)
                    if text:
                        entities = model.predict(text)
                        
                        st.subheader("Extracted Text Preview")
                        st.text_area("", value=text[:500] + "..." if len(text) > 500 else text, height=100)
                        
                        display_results(text, entities, output_format)
                    else:
                        st.error("Could not extract text from the file.")
    
    elif input_method == "Batch Processing":
        st.header("Batch Processing")
        
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Enter each text on a new line..."
        )
        
        if st.button("Process Batch", type="primary"):
            if batch_input.strip():
                texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
                
                with st.spinner(f"Processing {len(texts)} texts..."):
                    all_results = []
                    
                    for i, text in enumerate(texts):
                        entities = model.predict(text)
                        all_results.append({
                            'text_id': i + 1,
                            'text': text,
                            'entities': entities,
                            'entity_count': len(entities)
                        })
                    
                    display_batch_results(all_results, output_format)
            else:
                st.warning("Please enter texts to process.")

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file"""
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    try:
        if file_ext in ['.txt', '.md']:
            return str(uploaded_file.read(), "utf-8")
        
        elif file_ext == '.pdf' or file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            
            # Use Mistral OCR
            ocr_service = MistralOCR(os.getenv("MISTRAL_API_KEY"))
            
            if file_ext == '.pdf':
                text = ocr_service.extract_text_from_pdf(temp_path)
            else:
                text = ocr_service.extract_text_from_image(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            # Process text
            text_processor = TextProcessor()
            return text_processor.clean_text(text)
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def display_results(text, entities, output_format):
    """Display analysis results"""
    st.subheader("Results")
    
    if not entities:
        st.info("No entities found in the text.")
        return
    
    # Entity statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entities", len(entities))
    with col2:
        unique_types = len(set([entity['label'] for entity in entities]))
        st.metric("Unique Types", unique_types)
    with col3:
        avg_length = sum([len(entity['text'].split()) for entity in entities]) / len(entities)
        st.metric("Avg Entity Length", f"{avg_length:.1f} words")
    
    if output_format == "Structured View":
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            if entity['label'] not in entity_groups:
                entity_groups[entity['label']] = []
            entity_groups[entity['label']].append(entity['text'])
        
        for entity_type, entity_texts in entity_groups.items():
            with st.expander(f"{entity_type} ({len(entity_texts)} entities)", expanded=True):
                for text in set(entity_texts):  # Remove duplicates
                    st.write(f"• {text}")
    
    elif output_format == "JSON":
        json_output = {
            "text": text,
            "entities": entities,
            "entity_count": len(entities),
            "entity_types": list(set([entity['label'] for entity in entities]))
        }
        st.json(json_output)
        
        # Download button
        st.download_button(
            "Download JSON",
            json.dumps(json_output, indent=2),
            file_name="entities.json",
            mime="application/json"
        )
    
    elif output_format == "CSV":
        # Convert to DataFrame
        df_data = []
        for entity in entities:
            df_data.append({
                "Entity Text": entity['text'],
                "Entity Type": entity['label'],
                "Start Position": entity['start'],
                "End Position": entity['end']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name="entities.csv",
            mime="text/csv"
        )

def display_batch_results(all_results, output_format):
    """Display batch processing results"""
    st.subheader("Batch Results")
    
    # Overall statistics
    total_entities = sum([result['entity_count'] for result in all_results])
    st.metric("Total Entities Across All Texts", total_entities)
    
    if output_format == "Structured View":
        for result in all_results:
            with st.expander(f"Text {result['text_id']}: {result['text'][:100]}...", expanded=False):
                if result['entities']:
                    for entity in result['entities']:
                        st.write(f"**{entity['label']}**: {entity['text']}")
                else:
                    st.write("No entities found")
    
    elif output_format == "JSON":
        st.json(all_results)
        
        st.download_button(
            "Download Batch JSON",
            json.dumps(all_results, indent=2),
            file_name="batch_entities.json",
            mime="application/json"
        )
    
    elif output_format == "CSV":
        # Flatten all results
        df_data = []
        for result in all_results:
            for entity in result['entities']:
                df_data.append({
                    "Text ID": result['text_id'],
                    "Original Text": result['text'],
                    "Entity Text": entity['text'],
                    "Entity Type": entity['label'],
                    "Start Position": entity['start'],
                    "End Position": entity['end']
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Batch CSV",
                csv,
                file_name="batch_entities.csv",
                mime="text/csv"
            )
        else:
            st.info("No entities found in any of the texts.")

if __name__ == "__main__":
    main()