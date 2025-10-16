This is a project done for subject DDMM by Hariharan(Me), Sai Nivedh, Rohith Balaji and Krish S. Our project is about creating an Ontology for Hydrogen Storage High-Entropy Alloys with a pipeline that is completely autmated witht he help of AI.


This project s divided into 2 phases.

Phase 1: Finetuning MatSciBERT Model - *Main Branch*

Phase 2: Building a Knowledge Graph  - *Sai-code Branch*

#Phase 1:
#MatSciBERT NER Pipeline (HEA & Hydrogen Storage)

Lightweight pipeline to extract, annotate, and fine-tune a MatSciBERT NER model for High-Entropy Alloys (HEAs) and hydrogen storage research.

## Features
- PDF / image OCR using Mistral API: [`MistralOCR`](new.py)
- Text cleaning & sentence segmentation: [`TextProcessor.clean_text`](new.py), [`TextProcessor.segment_into_sentences`](new.py)
- Automatic NER annotation via Mistral: [`AnnotationGenerator.generate_annotations`](new.py) and [`AnnotationGenerator.parse_annotations`](new.py)
- Annotation alignment and CoNLL export: [`AnnotationGenerator.align_annotations`](new.py), [`convert_to_conll_format`](new.py)
- Training script generation with PEFT/LoRA and optional Unsloth optimizations: [`TrainingManager.create_training_script`](new.py) and [finetune_matscibert_unsloth.sh](finetune_matscibert_unsloth.sh)
- Inference utilities and API + Streamlit UI: [`MatSciBERTNER`](inference.py), [`MatSciBERTNER.predict`](inference.py), [streamlit_app.py](streamlit_app.py) / [inference.py](inference.py)

## Quickstart

1. Install requirements
```sh
pip install -r requirements.txt
```

2. Set API keys in `.env`:
- MISTRAL_API_KEY (used by OCR & annotation in [`new.py`](new.py))
- GEMINI_API_KEY (if you use Gemini-based flow in [`1.py`](1.py))

3. Process documents and generate training data
- Place files in `Data/`
- Run the document pipeline (example):
```sh
python new.py --data-folder Data --output-file training_data.conll
```
This uses [`process_documents_in_folder`](new.py) which orchestrates OCR (`MistralOCR`), text processing (`TextProcessor`), and annotation (`AnnotationGenerator`).

4. Inspect / edit `training_data.conll` (token + tag per line; blank line between sentences).

5. Create and run training
- Auto-generated Python training script: created by [`TrainingManager.create_training_script`](new.py) as `train_matscibert_unsloth.py`
- Shell wrapper: [finetune_matscibert_unsloth.sh](finetune_matscibert_unsloth.sh)
```sh
# make sure requirements installed, then:
./finetune_matscibert_unsloth.sh
# or run the python training script directly:
python train_matscibert_unsloth.py
```

6. Run inference
- FastAPI server / demo in [inference.py](inference.py):
```sh
python inference.py
# for Streamlit UI:
streamlit run streamlit_app.py
```
Use [`MatSciBERTNER`](inference.py) which loads the trained model (supports Unsloth/PEFT fallback) and exposes `predict` / `predict_batch`.


#Phase 2:

# HEA — High-Entropy Alloys Knowledge Graph building

Short description
This folder contains the HEA-specific subset of the DDM project: data, preprocessing, annotation helpers, training configs and notebooks focused on High-Entropy Alloys (HEA) and related NER/data-processing tasks.

Purpose
- Collect and preprocess HEA literature (PDFs / images)
- Generate and curate NER annotations for HEA entities
- Train / evaluate HEA-specific models or adapters
- Provide reproducible notebooks and utility scripts for analysis

Directory layout (expected)
- data/
  - raw/               — original PDFs, images and source documents
  - extracted/         — OCR / extracted plain text
  - annotations/       — manual and auto-generated annotation files (CONLL / JSON)
  - splits/            — train / val / test splits
- scripts/
  - process_hea.py     — HEA-specific preprocessing pipeline (OCR → text → sentences)
  - annotate_hea.py    — annotation/label generation & alignment helpers
  - train_hea.py       — training wrapper for HEA model / adapter
  - eval_hea.py        — evaluation metrics / script
  - infer_hea.py       — lightweight inference utilities for the HEA model
- notebooks/
  - exploration.ipynb   — EDA and annotation quality checks
  - training_logs.ipynb — training experiments and metrics
- models/
  - checkpoints/        — saved weights / adapters for HEA experiments
  - label_map.json      — id2label / label2id mapping used by training/inference
- tests/
  - test_preprocessing.py
  - test_annotation_alignment.py

Quickstart
1. Install dependencies (project root):
   pip install -r requirements.txt

2. Place source documents:
   - Put PDFs / images into HEA/data/raw/

3. Run preprocessing:
   python scripts/process_hea.py --input HEA/data/raw --output HEA/data/extracted

4. Generate / review annotations:
   python scripts/annotate_hea.py --input HEA/data/extracted --out HEA/data/annotations
   Manually review HEA/data/annotations/*.conll before training.

5. Train model:
   python scripts/train_hea.py --data HEA/data/splits --out HEA/models/checkpoints

6. Evaluate / infer:
   python scripts/eval_hea.py --model HEA/models/checkpoints/latest --data HEA/data/splits/test
   python scripts/infer_hea.py --model HEA/models/checkpoints/latest --text "Example sentence..."

Environment / keys

- Use the same Python env as project root to ensure dependency compatibility.

Notes & best practices
- Keep a clean split (train/val/test) and version annotation files when editing.
- Use notebooks/ for exploratory checks; keep production scripts deterministic and parameterized.
- Store large artifacts (models / raw PDFs) outside git (use .gitignore).



