

## A comprehensive system for extracting, processing, and storing High Entropy Alloy (HEA) knowledge from scientific literature PDFs using AI-powered text processing and graph database storage.



## Architecture

```
PDF Files → Mistral OCR → Gemini 2.5 Flash → Neo4j Graph Database
    ↓            ↓              ↓                    ↓
Raw PDFs → Text Content → HEA Sentences → Knowledge Graph
```


## Quick Start

### Prerequisites

1. **API Keys** - Set up environment variables:
   ```bash
   export MISTRAL_API_KEY="your_mistral_api_key"
   export GEMINI_API_KEY="your_gemini_api_key"
   export username="your_neo4j_username"
   export password="your_neo4j_password"
   ```

2. **Python Environment**:
   ```bash
   # Create virtual environment
   python -m venv ddmm
   source ddmm/bin/activate  # On Windows: ddmm\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Code Execution**:

```bash
python3 simple_pdf_to_neo4j_pipeline.py d1ee01543e.pdf --output-dir ./simple_results

```












