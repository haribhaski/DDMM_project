# HEA Knowledge Extractor 🔬

A comprehensive system for extracting, processing, and storing High Entropy Alloy (HEA) knowledge from scientific literature PDFs using AI-powered text processing and graph database storage.

## 🎯 Overview

This system automates knowledge extraction from HEA research papers through a three-stage pipeline:

1. **PDF Content Extraction** - Uses Mistral OCR to extract text from PDF files
2. **AI-Powered Processing** - Leverages Gemini 2.5 Flash for HEA-specific knowledge extraction
3. **Graph Storage** - Stores structured knowledge in Neo4j for advanced querying and retrieval

## 🏗️ Architecture

```
PDF Files → Mistral OCR → Gemini 2.5 Flash → Neo4j Graph Database
    ↓            ↓              ↓                    ↓
Raw PDFs → Text Content → HEA Sentences → Knowledge Graph
```

### Key Components

- **`pdf_extractor.py`** - Mistral OCR integration for PDF text extraction
- **`gemini_processor.py`** - Gemini-powered HEA knowledge processing
- **`neo4j_storage.py`** - Neo4j graph database integration
- **`hea_knowledge_extractor.py`** - Main orchestration script

## 🚀 Quick Start

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

### Installation

1. Clone or download the project files
2. Place your PDF files in the `Data/` directory
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables (see Prerequisites)

### Running the Pipeline

#### Full Pipeline Execution
```bash
# Run complete pipeline
python hea_knowledge_extractor.py --full-pipeline

# Force complete re-processing
python hea_knowledge_extractor.py --full-pipeline --force-all
```

#### Individual Components
```bash
# Extract PDFs only
python pdf_extractor.py

# Process with Gemini only
python gemini_processor.py

# Store in Neo4j only
python neo4j_storage.py
```

#### Querying Knowledge Base
```bash
# Search by keyword
python hea_knowledge_extractor.py --query keyword --keyword "hydrogen storage"

# Search by material
python hea_knowledge_extractor.py --query material --material "TiVZrNbHf"

# High confidence sentences
python hea_knowledge_extractor.py --query confidence --confidence 0.8

# Full text search
python hea_knowledge_extractor.py --query full_text --search-text "BCC crystal structure"

# Get statistics
python hea_knowledge_extractor.py --stats
```

## 📊 Features

### PDF Processing
- **Batch Processing** - Handles multiple PDFs automatically
- **Error Handling** - Robust error handling for failed extractions
- **Progress Tracking** - Real-time progress updates
- **Content Validation** - Verifies extraction quality

### HEA Knowledge Extraction
- **Domain-Specific Filtering** - Focuses on HEA and hydrogen storage content
- **Keyword Recognition** - Identifies HEA-specific terminology
- **Confidence Scoring** - Assigns relevance scores to extracted sentences
- **Deduplication** - Removes near-duplicate content
- **Entity Extraction** - Identifies materials, properties, and measurements

### Graph Database Storage
- **Structured Storage** - Organizes knowledge in graph format
- **Relationship Mapping** - Links sentences, keywords, materials, and properties
- **Advanced Querying** - Supports complex graph queries
- **Full-Text Search** - Enables semantic search capabilities
- **Performance Optimization** - Includes indexes and constraints

## 🔍 Knowledge Schema

The system creates the following node types in Neo4j:

- **Sentence** - Extracted HEA-relevant sentences
- **Document** - Source PDF files
- **Keyword** - HEA-specific keywords and terms
- **Material** - Identified HEA compositions
- **Property** - Numerical properties and measurements

### Relationships
- `FROM_DOCUMENT` - Links sentences to source documents
- `CONTAINS_KEYWORD` - Links sentences to relevant keywords
- `MENTIONS_MATERIAL` - Links sentences to material compositions
- `HAS_PROPERTY` - Links sentences to numerical properties

## 📈 Example Usage

### Python API
```python
from hea_knowledge_extractor import HEAKnowledgeExtractor

# Initialize extractor
extractor = HEAKnowledgeExtractor()

# Run full pipeline
results = extractor.run_full_pipeline()

# Query knowledge base
hea_sentences = extractor.query_knowledge("by_keyword", keyword="HEA")
high_conf_sentences = extractor.query_knowledge("by_confidence", min_confidence=0.8)
```

### Command Line Examples
```bash
# Process all PDFs and store in Neo4j
python hea_knowledge_extractor.py --full-pipeline

# Search for hydrogen storage information
python hea_knowledge_extractor.py --query keyword --keyword "hydrogen"

# Find high-quality extractions
python hea_knowledge_extractor.py --query confidence --confidence 0.9
```

## 🎯 HEA-Specific Features

### Targeted Keywords
- **Material Types**: HEA, high entropy alloy, MPEA
- **Hydrogen Related**: hydride, hydrogen storage, H2
- **Crystal Structures**: BCC, FCC, HCP
- **Measurements**: PCT, pressure-composition-temperature
- **Properties**: kinetics, thermodynamics, capacity

### Material Recognition
- Automatically identifies HEA compositions (e.g., TiVZrNbHf, CoCrFeMnNi)
- Extracts numerical properties with units
- Links materials to their properties and conditions

## 📋 Output Files

The system generates several output files:

- **`extracted_pdf_content.json`** - Raw extracted PDF content
- **`processed_hea_sentences.json`** - Processed HEA sentences
- **`hea_extraction.log`** - Detailed execution logs

## 🛠️ Configuration

### Environment Variables
```bash
# Required API keys
MISTRAL_API_KEY=your_mistral_api_key
GEMINI_API_KEY=your_gemini_api_key

# Neo4j credentials
username=your_neo4j_username
password=your_neo4j_password

# Optional: Logging level
LOG_LEVEL=INFO
```

### Customization
- Modify keyword lists in `gemini_processor.py`
- Adjust confidence thresholds
- Customize Neo4j schema in `neo4j_storage.py`
- Update extraction prompts for different domains

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   ValueError: MISTRAL_API_KEY not found in environment variables
   ```
   Solution: Set up environment variables correctly

2. **Neo4j Connection Issues**
   ```
   Failed to connect to Neo4j: ...
   ```
   Solution: Check Neo4j credentials and connection string

3. **PDF Processing Failures**
   ```
   Error processing PDF: ...
   ```
   Solution: Verify PDF files are not corrupted and accessible

### Performance Tips
- Process PDFs in smaller batches for large datasets
- Use `--force-reprocess` only when necessary
- Monitor API usage limits
- Optimize Neo4j queries for large datasets

## 📊 Performance Metrics

Typical performance on modern hardware:
- **PDF Extraction**: ~30-60 seconds per PDF
- **Gemini Processing**: ~10-20 seconds per document chunk
- **Neo4j Storage**: ~1-5 seconds per batch of sentences

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Mistral AI for OCR capabilities
- Google for Gemini 2.5 Flash API
- Neo4j for graph database technology
- The HEA research community for domain expertise

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review the logs in `hea_extraction.log`
3. Open an issue with detailed error information

---

**Happy Knowledge Extracting! 🚀**












