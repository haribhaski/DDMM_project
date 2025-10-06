#!/usr/bin/env python3
"""
HEA Knowledge Extractor
======================

A comprehensive system for extracting, processing, and storing High Entropy Alloy (HEA) 
knowledge from scientific literature PDFs.

Pipeline:
1. Extract content from PDFs using Mistral OCR
2. Process content with Gemini 2.5 Flash for HEA-specific knowledge extraction
3. Store structured knowledge in Neo4j graph database
4. Provide query interface for knowledge retrieval

Author: AI Assistant
Date: 2025-09-30
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import our custom modules
from pdf_extractor import PDFExtractor
from gemini_processor import GeminiHEAProcessor
from neo4j_storage import Neo4jHEAStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hea_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HEAKnowledgeExtractor:
    """Main orchestrator for the HEA knowledge extraction pipeline"""
    
    def __init__(self, data_dir: str = "/home/sai-nivedh-26/ddmm-proj/Data"):
        """Initialize the knowledge extractor"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path("/home/sai-nivedh-26/ddmm-proj")
        
        # Ensure directories exist
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = None
        self.gemini_processor = None
        self.neo4j_storage = None
        
        logger.info(f"HEAKnowledgeExtractor initialized with data dir: {data_dir}")
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing pipeline components...")
            
            # Initialize PDF extractor
            self.pdf_extractor = PDFExtractor()
            logger.info("✓ PDF Extractor initialized")
            
            # Initialize Gemini processor
            self.gemini_processor = GeminiHEAProcessor()
            logger.info("✓ Gemini Processor initialized")
            
            # Initialize Neo4j storage
            self.neo4j_storage = Neo4jHEAStorage()
            logger.info("✓ Neo4j Storage initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def extract_pdfs(self, force_reextract: bool = False) -> List[Dict]:
        """Extract content from all PDF files"""
        extracted_file = self.output_dir / "extracted_pdf_content.json"
        
        # Check if already extracted and not forcing re-extraction
        if extracted_file.exists() and not force_reextract:
            logger.info(f"Loading existing extracted content from {extracted_file}")
            with open(extracted_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logger.info("Starting PDF content extraction...")
        start_time = time.time()
        
        # Extract from all PDFs in data directory
        extracted_data = self.pdf_extractor.extract_from_directory(str(self.data_dir))
        
        # Save extracted data
        self.pdf_extractor.save_extracted_data(extracted_data, str(extracted_file))
        
        extraction_time = time.time() - start_time
        logger.info(f"PDF extraction completed in {extraction_time:.2f} seconds")
        
        return extracted_data
    
    def process_with_gemini(self, extracted_data: List[Dict], force_reprocess: bool = False) -> List[Dict]:
        """Process extracted content with Gemini for HEA knowledge extraction"""
        processed_file = self.output_dir / "processed_hea_sentences.json"
        
        # Check if already processed and not forcing re-processing
        if processed_file.exists() and not force_reprocess:
            logger.info(f"Loading existing processed content from {processed_file}")
            with open(processed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logger.info("Starting Gemini processing for HEA knowledge extraction...")
        start_time = time.time()
        
        # Process with Gemini
        processed_sentences = self.gemini_processor.process_extracted_pdfs(extracted_data)
        
        # Convert to serializable format
        processed_data = []
        for sentence in processed_sentences:
            processed_data.append({
                'original_text': sentence.original_text,
                'cleaned_text': sentence.cleaned_text,
                'tokens': sentence.tokens,
                'hea_keywords': sentence.hea_keywords,
                'confidence_score': sentence.confidence_score,
                'source_file': sentence.source_file,
                'page_number': sentence.page_number
            })
        
        # Save processed data
        self.gemini_processor.save_processed_data(processed_sentences, str(processed_file))
        
        processing_time = time.time() - start_time
        logger.info(f"Gemini processing completed in {processing_time:.2f} seconds")
        
        return processed_data
    
    def store_in_neo4j(self, processed_data: List[Dict], force_restore: bool = False) -> int:
        """Store processed knowledge in Neo4j"""
        logger.info("Starting Neo4j storage...")
        start_time = time.time()
        
        # Clear existing data if force_restore is True
        if force_restore:
            logger.info("Clearing existing Neo4j data...")
            with self.neo4j_storage.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Existing data cleared")
        
        # Store in Neo4j
        stored_count = self.neo4j_storage.store_hea_knowledge(processed_data)
        
        storage_time = time.time() - start_time
        logger.info(f"Neo4j storage completed in {storage_time:.2f} seconds")
        
        return stored_count
    
    def run_full_pipeline(self, force_reextract: bool = False, force_reprocess: bool = False, 
                         force_restore: bool = False) -> Dict:
        """Run the complete knowledge extraction pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING HEA KNOWLEDGE EXTRACTION PIPELINE")
        logger.info("=" * 60)
        
        pipeline_start_time = time.time()
        results = {}
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Step 1: Extract PDFs
            logger.info("\n" + "=" * 40)
            logger.info("STEP 1: PDF CONTENT EXTRACTION")
            logger.info("=" * 40)
            extracted_data = self.extract_pdfs(force_reextract)
            results['extracted_pdfs'] = len(extracted_data)
            results['successful_extractions'] = len([d for d in extracted_data 
                                                   if d["metadata"]["extraction_status"] == "success"])
            
            # Step 2: Process with Gemini
            logger.info("\n" + "=" * 40)
            logger.info("STEP 2: GEMINI HEA PROCESSING")
            logger.info("=" * 40)
            processed_data = self.process_with_gemini(extracted_data, force_reprocess)
            results['processed_sentences'] = len(processed_data)
            results['avg_confidence'] = sum(s['confidence_score'] for s in processed_data) / len(processed_data) if processed_data else 0
            
            # Step 3: Store in Neo4j
            logger.info("\n" + "=" * 40)
            logger.info("STEP 3: NEO4J STORAGE")
            logger.info("=" * 40)
            stored_count = self.store_in_neo4j(processed_data, force_restore)
            results['stored_in_neo4j'] = stored_count
            
            # Get final statistics
            stats = self.neo4j_storage.get_knowledge_statistics()
            results['neo4j_stats'] = stats
            
            # Calculate total time
            total_time = time.time() - pipeline_start_time
            results['total_time_seconds'] = total_time
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.neo4j_storage:
                self.neo4j_storage.close()
    
    def query_knowledge(self, query_type: str, **kwargs) -> List[Dict]:
        """Query the stored knowledge base"""
        if not self.neo4j_storage:
            self.neo4j_storage = Neo4jHEAStorage()
        
        return self.neo4j_storage.query_hea_knowledge(query_type, **kwargs)
    
    def print_pipeline_summary(self, results: Dict):
        """Print a comprehensive summary of the pipeline results"""
        print("\n" + "=" * 80)
        print("HEA KNOWLEDGE EXTRACTION PIPELINE SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 EXTRACTION RESULTS:")
        print(f"   • Total PDF files processed: {results.get('extracted_pdfs', 0)}")
        print(f"   • Successful extractions: {results.get('successful_extractions', 0)}")
        print(f"   • Failed extractions: {results.get('extracted_pdfs', 0) - results.get('successful_extractions', 0)}")
        
        print(f"\n🧠 PROCESSING RESULTS:")
        print(f"   • HEA-relevant sentences extracted: {results.get('processed_sentences', 0)}")
        print(f"   • Average confidence score: {results.get('avg_confidence', 0):.3f}")
        
        print(f"\n🗄️ STORAGE RESULTS:")
        print(f"   • Sentences stored in Neo4j: {results.get('stored_in_neo4j', 0)}")
        
        neo4j_stats = results.get('neo4j_stats', {})
        if neo4j_stats:
            print(f"   • Total sentences in database: {neo4j_stats.get('total_sentences', 0)}")
            print(f"   • Total documents: {neo4j_stats.get('total_documents', 0)}")
            print(f"   • Total keywords: {neo4j_stats.get('total_keywords', 0)}")
            print(f"   • Total materials identified: {neo4j_stats.get('total_materials', 0)}")
            print(f"   • High confidence sentences (>0.7): {neo4j_stats.get('high_confidence_sentences', 0)}")
        
        print(f"\n⏱️ PERFORMANCE:")
        print(f"   • Total pipeline time: {results.get('total_time_seconds', 0):.2f} seconds")
        print(f"   • Average time per PDF: {results.get('total_time_seconds', 0) / max(results.get('extracted_pdfs', 1), 1):.2f} seconds")
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully! 🎉")
        print("=" * 80)

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description="HEA Knowledge Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python hea_knowledge_extractor.py --full-pipeline
  
  # Force re-extraction and re-processing
  python hea_knowledge_extractor.py --full-pipeline --force-all
  
  # Query existing knowledge
  python hea_knowledge_extractor.py --query keyword --keyword "hydrogen storage"
  
  # Get statistics only
  python hea_knowledge_extractor.py --stats
        """
    )
    
    # Pipeline options
    parser.add_argument('--full-pipeline', action='store_true', 
                       help='Run the complete extraction pipeline')
    parser.add_argument('--force-reextract', action='store_true', 
                       help='Force re-extraction of PDFs')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force re-processing with Gemini')
    parser.add_argument('--force-restore', action='store_true', 
                       help='Force complete restoration of Neo4j database')
    parser.add_argument('--force-all', action='store_true', 
                       help='Force all operations (extract, process, restore)')
    
    # Query options
    parser.add_argument('--query', choices=['keyword', 'material', 'confidence', 'document', 'full_text'],
                       help='Query type for knowledge retrieval')
    parser.add_argument('--keyword', help='Keyword to search for')
    parser.add_argument('--material', help='Material to search for')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence score')
    parser.add_argument('--document', help='Document name to search in')
    parser.add_argument('--search-text', help='Full text search query')
    
    # Other options
    parser.add_argument('--stats', action='store_true', help='Show knowledge base statistics')
    parser.add_argument('--data-dir', default="/home/sai-nivedh-26/ddmm-proj/Data", 
                       help='Directory containing PDF files')
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = HEAKnowledgeExtractor(args.data_dir)
        
        if args.full_pipeline:
            # Set force flags
            force_reextract = args.force_reextract or args.force_all
            force_reprocess = args.force_reprocess or args.force_all
            force_restore = args.force_restore or args.force_all
            
            # Run pipeline
            results = extractor.run_full_pipeline(
                force_reextract=force_reextract,
                force_reprocess=force_reprocess,
                force_restore=force_restore
            )
            
            # Print summary
            extractor.print_pipeline_summary(results)
            
        elif args.query:
            # Query knowledge base
            query_params = {}
            if args.keyword:
                query_params['keyword'] = args.keyword
            if args.material:
                query_params['material'] = args.material
            if args.confidence:
                query_params['min_confidence'] = args.confidence
            if args.document:
                query_params['document'] = args.document
            if args.search_text:
                query_params['search_text'] = args.search_text
            
            results = extractor.query_knowledge(args.query, **query_params)
            
            print(f"\nQuery Results ({len(results)} found):")
            print("=" * 60)
            for i, result in enumerate(results[:10], 1):  # Show top 10
                print(f"\n{i}. [Confidence: {result.get('confidence', 'N/A'):.3f}]")
                print(f"   Source: {result.get('source', 'N/A')}")
                print(f"   Text: {result.get('text', 'N/A')[:200]}...")
                if 'keywords' in result:
                    print(f"   Keywords: {result['keywords']}")
            
            if len(results) > 10:
                print(f"\n... and {len(results) - 10} more results")
        
        elif args.stats:
            # Show statistics
            extractor.initialize_components()
            stats = extractor.neo4j_storage.get_knowledge_statistics()
            
            print("\nKnowledge Base Statistics:")
            print("=" * 40)
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()











