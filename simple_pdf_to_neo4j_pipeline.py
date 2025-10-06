#!/usr/bin/env python3
"""
Simple PDF to Neo4j Knowledge Graph Pipeline (No Image Analysis)

This script processes a PDF through a streamlined pipeline:
1. PDF Extractor (Mistral OCR) - Extract text only from PDF
2. Gemini Knowledge Extractor - Extract structured scientific knowledge
3. Neo4j Knowledge Uploader - Upload to Neo4j database

Usage:
    python simple_pdf_to_neo4j_pipeline.py path/to/document.pdf
    python simple_pdf_to_neo4j_pipeline.py path/to/document.pdf --clear-db
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List

# Import our pipeline components
from pdf_extractor import PDFExtractor
from gemini_knowledge_extractor import GeminiKnowledgeExtractor
from neo4j_knowledge_uploader import Neo4jKnowledgeGraphUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimplePDFToNeo4jPipeline:
    """Simple pipeline for processing PDF documents into Neo4j knowledge graphs (text only)"""
    
    def __init__(self, output_dir: str = None):
        """Initialize the simple pipeline"""
        self.output_dir = output_dir or os.getcwd()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline components
        self.pdf_extractor = None
        self.gemini_extractor = None
        self.neo4j_uploader = None
        
        logger.info(f"Simple pipeline initialized with output directory: {self.output_dir}")
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize PDF extractor (text only)
            logger.info("Initializing PDF text extractor...")
            self.pdf_extractor = PDFExtractor()
            
            # Initialize Gemini extractor
            logger.info("Initializing Gemini knowledge extractor...")
            self.gemini_extractor = GeminiKnowledgeExtractor()
            
            # Initialize Neo4j uploader
            logger.info("Initializing Neo4j uploader...")
            self.neo4j_uploader = Neo4jKnowledgeGraphUploader()
            
            logger.info("All pipeline components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            return False
    
    def stage1_extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Stage 1: Extract text from PDF using Mistral OCR (no images)"""
        try:
            logger.info("="*80)
            logger.info("STAGE 1: PDF TEXT EXTRACTION")
            logger.info("="*80)
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text content only
            result = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            if not result or result["metadata"]["extraction_status"] != "success":
                logger.error("Failed to extract text from PDF")
                return None
            
            # Get the extracted text - try different possible keys
            extracted_text = result.get("content", "")
            
            # If content is empty, try to extract from pages
            if not extracted_text and result.get("pages"):
                page_texts = []
                for page in result["pages"]:
                    if hasattr(page, 'markdown'):
                        page_texts.append(page.markdown)
                    elif hasattr(page, 'content'):
                        page_texts.append(page.content)
                    elif isinstance(page, str):
                        page_texts.append(page)
                extracted_text = "\n\n".join(page_texts)
            
            if not extracted_text.strip():
                logger.error("No text content extracted from PDF")
                return None
            
            # Save PDF extraction results
            pdf_name = Path(pdf_path).stem
            pdf_output_file = os.path.join(self.output_dir, f"{pdf_name}_pdf_extraction.json")
            
            import json
            with open(pdf_output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Also save as text file for easy reading
            text_output_file = os.path.join(self.output_dir, f"{pdf_name}_extracted_text.txt")
            with open(text_output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            logger.info(f"PDF text extraction completed successfully!")
            logger.info(f"  - Text length: {len(extracted_text):,} characters")
            logger.info(f"  - Results saved to: {pdf_output_file}")
            logger.info(f"  - Text saved to: {text_output_file}")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in Stage 1 (PDF extraction): {e}")
            return None
    
    def stage2_extract_knowledge(self, extracted_text: str, pdf_name: str) -> Optional[Dict]:
        """Stage 2: Extract structured knowledge using Gemini"""
        try:
            logger.info("="*80)
            logger.info("STAGE 2: GEMINI KNOWLEDGE EXTRACTION")
            logger.info("="*80)
            
            if not extracted_text.strip():
                logger.error("No text content available for knowledge extraction")
                return None
            
            logger.info(f"Extracting knowledge from {len(extracted_text):,} characters...")
            
            # Extract knowledge using Gemini
            knowledge_result = self.gemini_extractor.extract_knowledge_from_text(
                extracted_text, pdf_name
            )
            
            if not knowledge_result:
                logger.error("Failed to extract knowledge from content")
                return None
            
            # Wrap in list format expected by the pipeline
            knowledge_data = [knowledge_result]
            
            # Save Gemini results
            gemini_output_file = os.path.join(self.output_dir, f"{pdf_name}_gemini_knowledge.json")
            self.gemini_extractor.save_knowledge_data(knowledge_data, gemini_output_file)
            
            # Convert to Neo4j format
            neo4j_data = self.gemini_extractor.convert_to_neo4j_format(knowledge_data)
            
            # Save Neo4j data
            neo4j_output_file = os.path.join(self.output_dir, f"{pdf_name}_neo4j_data.json")
            self.gemini_extractor.save_neo4j_data(neo4j_data, neo4j_output_file)
            
            logger.info(f"Gemini knowledge extraction completed successfully!")
            logger.info(f"  - Entities extracted: {len(knowledge_result.get('entities', []))}")
            logger.info(f"  - Relationships extracted: {len(knowledge_result.get('relationships', []))}")
            logger.info(f"  - Key insights: {len(knowledge_result.get('key_insights', []))}")
            logger.info(f"  - Neo4j nodes: {len(neo4j_data['nodes'])}")
            logger.info(f"  - Neo4j relationships: {len(neo4j_data['relationships'])}")
            logger.info(f"  - Knowledge data saved to: {gemini_output_file}")
            logger.info(f"  - Neo4j data saved to: {neo4j_output_file}")
            
            return neo4j_data
            
        except Exception as e:
            logger.error(f"Error in Stage 2 (Gemini extraction): {e}")
            return None
    
    def stage3_upload_to_neo4j(self, neo4j_data: Dict, pdf_name: str, clear_existing: bool = False) -> bool:
        """Stage 3: Upload knowledge graph to Neo4j"""
        try:
            logger.info("="*80)
            logger.info("STAGE 3: NEO4J KNOWLEDGE GRAPH UPLOAD")
            logger.info("="*80)
            
            logger.info(f"Uploading knowledge graph for: {pdf_name}")
            logger.info(f"  - Nodes to upload: {len(neo4j_data.get('nodes', []))}")
            logger.info(f"  - Relationships to upload: {len(neo4j_data.get('relationships', []))}")
            logger.info(f"  - Clear existing database: {clear_existing}")
            
            # Upload to Neo4j
            success = self.neo4j_uploader.upload_knowledge_graph(neo4j_data, clear_existing=clear_existing)
            
            if success:
                logger.info("Neo4j upload completed successfully!")
                
                # Get database statistics
                stats = self.neo4j_uploader.get_database_stats()
                if stats:
                    logger.info(f"Neo4j Database Statistics:")
                    logger.info(f"  - Total nodes: {stats.get('total_nodes', 0)}")
                    logger.info(f"  - Total relationships: {stats.get('total_relationships', 0)}")
                    
                    if stats.get('node_counts'):
                        logger.info("  - Node counts by type:")
                        for label, count in stats['node_counts'].items():
                            if count > 0:  # Only show non-zero counts
                                logger.info(f"    {label}: {count}")
                    
                    if stats.get('relationship_counts'):
                        logger.info("  - Relationship counts by type:")
                        for rel_type, count in stats['relationship_counts'].items():
                            if count > 0:  # Only show non-zero counts
                                logger.info(f"    {rel_type}: {count}")
                
                return True
            else:
                logger.error("Failed to upload knowledge graph to Neo4j")
                return False
                
        except Exception as e:
            logger.error(f"Error in Stage 3 (Neo4j upload): {e}")
            return False
        finally:
            # Close Neo4j connection
            if self.neo4j_uploader:
                self.neo4j_uploader.close()
    
    def process_pdf(self, pdf_path: str, clear_db: bool = False) -> bool:
        """Process a PDF through the complete simple pipeline"""
        try:
            logger.info("="*80)
            logger.info("SIMPLE PDF TO NEO4J KNOWLEDGE GRAPH PIPELINE")
            logger.info("="*80)
            logger.info(f"Input PDF: {pdf_path}")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Clear existing database: {clear_db}")
            logger.info("="*80)
            
            pdf_name = Path(pdf_path).stem
            
            # Initialize components
            if not self.initialize_components():
                return False
            
            # Stage 1: PDF Text Extraction
            extracted_text = self.stage1_extract_pdf_text(pdf_path)
            if not extracted_text:
                return False
            
            # Stage 2: Gemini Knowledge Extraction
            neo4j_data = self.stage2_extract_knowledge(extracted_text, pdf_name)
            if not neo4j_data:
                return False
            
            # Stage 3: Neo4j Upload
            success = self.stage3_upload_to_neo4j(neo4j_data, pdf_name, clear_existing=clear_db)
            if not success:
                return False
            
            logger.info("="*80)
            logger.info("SIMPLE PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"PDF '{pdf_name}' has been successfully processed and uploaded to Neo4j!")
            logger.info(f"You can now query the knowledge graph using Neo4j Browser or Cypher queries.")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in simple pipeline: {e}")
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Simple PDF to Neo4j Knowledge Graph Pipeline (Text Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_pdf_to_neo4j_pipeline.py document.pdf
  python simple_pdf_to_neo4j_pipeline.py document.pdf --clear-db
  python simple_pdf_to_neo4j_pipeline.py document.pdf --output-dir ./results
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )
    
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear existing Neo4j database before upload"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save intermediate results (default: current directory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file does not exist: {args.pdf_path}")
        sys.exit(1)
    
    if not args.pdf_path.lower().endswith('.pdf'):
        logger.error(f"Input file must be a PDF: {args.pdf_path}")
        sys.exit(1)
    
    # Initialize and run pipeline
    try:
        pipeline = SimplePDFToNeo4jPipeline(output_dir=args.output_dir)
        success = pipeline.process_pdf(args.pdf_path, clear_db=args.clear_db)
        
        if success:
            logger.info("Simple pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Simple pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
