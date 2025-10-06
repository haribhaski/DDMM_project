import base64
import os
import json
from pathlib import Path
from mistralai import Mistral
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text content from PDF files using Mistral OCR"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the PDF extractor with Mistral API key"""
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        logger.info("PDFExtractor initialized successfully")
    
    def encode_pdf(self, pdf_path: str) -> Optional[str]:
        """Encode the PDF to base64"""
        try:
            with open(pdf_path, "rb") as pdf_file:
                encoded = base64.b64encode(pdf_file.read()).decode('utf-8')
                logger.info(f"Successfully encoded PDF: {pdf_path}")
                return encoded
        except FileNotFoundError:
            logger.error(f"File not found: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"Error encoding PDF {pdf_path}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[Dict]:
        """Extract text content from a single PDF using Mistral OCR"""
        try:
            # Encode PDF to base64
            base64_pdf = self.encode_pdf(pdf_path)
            if not base64_pdf:
                return None
            
            logger.info(f"Processing PDF with Mistral OCR: {pdf_path}")
            
            # Process with Mistral OCR
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}" 
                },
                include_image_base64=True
            )
            
            # Extract text content from the response
            extracted_content = {
                "file_name": Path(pdf_path).name,
                "file_path": pdf_path,
                "content": "",
                "pages": [],
                "metadata": {
                    "total_pages": 0,
                    "extraction_status": "success"
                }
            }
            
            # Process the OCR response and extract text
            combined_text = []
            
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                extracted_content["metadata"]["total_pages"] = len(ocr_response.pages)
                
                for page_num, page in enumerate(ocr_response.pages, 1):
                    page_text = ""
                    
                    # Try to get markdown content first
                    if hasattr(page, 'markdown') and page.markdown:
                        page_text = page.markdown
                    # Fallback to other possible text attributes
                    elif hasattr(page, 'content') and page.content:
                        page_text = page.content
                    elif hasattr(page, 'text') and page.text:
                        page_text = page.text
                    
                    if page_text:
                        combined_text.append(f"--- PAGE {page_num} ---\n\n{page_text}")
                        
                        # Store page info (serializable)
                        extracted_content["pages"].append({
                            "page_number": page_num,
                            "text": page_text,
                            "text_length": len(page_text)
                        })
                
                # Combine all page text
                extracted_content["content"] = "\n\n".join(combined_text)
            
            # If no pages, try to get content directly
            elif hasattr(ocr_response, 'content') and ocr_response.content:
                extracted_content["content"] = ocr_response.content
            elif hasattr(ocr_response, 'text') and ocr_response.text:
                extracted_content["content"] = ocr_response.text
            
            logger.info(f"Successfully extracted content from: {pdf_path}")
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {
                "file_name": Path(pdf_path).name,
                "file_path": pdf_path,
                "content": "",
                "pages": [],
                "metadata": {
                    "total_pages": 0,
                    "extraction_status": "failed",
                    "error": str(e)
                }
            }
    
    def extract_from_directory(self, directory_path: str) -> List[Dict]:
        """Extract text from all PDF files in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        extracted_data = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            result = self.extract_text_from_pdf(str(pdf_file))
            if result:
                extracted_data.append(result)
        
        logger.info(f"Successfully processed {len(extracted_data)} PDF files")
        return extracted_data
    
    def save_extracted_data(self, extracted_data: List[Dict], output_file: str):
        """Save extracted data to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Extracted data saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving extracted data: {e}")

def main():
    """Main function to test PDF extraction"""
    try:
        # Initialize extractor
        extractor = PDFExtractor()
        
        # Extract from Data directory
        data_dir = "/home/sai-nivedh-26/ddmm-proj/Data"
        extracted_data = extractor.extract_from_directory(data_dir)
        
        # Save results
        output_file = "/home/sai-nivedh-26/ddmm-proj/extracted_pdf_content.json"
        extractor.save_extracted_data(extracted_data, output_file)
        
        # Print summary
        total_files = len(extracted_data)
        successful_extractions = len([d for d in extracted_data if d["metadata"]["extraction_status"] == "success"])
        
        print(f"\nExtraction Summary:")
        print(f"Total PDF files: {total_files}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Failed extractions: {total_files - successful_extractions}")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()












