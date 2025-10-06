import base64
import os
import json
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk
from mistralai.models import OCRResponse
from typing import Dict, List, Optional, Tuple
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralEnhancedExtractor:
    """Enhanced PDF extractor using Mistral OCR with comprehensive image processing"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the enhanced extractor with Mistral API key"""
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "53K1SdAhnKtqqkbtazOkQjM5pVli39pT")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        logger.info("MistralEnhancedExtractor initialized successfully")
    
    def upload_and_process_pdf(self, pdf_path: str) -> Optional[OCRResponse]:
        """Upload PDF to Mistral and process with OCR including images"""
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.is_file():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            logger.info(f"Uploading PDF to Mistral OCR service: {pdf_file.name}")
            
            # Upload PDF file to Mistral's OCR service
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )
            
            # Get signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            
            # Process PDF with OCR, including embedded images
            logger.info(f"Processing PDF with OCR: {pdf_file.name}")
            pdf_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            
            logger.info(f"Successfully processed PDF: {pdf_file.name}")
            return pdf_response
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None
    
    def process_individual_image(self, image_base64: str, image_id: str) -> Optional[str]:
        """Process individual image using Mistral OCR to extract detailed information"""
        try:
            logger.info(f"Processing individual image: {image_id}")
            
            # Check if base64 data already has the data URL prefix
            if image_base64.startswith('data:image/'):
                base64_data_url = image_base64
            else:
                # Add data URL prefix if not present
                base64_data_url = f"data:image/jpeg;base64,{image_base64}"
            
            logger.debug(f"Using data URL: {base64_data_url[:50]}...")
            
            # Process image with OCR
            image_response = self.client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )
            
            # Extract text content from image
            if hasattr(image_response, 'pages') and image_response.pages:
                image_text = ""
                for page in image_response.pages:
                    if hasattr(page, 'markdown'):
                        image_text += page.markdown + "\n"
                
                if image_text.strip():
                    logger.info(f"Extracted text from image {image_id}: {len(image_text)} characters")
                    return image_text.strip()
            
            logger.warning(f"No text extracted from image: {image_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            return None
    
    def replace_images_with_insights(self, markdown_str: str, images_dict: dict) -> str:
        """
        Replace image placeholders in markdown with extracted insights from images.
        
        Args:
            markdown_str: Markdown text containing image placeholders
            images_dict: Dictionary mapping image IDs to base64 strings
        
        Returns:
            Markdown text with images replaced by their extracted insights
        """
        logger.info(f"Processing {len(images_dict)} images for insight extraction")
        
        for img_name, base64_str in images_dict.items():
            # Debug: Check the format of base64 string
            logger.debug(f"Image {img_name}: base64 starts with: {base64_str[:50]}...")
            
            # Extract insights from the image
            image_insights = self.process_individual_image(base64_str, img_name)
            
            if image_insights:
                # Create a detailed description to replace the image
                replacement_text = f"\n\n[IMAGE ANALYSIS - {img_name}]\n{image_insights}\n[END IMAGE ANALYSIS]\n\n"
            else:
                # Fallback if image processing fails
                replacement_text = f"\n\n[IMAGE - {img_name}]: Image content could not be extracted\n\n"
            
            # Replace image markdown with extracted insights
            image_pattern = f"![{re.escape(img_name)}]({re.escape(img_name)})"
            markdown_str = markdown_str.replace(image_pattern, replacement_text)
            
            # Also handle variations in image markdown format
            variations = [
                f"![{img_name}]({img_name})",
                f"![img-{img_name}](img-{img_name})",
                f"![]({img_name})",
            ]
            
            for variation in variations:
                if variation in markdown_str:
                    markdown_str = markdown_str.replace(variation, replacement_text)
        
        return markdown_str
    
    def get_combined_content_with_insights(self, ocr_response: OCRResponse) -> str:
        """
        Combine OCR text and image insights into a single coherent text document.
        
        Args:
            ocr_response: Response from OCR processing containing text and images
        
        Returns:
            Combined text with embedded image insights
        """
        combined_content = []
        
        # Process each page
        for page_num, page in enumerate(ocr_response.pages, 1):
            logger.info(f"Processing page {page_num} with {len(page.images)} images")
            
            # Extract images from page
            image_data = {}
            for img in page.images:
                image_data[img.id] = img.image_base64
            
            # Replace image placeholders with actual insights
            page_content = self.replace_images_with_insights(page.markdown, image_data)
            
            # Add page header
            page_header = f"\n\n--- PAGE {page_num} ---\n\n"
            combined_content.append(page_header + page_content)
        
        # Join all pages
        final_content = "\n".join(combined_content)
        
        # Clean up the content
        final_content = self.clean_extracted_content(final_content)
        
        logger.info(f"Generated combined content: {len(final_content)} characters")
        return final_content
    
    def clean_extracted_content(self, content: str) -> str:
        """Clean and normalize the extracted content"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Fix common OCR errors
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Add space between camelCase
        content = re.sub(r'(\d)([A-Za-z])', r'\1 \2', content)  # Space between numbers and letters
        
        # Clean up multiple spaces
        content = re.sub(r' +', ' ', content)
        
        # Ensure proper sentence endings
        content = re.sub(r'([.!?])([A-Z])', r'\1 \2', content)
        
        return content.strip()
    
    def extract_pdf_with_image_insights(self, pdf_path: str) -> Optional[Dict]:
        """
        Extract PDF content with comprehensive image analysis and insights.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Dictionary containing extracted content with image insights
        """
        try:
            # Process PDF with Mistral OCR
            ocr_response = self.upload_and_process_pdf(pdf_path)
            if not ocr_response:
                return None
            
            # Get combined content with image insights
            combined_content = self.get_combined_content_with_insights(ocr_response)
            
            # Prepare result structure
            result = {
                "file_name": Path(pdf_path).name,
                "file_path": pdf_path,
                "combined_content": combined_content,
                "metadata": {
                    "total_pages": len(ocr_response.pages),
                    "total_images_processed": sum(len(page.images) for page in ocr_response.pages),
                    "extraction_status": "success",
                    "content_length": len(combined_content),
                    "processing_method": "mistral_ocr_with_image_insights"
                },
                "pages_data": []
            }
            
            # Add detailed page information
            for page_num, page in enumerate(ocr_response.pages, 1):
                page_info = {
                    "page_number": page_num,
                    "original_markdown": page.markdown,
                    "images_count": len(page.images),
                    "image_ids": [img.id for img in page.images]
                }
                result["pages_data"].append(page_info)
            
            logger.info(f"Successfully extracted content from {pdf_path}")
            logger.info(f"  - Pages: {result['metadata']['total_pages']}")
            logger.info(f"  - Images processed: {result['metadata']['total_images_processed']}")
            logger.info(f"  - Content length: {result['metadata']['content_length']} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting PDF with image insights {pdf_path}: {e}")
            return {
                "file_name": Path(pdf_path).name,
                "file_path": pdf_path,
                "combined_content": "",
                "metadata": {
                    "total_pages": 0,
                    "total_images_processed": 0,
                    "extraction_status": "failed",
                    "error": str(e),
                    "processing_method": "mistral_ocr_with_image_insights"
                },
                "pages_data": []
            }
    
    def extract_from_directory(self, directory_path: str) -> List[Dict]:
        """Extract content from all PDF files in a directory with image insights"""
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        extracted_data = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            result = self.extract_pdf_with_image_insights(str(pdf_file))
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
    
    def save_combined_text(self, extracted_data: List[Dict], output_file: str):
        """Save only the combined text content to a text file for easy reading"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for data in extracted_data:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"FILE: {data['file_name']}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(data.get('combined_content', ''))
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"END OF {data['file_name']}\n")
                    f.write(f"{'='*80}\n\n")
            
            logger.info(f"Combined text content saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving combined text: {e}")

def main():
    """Main function to test the enhanced extraction"""
    try:
        # Initialize extractor
        extractor = MistralEnhancedExtractor()
        
        # Extract from Data directory
        data_dir = "/home/sai-nivedh-26/ddmm-proj/Data"
        extracted_data = extractor.extract_from_directory(data_dir)
        
        # Save results
        json_output_file = "/home/sai-nivedh-26/ddmm-proj/mistral_enhanced_extraction.json"
        text_output_file = "/home/sai-nivedh-26/ddmm-proj/mistral_enhanced_extraction.txt"
        
        extractor.save_extracted_data(extracted_data, json_output_file)
        extractor.save_combined_text(extracted_data, text_output_file)
        
        # Print summary
        total_files = len(extracted_data)
        successful_extractions = len([d for d in extracted_data if d["metadata"]["extraction_status"] == "success"])
        total_images = sum(d["metadata"]["total_images_processed"] for d in extracted_data)
        total_content_length = sum(d["metadata"]["content_length"] for d in extracted_data)
        
        print(f"\nEnhanced Extraction Summary:")
        print(f"{'='*50}")
        print(f"Total PDF files: {total_files}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Failed extractions: {total_files - successful_extractions}")
        print(f"Total images processed: {total_images}")
        print(f"Total content extracted: {total_content_length:,} characters")
        print(f"JSON results saved to: {json_output_file}")
        print(f"Text results saved to: {text_output_file}")
        print(f"{'='*50}")
        
        # Show sample of extracted content
        if extracted_data and successful_extractions > 0:
            sample_data = next(d for d in extracted_data if d["metadata"]["extraction_status"] == "success")
            sample_content = sample_data.get('combined_content', '')[:500]
            
            print(f"\nSample extracted content from {sample_data['file_name']}:")
            print(f"{'-'*50}")
            print(sample_content + "..." if len(sample_content) == 500 else sample_content)
            print(f"{'-'*50}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
