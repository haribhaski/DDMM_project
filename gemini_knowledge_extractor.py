import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from google import genai
from google.genai import types
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiKnowledgeExtractor:
    """Extract structured knowledge from scientific documents using Gemini 2.5 Flash"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini knowledge extractor"""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
        logger.info("GeminiKnowledgeExtractor initialized successfully")
    
    def get_knowledge_extraction_prompt(self) -> str:
        """Get comprehensive prompt template for extracting detailed scientific knowledge"""
        return """
You are an expert materials science knowledge extraction system. Your task is to analyze scientific documents and extract detailed, meaningful knowledge for creating a comprehensive materials science knowledge graph.

FOCUS AREAS:
- Materials composition, structure, and phases
- Processing methods and parameters
- Properties and their quantitative values
- Performance metrics and improvements
- Causal relationships between processing-structure-properties
- Experimental conditions and their effects

ENTITY TYPES TO EXTRACT:

1. **MATERIALS** - Specific materials, alloys, compounds, phases
   - Include: composition, crystal structure, phase percentages
   - Example: "Ti-Zr-Cr-Mn-Fe-Ni alloy with C14 Laves phase (95.5%)"

2. **PROPERTIES** - Measurable characteristics with values and units
   - Include: numerical values, units, measurement conditions
   - Example: "Hydrogen storage capacity: 1.77 wt.% at 30°C"

3. **PROCESSES** - Manufacturing, treatment, or testing methods
   - Include: parameters, conditions, duration, equipment
   - Example: "Annealing at 1200°C for 2h with 200°C/h heating rate"

4. **CONDITIONS** - Environmental or experimental parameters
   - Include: temperature, pressure, atmosphere, time
   - Example: "Argon atmosphere at 200 L/h flow rate"

5. **MEASUREMENTS** - Specific test results or data points
   - Include: values, units, test methods, conditions
   - Example: "Unit cell volume: 168.93 Å³ measured by XRD"

6. **PHENOMENA** - Physical or chemical effects observed
   - Include: mechanisms, causes, effects
   - Example: "Lattice expansion causing reduced desorption pressure"

RELATIONSHIP TYPES TO EXTRACT:

1. **COMPOSITION_CONTAINS** - Material contains specific elements/phases
2. **PROCESS_MODIFIES** - Process changes material structure/properties
3. **PROPERTY_DEPENDS_ON** - Property value depends on specific conditions
4. **CONDITION_AFFECTS** - Condition influences property or behavior
5. **PERFORMANCE_IMPROVES_WITH** - Performance enhancement relationships
6. **PERFORMANCE_DEGRADES_WITH** - Performance reduction relationships
7. **MEASUREMENT_SHOWS** - Measurement technique reveals specific values
8. **MECHANISM_CAUSES** - Physical mechanism causes observed effect
9. **OPTIMAL_RANGE** - Optimal parameter range for desired outcome
10. **PHASE_TRANSFORMS_TO** - Phase transformation relationships

EXTRACTION RULES:
- NO author information, publication details, or citation relationships
- NO generic "MENTIONED_IN" relationships
- Focus on quantitative data with units
- Extract specific numerical values and ranges
- Include experimental conditions for all measurements
- Capture cause-effect relationships between processing and properties
- Extract optimization insights and performance trade-offs

OUTPUT FORMAT:
{
  "document_metadata": {
    "title": "document title",
    "research_focus": "main research area",
    "material_system": "primary material studied"
  },
  "entities": [
    {
      "id": "unique_entity_id",
      "name": "entity_name",
      "type": "MATERIALS|PROPERTIES|PROCESSES|CONDITIONS|MEASUREMENTS|PHENOMENA",
      "description": "detailed technical description",
      "properties": {
        "numerical_value": "value with units",
        "measurement_conditions": "conditions under which measured",
        "uncertainty": "measurement uncertainty if given",
        "method": "measurement or processing method"
      }
    }
  ],
  "relationships": [
    {
      "id": "unique_relationship_id",
      "source_entity": "source_entity_id",
      "target_entity": "target_entity_id",
      "relationship_type": "specific_relationship_type",
      "description": "detailed description of the relationship",
      "properties": {
        "quantitative_effect": "numerical change or correlation",
        "conditions": "conditions under which relationship holds",
        "mechanism": "physical/chemical mechanism if known"
      },
      "evidence": "specific text supporting this relationship"
    }
  ],
  "key_insights": [
    {
      "insight": "important scientific finding",
      "quantitative_evidence": "numerical data supporting the insight",
      "practical_significance": "why this matters for applications",
      "related_entities": ["entity_id1", "entity_id2"]
    }
  ]
}

CRITICAL REQUIREMENTS:
1. Extract ONLY scientifically meaningful relationships
2. Include numerical values with proper units wherever possible
3. Focus on materials science knowledge, not bibliographic information
4. Capture optimization insights (best conditions, trade-offs)
5. Extract cause-effect relationships between processing-structure-properties
6. Include measurement uncertainties and experimental conditions
7. NO relationships to authors, institutions, or publication metadata

Now analyze the following scientific document text and extract detailed materials science knowledge:

"""

    def extract_knowledge_from_text(self, text: str, document_name: str = "") -> Optional[Dict]:
        """Extract structured knowledge from text using Gemini"""
        try:
            logger.info(f"Extracting knowledge from document: {document_name}")
            
            # Prepare the prompt with the text
            full_prompt = self.get_knowledge_extraction_prompt() + f"\n\nDOCUMENT TEXT:\n{text}"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=full_prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            
            logger.info("Sending request to Gemini for knowledge extraction...")
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )
            
            # Parse JSON response
            response_text = response.text
            logger.debug(f"Raw Gemini response: {response_text[:500]}...")
            
            try:
                knowledge_data = json.loads(response_text)
                logger.info(f"Successfully extracted knowledge structure")
                logger.info(f"  - Entities: {len(knowledge_data.get('entities', []))}")
                logger.info(f"  - Relationships: {len(knowledge_data.get('relationships', []))}")
                logger.info(f"  - Key findings: {len(knowledge_data.get('key_findings', []))}")
                
                # Add document metadata
                knowledge_data["source_document"] = document_name
                knowledge_data["extraction_timestamp"] = str(datetime.now())
                
                return knowledge_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting knowledge from text: {e}")
            return None
    
    def process_mistral_extraction_file(self, mistral_file_path: str) -> List[Dict]:
        """Process the Mistral extraction file and extract knowledge from each document"""
        try:
            # Load the Mistral extraction data
            with open(mistral_file_path, 'r', encoding='utf-8') as f:
                if mistral_file_path.endswith('.json'):
                    mistral_data = json.load(f)
                else:
                    # Handle text file format
                    content = f.read()
                    mistral_data = self.parse_text_extraction_file(content)
            
            knowledge_results = []
            
            if isinstance(mistral_data, list):
                # JSON format with multiple documents
                for doc_data in mistral_data:
                    doc_name = doc_data.get('file_name', 'unknown')
                    doc_content = doc_data.get('combined_content', '')
                    
                    if doc_content.strip():
                        knowledge = self.extract_knowledge_from_text(doc_content, doc_name)
                        if knowledge:
                            knowledge_results.append(knowledge)
            else:
                # Single document or text format
                knowledge = self.extract_knowledge_from_text(str(mistral_data), "extracted_document")
                if knowledge:
                    knowledge_results.append(knowledge)
            
            logger.info(f"Successfully processed {len(knowledge_results)} documents for knowledge extraction")
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Error processing Mistral extraction file: {e}")
            return []
    
    def parse_text_extraction_file(self, content: str) -> List[Dict]:
        """Parse text format extraction file into structured data"""
        documents = []
        
        # Split by file separators - look for the pattern with FILE: 
        file_pattern = r'={80,}\nFILE:\s*(.+?)\n={80,}(.*?)(?=\n={80,}\nEND OF|$)'
        matches = re.findall(file_pattern, content, re.DOTALL)
        
        for file_name, doc_content in matches:
            file_name = file_name.strip()
            doc_content = doc_content.strip()
            
            if doc_content:
                documents.append({
                    'file_name': file_name,
                    'combined_content': doc_content
                })
        
        # If no matches with the above pattern, try a simpler approach
        if not documents:
            # Split by the file separator lines
            sections = content.split('=' * 80)
            
            for i in range(len(sections)):
                section = sections[i].strip()
                if section.startswith('FILE:'):
                    # Extract filename
                    lines = section.split('\n')
                    file_name = lines[0].replace('FILE:', '').strip()
                    
                    # Get the content from the next section
                    if i + 1 < len(sections):
                        doc_content = sections[i + 1].strip()
                        
                        # Remove END OF markers
                        doc_content = re.sub(r'END OF .+', '', doc_content, flags=re.DOTALL).strip()
                        
                        if doc_content and not doc_content.startswith('END OF'):
                            documents.append({
                                'file_name': file_name,
                                'combined_content': doc_content
                            })
        
        return documents
    
    def save_knowledge_data(self, knowledge_data: List[Dict], output_file: str):
        """Save extracted knowledge data to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Knowledge data saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving knowledge data: {e}")
    
    def convert_to_neo4j_format(self, knowledge_data: List[Dict]) -> Dict[str, List]:
        """Convert extracted knowledge to Neo4j-compatible format"""
        neo4j_data = {
            "nodes": [],
            "relationships": []
        }
        
        entity_id_map = {}  # Track entity IDs across documents
        
        for doc_idx, doc_knowledge in enumerate(knowledge_data):
            doc_name = doc_knowledge.get("source_document", f"doc_{doc_idx}")
            
            # Process entities as nodes
            for entity in doc_knowledge.get("entities", []):
                node_id = f"{doc_name}_{entity['id']}"
                entity_id_map[entity['id']] = node_id
                
                # Use entity type as primary label
                labels = [entity['type']]
                
                node = {
                    "id": node_id,
                    "labels": labels,
                    "properties": {
                        "name": entity['name'],
                        "description": entity.get('description', ''),
                        **entity.get('properties', {})
                    }
                }
                
                neo4j_data["nodes"].append(node)
            
            # Process relationships - only scientific relationships
            for relationship in doc_knowledge.get("relationships", []):
                source_id = entity_id_map.get(relationship['source_entity'])
                target_id = entity_id_map.get(relationship['target_entity'])
                
                if source_id and target_id:
                    rel = {
                        "id": f"{doc_name}_{relationship['id']}",
                        "type": relationship['relationship_type'],
                        "source": source_id,
                        "target": target_id,
                        "properties": {
                            "description": relationship.get('description', ''),
                            "evidence": relationship.get('evidence', ''),
                            **relationship.get('properties', {})
                        }
                    }
                    neo4j_data["relationships"].append(rel)
        
        logger.info(f"Converted to Neo4j format:")
        logger.info(f"  - Nodes: {len(neo4j_data['nodes'])}")
        logger.info(f"  - Relationships: {len(neo4j_data['relationships'])}")
        
        return neo4j_data
    
    def save_neo4j_data(self, neo4j_data: Dict, output_file: str):
        """Save Neo4j-compatible data to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(neo4j_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Neo4j data saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving Neo4j data: {e}")

def main():
    """Main function to test knowledge extraction"""
    try:
        # Initialize extractor
        extractor = GeminiKnowledgeExtractor()
        
        # Process Mistral extraction file
        mistral_file = "/home/sai-nivedh-26/ddmm-proj/mistral_enhanced_extraction.txt"
        knowledge_data = extractor.process_mistral_extraction_file(mistral_file)
        
        if knowledge_data:
            # Save raw knowledge data
            knowledge_output = "/home/sai-nivedh-26/ddmm-proj/gemini_knowledge_extraction.json"
            extractor.save_knowledge_data(knowledge_data, knowledge_output)
            
            # Convert to Neo4j format
            neo4j_data = extractor.convert_to_neo4j_format(knowledge_data)
            
            # Save Neo4j data
            neo4j_output = "/home/sai-nivedh-26/ddmm-proj/neo4j_knowledge_graph_data.json"
            extractor.save_neo4j_data(neo4j_data, neo4j_output)
            
            # Print summary
            total_docs = len(knowledge_data)
            total_entities = sum(len(doc.get("entities", [])) for doc in knowledge_data)
            total_relationships = sum(len(doc.get("relationships", [])) for doc in knowledge_data)
            
            print(f"\nKnowledge Extraction Summary:")
            print(f"{'='*50}")
            print(f"Documents processed: {total_docs}")
            print(f"Total entities extracted: {total_entities}")
            print(f"Total relationships extracted: {total_relationships}")
            print(f"Neo4j nodes: {len(neo4j_data['nodes'])}")
            print(f"Neo4j relationships: {len(neo4j_data['relationships'])}")
            print(f"Knowledge data saved to: {knowledge_output}")
            print(f"Neo4j data saved to: {neo4j_output}")
            print(f"{'='*50}")
        else:
            print("No knowledge data extracted")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
