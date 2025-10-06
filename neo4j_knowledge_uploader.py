import json
import logging
from typing import Dict, List, Optional
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class Neo4jKnowledgeGraphUploader:
    """Upload structured knowledge graph data to Neo4j database"""
    
    def __init__(self):
        """Initialize Neo4j connection"""
        self.username = os.getenv("username")
        self.password = os.getenv("password")
        self.uri = "neo4j+s://f2f2b632.databases.neo4j.io"
        
        if not self.username or not self.password:
            raise ValueError("Neo4j username and password not found in environment variables")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.info("Neo4j connection established successfully")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """Clear all existing data in the database (use with caution)"""
        with self.driver.session() as session:
            try:
                result = session.run("MATCH (n) DETACH DELETE n")
                logger.info("Database cleared successfully")
                return True
            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                return False
    
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes for better performance"""
        constraints_and_indexes = [
            # Unique constraints for entity IDs
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            
            # Indexes for better query performance
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_name_index IF NOT EXISTS FOR (d:Document) ON (d.name)",
            
            # Indexes for material-specific properties
            "CREATE INDEX material_composition_index IF NOT EXISTS FOR (m:Material) ON (m.composition)",
            "CREATE INDEX property_value_index IF NOT EXISTS FOR (p:Property) ON (p.value)",
            "CREATE INDEX process_temperature_index IF NOT EXISTS FOR (pr:Process) ON (pr.temperature)",
        ]
        
        with self.driver.session() as session:
            for constraint_or_index in constraints_and_indexes:
                try:
                    session.run(constraint_or_index)
                    logger.debug(f"Created: {constraint_or_index}")
                except Exception as e:
                    logger.warning(f"Could not create constraint/index: {e}")
    
    def create_nodes_batch(self, nodes: List[Dict], upsert: bool = True) -> bool:
        """Create or update nodes in batches for better performance"""
        try:
            with self.driver.session() as session:
                # Group nodes by labels for more efficient creation
                nodes_by_labels = {}
                for node in nodes:
                    labels_key = tuple(sorted(node.get('labels', ['Entity'])))
                    if labels_key not in nodes_by_labels:
                        nodes_by_labels[labels_key] = []
                    nodes_by_labels[labels_key].append(node)
                
                total_created = 0
                total_updated = 0
                
                for labels, nodes_group in nodes_by_labels.items():
                    labels_str = ':'.join(labels)
                    
                    # Process nodes in batches of 1000
                    batch_size = 1000
                    for i in range(0, len(nodes_group), batch_size):
                        batch = nodes_group[i:i + batch_size]
                        
                        if upsert:
                            # Use MERGE for upsert functionality
                            query = f"""
                            UNWIND $nodes as node
                            MERGE (n:{labels_str} {{id: node.id}})
                            SET n += node.properties
                            """
                        else:
                            # Use CREATE for new nodes only
                            query = f"""
                            UNWIND $nodes as node
                            CREATE (n:{labels_str})
                            SET n = node.properties
                            SET n.id = node.id
                            """
                        
                        result = session.run(query, nodes=batch)
                        summary = result.consume()
                        batch_created = summary.counters.nodes_created
                        batch_updated = summary.counters.properties_set - batch_created
                        
                        total_created += batch_created
                        total_updated += batch_updated
                        
                        logger.debug(f"Processed {len(batch)} nodes with labels {labels_str}: {batch_created} created, {batch_updated} updated")
                
                if upsert:
                    logger.info(f"Successfully processed {total_created + total_updated} nodes: {total_created} created, {total_updated} updated")
                else:
                    logger.info(f"Successfully created {total_created} nodes")
                return True
                
        except Exception as e:
            logger.error(f"Error processing nodes: {e}")
            return False
    
    def create_relationships_batch(self, relationships: List[Dict], upsert: bool = True) -> bool:
        """Create or update relationships in batches"""
        try:
            with self.driver.session() as session:
                # Group relationships by type for more efficient creation
                rels_by_type = {}
                for rel in relationships:
                    rel_type = rel.get('type', 'RELATED_TO')
                    if rel_type not in rels_by_type:
                        rels_by_type[rel_type] = []
                    rels_by_type[rel_type].append(rel)
                
                total_created = 0
                total_updated = 0
                
                for rel_type, rels_group in rels_by_type.items():
                    # Process relationships in batches of 1000
                    batch_size = 1000
                    for i in range(0, len(rels_group), batch_size):
                        batch = rels_group[i:i + batch_size]
                        
                        if upsert:
                            # Use MERGE for upsert functionality
                            query = f"""
                            UNWIND $relationships as rel
                            MATCH (source {{id: rel.source}})
                            MATCH (target {{id: rel.target}})
                            MERGE (source)-[r:{rel_type}]->(target)
                            SET r += rel.properties
                            SET r.id = rel.id
                            """
                        else:
                            # Use CREATE for new relationships only
                            query = f"""
                            UNWIND $relationships as rel
                            MATCH (source {{id: rel.source}})
                            MATCH (target {{id: rel.target}})
                            CREATE (source)-[r:{rel_type}]->(target)
                            SET r = rel.properties
                            SET r.id = rel.id
                            """
                        
                        result = session.run(query, relationships=batch)
                        summary = result.consume()
                        batch_created = summary.counters.relationships_created
                        batch_updated = summary.counters.properties_set - batch_created
                        
                        total_created += batch_created
                        total_updated += batch_updated
                        
                        logger.debug(f"Processed {len(batch)} relationships of type {rel_type}: {batch_created} created, {batch_updated} updated")
                
                if upsert:
                    logger.info(f"Successfully processed {total_created + total_updated} relationships: {total_created} created, {total_updated} updated")
                else:
                    logger.info(f"Successfully created {total_created} relationships")
                return True
                
        except Exception as e:
            logger.error(f"Error processing relationships: {e}")
            return False
    
    def upload_knowledge_graph(self, neo4j_data: Dict, clear_existing: bool = False) -> bool:
        """Upload complete knowledge graph to Neo4j"""
        try:
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing database...")
                self.clear_database()
            
            # Create constraints and indexes
            logger.info("Creating constraints and indexes...")
            self.create_constraints_and_indexes()
            
            # Upload nodes (use upsert if not clearing database)
            nodes = neo4j_data.get('nodes', [])
            logger.info(f"Uploading {len(nodes)} nodes...")
            upsert_mode = not clear_existing
            if not self.create_nodes_batch(nodes, upsert=upsert_mode):
                return False
            
            # Upload relationships (use upsert if not clearing database)
            relationships = neo4j_data.get('relationships', [])
            logger.info(f"Uploading {len(relationships)} relationships...")
            if not self.create_relationships_batch(relationships, upsert=upsert_mode):
                return False
            
            # Verify upload
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                logger.info(f"Upload verification:")
                logger.info(f"  - Total nodes in database: {node_count}")
                logger.info(f"  - Total relationships in database: {rel_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading knowledge graph: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the current database"""
        try:
            with self.driver.session() as session:
                stats = {}
                
                # Node counts by label
                result = session.run("""
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                    YIELD value
                    RETURN label, value.count as count
                """)
                
                node_counts = {}
                for record in result:
                    node_counts[record["label"]] = record["count"]
                
                stats["node_counts"] = node_counts
                
                # Relationship counts by type
                result = session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {})
                    YIELD value
                    RETURN relationshipType, value.count as count
                """)
                
                rel_counts = {}
                for record in result:
                    rel_counts[record["relationshipType"]] = record["count"]
                
                stats["relationship_counts"] = rel_counts
                
                # Total counts
                total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                stats["total_nodes"] = total_nodes
                stats["total_relationships"] = total_rels
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def run_sample_queries(self):
        """Run sample queries to demonstrate the knowledge graph"""
        sample_queries = [
            {
                "name": "Materials and their properties",
                "query": """
                MATCH (m:Material)-[r:HAS_PROPERTY]->(p:Property)
                RETURN m.name as material, p.name as property, p.value as value
                LIMIT 10
                """
            },
            {
                "name": "Research processes and conditions",
                "query": """
                MATCH (proc:Process)-[r:OCCURS_AT]->(cond:Condition)
                RETURN proc.name as process, cond.type as condition_type, cond.value as value
                LIMIT 10
                """
            },
            {
                "name": "Document authors and their research",
                "query": """
                MATCH (d:Document)-[:RESEARCHED_BY]->(author)
                RETURN d.title as document, collect(author.name) as authors
                LIMIT 5
                """
            },
            {
                "name": "Material composition relationships",
                "query": """
                MATCH (m:Material)-[r:COMPOSED_OF]->(component)
                RETURN m.name as material, collect(component.name) as components
                LIMIT 10
                """
            }
        ]
        
        with self.driver.session() as session:
            for query_info in sample_queries:
                try:
                    logger.info(f"\nSample Query: {query_info['name']}")
                    logger.info("-" * 50)
                    
                    result = session.run(query_info['query'])
                    records = list(result)
                    
                    if records:
                        for record in records:
                            logger.info(f"  {dict(record)}")
                    else:
                        logger.info("  No results found")
                        
                except Exception as e:
                    logger.error(f"Error running query '{query_info['name']}': {e}")

def main():
    """Main function to upload knowledge graph data"""
    try:
        # Initialize uploader
        uploader = Neo4jKnowledgeGraphUploader()
        
        # Load Neo4j data
        neo4j_data_file = "/home/sai-nivedh-26/ddmm-proj/neo4j_knowledge_graph_data.json"
        
        if not os.path.exists(neo4j_data_file):
            logger.error(f"Neo4j data file not found: {neo4j_data_file}")
            logger.info("Please run gemini_knowledge_extractor.py first to generate the data")
            return
        
        with open(neo4j_data_file, 'r', encoding='utf-8') as f:
            neo4j_data = json.load(f)
        
        logger.info(f"Loaded Neo4j data:")
        logger.info(f"  - Nodes: {len(neo4j_data.get('nodes', []))}")
        logger.info(f"  - Relationships: {len(neo4j_data.get('relationships', []))}")
        
        # Upload to Neo4j
        success = uploader.upload_knowledge_graph(neo4j_data, clear_existing=True)
        
        if success:
            logger.info("Knowledge graph uploaded successfully!")
            
            # Get database statistics
            stats = uploader.get_database_stats()
            if stats:
                logger.info(f"\nDatabase Statistics:")
                logger.info(f"  - Total nodes: {stats.get('total_nodes', 0)}")
                logger.info(f"  - Total relationships: {stats.get('total_relationships', 0)}")
                
                if stats.get('node_counts'):
                    logger.info("  - Node counts by type:")
                    for label, count in stats['node_counts'].items():
                        logger.info(f"    {label}: {count}")
                
                if stats.get('relationship_counts'):
                    logger.info("  - Relationship counts by type:")
                    for rel_type, count in stats['relationship_counts'].items():
                        logger.info(f"    {rel_type}: {count}")
            
            # Run sample queries
            logger.info("\nRunning sample queries...")
            uploader.run_sample_queries()
            
        else:
            logger.error("Failed to upload knowledge graph")
        
        # Close connection
        uploader.close()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
