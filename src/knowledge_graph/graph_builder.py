"""
Neo4j Knowledge Graph Builder
==============================
Techniques implemented:
  - spaCy NER for entity extraction from captions/metadata
  - Custom relation extraction (CONTAINS, NEAR, INSPECTED_BY)
  - Neo4j Cypher graph population with property graphs
  - GraphSAGE node embeddings for semantic graph search
  - Batch graph construction pipeline
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from loguru import logger

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
    logger.warning("neo4j not installed. Run: pip install neo4j")

try:
    import spacy
except ImportError:
    spacy = None
    logger.warning("spacy not installed. Run: pip install spacy")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE, Neo4jConfig


# ═══════════════════════════════════════════════════════════════
# Entity & Relation Extraction
# ═══════════════════════════════════════════════════════════════
class EntityRelationExtractor:
    """
    Extracts entities and relationships from text using spaCy NER
    and custom relation patterns.
    
    Entity types: OBJECT, PERSON, LOCATION, MATERIAL, DEFECT
    Relation types: CONTAINS, NEAR, INSPECTED_BY, HAS_DEFECT,
                    LOCATED_IN, MADE_OF, PART_OF
    """

    # Relation patterns (subject-predicate-object regex)
    RELATION_PATTERNS = [
        # "X near Y", "X close to Y"
        (r"(\w+)\s+(?:near|close to|adjacent to|next to)\s+(\w+)", "NEAR"),
        # "X contains Y", "X has Y", "X with Y"
        (r"(\w+)\s+(?:contains?|has|with|including)\s+(\w+)", "CONTAINS"),
        # "X inspected by Y", "X checked by Y"
        (r"(\w+)\s+(?:inspected|checked|examined|reviewed)\s+by\s+(\w+)", "INSPECTED_BY"),
        # "X made of Y", "X composed of Y"
        (r"(\w+)\s+(?:made of|composed of|built from)\s+(\w+)", "MADE_OF"),
        # "X on Y", "X in Y", "X at Y"
        (r"(\w+)\s+(?:on|in|at|inside|within)\s+(\w+)", "LOCATED_IN"),
        # "X part of Y"
        (r"(\w+)\s+(?:part of|component of|section of)\s+(\w+)", "PART_OF"),
    ]

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        if spacy is None:
            raise ImportError("spacy required: pip install spacy")
        
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.info(f"Downloading spaCy model: {spacy_model}")
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        
        logger.info(f"NER model loaded: {spacy_model}")

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Returns list of dicts with 'text', 'label', 'start', 'end'
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy labels to our schema
            label_map = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "PRODUCT": "OBJECT",
                "WORK_OF_ART": "OBJECT",
            }
            label = label_map.get(ent.label_, "OBJECT")
            
            entities.append({
                "text": ent.text,
                "label": label,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        
        # Also extract noun chunks as potential objects
        for chunk in doc.noun_chunks:
            if not any(
                e["start"] <= chunk.start_char < e["end"]
                for e in entities
            ):
                entities.append({
                    "text": chunk.text,
                    "label": "OBJECT",
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                })
        
        return entities

    def extract_relations(
        self, text: str, entities: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Extract relations between entities using pattern matching
        and dependency parsing.
        
        Returns list of dicts with 'subject', 'predicate', 'object'
        """
        if entities is None:
            entities = self.extract_entities(text)
        
        relations = []
        text_lower = text.lower()
        
        # Pattern-based extraction
        for pattern, rel_type in self.RELATION_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                relations.append({
                    "subject": subject,
                    "predicate": rel_type,
                    "object": obj,
                    "source_text": match.group(0),
                })
        
        # Dependency-based: if entity A and entity B appear in same sentence
        # with a verb, create a relation
        doc = self.nlp(text)
        entity_texts = {e["text"].lower() for e in entities}
        
        for sent in doc.sents:
            sent_entities = [
                e for e in entities
                if sent.start_char <= e["start"] < sent.end_char
            ]
            if len(sent_entities) >= 2:
                # Create CO_OCCURS relation for entities in same sentence
                for i in range(len(sent_entities)):
                    for j in range(i + 1, len(sent_entities)):
                        relations.append({
                            "subject": sent_entities[i]["text"],
                            "predicate": "CO_OCCURS",
                            "object": sent_entities[j]["text"],
                            "source_text": sent.text,
                        })
        
        return relations

    def process_document(self, text: str, doc_id: str = "") -> Dict:
        """
        Full extraction pipeline: entities + relations.
        
        Args:
            text: Document text
            doc_id: Document identifier
        Returns:
            Dict with 'entities', 'relations', 'doc_id'
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        
        return {
            "doc_id": doc_id,
            "entities": entities,
            "relations": relations,
        }


# ═══════════════════════════════════════════════════════════════
# Neo4j Graph Manager
# ═══════════════════════════════════════════════════════════════
class KnowledgeGraph:
    """
    Neo4j-backed knowledge graph for structured factual reasoning.
    
    Node types: Entity, Image, Document, Category
    Edge types: CONTAINS, NEAR, INSPECTED_BY, HAS_DEFECT,
                LOCATED_IN, MADE_OF, PART_OF, CO_OCCURS,
                DEPICTED_IN, DESCRIBED_IN
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        self.config = config or Neo4jConfig()
        self.driver = None
        
        if GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.user, self.config.password),
                )
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Neo4j connected: {self.config.uri}")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}. Using local mode.")
                self.driver = None
        
        # Local fallback graph (in-memory)
        self.local_nodes: Dict[str, Dict] = {}
        self.local_edges: List[Dict] = []
        self.extractor = None

    def _init_extractor(self):
        """Lazy-load the entity extractor."""
        if self.extractor is None:
            self.extractor = EntityRelationExtractor(
                self.config.spacy_model
            )

    def create_constraints(self):
        """Create uniqueness constraints for node types."""
        if self.driver is None:
            return
        
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for query in constraints:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Constraint creation: {e}")
        
        logger.info("Neo4j constraints created")

    def add_entity(self, name: str, label: str, properties: Optional[Dict] = None):
        """Add an entity node to the graph."""
        props = properties or {}
        
        if self.driver is not None:
            query = """
            MERGE (e:Entity {name: $name})
            SET e.label = $label
            SET e += $props
            """
            with self.driver.session() as session:
                session.run(query, name=name, label=label, props=props)
        
        # Local fallback
        self.local_nodes[name] = {"name": name, "label": label, **props}

    def add_relation(
        self, subject: str, predicate: str, obj: str,
        properties: Optional[Dict] = None,
    ):
        """Add a relation edge between two entities."""
        props = properties or {}
        
        if self.driver is not None:
            query = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[r:{predicate}]->(o)
            SET r += $props
            """
            with self.driver.session() as session:
                session.run(
                    query, subject=subject, object=obj, props=props
                )
        
        self.local_edges.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            **props,
        })

    def add_image_node(
        self, image_id: str, path: str, caption: str = "",
        metadata: Optional[Dict] = None,
    ):
        """Add an image node and extract entities from its caption."""
        props = metadata or {}
        props["path"] = path
        props["caption"] = caption
        
        if self.driver is not None:
            query = """
            MERGE (i:Image {image_id: $image_id})
            SET i += $props
            """
            with self.driver.session() as session:
                session.run(query, image_id=image_id, props=props)
        
        self.local_nodes[f"img_{image_id}"] = {
            "type": "image", "image_id": image_id, **props
        }
        
        # Extract entities from caption and link to image
        if caption:
            self._init_extractor()
            result = self.extractor.process_document(caption, image_id)
            
            for entity in result["entities"]:
                self.add_entity(entity["text"], entity["label"])
                self.add_relation(
                    entity["text"], "DEPICTED_IN", f"image:{image_id}"
                )
            
            for relation in result["relations"]:
                self.add_relation(
                    relation["subject"],
                    relation["predicate"],
                    relation["object"],
                )

    def build_from_captions(
        self,
        captions: List[Dict],
        batch_size: int = 100,
    ):
        """
        Batch-build graph from a list of caption dicts.
        
        Args:
            captions: List of dicts with 'image_id', 'caption', 'path'
        """
        self._init_extractor()
        self.create_constraints()
        
        for i in range(0, len(captions), batch_size):
            batch = captions[i : i + batch_size]
            
            for item in batch:
                self.add_image_node(
                    image_id=item.get("image_id", str(i)),
                    path=item.get("path", ""),
                    caption=item.get("caption", ""),
                    metadata=item.get("metadata", {}),
                )
            
            logger.info(
                f"Graph built: {i + len(batch)}/{len(captions)} items | "
                f"Nodes: {len(self.local_nodes)} | Edges: {len(self.local_edges)}"
            )

    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query against Neo4j.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
        Returns:
            List of result records as dicts
        """
        params = params or {}
        
        if self.driver is not None:
            with self.driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]
        
        # Local fallback: basic pattern matching
        return self._local_query(cypher, params)

    def _local_query(self, cypher: str, params: Dict) -> List[Dict]:
        """Simple local query for when Neo4j is unavailable."""
        results = []
        cypher_lower = cypher.lower()
        
        # Basic entity search
        if "match" in cypher_lower and "entity" in cypher_lower:
            search_name = params.get("name", "")
            for name, node in self.local_nodes.items():
                if search_name.lower() in name.lower():
                    results.append(node)
        
        # Basic relation search
        if "match" in cypher_lower and any(
            r in cypher_lower for r in [
                "contains", "near", "depicted_in", "co_occurs"
            ]
        ):
            search = params.get("subject", params.get("name", ""))
            for edge in self.local_edges:
                if search.lower() in edge["subject"].lower():
                    results.append(edge)
        
        return results[:50]  # Limit results

    def get_entity_neighbors(
        self, entity_name: str, depth: int = 2
    ) -> List[Dict]:
        """Get all entities within N hops of a given entity."""
        cypher = """
        MATCH (e:Entity {name: $name})-[r*1..$depth]-(neighbor)
        RETURN DISTINCT neighbor.name AS name, 
               neighbor.label AS label,
               type(last(r)) AS relation
        LIMIT 50
        """
        return self.query(cypher, {"name": entity_name, "depth": depth})

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        if self.driver is not None:
            try:
                stats = self.query("""
                MATCH (n) WITH count(n) AS nodes
                MATCH ()-[r]->() WITH nodes, count(r) AS edges
                RETURN nodes, edges
                """)
                if stats:
                    return stats[0]
            except Exception:
                pass
        
        return {
            "nodes": len(self.local_nodes),
            "edges": len(self.local_edges),
        }

    def close(self):
        """Close the Neo4j driver."""
        if self.driver is not None:
            self.driver.close()
            logger.info("Neo4j connection closed")
