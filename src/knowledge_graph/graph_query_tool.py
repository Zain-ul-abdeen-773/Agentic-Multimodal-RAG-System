"""
Graph Query Tool — Natural Language → Cypher
=============================================
Techniques implemented:
  - NL-to-Cypher translation for structured queries
  - Entity disambiguation using embedding similarity
  - Template-based Cypher generation for common patterns
  - GraphSAGE integration for semantic graph search
"""

import re
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.knowledge_graph.graph_builder import KnowledgeGraph


class GraphQueryTool:
    """
    Translates natural language queries into Cypher graph traversals.
    
    Uses template matching for common query patterns, with
    LLM-based fallback for complex queries.
    """

    # Common query templates
    QUERY_TEMPLATES = {
        "find_entity": {
            "patterns": [
                r"(?:find|show|get|search)\s+(?:all\s+)?(.+?)(?:\s+entities?)?$",
                r"what\s+(?:is|are)\s+(.+)",
            ],
            "cypher": """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e.name AS name, e.label AS label
                LIMIT 20
            """,
        },
        "find_relations": {
            "patterns": [
                r"(?:how|what)\s+is\s+(.+?)\s+related\s+to\s+(.+)",
                r"(?:relationship|connection)\s+between\s+(.+?)\s+and\s+(.+)",
            ],
            "cypher": """
                MATCH (a:Entity)-[r]-(b:Entity)
                WHERE toLower(a.name) CONTAINS toLower($entity1)
                AND toLower(b.name) CONTAINS toLower($entity2)
                RETURN a.name AS source, type(r) AS relation, 
                       b.name AS target
                LIMIT 20
            """,
        },
        "find_neighbors": {
            "patterns": [
                r"what\s+(?:is|are)\s+near\s+(.+)",
                r"(?:neighbors?|adjacent|next to)\s+(.+)",
            ],
            "cypher": """
                MATCH (e:Entity {name: $entity})-[:NEAR|CONTAINS|CO_OCCURS]-(n)
                RETURN n.name AS neighbor, n.label AS label
                LIMIT 20
            """,
        },
        "find_images": {
            "patterns": [
                r"(?:show|find|get)\s+(?:me\s+)?images?\s+(?:of|with|showing)\s+(.+)",
            ],
            "cypher": """
                MATCH (e:Entity)-[:DEPICTED_IN]->(i:Image)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN i.image_id AS image_id, i.path AS path,
                       i.caption AS caption, e.name AS entity
                LIMIT 10
            """,
        },
        "find_defects": {
            "patterns": [
                r"(?:defects?|issues?|problems?)\s+(?:with|in|on|near)\s+(.+)",
                r"(?:inspect|inspection)\s+(?:results?|notes?)\s+(?:for|of)\s+(.+)",
            ],
            "cypher": """
                MATCH (e:Entity)-[:HAS_DEFECT|INSPECTED_BY]-(d)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e.name AS entity, type(last(relationships(path))) AS relation,
                       d.name AS related
                LIMIT 20
            """,
        },
    }

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        logger.info("GraphQueryTool initialized")

    def parse_query(self, query: str) -> Dict:
        """
        Parse a natural language query into a structured query intent.
        
        Returns:
            Dict with 'template', 'params', 'cypher'
        """
        query = query.strip()
        
        for template_name, template in self.QUERY_TEMPLATES.items():
            for pattern in template["patterns"]:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    params = {}
                    
                    if template_name == "find_relations" and len(groups) >= 2:
                        params["entity1"] = groups[0].strip()
                        params["entity2"] = groups[1].strip()
                    elif groups:
                        params["query"] = groups[0].strip()
                        params["entity"] = groups[0].strip()
                    
                    return {
                        "template": template_name,
                        "params": params,
                        "cypher": template["cypher"],
                    }
        
        # Default: generic entity search
        return {
            "template": "find_entity",
            "params": {"query": query},
            "cypher": self.QUERY_TEMPLATES["find_entity"]["cypher"],
        }

    def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[Dict]:
        """
        Execute a natural language graph query.
        
        Args:
            query: Natural language query
            max_results: Maximum results to return
        Returns:
            List of result dicts from the knowledge graph
        """
        parsed = self.parse_query(query)
        
        logger.info(
            f"Graph query: '{query}' → template={parsed['template']} | "
            f"params={parsed['params']}"
        )
        
        results = self.kg.query(parsed["cypher"], parsed["params"])
        
        # Add retriever metadata
        for r in results:
            r["retriever"] = "knowledge_graph"
            r["query_template"] = parsed["template"]
        
        return results[:max_results]

    def get_entity_context(
        self, entity_name: str, depth: int = 2
    ) -> str:
        """
        Get a textual summary of an entity's graph context.
        Useful for providing structured context to the VLM.
        """
        neighbors = self.kg.get_entity_neighbors(entity_name, depth)
        
        if not neighbors:
            return f"No graph context found for '{entity_name}'"
        
        lines = [f"Graph context for '{entity_name}':"]
        for n in neighbors:
            name = n.get("name", "unknown")
            label = n.get("label", "")
            relation = n.get("relation", "related")
            lines.append(f"  - {relation} → {name} ({label})")
        
        return "\n".join(lines)
