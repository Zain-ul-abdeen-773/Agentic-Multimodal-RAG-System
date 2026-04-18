"""
Agent Tool Definitions
======================
Wrapper tools that connect the LangGraph agent to retrieval modules.
Each tool has a detailed docstring that the LLM uses for routing.
"""

from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class ToolRegistry:
    """
    Registry that holds references to all retrieval modules
    and provides a unified interface for the agent.
    """

    def __init__(self):
        self.image_store = None       # FAISSHNSWStore for images
        self.text_store = None        # FAISSHNSWStore for text
        self.bm25 = None              # BM25Retriever
        self.graph_tool = None        # GraphQueryTool
        self.colbert = None           # ColBERTReranker
        self.clip_encoder = None      # CLIPEncoder
        self.rrf_fusion = None        # RRFFusion
        self.vlm = None               # VLMReasoner

    def register(self, **kwargs):
        """Register retrieval module instances."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Registered tool: {key}")
            else:
                logger.warning(f"Unknown tool: {key}")

    def search_images(self, query: str, top_k: int = 10) -> List[Dict]:
        """CLIP-based image search."""
        if self.clip_encoder is None or self.image_store is None:
            return [{"info": "Image search not configured"}]
        
        import torch
        import numpy as np
        
        # Encode query text with CLIP
        tokens = self.clip_encoder.tokenize([query]).to(
            next(self.clip_encoder.parameters()).device
        )
        with torch.no_grad():
            query_emb = self.clip_encoder.encode_text(tokens)
        
        query_np = query_emb.cpu().numpy()
        results = self.image_store.search(query_np[0], top_k=top_k)
        
        return results

    def search_text(self, query: str, top_k: int = 10) -> List[Dict]:
        """Hybrid BM25 + dense text search."""
        results = []
        
        # BM25 sparse search
        if self.bm25 is not None:
            bm25_results = self.bm25.search(query, top_k=top_k)
            results.extend(bm25_results)
        
        # Dense vector search
        if self.text_store is not None and self.clip_encoder is not None:
            import torch
            import numpy as np
            
            tokens = self.clip_encoder.tokenize([query]).to(
                next(self.clip_encoder.parameters()).device
            )
            with torch.no_grad():
                query_emb = self.clip_encoder.encode_text(tokens)
            
            query_np = query_emb.cpu().numpy()
            dense_results = self.text_store.search(query_np[0], top_k=top_k)
            results.extend(dense_results)
        
        return results

    def query_graph(self, query: str) -> List[Dict]:
        """Knowledge graph query."""
        if self.graph_tool is None:
            return [{"info": "Graph query not configured"}]
        return self.graph_tool.search(query)

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Full hybrid search with RRF fusion."""
        image_results = self.search_images(query, top_k=top_k)
        text_results = self.search_text(query, top_k=top_k)
        graph_results = self.query_graph(query)
        
        if self.rrf_fusion is not None:
            fused = self.rrf_fusion.fuse([
                image_results, text_results, graph_results
            ])
            
            # Optionally rerank with ColBERT
            if self.colbert is not None:
                fused = self.colbert.rerank(query, fused)
            
            return fused
        
        # Simple merge if no RRF
        all_results = image_results + text_results + graph_results
        return sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
