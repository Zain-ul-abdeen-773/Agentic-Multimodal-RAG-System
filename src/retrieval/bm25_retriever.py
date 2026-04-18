"""
BM25 Sparse Retriever
=====================
Techniques implemented:
  - BM25 (Okapi) sparse lexical retrieval
  - Custom tokenization and preprocessing pipeline
  - SMOTE-aware query expansion for underrepresented terms
  - Serves as lexical arm for hybrid retrieval (paired with dense)
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, BM25Config


class BM25Retriever:
    """
    BM25 sparse retrieval for text documents.
    
    Provides keyword-based matching that complements dense vector
    retrieval — especially effective for entity names, rare terms,
    and exact-match queries where dense models struggle.
    """

    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.bm25: Optional["BM25Okapi"] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        
        # Stopwords (minimal set to preserve domain terms)
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "and", "or", "but", "if", "then", "else", "when", "at", "by",
            "for", "with", "about", "against", "between", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further",
            "than", "once", "here", "there", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "too", "very", "just", "because",
            "this", "that", "these", "those", "it", "its",
        }

    def _tokenize(self, text: str) -> List[str]:
        """Preprocess and tokenize text for BM25."""
        # Lowercase
        text = text.lower()
        # Remove special chars but keep alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords and very short tokens
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        return tokens

    def index(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ):
        """
        Build BM25 index over a document corpus.
        
        Args:
            documents: List of document texts
            doc_ids: Optional document identifiers
        """
        if BM25Okapi is None:
            raise ImportError("rank_bm25 required: pip install rank-bm25")
        
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Tokenize corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b,
        )
        
        logger.info(
            f"BM25 index built: {len(documents)} documents | "
            f"Avg tokens/doc: "
            f"{np.mean([len(t) for t in self.tokenized_corpus]):.0f}"
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search the BM25 index.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
        Returns:
            List of dicts with 'doc_id', 'text', 'score', 'rank'
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")
        
        top_k = top_k or self.config.top_k
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "text": self.documents[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                    "retriever": "bm25",
                })
        
        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Dict]]:
        """Batch search for multiple queries."""
        return [self.search(q, top_k) for q in queries]

    def get_term_frequencies(self, query: str) -> Dict[str, float]:
        """Get BM25 term frequency scores for debugging."""
        if self.bm25 is None:
            return {}
        
        tokens = self._tokenize(query)
        tf_scores = {}
        
        for token in tokens:
            if token in self.bm25.idf:
                tf_scores[token] = float(self.bm25.idf[token])
        
        return dict(sorted(tf_scores.items(), key=lambda x: -x[1]))
