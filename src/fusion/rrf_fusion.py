"""
Reciprocal Rank Fusion (RRF) — Multi-Source Result Merging
==========================================================
Techniques implemented:
  - RRF with configurable k constant (default k=60)
  - Score normalization and deduplication
  - Bayes Optimal Error Rate analysis
  - Davies-Bouldin Index for retrieval cluster quality
  - Silhouette Score for result quality analysis
  - ∆RRF metric: improvement over single-best retriever
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from pathlib import Path
from loguru import logger

try:
    from sklearn.metrics import (
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.cluster import KMeans
except ImportError:
    davies_bouldin_score, silhouette_score = None, None
    KMeans = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, RRFConfig


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining results from multiple retrievers.
    
    RRF score for document d across retriever set R:
      Score(d) = Σ_{r ∈ R} 1 / (k + rank(r, d))
    
    Benefits:
      - Parameter-light (only k)
      - Scale-invariant (works with heterogeneous score ranges)
      - No score normalization needed
      - Theoretically grounded (Cormack et al., SIGIR 2009)
    """

    def __init__(self, config: Optional[RRFConfig] = None):
        self.config = config or RRFConfig()
        self.k = self.config.k
        
        # Track per-retriever metrics for ∆RRF
        self.retriever_mrr: Dict[str, List[float]] = defaultdict(list)
        self.fused_mrr: List[float] = []

    def fuse(
        self,
        ranked_lists: List[List[Dict]],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fuse multiple ranked lists using RRF.
        
        Args:
            ranked_lists: List of ranked result lists from different retrievers.
                         Each result dict must have an 'id' or 'doc_id' key.
            top_k: Number of top results to return
        Returns:
            Single fused and re-ranked result list
        """
        top_k = top_k or self.config.top_k_final
        
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, Dict] = {}
        
        for retriever_idx, ranked_list in enumerate(ranked_lists):
            for rank, doc in enumerate(ranked_list):
                # Get document identifier
                doc_id = str(
                    doc.get("id", doc.get("doc_id", doc.get("text", "")[:50]))
                )
                
                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)
                
                # Keep the richest document data
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc.copy()
                else:
                    # Merge metadata
                    for key, value in doc.items():
                        if key not in doc_data[doc_id] or not doc_data[doc_id][key]:
                            doc_data[doc_id][key] = value
        
        # Build sorted results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: -x[1])
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            result = doc_data.get(doc_id, {"id": doc_id})
            result["rrf_score"] = float(score)
            result["rrf_rank"] = rank + 1
            result["retriever"] = "rrf_fusion"
            results.append(result)
        
        logger.info(
            f"RRF fusion: {sum(len(r) for r in ranked_lists)} inputs → "
            f"{len(results)} fused results "
            f"(from {len(ranked_lists)} retrievers)"
        )
        
        return results

    def compute_delta_rrf(
        self,
        query: str,
        ranked_lists: List[List[Dict]],
        relevant_doc_id: str,
    ) -> Dict[str, float]:
        """
        Compute ∆RRF: improvement of RRF over the single best retriever.
        
        ∆RRF = MRR_rrf - max_i(MRR_Ri)
        
        A positive ∆RRF confirms multi-source fusion adds value.
        
        Args:
            query: Query string
            ranked_lists: Per-retriever results
            relevant_doc_id: Ground-truth relevant document ID
        Returns:
            Dict with 'delta_rrf', 'rrf_mrr', 'best_single_mrr'
        """
        # MRR for each individual retriever
        retriever_mrrs = {}
        for i, ranked_list in enumerate(ranked_lists):
            retriever_name = (
                ranked_list[0].get("retriever", f"retriever_{i}")
                if ranked_list else f"retriever_{i}"
            )
            
            mrr = 0.0
            for rank, doc in enumerate(ranked_list):
                doc_id = str(doc.get("id", doc.get("doc_id", "")))
                if doc_id == relevant_doc_id:
                    mrr = 1.0 / (rank + 1)
                    break
            
            retriever_mrrs[retriever_name] = mrr
            self.retriever_mrr[retriever_name].append(mrr)
        
        # MRR for fused results
        fused = self.fuse(ranked_lists)
        rrf_mrr = 0.0
        for rank, doc in enumerate(fused):
            doc_id = str(doc.get("id", doc.get("doc_id", "")))
            if doc_id == relevant_doc_id:
                rrf_mrr = 1.0 / (rank + 1)
                break
        
        self.fused_mrr.append(rrf_mrr)
        
        # ∆RRF
        best_single_mrr = max(retriever_mrrs.values()) if retriever_mrrs else 0
        delta_rrf = rrf_mrr - best_single_mrr
        
        return {
            "delta_rrf": float(delta_rrf),
            "rrf_mrr": float(rrf_mrr),
            "best_single_mrr": float(best_single_mrr),
            "per_retriever_mrr": retriever_mrrs,
        }

    def compute_aggregate_delta_rrf(self) -> Dict[str, float]:
        """
        Compute aggregate ∆RRF across all evaluated queries.
        """
        if not self.fused_mrr:
            return {"delta_rrf": 0.0, "n_queries": 0}
        
        avg_rrf_mrr = np.mean(self.fused_mrr)
        
        best_single_avg = 0.0
        for retriever_name, mrrs in self.retriever_mrr.items():
            avg = np.mean(mrrs) if mrrs else 0
            best_single_avg = max(best_single_avg, avg)
        
        delta = float(avg_rrf_mrr - best_single_avg)
        
        logger.info(
            f"Aggregate ∆RRF = {delta:.4f} | "
            f"RRF MRR = {avg_rrf_mrr:.4f} | "
            f"Best single MRR = {best_single_avg:.4f}"
        )
        
        return {
            "delta_rrf": delta,
            "rrf_mrr": float(avg_rrf_mrr),
            "best_single_mrr": float(best_single_avg),
            "n_queries": len(self.fused_mrr),
        }

    @staticmethod
    def bayes_optimal_error_rate(
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Estimate the Bayes Optimal Error Rate for retrieval results.
        
        Uses 1-NN classifier as proxy for the Bayes-optimal boundary.
        Lower is better — indicates embeddings are separable.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        knn = KNeighborsClassifier(n_neighbors=1)
        scores = cross_val_score(knn, embeddings, labels, cv=5)
        error_rate = 1.0 - np.mean(scores)
        
        logger.info(f"Bayes Optimal Error Rate estimate: {error_rate:.4f}")
        return float(error_rate)

    @staticmethod
    def cluster_quality_analysis(
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> Dict[str, float]:
        """
        Analyze retrieval result quality using clustering metrics:
          - Davies-Bouldin Index (lower is better)
          - Silhouette Score (-1 to 1, higher is better)
        """
        if davies_bouldin_score is None or KMeans is None:
            return {"error": "scikit-learn required"}
        
        if len(embeddings) < n_clusters + 1:
            return {"error": f"Need > {n_clusters} samples"}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        db_score = float(davies_bouldin_score(embeddings, cluster_labels))
        sil_score = float(silhouette_score(embeddings, cluster_labels))
        
        return {
            "davies_bouldin": db_score,
            "silhouette": sil_score,
            "n_clusters": n_clusters,
        }
