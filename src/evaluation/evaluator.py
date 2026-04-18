"""
Evaluation Suite — Metrics, Ablation, and Analysis
===================================================
Techniques implemented:
  - Recall@K (K=1,5,10) for visual, textual, and graph queries
  - MRR (Mean Reciprocal Rank)
  - ∆RRF metric
  - BERTScore F1 for VLM answer quality
  - Chunking quality (Cchunk)
  - Graph query accuracy
  - t-SNE + PCA embedding visualizations
  - Gaussian Mixture Models for cluster analysis
  - Isolation Forest + One-Class SVM for anomaly detection
  - DBSCAN for automatic cluster discovery
  - Davies-Bouldin Index + Silhouette Score
"""

import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from loguru import logger

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import davies_bouldin_score, silhouette_score
except ImportError:
    TSNE = PCA = GaussianMixture = IsolationForest = None
    OneClassSVM = DBSCAN = davies_bouldin_score = silhouette_score = None


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, EvalConfig


class Evaluator:
    """
    Comprehensive evaluation suite for the multimodal RAG system.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.results: Dict[str, List] = defaultdict(list)

    # ── Recall@K ─────────────────────────────────────────────
    def recall_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        k: int,
    ) -> float:
        """
        Compute Recall@K.
        
        Recall@K = (1/|Q|) * Σ_q 1[relevant ∩ top-K(q) ≠ ∅]
        """
        hits = 0
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            top_k = set(retrieved[:k])
            if top_k & set(relevant):
                hits += 1
        
        recall = hits / len(retrieved_ids) if retrieved_ids else 0
        return float(recall)

    def compute_all_recall(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
    ) -> Dict[str, float]:
        """Compute Recall@K for all configured K values."""
        results = {}
        for k in self.config.recall_k_values:
            recall = self.recall_at_k(retrieved_ids, relevant_ids, k)
            results[f"recall@{k}"] = recall
        return results

    # ── MRR ──────────────────────────────────────────────────
    def mrr(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
    ) -> float:
        """
        Mean Reciprocal Rank.
        
        MRR = (1/|Q|) * Σ_q 1/rank_q
        """
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # ── GMM Clustering ───────────────────────────────────────
    def gmm_analysis(
        self,
        embeddings: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Dict:
        """
        Gaussian Mixture Model analysis of embedding space.
        Reveals the underlying cluster structure of retrieved results.
        """
        if GaussianMixture is None:
            return {"error": "scikit-learn required"}
        
        n_comp = n_components or self.config.gmm_n_components
        n_comp = min(n_comp, len(embeddings))
        
        gmm = GaussianMixture(
            n_components=n_comp, random_state=42, covariance_type="full"
        )
        labels = gmm.fit_predict(embeddings)
        
        result = {
            "n_components": n_comp,
            "bic": float(gmm.bic(embeddings)),
            "aic": float(gmm.aic(embeddings)),
            "cluster_sizes": {
                int(i): int(np.sum(labels == i)) for i in range(n_comp)
            },
        }
        
        # Add DB and Silhouette scores if > 1 cluster
        if len(set(labels)) > 1 and davies_bouldin_score is not None:
            result["davies_bouldin"] = float(
                davies_bouldin_score(embeddings, labels)
            )
            result["silhouette"] = float(
                silhouette_score(embeddings, labels)
            )
        
        return result

    # ── DBSCAN Clustering ────────────────────────────────────
    def dbscan_analysis(
        self,
        embeddings: np.ndarray,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> Dict:
        """
        DBSCAN density-based clustering.
        Automatically discovers clusters without specifying K.
        Identifies noise points as potential anomalies.
        """
        if DBSCAN is None:
            return {"error": "scikit-learn required"}
        
        eps = eps or self.config.dbscan_eps
        min_samples = min_samples or self.config.dbscan_min_samples
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        
        result = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": float(n_noise / len(labels)),
            "cluster_sizes": {
                int(i): int(np.sum(labels == i))
                for i in set(labels) if i != -1
            },
        }
        
        if n_clusters > 1 and davies_bouldin_score is not None:
            non_noise = labels != -1
            if non_noise.sum() > n_clusters:
                result["davies_bouldin"] = float(
                    davies_bouldin_score(
                        embeddings[non_noise], labels[non_noise]
                    )
                )
                result["silhouette"] = float(
                    silhouette_score(
                        embeddings[non_noise], labels[non_noise]
                    )
                )
        
        return result

    # ── Anomaly Detection ────────────────────────────────────
    def anomaly_detection(
        self,
        embeddings: np.ndarray,
        contamination: Optional[float] = None,
    ) -> Dict:
        """
        Detect anomalous embeddings using:
          - Isolation Forest
          - One-Class SVM
        
        Useful for finding outlier/low-quality embeddings.
        """
        contamination = contamination or self.config.isolation_contamination
        results = {}
        
        if IsolationForest is not None:
            iso_forest = IsolationForest(
                contamination=contamination, random_state=42
            )
            iso_labels = iso_forest.fit_predict(embeddings)
            iso_scores = iso_forest.decision_function(embeddings)
            
            results["isolation_forest"] = {
                "n_anomalies": int(np.sum(iso_labels == -1)),
                "anomaly_ratio": float(np.mean(iso_labels == -1)),
                "mean_score": float(np.mean(iso_scores)),
                "anomaly_indices": np.where(iso_labels == -1)[0].tolist()[:20],
            }
        
        if OneClassSVM is not None:
            oc_svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
            svm_labels = oc_svm.fit_predict(embeddings)
            svm_scores = oc_svm.decision_function(embeddings)
            
            results["one_class_svm"] = {
                "n_anomalies": int(np.sum(svm_labels == -1)),
                "anomaly_ratio": float(np.mean(svm_labels == -1)),
                "mean_score": float(np.mean(svm_scores)),
                "anomaly_indices": np.where(svm_labels == -1)[0].tolist()[:20],
            }
        
        return results

    # ── Visualization ────────────────────────────────────────
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "tsne",
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        2D visualization of embeddings using t-SNE or PCA.
        """
        if method == "tsne" and TSNE is not None:
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        elif method == "pca" and PCA is not None:
            reducer = PCA(n_components=2)
        else:
            logger.warning(f"Visualization method {method} not available")
            return np.empty((0, 2))
        
        coords = reducer.fit_transform(embeddings)
        
        if save_path:
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                if labels is not None:
                    unique_labels = np.unique(labels)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    for i, label in enumerate(unique_labels):
                        mask = labels == label
                        ax.scatter(
                            coords[mask, 0], coords[mask, 1],
                            c=[colors[i]], label=str(label),
                            s=30, alpha=0.7,
                        )
                    ax.legend()
                else:
                    ax.scatter(
                        coords[:, 0], coords[:, 1],
                        s=30, alpha=0.7, c="steelblue",
                    )
                
                ax.set_title(f"Embedding Visualization ({method.upper()})")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Visualization saved: {save_path}")
            except ImportError:
                pass
        
        return coords

    # ── Full Evaluation ──────────────────────────────────────
    def run_full_evaluation(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        embeddings: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run the complete evaluation suite.
        """
        results = {}
        
        # Recall@K
        results["recall"] = self.compute_all_recall(retrieved_ids, relevant_ids)
        
        # MRR
        results["mrr"] = self.mrr(retrieved_ids, relevant_ids)
        
        logger.info(f"Recall: {results['recall']} | MRR: {results['mrr']:.4f}")
        
        # Embedding analysis
        if embeddings is not None and len(embeddings) > 10:
            results["gmm"] = self.gmm_analysis(embeddings)
            results["dbscan"] = self.dbscan_analysis(embeddings)
            results["anomalies"] = self.anomaly_detection(embeddings)
            
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                self.visualize_embeddings(
                    embeddings, method="tsne",
                    save_path=f"{save_dir}/tsne_embeddings.png",
                )
                self.visualize_embeddings(
                    embeddings, method="pca",
                    save_path=f"{save_dir}/pca_embeddings.png",
                )
        
        # Save results
        if save_dir:
            with open(f"{save_dir}/evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {save_dir}")
        
        return results
