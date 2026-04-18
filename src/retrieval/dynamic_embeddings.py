"""
Dynamic Embeddings — Incremental Index Updates
===============================================
Techniques implemented:
  - Incremental embedding updates without full re-indexing
  - Embedding versioning and cache invalidation
  - ONNX export for fast embedding inference
  - Batch processing with memory-efficient streaming
"""

import hashlib
import json
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx, ort = None, None
    logger.warning("onnx/onnxruntime not installed for ONNX export")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE


class EmbeddingVersion:
    """Track embedding versions for cache invalidation."""

    def __init__(self, version_file: str = "embedding_versions.json"):
        self.version_file = version_file
        self.versions: Dict[str, str] = {}  # doc_id → content_hash
        self._load()

    def _hash_content(self, content: str) -> str:
        """SHA256 hash of content for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def needs_update(self, doc_id: str, content: str) -> bool:
        """Check if document content has changed since last embedding."""
        new_hash = self._hash_content(content)
        return self.versions.get(doc_id) != new_hash

    def mark_updated(self, doc_id: str, content: str):
        """Record that a document has been embedded."""
        self.versions[doc_id] = self._hash_content(content)

    def mark_batch_updated(self, doc_ids: List[str], contents: List[str]):
        """Record batch of updates."""
        for doc_id, content in zip(doc_ids, contents):
            self.mark_updated(doc_id, content)

    def get_stale_docs(
        self, doc_ids: List[str], contents: List[str]
    ) -> List[int]:
        """Return indices of docs that need re-embedding."""
        return [
            i for i, (did, content) in enumerate(zip(doc_ids, contents))
            if self.needs_update(did, content)
        ]

    def _load(self):
        """Load version file."""
        if Path(self.version_file).exists():
            with open(self.version_file, "r") as f:
                self.versions = json.load(f)

    def save(self):
        """Save version file."""
        Path(self.version_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f)


class DynamicEmbeddingManager:
    """
    Manages incremental embedding updates:
      - Only re-embeds documents whose content has changed
      - Supports ONNX-exported models for fast CPU inference
      - Streaming batch processing for memory efficiency
    """

    def __init__(
        self,
        embedding_fn=None,
        version_dir: str = "models/versions",
    ):
        """
        Args:
            embedding_fn: Callable that takes List[str] → np.ndarray
            version_dir: Directory for version tracking files
        """
        self.embedding_fn = embedding_fn
        self.version_tracker = EmbeddingVersion(
            f"{version_dir}/embedding_versions.json"
        )
        self.onnx_session: Optional["ort.InferenceSession"] = None

    def set_onnx_model(self, onnx_path: str):
        """Load ONNX model for fast inference."""
        if ort is None:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        
        self.onnx_session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info(f"ONNX model loaded: {onnx_path}")

    @staticmethod
    def export_to_onnx(
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
    ):
        """
        Export a PyTorch embedding model to ONNX format.
        
        Args:
            model: PyTorch model to export
            dummy_input: Example input tensor for tracing
            output_path: Where to save the .onnx file
        """
        if onnx is None:
            raise ImportError("onnx required: pip install onnx")
        
        model.eval()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=14,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "embedding": {0: "batch_size"},
            },
        )
        logger.info(f"Model exported to ONNX: {output_path}")

    def compute_embeddings(
        self,
        doc_ids: List[str],
        contents: List[str],
        batch_size: int = 64,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute embeddings only for new or changed documents.
        
        Args:
            doc_ids: Document identifiers
            contents: Document texts
            batch_size: Batch size for embedding
            force_recompute: If True, recompute all embeddings
        Returns:
            (embeddings, updated_indices) — embeddings for changed docs
        """
        if force_recompute:
            stale_indices = list(range(len(doc_ids)))
        else:
            stale_indices = self.version_tracker.get_stale_docs(
                doc_ids, contents
            )
        
        if not stale_indices:
            logger.info("All embeddings up to date. No recomputation needed.")
            return np.empty((0, 0)), []
        
        logger.info(
            f"Recomputing {len(stale_indices)}/{len(doc_ids)} embeddings"
        )
        
        stale_contents = [contents[i] for i in stale_indices]
        
        # Compute in batches
        all_embeddings = []
        for i in range(0, len(stale_contents), batch_size):
            batch = stale_contents[i : i + batch_size]
            
            if self.onnx_session is not None:
                # ONNX inference path
                emb = self._onnx_embed(batch)
            elif self.embedding_fn is not None:
                emb = self.embedding_fn(batch)
            else:
                raise RuntimeError("No embedding function or ONNX model set")
            
            all_embeddings.append(emb)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Update version tracker
        stale_ids = [doc_ids[i] for i in stale_indices]
        self.version_tracker.mark_batch_updated(stale_ids, stale_contents)
        self.version_tracker.save()
        
        return embeddings, stale_indices

    def _onnx_embed(self, texts: List[str]) -> np.ndarray:
        """Run ONNX inference for text embedding."""
        if self.onnx_session is None:
            raise RuntimeError("ONNX session not loaded")
        
        # Basic tokenization (simplified — real impl would use proper tokenizer)
        # This is a placeholder — in practice, export tokenizer separately
        input_name = self.onnx_session.get_inputs()[0].name
        
        # For now, fallback to embedding_fn if ONNX input format is complex
        if self.embedding_fn is not None:
            return self.embedding_fn(texts)
        
        raise NotImplementedError(
            "ONNX tokenization not implemented. Use embedding_fn instead."
        )
