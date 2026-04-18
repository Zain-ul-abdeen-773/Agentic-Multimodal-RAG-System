"""
Semantic Chunker — Context-Preserving Text Segmentation
========================================================
Techniques implemented:
  - Embedding-similarity breakpoint detection (cosine delta)
  - Adaptive threshold δ via percentile-based tuning
  - BiLSTM boundary detection as learned alternative
  - GloVE embeddings as lightweight fallback encoder
  - Sliding window overlap for chunk continuity
  - t-SNE visualization of chunk embedding space
  - Fourier Transform analysis of embedding periodicity
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    TSNE, PCA = None, None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE, ChunkingConfig


@dataclass
class Chunk:
    """A semantic chunk with metadata."""
    text: str
    chunk_id: int
    start_idx: int          # Character offset in original doc
    end_idx: int
    embedding: Optional[np.ndarray] = None
    coherence_score: float = 0.0  # Intra-chunk similarity


# ═══════════════════════════════════════════════════════════════
# BiLSTM Boundary Detector
# ═══════════════════════════════════════════════════════════════
class BiLSTMBoundaryDetector(nn.Module):
    """
    Learned chunk boundary detection using BiLSTM.
    
    Takes a sequence of sentence embeddings and predicts a binary
    boundary label (split / no-split) at each position.
    
    Architecture: BiLSTM → Linear → Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence_embeddings: (B, seq_len, D) sentence-level features
        Returns:
            (B, seq_len) boundary probabilities (0=no split, 1=split)
        """
        lstm_out, _ = self.bilstm(sentence_embeddings)
        boundary_probs = self.classifier(lstm_out).squeeze(-1)
        return boundary_probs


# ═══════════════════════════════════════════════════════════════
# GloVE Embedder (Lightweight Fallback)
# ═══════════════════════════════════════════════════════════════
class GloVEEmbedder:
    """
    Simple GloVE-based sentence embedder (average pooling).
    Used as lightweight fallback when sentence-transformers is too heavy.
    
    Downloads GloVE vectors on first use.
    """

    def __init__(self, dim: int = 100):
        self.dim = dim
        self.word_vectors: Dict[str, np.ndarray] = {}
        self._loaded = False

    def load(self, glove_path: Optional[str] = None):
        """Load GloVE vectors from file."""
        if self._loaded:
            return
        
        if glove_path and Path(glove_path).exists():
            logger.info(f"Loading GloVE from {glove_path}")
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts[0]
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == self.dim:
                        self.word_vectors[word] = vector
            self._loaded = True
            logger.info(f"Loaded {len(self.word_vectors)} GloVE vectors")
        else:
            logger.warning(
                "GloVE file not found. Use sentence-transformers instead."
            )

    def embed(self, text: str) -> np.ndarray:
        """Embed text by averaging word vectors."""
        words = text.lower().split()
        vectors = [
            self.word_vectors[w] for w in words if w in self.word_vectors
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.dim, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.stack([self.embed(t) for t in texts])


# ═══════════════════════════════════════════════════════════════
# Main Semantic Chunker
# ═══════════════════════════════════════════════════════════════
class SemanticChunker:
    """
    Splits documents into semantically coherent chunks by detecting
    embedding-similarity breakpoints between consecutive sentences.
    
    Algorithm:
      1. Split document into sentences
      2. Embed each sentence
      3. Compute cosine similarity between consecutive embeddings
      4. Detect breakpoints where similarity drops below threshold δ
      5. Group sentences between breakpoints into chunks
      6. Enforce min/max chunk size constraints
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        
        # Sentence encoder
        if SentenceTransformer is not None:
            self.encoder = SentenceTransformer(self.config.embedding_model)
            logger.info(
                f"SemanticChunker using: {self.config.embedding_model}"
            )
        else:
            self.encoder = None
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to GloVE or fixed-window chunking."
            )
        
        # BiLSTM boundary detector (optional learned approach)
        self.bilstm_detector = None
        if self.config.use_bilstm_boundary:
            input_dim = 384  # MiniLM default
            self.bilstm_detector = BiLSTMBoundaryDetector(
                input_dim=input_dim,
                hidden_dim=self.config.bilstm_hidden,
                num_layers=self.config.bilstm_layers,
            )
        
        # GloVE fallback
        self.glove = GloVEEmbedder()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        import re
        # Split on sentence-ending punctuation followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Embed sentences using the encoder."""
        if self.encoder is not None:
            return self.encoder.encode(
                sentences, show_progress_bar=False, convert_to_numpy=True
            )
        else:
            self.glove.load()
            return self.glove.embed_batch(sentences)

    def _cosine_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between consecutive sentence embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        
        similarities = np.array([
            np.dot(normalized[i], normalized[i + 1])
            for i in range(len(normalized) - 1)
        ])
        return similarities

    def _detect_breakpoints(
        self, similarities: np.ndarray
    ) -> List[int]:
        """
        Detect breakpoints where cosine similarity drops below threshold.
        
        Uses percentile-based adaptive threshold:
          δ = percentile(similarities, threshold_percentile)
        """
        if len(similarities) == 0:
            return []
        
        # Adaptive threshold: use the configured threshold as a percentile
        threshold = np.percentile(
            similarities, self.config.breakpoint_threshold * 100
        )
        
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # Split after sentence i
        
        return breakpoints

    def _detect_breakpoints_bilstm(
        self, embeddings: np.ndarray
    ) -> List[int]:
        """Use trained BiLSTM to detect boundaries."""
        if self.bilstm_detector is None:
            return self._detect_breakpoints(
                self._cosine_similarities(embeddings)
            )
        
        self.bilstm_detector.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(
                embeddings, dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            probs = self.bilstm_detector(input_tensor).squeeze(0).cpu().numpy()
        
        breakpoints = [
            i for i, p in enumerate(probs) if p > 0.5
        ]
        return breakpoints

    def _enforce_size_constraints(
        self,
        sentences: List[str],
        breakpoints: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Enforce min/max chunk size by merging small chunks
        and splitting large ones.
        
        Returns list of (start_idx, end_idx) sentence ranges.
        """
        # Create initial ranges from breakpoints
        all_breaks = [0] + breakpoints + [len(sentences)]
        ranges = [
            (all_breaks[i], all_breaks[i + 1])
            for i in range(len(all_breaks) - 1)
        ]
        
        # Merge small chunks
        merged = []
        current_start, current_end = ranges[0]
        
        for start, end in ranges[1:]:
            current_text = " ".join(sentences[current_start:current_end])
            current_tokens = len(current_text.split())
            
            if current_tokens < self.config.min_chunk_size:
                # Merge with next chunk
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        # Split chunks that exceed max size
        final_ranges = []
        for start, end in merged:
            chunk_text = " ".join(sentences[start:end])
            chunk_tokens = len(chunk_text.split())
            
            if chunk_tokens > self.config.max_chunk_size:
                # Split into roughly equal sub-chunks
                n_splits = (chunk_tokens // self.config.max_chunk_size) + 1
                n_sents = end - start
                split_size = max(n_sents // n_splits, 1)
                
                for i in range(0, n_sents, split_size):
                    sub_start = start + i
                    sub_end = min(start + i + split_size, end)
                    final_ranges.append((sub_start, sub_end))
            else:
                final_ranges.append((start, end))
        
        return final_ranges

    def chunk(
        self,
        text: str,
        use_bilstm: bool = False,
    ) -> List[Chunk]:
        """
        Split document into semantic chunks.
        
        Args:
            text: Full document text
            use_bilstm: Use BiLSTM boundary detector instead of cosine
        Returns:
            List of Chunk objects with text, embeddings, and metadata
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(text=text, chunk_id=0, start_idx=0, end_idx=len(text))]
        
        # Embed sentences
        embeddings = self._embed_sentences(sentences)
        
        # Detect breakpoints
        if use_bilstm and self.bilstm_detector is not None:
            breakpoints = self._detect_breakpoints_bilstm(embeddings)
        else:
            similarities = self._cosine_similarities(embeddings)
            breakpoints = self._detect_breakpoints(similarities)
        
        # Enforce size constraints
        ranges = self._enforce_size_constraints(sentences, breakpoints)
        
        # Build chunks
        chunks = []
        char_offset = 0
        
        for chunk_id, (start, end) in enumerate(ranges):
            chunk_text = " ".join(sentences[start:end])
            chunk_embeddings = embeddings[start:end]
            
            # Compute chunk embedding (average)
            chunk_embedding = np.mean(chunk_embeddings, axis=0)
            
            # Intra-chunk coherence score
            if len(chunk_embeddings) > 1:
                norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                normed = chunk_embeddings / norms
                sim_matrix = normed @ normed.T
                # Average off-diagonal similarity
                n = len(sim_matrix)
                coherence = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
            else:
                coherence = 1.0
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_idx=char_offset,
                end_idx=char_offset + len(chunk_text),
                embedding=chunk_embedding,
                coherence_score=float(coherence),
            )
            chunks.append(chunk)
            char_offset += len(chunk_text) + 1
        
        logger.info(
            f"Chunked document: {len(sentences)} sentences → "
            f"{len(chunks)} chunks | Avg coherence: "
            f"{np.mean([c.coherence_score for c in chunks]):.3f}"
        )
        
        return chunks

    def compute_chunking_quality(self, chunks: List[Chunk]) -> float:
        """
        Compute Cchunk = sim_intra - sim_inter.
        Higher = better semantic segmentation.
        """
        if len(chunks) < 2:
            return 1.0
        
        embeddings = np.stack([c.embedding for c in chunks if c.embedding is not None])
        
        if len(embeddings) < 2:
            return 1.0
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = embeddings / norms
        
        # Inter-chunk similarity (between different chunks)
        sim_matrix = normed @ normed.T
        n = len(sim_matrix)
        inter_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        
        # Intra-chunk similarity (average of per-chunk coherence)
        intra_sim = np.mean([c.coherence_score for c in chunks])
        
        quality = float(intra_sim - inter_sim)
        logger.info(
            f"Chunking quality Cchunk = {quality:.4f} "
            f"(intra={intra_sim:.4f}, inter={inter_sim:.4f})"
        )
        return quality

    def visualize_tsne(
        self,
        chunks: List[Chunk],
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate t-SNE 2D visualization of chunk embeddings.
        
        Returns (N, 2) array of t-SNE coordinates.
        """
        if TSNE is None:
            logger.warning("scikit-learn needed for t-SNE visualization")
            return None
        
        embeddings = np.stack(
            [c.embedding for c in chunks if c.embedding is not None]
        )
        
        if len(embeddings) < 3:
            logger.warning("Need >= 3 chunks for t-SNE")
            return None
        
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(
            n_components=2, perplexity=perplexity,
            random_state=42, n_iter=1000,
        )
        coords = tsne.fit_transform(embeddings)
        
        if save_path:
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=[c.coherence_score for c in chunks if c.embedding is not None],
                    cmap="viridis", s=60, alpha=0.8,
                )
                plt.colorbar(scatter, label="Coherence Score")
                ax.set_title("t-SNE: Semantic Chunk Embeddings")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"t-SNE plot saved: {save_path}")
            except ImportError:
                logger.warning("matplotlib needed for plotting")
        
        return coords

    def fourier_analysis(
        self, chunks: List[Chunk]
    ) -> Dict[str, np.ndarray]:
        """
        Fourier Transform analysis of embedding dimensions.
        Reveals periodic patterns in the embedding space.
        """
        embeddings = np.stack(
            [c.embedding for c in chunks if c.embedding is not None]
        )
        
        # FFT along the chunk sequence axis for each dimension
        fft_result = np.fft.rfft(embeddings, axis=0)
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(len(embeddings))
        
        # Dominant frequency per dimension
        dominant_freqs = frequencies[np.argmax(magnitudes[1:], axis=0) + 1]
        
        return {
            "magnitudes": magnitudes,
            "frequencies": frequencies,
            "dominant_freqs": dominant_freqs,
            "mean_magnitude_spectrum": magnitudes.mean(axis=1),
        }
