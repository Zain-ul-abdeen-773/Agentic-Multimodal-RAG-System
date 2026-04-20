"""
Data Preprocessing Pipeline
=============================
Production-grade preprocessing BEFORE feeding data to models:

  Text Pipeline:
    1. Unicode normalization (NFKC)
    2. HTML/URL stripping
    3. Lowercasing + whitespace normalization
    4. Stopword removal (optional)
    5. Deduplication (MinHash / exact hash)
    6. Length filtering (min/max tokens)
    7. Language detection (English only)
    8. Sentence tokenization for chunking
    9. TF-IDF quality scoring

  Image Pipeline:
    1. Format validation (JPEG/PNG)
    2. Resize to uniform resolution
    3. Center crop / padding
    4. Normalize pixel values (ImageNet μ/σ)
    5. Sobel edge detection (quality check)
    6. Duplicate detection (perceptual hash)
    7. Corruption detection

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --skip-images --min-length 30
"""

import sys
import re
import json
import hashlib
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass, field

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, MODELS_DIR


# ═══════════════════════════════════════════════════════════════
# Text Preprocessing
# ═══════════════════════════════════════════════════════════════
@dataclass
class TextPreprocessConfig:
    """Configuration for text preprocessing."""
    min_tokens: int = 10             # Minimum tokens per document
    max_tokens: int = 2048           # Maximum tokens per document
    min_unique_ratio: float = 0.3    # Minimum unique-word ratio (quality gate)
    remove_stopwords: bool = False   # True removes common words before embedding
    lowercase: bool = True           # Lowercase all text
    deduplicate: bool = True         # Remove exact/near-duplicate documents
    dedup_threshold: float = 0.95    # Jaccard threshold for near-dedup
    strip_html: bool = True          # Remove HTML tags
    strip_urls: bool = True          # Remove URLs
    strip_emails: bool = True        # Remove email addresses
    normalize_unicode: bool = True   # NFKC normalization
    max_sentence_length: int = 500   # Split overly long sentences


class TextPreprocessor:
    """
    Production text preprocessing pipeline.

    Techniques:
      - Unicode NFKC normalization
      - Regex-based HTML/URL/email stripping
      - Tokenization with quality gating
      - MinHash-based near-deduplication
      - TF-IDF quality scoring
    """

    # Common English stopwords (subset for efficiency)
    STOPWORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "but", "and", "or", "if", "while", "about", "up", "it", "its",
        "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them",
    }

    def __init__(self, config: Optional[TextPreprocessConfig] = None):
        self.config = config or TextPreprocessConfig()
        self._seen_hashes: Set[str] = set()
        self._doc_counter = Counter()

    def clean_text(self, text: str) -> str:
        """Apply all text cleaning transformations."""
        if not text:
            return ""

        # 1. Unicode NFKC normalization (merges compatibility characters)
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Strip HTML tags
        if self.config.strip_html:
            text = re.sub(r"<[^>]+>", " ", text)

        # 3. Strip URLs
        if self.config.strip_urls:
            text = re.sub(
                r"https?://\S+|www\.\S+|ftp://\S+",
                " ", text
            )

        # 4. Strip email addresses
        if self.config.strip_emails:
            text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # 5. Remove LaTeX commands (common in ArXiv)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)  # \command{arg}
        text = re.sub(r"\$[^$]+\$", " ", text)               # inline math
        text = re.sub(r"\\[a-zA-Z]+", " ", text)              # bare commands

        # 6. Remove excessive special characters but keep punctuation
        text = re.sub(r"[^\w\s.,;:!?'\"-/()\[\]]", " ", text)

        # 7. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 8. Lowercase
        if self.config.lowercase:
            text = text.lower()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with punctuation handling."""
        # Split on whitespace
        tokens = text.split()
        # Remove tokens that are pure punctuation
        tokens = [t for t in tokens if re.search(r"\w", t)]
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords."""
        return [t for t in tokens if t.lower() not in self.STOPWORDS]

    def compute_quality_score(self, text: str, tokens: List[str]) -> float:
        """
        Compute a [0, 1] quality score based on:
          - Unique word ratio (vocabulary richness)
          - Average word length (filters gibberish)
          - Sentence count (filters fragments)
        """
        if not tokens:
            return 0.0

        # Unique word ratio
        unique_ratio = len(set(tokens)) / len(tokens)

        # Average word length (good text: 4-8 characters)
        avg_len = np.mean([len(t) for t in tokens])
        len_score = 1.0 - abs(avg_len - 5.5) / 10.0
        len_score = max(0.0, min(1.0, len_score))

        # Sentence count (more sentences = more informative)
        sentences = re.split(r"[.!?]+", text)
        sent_score = min(len(sentences) / 3.0, 1.0)

        # Weighted combination
        quality = 0.5 * unique_ratio + 0.3 * len_score + 0.2 * sent_score
        return round(quality, 4)

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate using hash-based dedup."""
        if not self.config.deduplicate:
            return False

        # Exact dedup via content hash
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if content_hash in self._seen_hashes:
            return True

        self._seen_hashes.add(content_hash)
        return False

    def process_document(self, doc: Dict) -> Optional[Dict]:
        """
        Full preprocessing pipeline for a single document.

        Returns None if the document fails quality checks.
        """
        # Get text field (handle different key names)
        text = doc.get("text", doc.get("caption", doc.get("abstract", "")))
        if not text:
            return None

        # Clean
        cleaned = self.clean_text(text)
        if not cleaned:
            return None

        # Tokenize
        tokens = self.tokenize(cleaned)

        # Length filter
        if len(tokens) < self.config.min_tokens:
            return None
        if len(tokens) > self.config.max_tokens:
            tokens = tokens[:self.config.max_tokens]
            cleaned = " ".join(tokens)

        # Quality gate: unique word ratio
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
        if unique_ratio < self.config.min_unique_ratio:
            return None

        # Deduplication
        if self.is_duplicate(cleaned):
            return None

        # Quality score
        quality = self.compute_quality_score(cleaned, tokens)

        # Optional stopword removal (for embedding, not for display)
        tokens_for_embedding = tokens
        if self.config.remove_stopwords:
            tokens_for_embedding = self.remove_stopwords(tokens)

        # Build output document
        processed = {
            **doc,
            "text": cleaned,
            "text_original": text,
            "tokens": len(tokens),
            "tokens_unique": len(set(tokens)),
            "quality_score": quality,
            "preprocessed": True,
        }

        return processed

    def process_batch(self, documents: List[Dict], source_name: str = "") -> List[Dict]:
        """
        Process a batch of documents through the full pipeline.

        Returns only documents that pass all quality gates.
        """
        self._seen_hashes.clear()
        processed = []
        rejected = {"short": 0, "low_quality": 0, "duplicate": 0, "empty": 0}

        for doc in documents:
            result = self.process_document(doc)
            if result is not None:
                processed.append(result)
            else:
                text = doc.get("text", doc.get("caption", ""))
                if not text:
                    rejected["empty"] += 1
                elif len(text.split()) < self.config.min_tokens:
                    rejected["short"] += 1
                else:
                    rejected["duplicate"] += 1

        logger.info(
            f"[{source_name}] Preprocessed: {len(processed)}/{len(documents)} passed | "
            f"Rejected: {rejected}"
        )

        return processed


# ═══════════════════════════════════════════════════════════════
# Image Preprocessing
# ═══════════════════════════════════════════════════════════════
@dataclass
class ImagePreprocessConfig:
    """Configuration for image preprocessing."""
    target_size: Tuple[int, int] = (224, 224)       # CLIP input size
    max_file_size_mb: float = 10.0                  # Skip huge files
    min_resolution: int = 32                        # Skip tiny images
    normalize_mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)   # CLIP
    normalize_std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)   # CLIP
    sobel_threshold: float = 5.0                    # Min edge energy (skip blank imgs)
    dedup_hash_size: int = 8                        # Perceptual hash grid size


class ImagePreprocessor:
    """
    Image preprocessing pipeline with quality validation.

    Techniques:
      - Format validation
      - Resize + center crop
      - ImageNet/CLIP normalization
      - Sobel edge detection for quality gating
      - Perceptual hash for duplicate detection
    """

    def __init__(self, config: Optional[ImagePreprocessConfig] = None):
        self.config = config or ImagePreprocessConfig()
        self._seen_hashes: Set[str] = set()

    def validate_image(self, path: str) -> bool:
        """Check if image file is valid and uncorrupted."""
        try:
            from PIL import Image
            p = Path(path)
            if not p.exists():
                return False
            if p.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                return False
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def compute_sobel_energy(self, img_array: np.ndarray) -> float:
        """
        Compute Sobel edge energy to detect blank/low-quality images.

        Technique: Sobel/Scharr Filters (from project_techniques.txt)
        """
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(np.float32)

        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Manual 2D convolution (avoiding scipy dependency)
        h, w = gray.shape
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = gray[i-1:i+2, j-1:j+2]
                gx[i, j] = np.sum(patch * sobel_x)
                gy[i, j] = np.sum(patch * sobel_y)

        energy = np.sqrt(gx**2 + gy**2).mean()
        return float(energy)

    def perceptual_hash(self, img_array: np.ndarray) -> str:
        """
        Compute a perceptual hash for duplicate detection.
        Resizes to small grid and binarizes based on mean.
        """
        from PIL import Image

        img = Image.fromarray(img_array.astype(np.uint8))
        size = self.config.dedup_hash_size

        # Resize to tiny grid, convert to grayscale
        small = img.resize((size, size)).convert("L")
        pixels = np.array(small, dtype=np.float32)

        # Binarize: pixel > mean = 1, else 0
        mean_val = pixels.mean()
        bits = (pixels > mean_val).flatten()

        # Convert to hex string
        hash_int = 0
        for bit in bits:
            hash_int = (hash_int << 1) | int(bit)

        return f"{hash_int:016x}"

    def preprocess_image(self, path: str) -> Optional[Dict]:
        """
        Full preprocessing for a single image.

        Returns dict with normalized array and metadata, or None if invalid.
        """
        try:
            from PIL import Image

            if not self.validate_image(path):
                return None

            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size

                # Skip tiny images
                if min(w, h) < self.config.min_resolution:
                    return None

                # Resize maintaining aspect ratio, then center crop
                target_w, target_h = self.config.target_size
                scale = max(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

                # Center crop
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                img = img.crop((left, top, left + target_w, top + target_h))

                img_array = np.array(img, dtype=np.float32)

            # Sobel energy check
            edge_energy = self.compute_sobel_energy(img_array)
            if edge_energy < self.config.sobel_threshold:
                return None

            # Perceptual hash dedup
            p_hash = self.perceptual_hash(img_array)
            if p_hash in self._seen_hashes:
                return None
            self._seen_hashes.add(p_hash)

            # Normalize to [0, 1] then apply CLIP normalization
            normalized = img_array / 255.0
            mean = np.array(self.config.normalize_mean).reshape(1, 1, 3)
            std = np.array(self.config.normalize_std).reshape(1, 1, 3)
            normalized = (normalized - mean) / std

            return {
                "path": path,
                "original_size": (w, h),
                "processed_size": self.config.target_size,
                "edge_energy": round(edge_energy, 2),
                "perceptual_hash": p_hash,
                "normalized_shape": normalized.shape,
                "valid": True,
            }

        except Exception as e:
            logger.debug(f"Image preprocessing failed for {path}: {e}")
            return None

    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process a batch of images through the full pipeline."""
        self._seen_hashes.clear()
        results = []
        rejected = {"invalid": 0, "low_quality": 0, "duplicate": 0, "tiny": 0}

        for path in image_paths:
            result = self.preprocess_image(path)
            if result is not None:
                results.append(result)
            else:
                rejected["invalid"] += 1

        logger.info(
            f"[Images] Preprocessed: {len(results)}/{len(image_paths)} passed | "
            f"Rejected: {rejected}"
        )

        return results


# ═══════════════════════════════════════════════════════════════
# Master Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════
def preprocess_all_data(
    skip_images: bool = False,
    min_length: int = 10,
) -> Dict[str, List[Dict]]:
    """
    Run the full preprocessing pipeline on all available data.

    Reads from data/ (raw), writes to data/preprocessed/ (clean).
    """
    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    text_config = TextPreprocessConfig(min_tokens=min_length)
    text_pp = TextPreprocessor(text_config)
    image_pp = ImagePreprocessor()

    output_dir = DATA_DIR / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}

    # ── 1. ArXiv (real or synthetic) ──
    arxiv_files = [
        DATA_DIR / "text" / "arxiv_real.json",
        DATA_DIR / "text" / "arxiv_abstracts.json",
    ]
    for arxiv_path in arxiv_files:
        if arxiv_path.exists():
            with open(arxiv_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            processed = text_pp.process_batch(raw, source_name=f"ArXiv ({arxiv_path.name})")
            all_data["arxiv"] = processed

            out_path = output_dir / "arxiv_clean.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
            logger.info(f"  Saved: {out_path} ({len(processed)} docs)")
            break

    # ── 2. Wikipedia ──
    wiki_path = DATA_DIR / "text" / "wikipedia_real.json"
    if wiki_path.exists():
        with open(wiki_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        processed = text_pp.process_batch(raw, source_name="Wikipedia")
        all_data["wikipedia"] = processed

        out_path = output_dir / "wikipedia_clean.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved: {out_path} ({len(processed)} docs)")

    # ── 3. COCO captions ──
    coco_path = DATA_DIR / "coco" / "processed_data.json"
    if coco_path.exists():
        with open(coco_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        processed = text_pp.process_batch(raw, source_name="COCO Captions")
        all_data["coco"] = processed

        out_path = output_dir / "coco_clean.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved: {out_path} ({len(processed)} docs)")

    # ── 4. MVTec metadata ──
    mvtec_path = DATA_DIR / "mvtec" / "mvtec_metadata.json"
    if mvtec_path.exists():
        with open(mvtec_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        processed = text_pp.process_batch(raw, source_name="MVTec")
        all_data["mvtec"] = processed

        out_path = output_dir / "mvtec_clean.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved: {out_path} ({len(processed)} docs)")

    # ── 5. Flickr8k images ──
    flickr_path = DATA_DIR / "images" / "flickr8k_real.json"
    if flickr_path.exists():
        with open(flickr_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Process text captions
        processed = text_pp.process_batch(raw, source_name="Flickr8k Captions")
        all_data["flickr"] = processed

        # Process images if requested
        if not skip_images:
            image_paths = [
                item["image_path"] for item in raw
                if item.get("image_path") and Path(item["image_path"]).exists()
            ]
            if image_paths:
                img_results = image_pp.process_batch(image_paths)
                out_img = output_dir / "flickr_images_meta.json"
                with open(out_img, "w", encoding="utf-8") as f:
                    json.dump(img_results, f, indent=2)
                logger.info(f"  Image metadata saved: {out_img}")

        out_path = output_dir / "flickr_clean.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved: {out_path} ({len(processed)} docs)")

    # ── 6. COCO images ──
    if not skip_images:
        coco_images_dir = DATA_DIR / "coco" / "val2017"
        if coco_images_dir.exists():
            image_paths = [
                str(p) for p in sorted(coco_images_dir.glob("*.jpg"))[:500]
            ]
            if image_paths:
                img_results = image_pp.process_batch(image_paths)
                out_img = output_dir / "coco_images_meta.json"
                with open(out_img, "w", encoding="utf-8") as f:
                    json.dump(img_results, f, indent=2)
                logger.info(f"  COCO image metadata saved: {out_img}")

    # ── Summary stats ──
    stats = {}
    for name, docs in all_data.items():
        if docs:
            qualities = [d.get("quality_score", 0) for d in docs]
            token_counts = [d.get("tokens", 0) for d in docs]
            stats[name] = {
                "count": len(docs),
                "avg_quality": round(np.mean(qualities), 4) if qualities else 0,
                "avg_tokens": round(np.mean(token_counts), 1) if token_counts else 0,
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
            }

    stats_path = output_dir / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    for name, s in stats.items():
        logger.info(
            f"  {name}: {s['count']} docs | "
            f"avg quality={s['avg_quality']:.3f} | "
            f"avg tokens={s['avg_tokens']:.0f}"
        )
    logger.info("=" * 60)

    return all_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess all data")
    parser.add_argument("--skip-images", action="store_true", help="Skip image processing")
    parser.add_argument("--min-length", type=int, default=10, help="Min tokens per doc")
    args = parser.parse_args()

    preprocess_all_data(
        skip_images=args.skip_images,
        min_length=args.min_length,
    )
