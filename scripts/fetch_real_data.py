"""
Real Data Fetcher — Live API Sources
=====================================
Fetches REAL data from free public APIs (no API key required):
  1. ArXiv API    — real research paper abstracts (OAI-PMH / REST)
  2. Wikipedia API — real encyclopedia articles
  3. Flickr8k     — real image-caption pairs (via HuggingFace)

Usage:
    python scripts/fetch_real_data.py
    python scripts/fetch_real_data.py --arxiv 500 --wiki 300 --flickr 200
"""

import sys
import json
import time
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════
@dataclass
class TextDocument:
    """Standard document format shared across all sources."""
    doc_id: str
    title: str
    text: str
    source: str
    url: str = ""
    category: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ImageCaptionPair:
    """Standard image-caption pair."""
    image_id: str
    caption: str
    all_captions: list = None
    image_path: str = ""
    image_url: str = ""
    source: str = ""

    def __post_init__(self):
        if self.all_captions is None:
            self.all_captions = [self.caption]


# ═══════════════════════════════════════════════════════════════
# 1. ArXiv API (REST / Atom Feed)
# ═══════════════════════════════════════════════════════════════
def fetch_arxiv(
    n_papers: int = 500,
    categories: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Fetch real research paper abstracts from ArXiv REST API.

    Args:
        n_papers: Number of papers to fetch
        categories: ArXiv categories (e.g. cs.CV, cs.CL, cs.LG)
        save_path: Path to save the JSON output

    Returns:
        List of document dicts
    """
    if save_path is None:
        save_path = DATA_DIR / "text" / "arxiv_real.json"

    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"ArXiv data already cached: {len(data)} papers")
        return data

    if categories is None:
        categories = ["cs.CV", "cs.CL", "cs.LG", "cs.AI", "cs.IR"]

    cat_query = "+OR+".join([f"cat:{c}" for c in categories])
    base_url = "http://export.arxiv.org/api/query"

    all_papers = []
    batch_size = 100   # ArXiv max per request
    retries = 3

    logger.info(f"Fetching {n_papers} ArXiv papers from: {categories}")

    for start in range(0, n_papers, batch_size):
        remaining = min(batch_size, n_papers - start)
        params = urllib.parse.urlencode({
            "search_query": cat_query,
            "start": start,
            "max_results": remaining,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{base_url}?{params}"

        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AgenticRAG/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read().decode("utf-8")

                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom",
                      "arxiv": "http://arxiv.org/schemas/atom"}

                entries = root.findall("atom:entry", ns)
                if not entries:
                    logger.warning(f"No entries at offset {start}")
                    break

                for entry in entries:
                    title_el = entry.find("atom:title", ns)
                    summary_el = entry.find("atom:summary", ns)
                    id_el = entry.find("atom:id", ns)
                    published_el = entry.find("atom:published", ns)

                    title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
                    abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
                    arxiv_id = id_el.text.strip() if id_el is not None else ""
                    published = published_el.text.strip() if published_el is not None else ""

                    # Extract categories
                    cat_els = entry.findall("atom:category", ns)
                    cats = [c.get("term", "") for c in cat_els]

                    # Extract authors
                    author_els = entry.findall("atom:author", ns)
                    authors = []
                    for a in author_els:
                        name_el = a.find("atom:name", ns)
                        if name_el is not None:
                            authors.append(name_el.text.strip())

                    if abstract and len(abstract) > 50:
                        doc_id = arxiv_id.split("/")[-1] if "/" in arxiv_id else arxiv_id
                        all_papers.append({
                            "doc_id": f"arxiv_{doc_id}",
                            "title": title,
                            "text": abstract,
                            "source": "arxiv_api",
                            "url": arxiv_id,
                            "category": cats[0] if cats else "",
                            "metadata": {
                                "authors": authors[:5],
                                "published": published,
                                "categories": cats,
                            },
                        })

                logger.info(f"  ArXiv batch {start}-{start + len(entries)}: "
                            f"{len(entries)} papers fetched")
                break  # Success, exit retry loop

            except Exception as e:
                logger.warning(f"  ArXiv request failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))

        # ArXiv rate limit: 1 request per 3 seconds
        time.sleep(3)

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    logger.info(f"ArXiv: {len(all_papers)} real papers saved to {save_path}")
    return all_papers


# ═══════════════════════════════════════════════════════════════
# 2. Wikipedia API (MediaWiki REST)
# ═══════════════════════════════════════════════════════════════
def fetch_wikipedia(
    n_articles: int = 300,
    topics: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Fetch real Wikipedia article summaries via the MediaWiki API.

    Args:
        n_articles: Number of articles to fetch
        topics: Seed topics for article discovery
        save_path: Output JSON path

    Returns:
        List of document dicts
    """
    if save_path is None:
        save_path = DATA_DIR / "text" / "wikipedia_real.json"

    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Wikipedia data already cached: {len(data)} articles")
        return data

    if topics is None:
        topics = [
            "Artificial intelligence", "Deep learning", "Computer vision",
            "Natural language processing", "Convolutional neural network",
            "Transformer (machine learning)", "BERT (language model)",
            "Generative adversarial network", "Reinforcement learning",
            "Recurrent neural network", "Object detection",
            "Image segmentation", "Neural network", "Backpropagation",
            "Gradient descent", "Knowledge graph", "Information retrieval",
            "Machine learning", "Feature extraction", "Transfer learning",
            "Graph neural network", "Attention (machine learning)",
            "Autoencoder", "Diffusion model", "Retrieval-augmented generation",
            "Self-supervised learning", "Few-shot learning",
            "Support vector machine", "Random forest", "Bayesian network",
            "Principal component analysis", "K-means clustering",
            "Anomaly detection", "Dimensionality reduction",
            "Neural architecture search", "Federated learning",
            "Model compression", "Knowledge distillation",
            "Data augmentation", "Batch normalization",
            "Dropout (neural networks)", "Residual network",
            "Optical character recognition", "Speech recognition",
            "Facial recognition system", "Autonomous vehicle",
            "Robotics", "Edge computing", "Quantum computing",
        ]

    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    search_url = "https://en.wikipedia.org/w/api.php"

    all_articles = []
    seen_ids = set()

    logger.info(f"Fetching up to {n_articles} Wikipedia articles...")

    # Phase 1: Fetch articles for each seed topic
    for topic in topics:
        if len(all_articles) >= n_articles:
            break

        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"{base_url}{encoded}"

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "AgenticRAG/1.0 (education project)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            page_id = str(data.get("pageid", ""))
            if page_id in seen_ids or not page_id:
                continue

            extract = data.get("extract", "")
            if len(extract) < 50:
                continue

            seen_ids.add(page_id)
            all_articles.append({
                "doc_id": f"wiki_{page_id}",
                "title": data.get("title", topic),
                "text": extract,
                "source": "wikipedia_api",
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "category": "science_technology",
                "metadata": {
                    "description": data.get("description", ""),
                    "page_id": page_id,
                    "lang": "en",
                },
            })

        except Exception as e:
            logger.debug(f"  Wiki fetch failed for '{topic}': {e}")

        time.sleep(0.5)  # Be polite to API

    # Phase 2: Get MORE articles via search (related pages)
    if len(all_articles) < n_articles:
        logger.info(f"  Phase 2: Expanding via search ({len(all_articles)}/{n_articles})...")
        search_topics = [
            "neural network applications", "machine learning algorithms",
            "image classification", "text generation", "sentiment analysis",
            "recommendation system", "time series prediction",
            "medical image analysis", "autonomous driving sensors",
            "language model", "embedding space", "vector database",
        ]

        for query in search_topics:
            if len(all_articles) >= n_articles:
                break

            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 20,
                "format": "json",
            })
            url = f"{search_url}?{params}"

            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "AgenticRAG/1.0 (education project)"
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    results = json.loads(resp.read().decode("utf-8"))

                for item in results.get("query", {}).get("search", []):
                    if len(all_articles) >= n_articles:
                        break

                    page_title = item.get("title", "")
                    encoded = urllib.parse.quote(page_title.replace(" ", "_"))
                    sum_url = f"{base_url}{encoded}"

                    try:
                        req2 = urllib.request.Request(sum_url, headers={
                            "User-Agent": "AgenticRAG/1.0",
                            "Accept": "application/json",
                        })
                        with urllib.request.urlopen(req2, timeout=10) as resp2:
                            data = json.loads(resp2.read().decode("utf-8"))

                        page_id = str(data.get("pageid", ""))
                        if page_id in seen_ids or not page_id:
                            continue

                        extract = data.get("extract", "")
                        if len(extract) < 50:
                            continue

                        seen_ids.add(page_id)
                        all_articles.append({
                            "doc_id": f"wiki_{page_id}",
                            "title": data.get("title", page_title),
                            "text": extract,
                            "source": "wikipedia_api",
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "category": "science_technology",
                            "metadata": {
                                "description": data.get("description", ""),
                                "page_id": page_id,
                                "lang": "en",
                            },
                        })
                    except Exception:
                        pass

                    time.sleep(0.3)

            except Exception as e:
                logger.debug(f"  Search failed for '{query}': {e}")

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Wikipedia: {len(all_articles)} real articles saved to {save_path}")
    return all_articles


# ═══════════════════════════════════════════════════════════════
# 3. Flickr8k Image-Caption Pairs (via HuggingFace)
# ═══════════════════════════════════════════════════════════════
def fetch_flickr8k(
    n_pairs: int = 200,
    download_images: bool = True,
    save_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Fetch real image-caption pairs from the Flickr8k dataset (HuggingFace).

    This gives us REAL images with REAL human-written captions.

    Args:
        n_pairs: How many image-caption pairs to fetch
        download_images: Whether to save images to disk
        save_path: Output JSON path

    Returns:
        List of image-caption pair dicts
    """
    if save_path is None:
        save_path = DATA_DIR / "images" / "flickr8k_real.json"

    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Flickr8k data already cached: {len(data)} pairs")
        return data

    logger.info(f"Fetching {n_pairs} Flickr8k image-caption pairs from HuggingFace...")

    try:
        from datasets import load_dataset

        ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)

        images_dir = DATA_DIR / "images" / "flickr8k"
        images_dir.mkdir(parents=True, exist_ok=True)

        all_pairs = []
        seen = set()

        for i, item in enumerate(ds):
            if len(all_pairs) >= n_pairs:
                break

            caption_list = item.get("caption", [])
            if isinstance(caption_list, str):
                caption_list = [caption_list]
            if not caption_list:
                continue

            # Generate a stable ID from the caption
            cap_hash = hashlib.md5(caption_list[0].encode()).hexdigest()[:8]
            image_id = f"flickr_{cap_hash}"

            if image_id in seen:
                continue
            seen.add(image_id)

            image_path = ""
            if download_images and item.get("image"):
                try:
                    img = item["image"]
                    img_file = images_dir / f"{image_id}.jpg"
                    img.save(str(img_file), "JPEG", quality=85)
                    image_path = str(img_file)
                except Exception:
                    pass

            all_pairs.append({
                "image_id": image_id,
                "caption": caption_list[0],
                "all_captions": caption_list[:5],
                "image_path": image_path,
                "source": "flickr8k_real",
            })

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1} items, kept {len(all_pairs)} pairs")

        # Save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Flickr8k: {len(all_pairs)} real image-caption pairs saved")
        return all_pairs

    except ImportError:
        logger.warning("datasets library not installed. Run: pip install datasets")
        return []
    except Exception as e:
        logger.warning(f"Flickr8k fetch failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# Master Fetch Function
# ═══════════════════════════════════════════════════════════════
def fetch_all_real_data(
    n_arxiv: int = 500,
    n_wiki: int = 300,
    n_flickr: int = 200,
) -> Dict[str, List[Dict]]:
    """Fetch all real data from live APIs."""
    logger.info("=" * 60)
    logger.info("FETCHING REAL DATA FROM LIVE APIs")
    logger.info("=" * 60)

    datasets = {}

    # 1. ArXiv
    logger.info("\n[1/3] ArXiv API — real research papers...")
    datasets["arxiv"] = fetch_arxiv(n_papers=n_arxiv)

    # 2. Wikipedia
    logger.info("\n[2/3] Wikipedia API — real encyclopedia articles...")
    datasets["wikipedia"] = fetch_wikipedia(n_articles=n_wiki)

    # 3. Flickr8k
    logger.info("\n[3/3] Flickr8k — real image-caption pairs...")
    datasets["flickr"] = fetch_flickr8k(n_pairs=n_flickr)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("REAL DATA SUMMARY:")
    total = 0
    for name, data in datasets.items():
        logger.info(f"  {name}: {len(data)} entries")
        total += len(data)
    logger.info(f"  TOTAL: {total} real data entries")
    logger.info("=" * 60)

    return datasets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch real data from live APIs")
    parser.add_argument("--arxiv", type=int, default=500, help="Number of ArXiv papers")
    parser.add_argument("--wiki", type=int, default=300, help="Number of Wikipedia articles")
    parser.add_argument("--flickr", type=int, default=200, help="Number of Flickr8k pairs")
    args = parser.parse_args()

    fetch_all_real_data(
        n_arxiv=args.arxiv,
        n_wiki=args.wiki,
        n_flickr=args.flickr,
    )
