"""
Dataset Downloader — MS-COCO, MVTec AD, ArXiv Abstracts
========================================================
Downloads and prepares all three datasets for the RAG pipeline.
"""

import os
import json
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Dict
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT, DATA_DIR


# ═══════════════════════════════════════════════════════════════
# MS-COCO val2017 (images + captions)
# ═══════════════════════════════════════════════════════════════
def download_coco(max_images: int = 1000):
    """
    Download MS-COCO val2017 images and captions.
    Uses a subset for feasibility (default 1000 images).
    """
    coco_dir = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    captions_file = coco_dir / "captions_val2017.json"
    images_dir = coco_dir / "val2017"

    # Download captions annotation
    if not captions_file.exists():
        logger.info("Downloading COCO captions annotation...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = coco_dir / "annotations.zip"

        try:
            urllib.request.urlretrieve(url, str(zip_path))
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                # Extract only captions_val2017.json
                for member in zf.namelist():
                    if "captions_val2017" in member:
                        zf.extract(member, str(coco_dir))
            # Move from nested annotations/ folder
            nested = coco_dir / "annotations" / "captions_val2017.json"
            if nested.exists():
                shutil.move(str(nested), str(captions_file))
                shutil.rmtree(str(coco_dir / "annotations"), ignore_errors=True)
            zip_path.unlink(missing_ok=True)
            logger.info("COCO captions downloaded")
        except Exception as e:
            logger.warning(f"COCO caption download failed: {e}")
            # Create synthetic captions as fallback
            _create_synthetic_coco_captions(captions_file, max_images)
            return _load_coco_data(captions_file, max_images)
    
    # Download images (subset)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if captions_file.exists():
        with open(captions_file, "r") as f:
            coco_data = json.load(f)
        
        images_meta = coco_data.get("images", [])[:max_images]
        annotations = coco_data.get("annotations", [])
        
        # Build image_id → captions mapping
        caption_map = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in caption_map:
                caption_map[img_id] = []
            caption_map[img_id].append(ann["caption"])
        
        downloaded = 0
        data_entries = []
        
        for img_meta in images_meta:
            img_id = img_meta["id"]
            filename = img_meta["file_name"]
            img_path = images_dir / filename
            
            if not img_path.exists():
                url = f"http://images.cocodataset.org/val2017/{filename}"
                try:
                    urllib.request.urlretrieve(url, str(img_path))
                    downloaded += 1
                    if downloaded % 50 == 0:
                        logger.info(f"Downloaded {downloaded}/{len(images_meta)} COCO images")
                except Exception as e:
                    logger.debug(f"Failed to download {filename}: {e}")
                    continue
            
            captions = caption_map.get(img_id, ["No caption available"])
            data_entries.append({
                "image_id": str(img_id),
                "path": str(img_path),
                "caption": captions[0],  # Primary caption
                "all_captions": captions,
                "filename": filename,
                "source": "coco_val2017",
            })
        
        logger.info(f"COCO ready: {len(data_entries)} entries, {downloaded} new downloads")
        
        # Save processed data
        processed_path = coco_dir / "processed_data.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)
        
        return data_entries
    
    return _load_coco_data(captions_file, max_images)


def _create_synthetic_coco_captions(path: Path, n: int = 1000):
    """Create synthetic COCO-like captions for offline development."""
    import random
    random.seed(42)
    
    objects = ["cat", "dog", "car", "person", "bicycle", "bus", "train",
               "bird", "horse", "sheep", "cow", "elephant", "bear",
               "zebra", "giraffe", "backpack", "umbrella", "handbag",
               "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "skateboard", "surfboard", "tennis racket",
               "bottle", "wine glass", "cup", "fork", "knife", "spoon"]
    
    scenes = ["on a street", "in a park", "near a building", "on a beach",
              "in a kitchen", "in a living room", "on a field", "near water",
              "in a forest", "on a sidewalk", "at a station", "in a garden"]
    
    actions = ["standing", "sitting", "walking", "running", "playing",
               "eating", "looking at", "holding", "riding", "next to"]
    
    data = {"images": [], "annotations": []}
    for i in range(n):
        img_id = 100000 + i
        obj1 = random.choice(objects)
        obj2 = random.choice(objects)
        scene = random.choice(scenes)
        action = random.choice(actions)
        
        data["images"].append({
            "id": img_id,
            "file_name": f"synthetic_{img_id:06d}.jpg",
        })
        data["annotations"].append({
            "image_id": img_id,
            "caption": f"A {obj1} {action} a {obj2} {scene}",
        })
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Created {n} synthetic COCO captions at {path}")


def _load_coco_data(captions_file: Path, max_images: int) -> List[Dict]:
    """Load processed COCO data."""
    processed = captions_file.parent / "processed_data.json"
    if processed.exists():
        with open(processed, "r") as f:
            return json.load(f)[:max_images]
    return []


# ═══════════════════════════════════════════════════════════════
# ArXiv Abstracts (text corpus for semantic chunking)
# ═══════════════════════════════════════════════════════════════
def download_arxiv_abstracts(n_abstracts: int = 1000) -> List[Dict]:
    """
    Create a corpus of ArXiv-style abstracts for text retrieval.
    Uses HuggingFace datasets if available, otherwise generates
    domain-relevant synthetic abstracts.
    """
    arxiv_dir = DATA_DIR / "text"
    arxiv_dir.mkdir(parents=True, exist_ok=True)
    output_file = arxiv_dir / "arxiv_abstracts.json"
    
    if output_file.exists():
        with open(output_file, "r") as f:
            data = json.load(f)
        logger.info(f"ArXiv abstracts loaded: {len(data)} entries")
        return data[:n_abstracts]
    
    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        logger.info("Loading ArXiv abstracts from HuggingFace...")
        ds = load_dataset("ccdv/arxiv-classification", split="test", trust_remote_code=True)
        entries = []
        for i, item in enumerate(ds):
            if i >= n_abstracts:
                break
            entries.append({
                "doc_id": f"arxiv_{i:05d}",
                "text": item.get("text", item.get("abstract", "")),
                "title": item.get("title", f"Document {i}"),
                "source": "arxiv",
            })
        
        if entries:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            logger.info(f"ArXiv abstracts saved: {len(entries)} entries")
            return entries
    except Exception as e:
        logger.warning(f"HuggingFace dataset load failed: {e}")
    
    # Fallback: generate domain-relevant abstracts
    logger.info("Generating synthetic ML/DL research abstracts...")
    entries = _generate_synthetic_abstracts(n_abstracts)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    return entries


def _generate_synthetic_abstracts(n: int) -> List[Dict]:
    """Generate realistic ML/DL research abstracts."""
    import random
    random.seed(42)
    
    topics = [
        ("Transformer architectures", "attention mechanisms", "self-attention", "multi-head attention"),
        ("Contrastive learning", "representation learning", "SimCLR", "MoCo"),
        ("Object detection", "YOLO", "Faster R-CNN", "anchor-free detectors"),
        ("Image segmentation", "U-Net", "Mask R-CNN", "panoptic segmentation"),
        ("Natural language processing", "BERT", "GPT", "language modeling"),
        ("Generative models", "GANs", "diffusion models", "VAEs"),
        ("Reinforcement learning", "policy gradient", "Q-learning", "PPO"),
        ("Graph neural networks", "GCN", "GraphSAGE", "message passing"),
        ("Federated learning", "privacy-preserving", "differential privacy", "distributed training"),
        ("Neural architecture search", "AutoML", "NAS", "efficient architectures"),
        ("Knowledge distillation", "model compression", "pruning", "quantization"),
        ("Multimodal learning", "CLIP", "vision-language", "cross-modal retrieval"),
        ("Anomaly detection", "one-class SVM", "isolation forest", "autoencoders"),
        ("Time series forecasting", "LSTM", "temporal convolutions", "attention"),
        ("3D vision", "point clouds", "NeRF", "depth estimation"),
        ("Medical imaging", "pathology", "radiology", "diagnostic AI"),
        ("Retrieval-augmented generation", "RAG", "dense retrieval", "knowledge grounding"),
        ("Few-shot learning", "meta-learning", "MAML", "prototypical networks"),
        ("Self-supervised learning", "pretext tasks", "masked modeling", "DINO"),
        ("Efficient inference", "ONNX", "TensorRT", "edge deployment"),
    ]
    
    methods = [
        "We propose a novel approach that",
        "This paper introduces a framework that",
        "We present a method that",
        "In this work, we develop a system that",
        "Our approach leverages",
    ]
    
    results_template = [
        "Experiments on {} demonstrate state-of-the-art performance, achieving {}% accuracy.",
        "We evaluate on {} and show improvements of {}% over previous baselines.",
        "Our method achieves {}% on {} benchmark, outperforming existing approaches.",
        "Extensive experiments on {} validate our approach with {}% improvement.",
    ]
    
    datasets_names = ["ImageNet", "COCO", "Pascal VOC", "Cityscapes", "SQuAD",
                      "GLUE", "WikiText", "Common Crawl", "OpenImages", "LVIS"]
    
    entries = []
    for i in range(n):
        topic_group = random.choice(topics)
        topic = topic_group[0]
        details = random.sample(topic_group[1:], min(2, len(topic_group)-1))
        method = random.choice(methods)
        dataset = random.choice(datasets_names)
        accuracy = round(random.uniform(85, 99), 1)
        
        abstract = (
            f"{method} combines {details[0]} with "
            f"{'and '.join(details[1:]) if len(details) > 1 else 'advanced techniques'} "
            f"for improved {topic.lower()}. "
            f"{random.choice(results_template).format(dataset, accuracy)} "
            f"Our implementation is publicly available and enables "
            f"reproducible research in {topic.lower()}."
        )
        
        entries.append({
            "doc_id": f"arxiv_{i:05d}",
            "text": abstract,
            "title": f"Advances in {topic}: A {details[0].title()} Approach",
            "source": "synthetic_arxiv",
            "topic": topic,
        })
    
    logger.info(f"Generated {n} synthetic research abstracts")
    return entries


# ═══════════════════════════════════════════════════════════════
# MVTec AD (industrial defect — synthetic metadata)
# ═══════════════════════════════════════════════════════════════
def create_mvtec_metadata(n_entries: int = 500) -> List[Dict]:
    """
    Create synthetic MVTec-AD-style inspection metadata.
    (Full MVTec download requires manual registration)
    """
    mvtec_dir = DATA_DIR / "mvtec"
    mvtec_dir.mkdir(parents=True, exist_ok=True)
    output_file = mvtec_dir / "mvtec_metadata.json"
    
    if output_file.exists():
        with open(output_file, "r") as f:
            return json.load(f)
    
    import random
    random.seed(42)
    
    categories = ["bottle", "cable", "capsule", "carpet", "grid",
                   "hazelnut", "leather", "metal_nut", "pill", "screw",
                   "tile", "toothbrush", "transistor", "wood", "zipper"]
    
    defect_types = ["scratch", "crack", "hole", "contamination", "bent",
                    "broken", "color_defect", "cut", "fold", "glue_strip",
                    "metal_contamination", "thread", "combined", "poke"]
    
    operators = ["John Smith", "Maria Garcia", "Ahmed Hassan",
                 "Wei Chen", "Sarah Johnson", "Raj Patel",
                 "Elena Volkov", "Carlos Mendez"]
    
    machines = ["CNC-A1", "CNC-B2", "Press-C3", "Weld-D4",
                "Lathe-E5", "Mill-F6", "Grind-G7"]
    
    entries = []
    for i in range(n_entries):
        category = random.choice(categories)
        is_defective = random.random() < 0.3
        
        entry = {
            "image_id": f"mvtec_{i:05d}",
            "category": category,
            "is_defective": is_defective,
            "defect_type": random.choice(defect_types) if is_defective else "good",
            "inspector": random.choice(operators),
            "machine": random.choice(machines),
            "inspection_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "severity": random.choice(["low", "medium", "high"]) if is_defective else "none",
            "caption": (
                f"A {category} with {'a ' + random.choice(defect_types) + ' defect' if is_defective else 'no visible defects'} "
                f"inspected by {random.choice(operators)} using machine {random.choice(machines)}"
            ),
            "path": f"data/mvtec/{category}/{'defect' if is_defective else 'good'}/{i:05d}.png",
            "source": "mvtec_ad",
        }
        entries.append(entry)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    
    logger.info(f"MVTec metadata created: {n_entries} entries ({sum(1 for e in entries if e['is_defective'])} defective)")
    return entries


# ═══════════════════════════════════════════════════════════════
# Main Download All
# ═══════════════════════════════════════════════════════════════
def download_all(
    max_coco: int = 500,
    max_arxiv: int = 1000,
    max_mvtec: int = 500,
) -> Dict[str, List[Dict]]:
    """Download/prepare all datasets."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING/PREPARING ALL DATASETS")
    logger.info("=" * 60)
    
    datasets = {}
    
    # COCO
    logger.info("\n[1/3] MS-COCO val2017...")
    datasets["coco"] = download_coco(max_images=max_coco)
    
    # ArXiv
    logger.info("\n[2/3] ArXiv Abstracts...")
    datasets["arxiv"] = download_arxiv_abstracts(n_abstracts=max_arxiv)
    
    # MVTec
    logger.info("\n[3/3] MVTec AD Metadata...")
    datasets["mvtec"] = create_mvtec_metadata(n_entries=max_mvtec)
    
    # Summary
    logger.info("\n" + "=" * 60)
    for name, data in datasets.items():
        logger.info(f"  {name}: {len(data)} entries")
    logger.info("=" * 60)
    
    return datasets


if __name__ == "__main__":
    download_all()
