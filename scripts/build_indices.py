"""
Build All Indices — Connects FAISS, BM25, Neo4j KG
====================================================
Takes preprocessed data and builds all retrieval indices:
  1. CLIP text embeddings (batch, AMP, GPU)
  2. FAISS HNSW index (dense vectors)
  3. BM25 index (sparse tokens)
  4. Neo4j Knowledge Graph (entities + relations)
  5. Graph Query Tool wiring

After this script, ALL retrieval backends are connected and ready
for the ReAct agent to query.

Usage:
    python scripts/build_indices.py
    python scripts/build_indices.py --source preprocessed   # use preprocessed data
    python scripts/build_indices.py --source raw             # use raw data
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, DEVICE, DATA_DIR, MODELS_DIR, RESULTS_DIR


def load_all_data(source: str = "preprocessed") -> Dict[str, List[Dict]]:
    """
    Load data from either preprocessed or raw directories.

    Args:
        source: 'preprocessed' for cleaned data, 'raw' for original
    """
    data = {}

    if source == "preprocessed":
        pp_dir = DATA_DIR / "preprocessed"
        file_map = {
            "arxiv": pp_dir / "arxiv_clean.json",
            "wikipedia": pp_dir / "wikipedia_clean.json",
            "coco": pp_dir / "coco_clean.json",
            "mvtec": pp_dir / "mvtec_clean.json",
            "flickr": pp_dir / "flickr_clean.json",
        }
    else:
        file_map = {
            "arxiv": DATA_DIR / "text" / "arxiv_real.json",
            "wikipedia": DATA_DIR / "text" / "wikipedia_real.json",
            "coco": DATA_DIR / "coco" / "processed_data.json",
            "mvtec": DATA_DIR / "mvtec" / "mvtec_metadata.json",
            "flickr": DATA_DIR / "images" / "flickr8k_real.json",
        }

    # Also check synthetic fallbacks
    fallbacks = {
        "arxiv": DATA_DIR / "text" / "arxiv_abstracts.json",
    }

    for name, path in file_map.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            data[name] = items
            logger.info(f"  Loaded {name}: {len(items)} entries from {path.name}")
        elif name in fallbacks and fallbacks[name].exists():
            with open(fallbacks[name], "r", encoding="utf-8") as f:
                items = json.load(f)
            data[name] = items
            logger.info(f"  Loaded {name}: {len(items)} entries from fallback")

    return data


def step_1_embed_text(data: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
    """Generate CLIP text embeddings for ALL text sources."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Generating CLIP Embeddings")
    logger.info("=" * 60)

    from src.embeddings.clip_encoder import CLIPEncoder, embed_texts

    config = get_config()
    encoder = CLIPEncoder(config.clip)
    encoder = encoder.to(DEVICE)
    encoder.eval()
    logger.info(f"CLIP loaded on {DEVICE}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = {}

    for source_name, items in data.items():
        if not items:
            continue

        # Get text field
        texts = []
        for item in items:
            text = item.get("text", item.get("caption", ""))
            if text:
                texts.append(text)

        if not texts:
            continue

        logger.info(f"  Embedding {len(texts)} texts from {source_name}...")
        emb = embed_texts(encoder, texts, batch_size=128)
        emb_np = emb.numpy() if hasattr(emb, "numpy") else np.array(emb)

        embeddings[source_name] = emb_np
        np.save(str(MODELS_DIR / f"{source_name}_embeddings.npy"), emb_np)
        logger.info(f"  {source_name}: {emb_np.shape} saved")

    return embeddings


def step_2_build_faiss(
    data: Dict[str, List[Dict]],
    embeddings: Dict[str, np.ndarray],
) -> "FAISSHNSWStore":
    """Build a unified FAISS HNSW index covering ALL data sources."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Building FAISS HNSW Index")
    logger.info("=" * 60)

    from src.retrieval.faiss_hnsw_store import FAISSHNSWStore

    config = get_config()

    # Combine all embeddings + metadata into one unified index
    all_emb_parts = []
    all_meta = []

    for source_name, items in data.items():
        if source_name not in embeddings:
            continue

        emb = embeddings[source_name]
        n = min(len(items), len(emb))

        for i in range(n):
            item = items[i]
            text = item.get("text", item.get("caption", ""))
            doc_id = item.get("doc_id", item.get("image_id", f"{source_name}_{i}"))
            title = item.get("title", "")

            all_meta.append({
                "doc_id": str(doc_id),
                "text": text[:500],  # Truncate for metadata storage
                "title": title,
                "source": source_name,
            })

        all_emb_parts.append(emb[:n])

    if not all_emb_parts:
        logger.error("No embeddings to index!")
        return None

    all_emb = np.concatenate(all_emb_parts, axis=0).astype(np.float32)
    dim = all_emb.shape[1]

    logger.info(f"  Total vectors: {all_emb.shape[0]} × {dim}")

    # Build HNSW index
    store = FAISSHNSWStore(dim=dim, config=config.faiss, store_name="unified")
    store.add(all_emb, all_meta)

    # Save
    idx_path = str(MODELS_DIR / "faiss_unified.index")
    meta_path = str(MODELS_DIR / "faiss_unified_metadata.json")
    store.save(idx_path, meta_path)

    # Quick validation search
    test_results = store.search(all_emb[0], top_k=3)
    logger.info(f"  Index built: {store.n_vectors} vectors | "
                f"Validation search returned {len(test_results)} results")

    return store


def step_3_build_bm25(data: Dict[str, List[Dict]]) -> "BM25Retriever":
    """Build a unified BM25 sparse text index."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Building BM25 Index")
    logger.info("=" * 60)

    from src.retrieval.bm25_retriever import BM25Retriever

    config = get_config()

    all_docs = []
    all_ids = []

    for source_name, items in data.items():
        for item in items:
            text = item.get("text", item.get("caption", ""))
            doc_id = item.get("doc_id", item.get("image_id", ""))
            if text:
                all_docs.append(text)
                all_ids.append(str(doc_id))

    bm25 = BM25Retriever(config.bm25)
    bm25.index(all_docs, all_ids)

    # Validation
    test_query = "neural network deep learning"
    results = bm25.search(test_query, top_k=3)
    logger.info(f"  BM25 built: {len(all_docs)} documents | "
                f"Validation: '{test_query}' → {len(results)} results")

    return bm25


def step_4_build_knowledge_graph(data: Dict[str, List[Dict]]):
    """Build Neo4j knowledge graph from all caption/text data."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Building Knowledge Graph")
    logger.info("=" * 60)

    from src.knowledge_graph.graph_builder import KnowledgeGraph
    from src.knowledge_graph.graph_query_tool import GraphQueryTool

    config = get_config()
    kg = KnowledgeGraph(config.neo4j)

    # Build captions list from ALL sources
    all_captions = []

    # COCO / Flickr captions (image descriptions)
    for source in ["coco", "flickr"]:
        for item in data.get(source, [])[:300]:
            all_captions.append({
                "image_id": item.get("image_id", item.get("doc_id", "")),
                "caption": item.get("text", item.get("caption", "")),
                "path": item.get("image_path", item.get("path", "")),
            })

    # MVTec (industrial defect reports)
    for item in data.get("mvtec", [])[:200]:
        all_captions.append({
            "image_id": item.get("image_id", ""),
            "caption": item.get("text", item.get("caption", "")),
            "path": "",
            "metadata": {
                "category": item.get("category", ""),
                "defect_type": item.get("defect_type", ""),
                "inspector": item.get("inspector", ""),
                "machine": item.get("machine", ""),
            },
        })

    # ArXiv / Wikipedia (knowledge articles) — first sentence as "caption"
    for source in ["arxiv", "wikipedia"]:
        for item in data.get(source, [])[:200]:
            text = item.get("text", "")
            # Use first sentence as caption for NER
            first_sent = text.split(".")[0] + "." if "." in text else text[:200]
            all_captions.append({
                "image_id": item.get("doc_id", ""),
                "caption": first_sent,
                "path": "",
                "metadata": {
                    "title": item.get("title", ""),
                    "source": source,
                    "category": item.get("category", ""),
                },
            })

    if all_captions:
        kg.build_from_captions(all_captions, batch_size=50)

    stats = kg.get_stats()
    logger.info(f"  Graph built: {stats}")

    # Wire up query tool
    graph_tool = GraphQueryTool(kg)

    # Validation
    test_results = graph_tool.search("find neural network")
    logger.info(f"  Graph query validation: {len(test_results)} results")

    return kg, graph_tool


def step_5_wire_agent(store, bm25, graph_tool):
    """Initialize the agent with all connected retrieval backends."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Wiring Agent Tools")
    logger.info("=" * 60)

    from src.agent.tools import ToolRegistry
    from src.fusion.rrf_fusion import RRFFusion

    config = get_config()

    registry = ToolRegistry()
    registry.text_store = store
    registry.bm25 = bm25
    registry.graph_tool = graph_tool
    registry.rrf_fusion = RRFFusion(config.rrf)

    # Test the hybrid search (RRF fusion of dense + sparse)
    test_query = "deep learning convolutional neural network"
    dense_results = store.search(
        store.index.reconstruct(0), top_k=20
    ) if store else []
    sparse_results = bm25.search(test_query, top_k=20) if bm25 else []
    graph_results = graph_tool.search(test_query) if graph_tool else []

    logger.info(f"  Dense search: {len(dense_results)} results")
    logger.info(f"  Sparse search: {len(sparse_results)} results")
    logger.info(f"  Graph search: {len(graph_results)} results")

    if dense_results and sparse_results:
        fused = registry.rrf_fusion.fuse(
            [dense_results, sparse_results], top_k=10
        )
        logger.info(f"  RRF fused: {len(fused)} results (from dense+sparse)")

    logger.info("  All retrieval backends CONNECTED ✓")

    return registry


def main(source: str = "preprocessed"):
    """Run the full index building pipeline."""
    t0 = time.time()

    logger.info("🔧 INDEX BUILDER — Connecting All Retrieval Backends")
    logger.info(f"   Device: {DEVICE} | Source: {source}")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    data = load_all_data(source=source)

    if not data:
        logger.error("No data found! Run fetch_real_data.py or download_data.py first.")
        return

    # Step 1: CLIP embeddings
    embeddings = step_1_embed_text(data)

    # Step 2: FAISS HNSW
    store = step_2_build_faiss(data, embeddings)

    # Step 3: BM25
    bm25 = step_3_build_bm25(data)

    # Step 4: Knowledge Graph
    kg, graph_tool = step_4_build_knowledge_graph(data)

    # Step 5: Wire everything
    registry = step_5_wire_agent(store, bm25, graph_tool)

    # Save connection manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(DEVICE),
        "source": source,
        "data_sources": {k: len(v) for k, v in data.items()},
        "embeddings": {k: list(v.shape) for k, v in embeddings.items()},
        "faiss_vectors": store.n_vectors if store else 0,
        "bm25_documents": len(bm25.documents) if hasattr(bm25, "documents") else 0,
        "kg_stats": kg.get_stats() if kg else {},
        "index_files": {
            "faiss_index": str(MODELS_DIR / "faiss_unified.index"),
            "faiss_metadata": str(MODELS_DIR / "faiss_unified_metadata.json"),
        },
    }

    manifest_path = MODELS_DIR / "index_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ ALL INDICES BUILT & CONNECTED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"   FAISS: {store.n_vectors if store else 0} vectors")
    logger.info(f"   BM25:  {manifest.get('bm25_documents', 0)} documents")
    logger.info(f"   KG:    {kg.get_stats() if kg else 'N/A'}")
    logger.info(f"   Manifest: {manifest_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build all retrieval indices")
    parser.add_argument(
        "--source", type=str, default="preprocessed",
        choices=["preprocessed", "raw"],
        help="Data source: 'preprocessed' (cleaned) or 'raw' (original)",
    )
    args = parser.parse_args()
    main(source=args.source)
