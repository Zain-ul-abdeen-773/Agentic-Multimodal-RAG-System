"""
Main Pipeline — End-to-End Agentic Multimodal RAG
==================================================
Orchestrates: data loading → embedding → indexing → graph → agent → demo
Run: python scripts/run_pipeline.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, DEVICE, PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR


def step_1_prepare_data():
    """Step 1: Download/prepare datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing Datasets")
    logger.info("=" * 60)
    
    from scripts.download_data import download_all
    datasets = download_all(max_coco=500, max_arxiv=1000, max_mvtec=500)
    
    logger.info(f"Datasets ready: {list(datasets.keys())}")
    return datasets


def step_2_embed_data(datasets: dict):
    """Step 2: Generate CLIP embeddings for images and text."""
    logger.info("=" * 60)
    logger.info("STEP 2: Generating CLIP Embeddings")
    logger.info("=" * 60)
    
    from src.embeddings.clip_encoder import CLIPEncoder, embed_texts
    
    config = get_config()
    encoder = CLIPEncoder(config.clip)
    encoder = encoder.to(DEVICE)
    encoder.eval()
    
    embeddings = {}
    
    # ── Embed COCO captions ──
    coco = datasets.get("coco", [])
    if coco:
        coco_captions = [item["caption"] for item in coco]
        logger.info(f"Embedding {len(coco_captions)} COCO captions...")
        coco_text_emb = embed_texts(encoder, coco_captions, batch_size=128)
        embeddings["coco_text"] = coco_text_emb.numpy()
        logger.info(f"COCO text embeddings: {coco_text_emb.shape}")
    
    # ── Embed COCO images (if downloaded) ──
    coco_with_images = [item for item in coco if Path(item.get("path", "")).exists()]
    if coco_with_images:
        from src.embeddings.clip_encoder import embed_images
        image_paths = [item["path"] for item in coco_with_images]
        logger.info(f"Embedding {len(image_paths)} COCO images...")
        coco_img_emb = embed_images(encoder, image_paths, batch_size=64)
        embeddings["coco_images"] = coco_img_emb.numpy()
        logger.info(f"COCO image embeddings: {coco_img_emb.shape}")
    
    # ── Embed ArXiv abstracts ──
    arxiv = datasets.get("arxiv", [])
    if arxiv:
        arxiv_texts = [item["text"] for item in arxiv]
        logger.info(f"Embedding {len(arxiv_texts)} ArXiv abstracts...")
        arxiv_emb = embed_texts(encoder, arxiv_texts, batch_size=128)
        embeddings["arxiv_text"] = arxiv_emb.numpy()
        logger.info(f"ArXiv embeddings: {arxiv_emb.shape}")
    
    # ── Embed MVTec captions ──
    mvtec = datasets.get("mvtec", [])
    if mvtec:
        mvtec_captions = [item["caption"] for item in mvtec]
        logger.info(f"Embedding {len(mvtec_captions)} MVTec captions...")
        mvtec_emb = embed_texts(encoder, mvtec_captions, batch_size=128)
        embeddings["mvtec_text"] = mvtec_emb.numpy()
        logger.info(f"MVTec embeddings: {mvtec_emb.shape}")
    
    # Save embeddings
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, emb in embeddings.items():
        np.save(str(MODELS_DIR / f"{name}_embeddings.npy"), emb)
    
    logger.info(f"All embeddings saved to {MODELS_DIR}")
    return embeddings, encoder


def step_3_build_indices(datasets: dict, embeddings: dict):
    """Step 3: Build FAISS HNSW indices."""
    logger.info("=" * 60)
    logger.info("STEP 3: Building FAISS HNSW Indices")
    logger.info("=" * 60)
    
    from src.retrieval.faiss_hnsw_store import FAISSHNSWStore
    config = get_config()
    
    stores = {}
    
    # ── Text index (COCO + ArXiv + MVTec captions) ──
    all_text_emb = []
    all_text_meta = []
    
    for key, dataset_name in [("coco", "coco"), ("arxiv", "arxiv"), ("mvtec", "mvtec")]:
        emb_key = f"{key}_text"
        if emb_key in embeddings:
            all_text_emb.append(embeddings[emb_key])
            for item in datasets.get(dataset_name, []):
                all_text_meta.append({
                    "text": item.get("text", item.get("caption", "")),
                    "doc_id": item.get("doc_id", item.get("image_id", "")),
                    "source": item.get("source", dataset_name),
                })
    
    if all_text_emb:
        text_embeddings = np.concatenate(all_text_emb, axis=0)
        # Truncate metadata to match embeddings
        all_text_meta = all_text_meta[:len(text_embeddings)]
        
        dim = text_embeddings.shape[1]
        text_store = FAISSHNSWStore(dim=dim, config=config.faiss, store_name="text")
        text_store.add(text_embeddings, all_text_meta)
        
        text_idx_path = str(MODELS_DIR / "faiss_text.index")
        text_meta_path = str(MODELS_DIR / "faiss_text_metadata.json")
        text_store.save(text_idx_path, text_meta_path)
        
        stores["text"] = text_store
        logger.info(f"Text index built: {text_store.n_vectors} vectors")
    
    # ── Image index ──
    if "coco_images" in embeddings:
        img_emb = embeddings["coco_images"]
        coco_with_images = [
            item for item in datasets.get("coco", [])
            if Path(item.get("path", "")).exists()
        ]
        img_meta = [
            {
                "path": item["path"],
                "caption": item["caption"],
                "image_id": item["image_id"],
                "source": "coco",
            }
            for item in coco_with_images[:len(img_emb)]
        ]
        
        dim = img_emb.shape[1]
        img_store = FAISSHNSWStore(dim=dim, config=config.faiss, store_name="image")
        img_store.add(img_emb, img_meta)
        
        img_idx_path = str(MODELS_DIR / "faiss_image.index")
        img_meta_path = str(MODELS_DIR / "faiss_image_metadata.json")
        img_store.save(img_idx_path, img_meta_path)
        
        stores["image"] = img_store
        logger.info(f"Image index built: {img_store.n_vectors} vectors")
    
    return stores


def step_4_build_bm25(datasets: dict):
    """Step 4: Build BM25 sparse index."""
    logger.info("=" * 60)
    logger.info("STEP 4: Building BM25 Index")
    logger.info("=" * 60)
    
    from src.retrieval.bm25_retriever import BM25Retriever
    config = get_config()
    
    # Combine all text documents
    all_docs = []
    all_ids = []
    
    for item in datasets.get("coco", []):
        all_docs.append(item.get("caption", ""))
        all_ids.append(item.get("image_id", ""))
    
    for item in datasets.get("arxiv", []):
        all_docs.append(item.get("text", ""))
        all_ids.append(item.get("doc_id", ""))
    
    for item in datasets.get("mvtec", []):
        all_docs.append(item.get("caption", ""))
        all_ids.append(item.get("image_id", ""))
    
    bm25 = BM25Retriever(config.bm25)
    bm25.index(all_docs, all_ids)
    
    logger.info(f"BM25 index built: {len(all_docs)} documents")
    return bm25


def step_5_semantic_chunking(datasets: dict):
    """Step 5: Apply semantic chunking to ArXiv abstracts."""
    logger.info("=" * 60)
    logger.info("STEP 5: Semantic Chunking")
    logger.info("=" * 60)
    
    from src.retrieval.semantic_chunker import SemanticChunker
    config = get_config()
    
    chunker = SemanticChunker(config.chunking)
    
    arxiv = datasets.get("arxiv", [])
    all_chunks = []
    
    for item in arxiv[:100]:  # Chunk first 100 for demo
        text = item.get("text", "")
        if len(text) > 50:
            chunks = chunker.chunk(text)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": item.get("doc_id", ""),
                    "coherence": chunk.coherence_score,
                })
    
    # Compute overall chunking quality
    if all_chunks:
        from src.retrieval.semantic_chunker import Chunk
        dummy_chunks = []
        for item in arxiv[:20]:
            text = item.get("text", "")
            if len(text) > 50:
                chunks = chunker.chunk(text)
                dummy_chunks.extend(chunks)
        
        if dummy_chunks:
            quality = chunker.compute_chunking_quality(dummy_chunks)
            logger.info(f"Chunking quality (Cchunk): {quality:.4f}")
        
        # Save t-SNE visualization
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if dummy_chunks and len(dummy_chunks) > 5:
            chunker.visualize_tsne(
                dummy_chunks,
                save_path=str(RESULTS_DIR / "chunk_tsne.png"),
            )
    
    logger.info(f"Semantic chunking done: {len(all_chunks)} chunks from {len(arxiv[:100])} docs")
    return all_chunks


def step_6_build_knowledge_graph(datasets: dict):
    """Step 6: Build Neo4j knowledge graph."""
    logger.info("=" * 60)
    logger.info("STEP 6: Building Knowledge Graph")
    logger.info("=" * 60)
    
    from src.knowledge_graph.graph_builder import KnowledgeGraph
    from src.knowledge_graph.graph_query_tool import GraphQueryTool
    config = get_config()
    
    kg = KnowledgeGraph(config.neo4j)
    
    # Build from COCO captions + MVTec metadata
    all_captions = []
    
    for item in datasets.get("coco", [])[:200]:  # First 200 COCO
        all_captions.append({
            "image_id": item.get("image_id", ""),
            "caption": item.get("caption", ""),
            "path": item.get("path", ""),
        })
    
    for item in datasets.get("mvtec", [])[:200]:  # First 200 MVTec
        all_captions.append({
            "image_id": item.get("image_id", ""),
            "caption": item.get("caption", ""),
            "path": item.get("path", ""),
            "metadata": {
                "category": item.get("category", ""),
                "defect_type": item.get("defect_type", ""),
                "inspector": item.get("inspector", ""),
                "machine": item.get("machine", ""),
            },
        })
    
    if all_captions:
        kg.build_from_captions(all_captions, batch_size=50)
    
    stats = kg.get_stats()
    logger.info(f"Knowledge graph built: {stats}")
    
    graph_tool = GraphQueryTool(kg)
    
    return kg, graph_tool


def step_7_initialize_agent(stores, bm25, graph_tool, encoder):
    """Step 7: Initialize LangGraph ReAct agent."""
    logger.info("=" * 60)
    logger.info("STEP 7: Initializing Agent")
    logger.info("=" * 60)
    
    from src.agent.tools import ToolRegistry
    from src.agent.react_agent import AgenticRAG
    from src.fusion.rrf_fusion import RRFFusion
    config = get_config()
    
    # Build tool registry
    registry = ToolRegistry()
    registry.clip_encoder = encoder
    registry.bm25 = bm25
    registry.image_store = stores.get("image")
    registry.text_store = stores.get("text")
    registry.graph_tool = graph_tool
    registry.rrf_fusion = RRFFusion(config.rrf)
    
    # Try to load ColBERT reranker
    try:
        from src.retrieval.colbert_reranker import ColBERTReranker
        colbert = ColBERTReranker(config.colbert).to(DEVICE)
        colbert.quantize()
        registry.colbert = colbert
        logger.info("ColBERT reranker loaded")
    except Exception as e:
        logger.warning(f"ColBERT not loaded: {e}")
    
    # Initialize agent
    agent = AgenticRAG(
        config=config.agent,
        image_searcher=registry,
        text_searcher=registry,
        graph_querier=graph_tool,
        hybrid_searcher=registry.hybrid_search,
    )
    
    logger.info("Agent initialized")
    return agent, registry


def step_8_run_evaluation(stores, bm25, datasets, embeddings):
    """Step 8: Run evaluation suite."""
    logger.info("=" * 60)
    logger.info("STEP 8: Running Evaluation")
    logger.info("=" * 60)
    
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.ablation import AblationStudy
    from src.fusion.rrf_fusion import RRFFusion
    config = get_config()
    
    evaluator = Evaluator(config.eval)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # ── Test text retrieval ──
    text_store = stores.get("text")
    if text_store and "coco_text" in embeddings:
        logger.info("Evaluating text retrieval...")
        
        # Use first 50 queries as test
        test_queries = embeddings["coco_text"][:50]
        
        # Get retrieved results
        retrieved_ids = []
        relevant_ids = []
        
        coco = datasets.get("coco", [])
        for i in range(min(50, len(test_queries))):
            search_results = text_store.search(test_queries[i], top_k=10)
            retrieved = [str(r.get("doc_id", r.get("id", ""))) for r in search_results]
            retrieved_ids.append(retrieved)
            
            # Self-retrieval ground truth
            relevant_ids.append([coco[i].get("image_id", str(i))])
        
        recall_results = evaluator.compute_all_recall(retrieved_ids, relevant_ids)
        mrr = evaluator.mrr(retrieved_ids, relevant_ids)
        
        results["text_retrieval"] = {
            "recall": recall_results,
            "mrr": mrr,
        }
        logger.info(f"Text retrieval — Recall: {recall_results} | MRR: {mrr:.4f}")
    
    # ── Embedding analysis ──
    if "coco_text" in embeddings:
        emb = embeddings["coco_text"][:500]
        
        results["gmm"] = evaluator.gmm_analysis(emb, n_components=5)
        results["dbscan"] = evaluator.dbscan_analysis(emb)
        results["anomalies"] = evaluator.anomaly_detection(emb)
        
        # Visualizations
        evaluator.visualize_embeddings(
            emb, method="tsne",
            save_path=str(RESULTS_DIR / "tsne_text_embeddings.png"),
        )
        evaluator.visualize_embeddings(
            emb, method="pca",
            save_path=str(RESULTS_DIR / "pca_text_embeddings.png"),
        )
        
        logger.info(f"GMM analysis: {results['gmm']}")
        logger.info(f"DBSCAN analysis: {results['dbscan']}")
    
    # ── RRF Fusion evaluation ──
    if text_store and bm25:
        logger.info("Evaluating RRF fusion...")
        rrf = RRFFusion(config.rrf)
        
        coco = datasets.get("coco", [])
        for i in range(min(20, len(coco))):
            query_text = coco[i].get("caption", "")
            if not query_text:
                continue
            
            bm25_results = bm25.search(query_text, top_k=20)
            
            query_emb = embeddings["coco_text"][i:i+1]
            dense_results = text_store.search(query_emb[0], top_k=20)
            
            delta = rrf.compute_delta_rrf(
                query_text,
                [bm25_results, dense_results],
                relevant_doc_id=str(coco[i].get("image_id", "")),
            )
        
        aggregate = rrf.compute_aggregate_delta_rrf()
        results["rrf"] = aggregate
        logger.info(f"∆RRF: {aggregate}")
    
    # Save all results
    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"All results saved to {RESULTS_DIR}")
    return results


def step_9_launch_demo(registry):
    """Step 9: Launch Gradio demo."""
    logger.info("=" * 60)
    logger.info("STEP 9: Launching Gradio Demo")
    logger.info("=" * 60)
    
    from src.app import create_app
    app = create_app(registry)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════
def main(skip_demo: bool = False):
    """Run the full pipeline."""
    start = time.time()
    
    logger.info("🚀 AGENTIC MULTIMODAL RAG PIPELINE")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Project: {PROJECT_ROOT}")
    
    # Step 1: Data
    datasets = step_1_prepare_data()
    
    # Step 2: Embeddings
    embeddings, encoder = step_2_embed_data(datasets)
    
    # Step 3: FAISS indices
    stores = step_3_build_indices(datasets, embeddings)
    
    # Step 4: BM25
    bm25 = step_4_build_bm25(datasets)
    
    # Step 5: Semantic chunking
    chunks = step_5_semantic_chunking(datasets)
    
    # Step 6: Knowledge graph
    kg, graph_tool = step_6_build_knowledge_graph(datasets)
    
    # Step 7: Agent
    agent, registry = step_7_initialize_agent(stores, bm25, graph_tool, encoder)
    
    # Step 8: Evaluation
    results = step_8_run_evaluation(stores, bm25, datasets, embeddings)
    
    # Summary
    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 PIPELINE COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")
    
    # Step 9: Demo (optional)
    if not skip_demo:
        step_9_launch_demo(registry)
    
    return {
        "datasets": {k: len(v) for k, v in datasets.items()},
        "embeddings": {k: v.shape for k, v in embeddings.items()},
        "stores": {k: v.n_vectors for k, v in stores.items()},
        "evaluation": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agentic Multimodal RAG Pipeline")
    parser.add_argument("--skip-demo", action="store_true", help="Skip Gradio launch")
    parser.add_argument("--step", type=int, default=0, help="Run only a specific step (1-9)")
    args = parser.parse_args()
    
    if args.step > 0:
        logger.info(f"Running step {args.step} only")
        # Individual step execution would go here
    else:
        main(skip_demo=args.skip_demo)
