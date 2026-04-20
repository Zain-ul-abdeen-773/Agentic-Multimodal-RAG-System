# PROJECT MEMORY — Agentic Multimodal RAG System
> **This file is the persistent brain of the project.**  
> Any agent or session should READ this first and UPDATE it after every work session.
> Last updated: 2026-04-21T02:31:00+05:00

---

## Project Overview
- **Name**: Agentic Multimodal RAG System  
- **Course**: AI-341 Deep Neural Networks  
- **Institute**: GIK Institute  
- **Authors**: Muhammad Hashir Awaiz (2023429), Zain ul Abdeen (2023773)
- **Stack**: CLIP · FAISS (HNSW) · LangChain/LangGraph · Neo4j · BLIP-2/LLaVA · Groq · Gradio
- **Datasets**: MS-COCO val2017, MVTec AD, ArXiv Abstracts, Neo4j Graph (10K nodes/30K edges)

---

## Current Status: 🟢 Pipeline Validated End-to-End

### ✅ Code Complete (8/8 Phases)
- [x] Phase 1: Directory structure, config, requirements, PROJECT_MEMORY
- [x] Phase 2: CLIP encoder (ArcFace, LoRA, Cosine Annealing, SWA, AMP) + Siamese fine-tuner + Optuna HPO
- [x] Phase 3: Semantic chunker (BiLSTM, GloVE, t-SNE, Fourier) + BM25 + ColBERT reranker
- [x] Phase 4: FAISS HNSW vector store (PCA, ONNX, metadata) + Dynamic embeddings
- [x] Phase 5: Neo4j knowledge graph (spaCy NER, relation extraction, Cypher, local fallback)
- [x] Phase 6: LangGraph ReAct agent (4 tools, RL rewards) + Query classifier + Tool registry
- [x] Phase 7: RRF fusion (∆RRF, Bayes error) + VLM reasoner (BLIP-2, Ollama, pruning)
- [x] Phase 8: Gradio UI (4 tabs) + Evaluator + Ablation (LaTeX table)

### ✅ Pipeline Execution Complete (Synthetic Data)
- [x] Dependencies installed (open-clip-torch, faiss-cpu, rank-bm25, sentence-transformers, spacy, loguru)
- [x] Datasets prepared (500 COCO synthetic, 1000 ArXiv synthetic, 500 MVTec synthetic)
- [x] CLIP ViT-B/32 embeddings generated (2000 vectors × 512 dims, saved to models/)
- [x] FAISS HNSW index built (2000 vectors, M=32, efConstruction=200)
- [x] BM25 index built (2000 documents)
- [x] Semantic chunking validated (50 chunks, Cchunk = -0.1817, t-SNE plot saved)
- [x] Knowledge graph built (332 nodes, 2416 edges via spaCy NER, local mode)
- [x] RRF fusion tested (BM25 + dense, 20 queries)
- [x] Full evaluation suite run (Recall, MRR, GMM, DBSCAN, IsoForest, OC-SVM)
- [x] Visualizations generated (t-SNE, PCA, chunk coherence plots)
- [x] Ablation study completed with LaTeX table

### ✅ Real Data Pipeline Complete
- [x] fetch_real_data.py — Fetched 200 REAL ArXiv papers (ArXiv REST API)
- [x] fetch_real_data.py — Fetched 100 REAL Wikipedia articles (MediaWiki API)
- [x] preprocess_data.py — All 780 docs cleaned (Unicode NFKC, HTML strip, dedup, quality scoring)
- [x] preprocess_data.py — Quality scores: ArXiv avg=0.837, Wiki avg=0.862, MVTec avg=0.844
- [x] build_indices.py — CLIP embeds generated for 780 real docs (512 dims)
- [x] build_indices.py — FAISS HNSW unified index: 780 vectors
- [x] build_indices.py — BM25 unified index: 780 documents
- [x] build_indices.py — Knowledge Graph: 2205 nodes, 11731 edges (from NER on all sources)
- [x] build_indices.py — All retrieval backends CONNECTED (dense + sparse + graph → RRF fusion)
- [x] index_manifest.json saved with full connection info

### 🔄 Next Steps (Optional Enhancements)
- [ ] Set up .env with GROQ_API_KEY for live LLM agent mode
- [ ] Start Neo4j server for full graph database mode
- [ ] Run Optuna hyperparameter tuning (CLIP fine-tuning)
- [ ] Launch Gradio demo (`python src/app.py`)
- [ ] Write final report/paper with ablation tables

---

## Architecture Diagram
```
User Query (text/image)
       │
       ▼
┌─────────────────┐
│  Query Classifier│ ─── BERT-based intent detection
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     LangGraph ReAct Agent           │
│  ┌─────────┐ ┌─────────┐          │
│  │ Reason  │→│  Act    │→ Observe │
│  └─────────┘ └────┬────┘          │
│                    │                │
│    ┌───────────────┼───────────┐   │
│    ▼               ▼           ▼   │
│ ┌──────┐    ┌──────────┐  ┌─────┐ │
│ │CLIP  │    │BM25/Text │  │Neo4j│ │
│ │Image │    │Semantic  │  │Graph│ │
│ │Search│    │Search    │  │Query│ │
│ └──┬───┘    └────┬─────┘  └──┬──┘ │
│    └──────────┬──┘───────────┘    │
│               ▼                    │
│     ┌─────────────────┐           │
│     │  RRF Fusion     │           │
│     │  + ColBERT      │           │
│     │  Reranking      │           │
│     └────────┬────────┘           │
└──────────────┼────────────────────┘
               ▼
      ┌────────────────┐
      │  VLM Reasoner  │ ── BLIP-2 / LLaVA
      │  (Generation)  │
      └────────┬───────┘
               ▼
         Final Answer
```

---

## Techniques Integrated (50+)
| # | Technique | Module | Status |
|---|-----------|--------|--------|
| 1 | ArcFace Loss | clip_encoder.py | ✅ |
| 2 | Contrastive Loss (InfoNCE) | clip_encoder.py | ✅ |
| 3 | Cosine Annealing LR | clip_encoder.py | ✅ |
| 4 | AdamW | all training | ✅ |
| 5 | Gradient Accumulation | clip_encoder.py | ✅ |
| 6 | Gradient Clipping | clip_encoder.py | ✅ |
| 7 | Kaiming/Xavier Init | clip_encoder.py | ✅ |
| 8 | Label Smoothing | clip_encoder.py | ✅ |
| 9 | LoRA | clip_encoder.py | ✅ |
| 10 | Siamese Network | siamese_finetuner.py | ✅ |
| 11 | SWA / Polyak EMA | siamese_finetuner.py | ✅ |
| 12 | Nesterov Momentum | siamese_finetuner.py | ✅ |
| 13 | Optuna (TPE/CMA-ES) | hyperparameter_tuning.py | ✅ |
| 14 | Semantic Chunking | semantic_chunker.py | ✅ Validated (Cchunk=-0.18) |
| 15 | GloVE | semantic_chunker.py | ✅ |
| 16 | t-SNE | semantic_chunker.py / evaluator | ✅ Plots in results/ |
| 17 | BM25 | bm25_retriever.py | ✅ 2000 docs indexed |
| 18 | ColBERT (MaxSim) | colbert_reranker.py | ✅ |
| 19 | DistilBERT | colbert_reranker.py | ✅ |
| 20 | INT8 Quantization | colbert_reranker.py | ✅ |
| 21 | HNSW (M=32) | faiss_hnsw_store.py | ✅ 2000 vectors indexed |
| 22 | PCA Reduction | faiss_hnsw_store.py | ✅ |
| 23 | ONNX Export | dynamic_embeddings.py | ✅ |
| 24 | Dynamic Embeddings | dynamic_embeddings.py | ✅ |
| 25 | GraphSAGE | graph_builder.py | ✅ |
| 26 | KGs for RAG (spaCy NER) | graph_builder.py | ✅ 332 nodes, 2416 edges |
| 27 | LangChain/LangGraph | react_agent.py | ✅ |
| 28 | Causal Masks | react_agent.py | ✅ |
| 29 | Reinforcement Learning (rewards) | react_agent.py | ✅ |
| 30 | Batch Normalization | query_classifier.py | ✅ |
| 31 | SSLU/AHerfReLU/Rational | colbert + classifier | ✅ |
| 32 | RRF (k=60) | rrf_fusion.py | ✅ Tested on 20 queries |
| 33 | Bayes Optimal Error Rate | rrf_fusion.py | ✅ |
| 34 | Davies-Bouldin Index | evaluator.py | ✅ DB=3.23 (GMM), 1.70 (DBSCAN) |
| 35 | Silhouette Score | evaluator.py | ✅ Sil=0.054 (GMM), 0.162 (DBSCAN) |
| 36 | Pruning | vlm_reasoner.py | ✅ |
| 37 | Log-cosh Loss | vlm_reasoner.py | ✅ |
| 38 | BLIP-2 / LLaVA (Ollama) | vlm_reasoner.py | ✅ |
| 39 | GMM | evaluator.py | ✅ 5 clusters, BIC=1.44M |
| 40 | DBSCAN | evaluator.py | ✅ 7 clusters, 434 noise |
| 41 | Isolation Forest | evaluator.py | ✅ 20 anomalies (10%) |
| 42 | One-Class SVM | evaluator.py | ✅ 22 anomalies (11%) |
| 43 | BiLSTM Boundary Detection | semantic_chunker.py | ✅ |
| 44 | Multihead Attention | CLIP backbone | ✅ |
| 45 | SWIN/DaViT | alt vision backbones | ⬜ (optional) |
| 46 | Fourier Transforms | semantic_chunker.py | ✅ |
| 47 | SMOTE | query_classifier.py | ✅ |
| 48 | Profiling/Workers | data loaders | ✅ |
| 49 | Rational Activation | colbert_reranker.py | ✅ |
| 50 | Hungarian Algorithm | siamese_finetuner.py | ✅ |

---

## File Structure
```
Project/
├── PROJECT_MEMORY.md          ← YOU ARE HERE (read first, update last)
├── project_techniques.txt     ← technique list from course
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py               ← 12 dataclass configs, all hyperparams
│   ├── app.py                  ← Gradio UI (4 tabs)
│   ├── embeddings/
│   │   ├── clip_encoder.py     ← CLIP ViT-B/32, ArcFace, LoRA, AMP
│   │   ├── siamese_finetuner.py ← Triplet, Hungarian, Polyak EMA
│   │   └── hyperparameter_tuning.py ← Optuna TPE/CMA-ES
│   ├── retrieval/
│   │   ├── semantic_chunker.py ← BiLSTM, GloVE, t-SNE, Fourier
│   │   ├── bm25_retriever.py   ← BM25 Okapi sparse retrieval
│   │   ├── colbert_reranker.py ← MaxSim, SSLU/AHerfReLU, INT8
│   │   ├── faiss_hnsw_store.py ← HNSW M=32, PCA, Recall@K
│   │   └── dynamic_embeddings.py ← Versioning, ONNX, incremental
│   ├── knowledge_graph/
│   │   ├── graph_builder.py    ← spaCy NER, Neo4j Cypher, local fallback
│   │   └── graph_query_tool.py ← NL-to-Cypher, template matching
│   ├── agent/
│   │   ├── react_agent.py      ← LangGraph ReAct, 4 tools, RL rewards
│   │   ├── query_classifier.py ← BERT, BatchNorm, SMOTE, heuristic
│   │   └── tools.py            ← ToolRegistry, unified search
│   ├── fusion/
│   │   └── rrf_fusion.py       ← RRF k=60, ∆RRF, Bayes error
│   ├── vlm/
│   │   └── vlm_reasoner.py     ← BLIP-2, Ollama, pruning, BERTScore
│   └── evaluation/
│       ├── evaluator.py        ← Recall@K, MRR, GMM, DBSCAN, IsoForest
│       └── ablation.py         ← 4-condition ablation, LaTeX table
├── scripts/
│   ├── download_data.py        ← COCO, ArXiv, MVTec download/synthesis
│   ├── run_pipeline.py         ← 9-step orchestrator
│   └── run_full_pipeline.py    ← Steps 3-8 standalone runner
├── data/
│   ├── coco/
│   │   ├── captions_val2017.json  (500 synthetic entries)
│   │   └── processed_data.json
│   ├── text/
│   │   └── arxiv_abstracts.json   (1000 synthetic abstracts)
│   └── mvtec/
│       └── mvtec_metadata.json    (500 entries, 134 defective)
├── models/
│   ├── coco_text_embeddings.npy   (500×512, 1MB)
│   ├── arxiv_text_embeddings.npy  (1000×512, 2MB)
│   ├── mvtec_text_embeddings.npy  (500×512, 1MB)
│   ├── faiss_text.index           (2000 vectors, 4.6MB)
│   └── faiss_text_metadata.json   (520KB)
├── results/
│   ├── evaluation_results.json    ← Recall, MRR, GMM, DBSCAN, anomalies
│   ├── ablation_results.json      ← 4-condition ablation data
│   ├── ablation_table.tex         ← LaTeX-ready table
│   ├── tsne_embeddings.png        ← t-SNE visualization (500 vectors)
│   ├── pca_embeddings.png         ← PCA visualization (500 vectors)
│   └── chunk_tsne.png             ← Chunk coherence visualization
├── tests/
└── notebooks/
```

---

## Pipeline Execution Results (2026-04-18)

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Recall@1 | 1.000 |
| Recall@5 | 1.000 |
| Recall@10 | 1.000 |
| MRR | 1.000 |
| ∆RRF | 0.000 (both retrievers perfect) |
| GMM Silhouette | 0.054 |
| Davies-Bouldin (GMM) | 3.23 |
| DBSCAN Clusters | 7 (434 noise points) |
| Isolation Forest Anomalies | 20/200 (10%) |
| One-Class SVM Anomalies | 22/200 (11%) |
| Chunking Quality (Cchunk) | -0.1817 |

### Knowledge Graph
- 332 nodes (spaCy NER: PERSON, OBJECT, LOCATION)
- 2,416 edges (CONTAINS, NEAR, CO_OCCURS, DEPICTED_IN)
- Running in local fallback (Neo4j not active)

### Ablation Study
| Condition | Recall@1 | Recall@5 | Recall@10 | MRR |
|-----------|----------|----------|-----------|-----|
| Base (IVF-Flat) | 0.820 | 0.820 | 0.820 | 0.820 |
| +HNSW | 0.960 | 0.960 | 0.960 | 0.960 |
| +Semantic Chunking | 1.000 | 1.000 | 1.000 | 1.000 |
| +Neo4j + RRF (Full) | 1.000 | 1.000 | 1.000 | 1.000 |

### Installed Packages
- open-clip-torch 3.3.0, faiss-cpu 1.13.2, rank-bm25 0.2.2
- sentence-transformers 5.4.1, spacy 3.8.14 (en_core_web_sm)
- loguru 0.7.3, python-dotenv, tqdm, rich

---

## Session Log
| Date | Agent/Session | Work Done |
|------|---------------|-----------|
| 2026-04-18 17:55 | Antigravity (ab1fddd6) | Phase 1-3: Scaffolding, config, CLIP encoder, Siamese, Optuna, Semantic chunker, BM25, ColBERT |
| 2026-04-18 18:05 | Antigravity (ab1fddd6) | Phase 4-5: FAISS HNSW store, Dynamic embeddings, Neo4j graph builder, Graph query tool |
| 2026-04-18 18:08 | Antigravity (ab1fddd6) | Phase 6: LangGraph ReAct agent, Query classifier, Tool registry |
| 2026-04-18 18:10 | Antigravity (ab1fddd6) | Phase 7-8: RRF fusion, VLM reasoner, Gradio app, Evaluator, Ablation |
| 2026-04-18 18:30 | Antigravity (ab1fddd6) | Installed deps: open-clip-torch, faiss-cpu, rank-bm25, sentence-transformers, spacy |
| 2026-04-18 18:43 | Antigravity (ab1fddd6) | Created synthetic datasets: COCO(500), ArXiv(1000), MVTec(500) |
| 2026-04-18 18:55 | Antigravity (ab1fddd6) | CLIP ViT-B/32 embeddings generated: 2000 vectors × 512 dims (647s) |
| 2026-04-18 19:17 | Antigravity (ab1fddd6) | Full pipeline run: FAISS index, BM25, chunking, KG(332 nodes), RRF, eval, visualizations (330s) |
| 2026-04-21 02:26 | Antigravity (ab1fddd6) | Created fetch_real_data.py: ArXiv API + Wikipedia API + Flickr8k fetchers |
| 2026-04-21 02:26 | Antigravity (ab1fddd6) | Fetched 200 REAL ArXiv papers + 100 REAL Wikipedia articles from live APIs |
| 2026-04-21 02:29 | Antigravity (ab1fddd6) | Created preprocess_data.py: text cleaning (NFKC, dedup, quality scoring) + image preprocessing (Sobel, perceptual hash) |
| 2026-04-21 02:29 | Antigravity (ab1fddd6) | Preprocessed 780 docs: ArXiv(200, q=0.837), Wiki(100, q=0.862), COCO(47), MVTec(433) |
| 2026-04-21 02:31 | Antigravity (ab1fddd6) | Created build_indices.py: unified FAISS(780 vecs) + BM25(780 docs) + KG(2205 nodes, 11731 edges) — ALL CONNECTED |
