# PROJECT MEMORY — Agentic Multimodal RAG System
> **This file is the persistent brain of the project.**  
> Any agent or session should READ this first and UPDATE it after every work session.
> Last updated: 2026-04-18T17:55:00+05:00

---

## Project Overview
- **Name**: Agentic Multimodal RAG System  
- **Course**: AI-341 Deep Neural Networks  
- **Institute**: GIK Institute  
- **Authors**: Muhammad Hashir Awaiz (2023429), Zain ul Abdeen (2023773)
- **Stack**: CLIP · FAISS (HNSW) · LangChain/LangGraph · Neo4j · BLIP-2/LLaVA · Groq · Gradio
- **Datasets**: MS-COCO val2017, MVTec AD, ArXiv Abstracts, Neo4j Graph (10K nodes/30K edges)

---

## Current Status: 🟡 Phase 1 — Scaffolding (IN PROGRESS)

### ✅ Completed
- [x] Phase 1: Directory structure created
- [x] Phase 1: PROJECT_MEMORY.md created
- [x] Phase 1: requirements.txt created
- [x] Phase 1: src/config.py created
- [x] Phase 1: src/__init__.py + all subpackage inits created
- [x] Phase 1: .env.example created

### 🔄 In Progress
- [ ] Phase 2: CLIP embedding pipeline + fine-tuning

### 📋 Remaining Phases
- [ ] Phase 2: CLIP Embedding Pipeline + Fine-Tuning (ArcFace, LoRA, Siamese, Optuna)
- [ ] Phase 3: Semantic Chunking + BM25 + ColBERT Reranker
- [ ] Phase 4: HNSW-Indexed FAISS Vector Store
- [ ] Phase 5: Neo4j Knowledge Graph (GraphSAGE, spaCy NER)
- [ ] Phase 6: LangGraph ReAct Agent (4 tools, query classifier)
- [ ] Phase 7: RRF Fusion + VLM Integration (BLIP-2/LLaVA)
- [ ] Phase 8: Gradio Demo UI + Full Evaluation Suite

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
| 1 | ArcFace Loss | clip_encoder.py | ⬜ |
| 2 | Contrastive Loss | clip_encoder.py | ⬜ |
| 3 | Cosine Annealing LR | clip_encoder.py | ⬜ |
| 4 | AdamW | all training | ⬜ |
| 5 | Gradient Accumulation | clip_encoder.py | ⬜ |
| 6 | Gradient Clipping | clip_encoder.py | ⬜ |
| 7 | Kaiming/Xavier Init | clip_encoder.py | ⬜ |
| 8 | Label Smoothing | clip_encoder.py | ⬜ |
| 9 | LoRA | clip_encoder.py | ⬜ |
| 10 | Siamese Network | siamese_finetuner.py | ⬜ |
| 11 | SWA | siamese_finetuner.py | ⬜ |
| 12 | Nesterov Momentum | siamese_finetuner.py | ⬜ |
| 13 | Optuna | hyperparameter_tuning.py | ⬜ |
| 14 | Semantic Chunking | semantic_chunker.py | ⬜ |
| 15 | GloVE | semantic_chunker.py | ⬜ |
| 16 | t-SNE | semantic_chunker.py / evaluator | ⬜ |
| 17 | BM25 | bm25_retriever.py | ⬜ |
| 18 | ColBERT | colbert_reranker.py | ⬜ |
| 19 | DistilBERT | colbert_reranker.py | ⬜ |
| 20 | Quantization Training | colbert_reranker.py | ⬜ |
| 21 | HNSW | faiss_hnsw_store.py | ⬜ |
| 22 | PCA | faiss_hnsw_store.py | ⬜ |
| 23 | ONNX Export | faiss_hnsw_store.py | ⬜ |
| 24 | Dynamic Embeddings | dynamic_embeddings.py | ⬜ |
| 25 | GraphSAGE | graph_builder.py | ⬜ |
| 26 | KGs for RAG | graph_query_tool.py | ⬜ |
| 27 | LangChain/LangGraph | react_agent.py | ⬜ |
| 28 | Causal Masks | react_agent.py | ⬜ |
| 29 | Reinforcement Learning | react_agent.py | ⬜ |
| 30 | Batch Normalization | query_classifier.py | ⬜ |
| 31 | SSLU/AHerfReLU | query_classifier.py | ⬜ |
| 32 | RRF | rrf_fusion.py | ⬜ |
| 33 | Bayes Optimal Error | rrf_fusion.py | ⬜ |
| 34 | Davies-Bouldin Index | evaluator.py | ⬜ |
| 35 | Silhouette Score | evaluator.py | ⬜ |
| 36 | Pruning | vlm_reasoner.py | ⬜ |
| 37 | Log-cosh Loss | vlm_reasoner.py | ⬜ |
| 38 | BLIP-2 / LLaVA | vlm_reasoner.py | ⬜ |
| 39 | GMM | evaluator.py | ⬜ |
| 40 | DBSCAN | evaluator.py | ⬜ |
| 41 | Isolation Forest | evaluator.py | ⬜ |
| 42 | One-Class SVM | evaluator.py | ⬜ |
| 43 | BiLSTM | semantic_chunker.py | ⬜ |
| 44 | Multihead Attention | multiple modules | ⬜ |
| 45 | SWIN/DaViT | alt vision backbones | ⬜ |
| 46 | Fourier Transforms | embedding analysis | ⬜ |
| 47 | SMOTE | data balancing | ⬜ |
| 48 | Profiling/Workers | data loaders | ⬜ |
| 49 | Rational Activation | experimental | ⬜ |
| 50 | Hungarian Algorithm | assignment matching | ⬜ |

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
│   ├── config.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── clip_encoder.py
│   │   ├── siamese_finetuner.py
│   │   └── hyperparameter_tuning.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── semantic_chunker.py
│   │   ├── bm25_retriever.py
│   │   ├── colbert_reranker.py
│   │   ├── faiss_hnsw_store.py
│   │   └── dynamic_embeddings.py
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   └── graph_query_tool.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react_agent.py
│   │   ├── query_classifier.py
│   │   └── tools.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── rrf_fusion.py
│   ├── vlm/
│   │   ├── __init__.py
│   │   └── vlm_reasoner.py
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py
│       └── ablation.py
├── data/
│   ├── images/
│   ├── text/
│   └── graph/
├── models/
├── results/
├── tests/
└── notebooks/
```

---

## Session Log
| Date | Agent/Session | Work Done |
|------|---------------|-----------|
| 2026-04-18 | Antigravity (ab1fddd6) | Phase 1: Full scaffolding, config, requirements, memory file |
