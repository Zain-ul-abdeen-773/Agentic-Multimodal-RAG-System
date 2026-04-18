"""
Agentic Multimodal RAG System — Centralized Configuration
==========================================================
All hyperparameters, paths, and model settings in one place.
Techniques: Cosine Annealing LR, AdamW, Gradient Accumulation,
Gradient Clipping, Label Smoothing, HNSW params, RRF k, etc.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Project Root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ── Device Configuration ─────────────────────────────────────
def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ── CLIP Embedding Config ────────────────────────────────────
@dataclass
class CLIPConfig:
    """Configuration for CLIP encoder and fine-tuning."""
    # Model
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    embedding_dim: int = 512
    
    # LoRA fine-tuning
    lora_enabled: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    # ArcFace head
    arcface_scale: float = 30.0
    arcface_margin: float = 0.5
    
    # Training
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01            # AdamW
    warmup_steps: int = 500
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 4   # Effective batch = 128
    max_grad_norm: float = 1.0             # Gradient clipping
    mixed_precision: bool = True           # AMP
    
    # Cosine Annealing LR with Warm Restarts
    lr_scheduler: str = "cosine_annealing_warm_restarts"
    cosine_T_0: int = 5                    # Initial restart period
    cosine_T_mult: int = 2                 # Period multiplier  
    cosine_eta_min: float = 1e-7           # Minimum LR
    
    # Stochastic Weight Averaging
    swa_enabled: bool = True
    swa_start_epoch: int = 15
    swa_lr: float = 1e-5
    
    # Weight Initialization
    init_method: str = "kaiming"           # kaiming | xavier
    
    # Contrastive Loss
    contrastive_temperature: float = 0.07
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


# ── Siamese Network Config ───────────────────────────────────
@dataclass
class SiameseConfig:
    """Configuration for Siamese fine-tuning with hard-negative mining."""
    margin: float = 1.0
    triplet_margin: float = 0.3
    mining_strategy: str = "semi-hard"     # hard | semi-hard | easy
    nesterov_momentum: float = 0.9
    polyak_decay: float = 0.999            # Polyak averaging EMA decay
    num_negatives: int = 5


# ── Semantic Chunking Config ─────────────────────────────────
@dataclass
class ChunkingConfig:
    """Embedding-similarity-based semantic chunking."""
    breakpoint_threshold: float = 0.3      # Cosine similarity delta for split
    min_chunk_size: int = 100              # Minimum tokens per chunk
    max_chunk_size: int = 512              # Maximum tokens per chunk
    overlap_tokens: int = 50               # Sliding window overlap
    embedding_model: str = "all-MiniLM-L6-v2"
    use_bilstm_boundary: bool = True       # BiLSTM boundary detection
    bilstm_hidden: int = 128
    bilstm_layers: int = 2


# ── BM25 Config ──────────────────────────────────────────────
@dataclass
class BM25Config:
    """BM25 sparse retrieval parameters."""
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 50


# ── ColBERT Reranker Config ──────────────────────────────────
@dataclass
class ColBERTConfig:
    """ColBERT late-interaction reranking."""
    model_name: str = "distilbert-base-uncased"
    max_query_length: int = 32
    max_doc_length: int = 180
    embedding_dim: int = 128              # Projected dimension
    top_k_rerank: int = 30                # Rerank top-K from fusion
    quantize: bool = True                 # INT8 quantization for deployment


# ── FAISS HNSW Config ────────────────────────────────────────
@dataclass
class FAISSConfig:
    """HNSW index parameters for FAISS."""
    # HNSW tuning
    hnsw_M: int = 32                       # Number of neighbors per layer
    hnsw_ef_construction: int = 200        # Build-time search width
    hnsw_ef_search: int = 128              # Query-time search width
    
    # Index paths
    image_index_path: str = str(MODELS_DIR / "faiss_image.index")
    text_index_path: str = str(MODELS_DIR / "faiss_text.index")
    metadata_path: str = str(MODELS_DIR / "metadata.db")
    
    # PCA
    pca_enabled: bool = False
    pca_components: int = 256              # Reduce 512 → 256
    
    # Retrieval
    top_k: int = 50


# ── Neo4j Knowledge Graph Config ────────────────────────────
@dataclass
class Neo4jConfig:
    """Neo4j connection and graph construction settings."""
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = "neo4j"
    
    # NER
    spacy_model: str = "en_core_web_sm"
    
    # GraphSAGE
    graphsage_hidden: int = 128
    graphsage_layers: int = 2
    graphsage_aggregator: str = "mean"     # mean | pool | lstm


# ── Agent Config ─────────────────────────────────────────────
@dataclass
class AgentConfig:
    """LangGraph ReAct agent settings."""
    llm_model: str = "llama3-70b-8192"     # Groq model
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    temperature: float = 0.1
    max_iterations: int = 5                # Max ReAct loops
    max_tokens: int = 4096
    
    # Query classifier
    classifier_model: str = "bert-base-uncased"
    num_query_classes: int = 4             # visual, textual, hybrid, graph
    classifier_activation: str = "sslu"    # sslu | aherfrelu | relu


# ── RRF Fusion Config ────────────────────────────────────────
@dataclass
class RRFConfig:
    """Reciprocal Rank Fusion parameters."""
    k: int = 60                            # RRF constant
    top_k_final: int = 10                  # Final results to return


# ── VLM Config ───────────────────────────────────────────────
@dataclass
class VLMConfig:
    """Vision-Language Model for answer generation."""
    model_name: str = "Salesforce/blip2-opt-2.7b"
    ollama_model: str = "llava:7b"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    use_ollama: bool = True                # True=Ollama, False=HF BLIP-2
    quantize_bits: int = 8                 # INT8 quantization
    prune_ratio: float = 0.2              # Prune 20% of weights
    max_new_tokens: int = 512


# ── Evaluation Config ────────────────────────────────────────
@dataclass
class EvalConfig:
    """Evaluation and ablation settings."""
    recall_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    num_test_queries: int = 50
    ablation_conditions: List[str] = field(
        default_factory=lambda: [
            "base",                        # CLIP + IVF-Flat + fixed chunking
            "+hnsw",                       # + HNSW index
            "+semantic_chunking",          # + semantic chunking
            "+neo4j_rrf",                  # Full system
        ]
    )
    # Clustering analysis
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    gmm_n_components: int = 5
    isolation_contamination: float = 0.1


# ── Optuna Hyperparameter Search Config ──────────────────────
@dataclass
class OptunaConfig:
    """Automated hyperparameter tuning via Optuna."""
    n_trials: int = 50
    timeout_seconds: int = 3600            # 1 hour max
    sampler: str = "tpe"                   # tpe | random | cmaes
    pruner: str = "median"                 # median | hyperband
    
    # Search ranges
    lr_range: tuple = (1e-6, 1e-3)
    weight_decay_range: tuple = (1e-4, 1e-1)
    lora_rank_range: tuple = (4, 32)
    batch_size_choices: List[int] = field(
        default_factory=lambda: [16, 32, 64]
    )
    temperature_range: tuple = (0.01, 0.2)


# ── Master Config ────────────────────────────────────────────
@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    siamese: SiameseConfig = field(default_factory=SiameseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    colbert: ColBERTConfig = field(default_factory=ColBERTConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rrf: RRFConfig = field(default_factory=RRFConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    
    # Global
    seed: int = 42
    device: torch.device = field(default_factory=get_device)
    log_level: str = "INFO"


def get_config() -> Config:
    """Factory function to create the default configuration."""
    return Config()
