"""
Agentic Multimodal RAG System
==============================
AI-341 Deep Neural Networks | GIK Institute
CLIP · FAISS (HNSW) · LangGraph · Neo4j · BLIP-2/LLaVA · Groq · Gradio
"""

__version__ = "0.1.0"
__authors__ = ["Muhammad Hashir Awaiz (2023429)", "Zain ul Abdeen (2023773)"]

from src.config import get_config, Config, DEVICE

__all__ = ["get_config", "Config", "DEVICE"]
