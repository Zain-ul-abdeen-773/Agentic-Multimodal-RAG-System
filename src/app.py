"""
Gradio Demo Application — Agentic Multimodal RAG
=================================================
Full demo UI with:
  - Text query tab
  - Image upload + query tab
  - Hybrid query tab
  - Knowledge graph query tab
  - Agent reasoning trace visualization
  - Retrieval result cards
"""

import json
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import gradio as gr
except ImportError:
    gr = None
    logger.warning("gradio not installed. Run: pip install gradio")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.agent.react_agent import AgenticRAG
from src.agent.query_classifier import QueryClassifier
from src.agent.tools import ToolRegistry


def create_app(tool_registry: Optional[ToolRegistry] = None):
    """
    Create the Gradio demo application.
    
    Args:
        tool_registry: Initialized ToolRegistry with all modules
    Returns:
        Gradio Blocks application
    """
    if gr is None:
        raise ImportError("gradio required: pip install gradio")
    
    config = get_config()
    
    # Initialize agent
    agent = AgenticRAG(
        config=config.agent,
        image_searcher=tool_registry.image_store if tool_registry else None,
        text_searcher=tool_registry.bm25 if tool_registry else None,
        graph_querier=tool_registry.graph_tool if tool_registry else None,
        hybrid_searcher=(
            tool_registry.hybrid_search if tool_registry else None
        ),
    )
    
    # Query classifier
    try:
        classifier = QueryClassifier(config=config.agent)
    except Exception:
        classifier = None
    
    def process_query(
        query: str,
        image=None,
        query_type: str = "auto",
    ) -> Tuple[str, str, str]:
        """
        Process a user query through the full pipeline.
        
        Returns: (answer, reasoning_trace, query_classification)
        """
        if not query.strip():
            return "Please enter a query.", "", ""
        
        # Classify query
        classification = ""
        if classifier is not None and query_type == "auto":
            cls_result = classifier._heuristic_classify(query)
            classification = json.dumps(cls_result, indent=2)
            logger.info(f"Query classified as: {cls_result['type']}")
        
        # Run agent
        result = agent.invoke(query)
        
        answer = result.get("answer", "No answer generated")
        trace = "\n---\n".join(result.get("reasoning_trace", []))
        tools = ", ".join(result.get("tools_used", []))
        
        reasoning = f"**Tools used:** {tools}\n\n**Trace:**\n{trace}"
        
        return answer, reasoning, classification
    
    # ── Build Gradio UI ──────────────────────────────────────
    with gr.Blocks(
        title="Agentic Multimodal RAG",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
    ) as app:
        gr.Markdown(
            "# 🧠 Agentic Multimodal RAG System\n"
            "**CLIP · FAISS (HNSW) · LangGraph · Neo4j · BLIP-2/LLaVA**\n\n"
            "Query images, documents, and knowledge graphs using natural language. "
            "An intelligent agent selects the best retrieval strategy automatically."
        )
        
        with gr.Tabs():
            # ── Text Query Tab ──
            with gr.Tab("💬 Text Query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Your Query",
                            placeholder="e.g., Show me images of surface cracks near welding joints",
                            lines=3,
                        )
                        text_btn = gr.Button("🔍 Search", variant="primary")
                    with gr.Column(scale=1):
                        query_type_dropdown = gr.Dropdown(
                            choices=["auto", "visual", "textual", "hybrid", "graph"],
                            value="auto",
                            label="Query Type",
                        )
                
                with gr.Row():
                    text_answer = gr.Markdown(label="Answer")
                
                with gr.Accordion("🔧 Agent Reasoning Trace", open=False):
                    text_trace = gr.Markdown()
                
                with gr.Accordion("📊 Query Classification", open=False):
                    text_classification = gr.Code(language="json")
                
                text_btn.click(
                    process_query,
                    inputs=[text_input, gr.State(None), query_type_dropdown],
                    outputs=[text_answer, text_trace, text_classification],
                )
            
            # ── Image + Query Tab ──
            with gr.Tab("🖼️ Image + Query"):
                with gr.Row():
                    img_input = gr.Image(label="Upload Image", type="filepath")
                    with gr.Column():
                        img_query = gr.Textbox(
                            label="Query about this image",
                            placeholder="What defects are visible in this image?",
                            lines=2,
                        )
                        img_btn = gr.Button("🔍 Analyze", variant="primary")
                
                img_answer = gr.Markdown(label="Answer")
                
                with gr.Accordion("🔧 Reasoning Trace", open=False):
                    img_trace = gr.Markdown()
                
                img_btn.click(
                    process_query,
                    inputs=[img_query, img_input, gr.State("visual")],
                    outputs=[img_answer, img_trace, gr.State("")],
                )
            
            # ── Knowledge Graph Tab ──
            with gr.Tab("🕸️ Knowledge Graph"):
                with gr.Row():
                    graph_input = gr.Textbox(
                        label="Graph Query",
                        placeholder="Who inspected the welding joints?",
                        lines=2,
                    )
                    graph_btn = gr.Button("🔍 Query Graph", variant="primary")
                
                graph_answer = gr.Markdown(label="Answer")
                
                with gr.Accordion("🔧 Reasoning Trace", open=False):
                    graph_trace = gr.Markdown()
                
                graph_btn.click(
                    process_query,
                    inputs=[graph_input, gr.State(None), gr.State("graph")],
                    outputs=[graph_answer, graph_trace, gr.State("")],
                )
            
            # ── System Info Tab ──
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown(
                    "## Architecture\n"
                    "- **Embedding**: CLIP ViT-B/32 + LoRA fine-tuning\n"
                    "- **Vector Store**: FAISS HNSW (M=32, efSearch=128)\n"
                    "- **Sparse Retrieval**: BM25 (Okapi)\n"
                    "- **Reranker**: ColBERT (DistilBERT)\n"
                    "- **Knowledge Graph**: Neo4j (Cypher)\n"
                    "- **Fusion**: Reciprocal Rank Fusion (k=60)\n"
                    "- **Agent**: LangGraph ReAct (4 tools)\n"
                    "- **VLM**: BLIP-2 / LLaVA (Ollama)\n"
                    "- **LLM**: Groq (LLaMA-3-70B)\n\n"
                    "## Techniques (50+)\n"
                    "ArcFace, LoRA, Cosine Annealing LR, AdamW, "
                    "Gradient Accumulation, Label Smoothing, Siamese Networks, "
                    "Hard-Negative Mining, Optuna HPO, Semantic Chunking, "
                    "BiLSTM Boundaries, GloVE, BM25, ColBERT MaxSim, "
                    "HNSW, PCA, ONNX Export, GraphSAGE, RRF, "
                    "DBSCAN, GMM, Isolation Forest, BERTScore, "
                    "and many more..."
                )
    
    return app


def main():
    """Launch the Gradio application."""
    # Initialize with empty registry (modules loaded separately)
    registry = ToolRegistry()
    app = create_app(registry)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
