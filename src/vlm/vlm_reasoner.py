"""
VLM Reasoner — Vision-Language Model for Answer Generation
==========================================================
Techniques implemented:
  - BLIP-2 integration via HuggingFace Transformers
  - LLaVA integration via Ollama local server
  - Model pruning for inference speedup
  - INT8 quantization for deployment
  - BERTScore evaluation against reference answers
  - Log-cosh loss for answer quality regression
  - Context window construction from retrieved results
"""

import torch
import torch.nn as nn
import requests
import base64
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger
from PIL import Image

try:
    from transformers import (
        Blip2Processor,
        Blip2ForConditionalGeneration,
        BitsAndBytesConfig,
    )
except ImportError:
    Blip2Processor, Blip2ForConditionalGeneration = None, None
    BitsAndBytesConfig = None

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE, VLMConfig


# ═══════════════════════════════════════════════════════════════
# Log-Cosh Loss (smooth L1 approximation)
# ═══════════════════════════════════════════════════════════════
class LogCoshLoss(nn.Module):
    """
    Log-cosh loss: L = log(cosh(y_pred - y_true))
    
    Behaves like L2 for small errors, L1 for large errors.
    Smoother than Huber loss, twice differentiable everywhere.
    Useful for regression tasks (e.g., answer quality scoring).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


# ═══════════════════════════════════════════════════════════════
# Model Pruning Utilities
# ═══════════════════════════════════════════════════════════════
class ModelPruner:
    """
    Structured and unstructured pruning for neural networks.
    Reduces model size and inference cost.
    """

    @staticmethod
    def prune_model(
        model: nn.Module,
        prune_ratio: float = 0.2,
        method: str = "l1_unstructured",
    ) -> nn.Module:
        """
        Apply magnitude-based pruning to linear layers.
        
        Args:
            model: Model to prune
            prune_ratio: Fraction of weights to zero out
            method: Pruning method (l1_unstructured, random)
        Returns:
            Pruned model (in-place)
        """
        import torch.nn.utils.prune as prune
        
        pruned_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if method == "l1_unstructured":
                    prune.l1_unstructured(
                        module, name="weight", amount=prune_ratio
                    )
                elif method == "random":
                    prune.random_unstructured(
                        module, name="weight", amount=prune_ratio
                    )
                
                # Make pruning permanent
                prune.remove(module, "weight")
                pruned_count += 1
        
        logger.info(
            f"Pruned {pruned_count} layers at {prune_ratio*100:.0f}% sparsity"
        )
        return model


# ═══════════════════════════════════════════════════════════════
# VLM Reasoner
# ═══════════════════════════════════════════════════════════════
class VLMReasoner:
    """
    Vision-Language Model for synthesizing final answers from
    retrieved multi-modal context.
    
    Supports:
      - BLIP-2 (HuggingFace, local GPU)
      - LLaVA (Ollama, local CPU/GPU server)
      
    The VLM receives:
      1. Retrieved images
      2. Retrieved text passages
      3. Knowledge graph context
      4. User query
    And generates a grounded, evidence-backed natural language answer.
    """

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self.blip_model = None
        self.blip_processor = None
        self.log_cosh_loss = LogCoshLoss()
        
        logger.info(
            f"VLMReasoner initialized | Mode: "
            f"{'Ollama' if self.config.use_ollama else 'BLIP-2'}"
        )

    def load_blip2(self):
        """Load BLIP-2 model with optional quantization."""
        if Blip2Processor is None:
            raise ImportError("transformers required for BLIP-2")
        
        self.blip_processor = Blip2Processor.from_pretrained(
            self.config.model_name
        )
        
        # INT8 quantization
        quantization_config = None
        if self.config.quantize_bits == 8 and BitsAndBytesConfig is not None:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        load_kwargs = {"device_map": "auto"}
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        
        try:
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model_name, **load_kwargs
            )
        except Exception as e:
            logger.warning(f"BLIP-2 quantized load failed ({e}), loading in float32")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model_name
            ).to(DEVICE)
        
        # Apply pruning if configured
        if self.config.prune_ratio > 0:
            ModelPruner.prune_model(
                self.blip_model, self.config.prune_ratio
            )
        
        logger.info(f"BLIP-2 loaded: {self.config.model_name}")

    def _build_context(
        self,
        query: str,
        retrieved_results: List[Dict],
        graph_context: str = "",
    ) -> str:
        """
        Construct context window from retrieved results.
        """
        context_parts = []
        
        # Text results
        text_results = [
            r for r in retrieved_results
            if r.get("retriever") in ("bm25", "faiss_hnsw_text", "rrf_fusion")
            and "text" in r
        ]
        if text_results:
            context_parts.append("=== Retrieved Text ===")
            for i, r in enumerate(text_results[:5], 1):
                context_parts.append(f"[{i}] {r['text'][:500]}")
        
        # Image captions
        image_results = [
            r for r in retrieved_results if "caption" in r
        ]
        if image_results:
            context_parts.append("\n=== Image Descriptions ===")
            for i, r in enumerate(image_results[:5], 1):
                context_parts.append(
                    f"[Image {i}] {r.get('caption', 'No caption')}"
                )
        
        # Graph context
        if graph_context:
            context_parts.append(f"\n=== Knowledge Graph ===\n{graph_context}")
        
        context = "\n".join(context_parts)
        
        prompt = (
            f"Based on the following retrieved evidence, answer the query.\n\n"
            f"Query: {query}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Provide a comprehensive, grounded answer citing the evidence."
        )
        
        return prompt

    def generate_blip2(
        self,
        image: Optional[Image.Image],
        prompt: str,
    ) -> str:
        """Generate answer using BLIP-2."""
        if self.blip_model is None:
            self.load_blip2()
        
        inputs = self.blip_processor(
            images=image, text=prompt, return_tensors="pt"
        )
        
        # Move to device
        device = next(self.blip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.blip_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=3,
                temperature=0.7,
            )
        
        answer = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return answer

    def generate_ollama(
        self,
        image_path: Optional[str],
        prompt: str,
    ) -> str:
        """Generate answer using Ollama (LLaVA/local VLM)."""
        url = f"{self.config.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": self.config.max_new_tokens,
            },
        }
        
        # Attach image if provided
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            payload["images"] = [img_b64]
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response generated")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"VLM generation failed: {e}"

    def generate(
        self,
        query: str,
        retrieved_results: List[Dict],
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        graph_context: str = "",
    ) -> str:
        """
        Generate an answer using the VLM.
        
        Args:
            query: User query
            retrieved_results: Results from retrieval pipeline
            image: PIL Image (for BLIP-2)
            image_path: Path to image file (for Ollama)
            graph_context: Text from knowledge graph
        Returns:
            Generated answer string
        """
        prompt = self._build_context(query, retrieved_results, graph_context)
        
        if self.config.use_ollama:
            return self.generate_ollama(image_path, prompt)
        else:
            return self.generate_blip2(image, prompt)

    def evaluate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate VLM answer quality using BERTScore F1.
        
        Measures semantic similarity between generated and reference answers.
        """
        if bert_score_fn is None:
            logger.warning("bert_score not installed")
            return {"error": "bert_score not installed"}
        
        P, R, F1 = bert_score_fn(
            predictions, references,
            lang="en", verbose=False,
        )
        
        results = {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
        
        logger.info(f"BERTScore F1: {results['bertscore_f1']:.4f}")
        return results
