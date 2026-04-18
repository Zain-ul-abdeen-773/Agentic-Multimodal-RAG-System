"""
Ablation Study Automation
=========================
Automates the 4-condition ablation study:
  1. Base: CLIP + IVF-Flat + fixed chunking
  2. +HNSW: Replace IVF-Flat with HNSW
  3. +Semantic Chunking: Add semantic chunking
  4. +Neo4j + RRF: Full system with KG and fusion
"""

import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, EvalConfig
from src.evaluation.evaluator import Evaluator


class AblationStudy:
    """
    Runs ablation experiments isolating the contribution
    of each system component.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.evaluator = Evaluator(config)
        self.results: Dict[str, Dict] = {}

    def run_condition(
        self,
        condition_name: str,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run evaluation for a single ablation condition."""
        logger.info(f"Running ablation condition: {condition_name}")
        
        results = {
            "condition": condition_name,
            "recall": self.evaluator.compute_all_recall(
                retrieved_ids, relevant_ids
            ),
            "mrr": self.evaluator.mrr(retrieved_ids, relevant_ids),
        }
        
        if embeddings is not None and len(embeddings) > 5:
            results["gmm"] = self.evaluator.gmm_analysis(embeddings)
        
        self.results[condition_name] = results
        return results

    def generate_latex_table(self) -> str:
        """
        Generate a LaTeX table from ablation results.
        Suitable for direct inclusion in the report.
        """
        if not self.results:
            return "% No ablation results available"
        
        header = (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\caption{Ablation Study Results}\n"
            "\\label{tab:ablation}\n"
            "\\begin{tabular}{|l|c|c|c|c|}\n"
            "\\hline\n"
            "\\textbf{Condition} & \\textbf{Recall@1} & "
            "\\textbf{Recall@5} & \\textbf{Recall@10} & "
            "\\textbf{MRR} \\\\\n"
            "\\hline\n"
        )
        
        rows = []
        for name, result in self.results.items():
            recall = result.get("recall", {})
            mrr = result.get("mrr", 0)
            row = (
                f"{name} & "
                f"{recall.get('recall@1', 0):.3f} & "
                f"{recall.get('recall@5', 0):.3f} & "
                f"{recall.get('recall@10', 0):.3f} & "
                f"{mrr:.3f} \\\\"
            )
            rows.append(row)
        
        footer = (
            "\n\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        
        return header + "\n".join(rows) + footer

    def save_results(self, save_dir: str):
        """Save ablation results to disk."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # JSON results
        with open(f"{save_dir}/ablation_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # LaTeX table
        latex = self.generate_latex_table()
        with open(f"{save_dir}/ablation_table.tex", "w") as f:
            f.write(latex)
        
        logger.info(f"Ablation results saved to {save_dir}")
