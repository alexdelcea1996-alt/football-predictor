"""
Evaluation report generator.
"""

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from football_predictor.evaluation.metrics import calculate_all_metrics


class EvaluationReporter:
    """
    Generates comprehensive evaluation reports.
    """
    
    def __init__(self) -> None:
        self.results: dict[str, Any] = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        dataset_name: str = "test",
    ) -> dict[str, Any]:
        """Run evaluation and store results."""
        metrics = calculate_all_metrics(y_true, y_proba)
        
        # Confusion matrix
        y_pred = np.argmax(y_proba, axis=1)
        labels = ["Home Win", "Draw", "Away Win"]
        
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        
        # Outcome distribution
        outcome_dist = {
            "actual": {
                labels[i]: int((y_true == i).sum()) for i in range(3)
            },
            "predicted": {
                labels[i]: int((y_pred == i).sum()) for i in range(3)
            },
        }
        
        self.results[dataset_name] = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "outcome_distribution": outcome_dist,
            "n_samples": len(y_true),
        }
        
        return self.results[dataset_name]
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = ["=" * 50, "EVALUATION SUMMARY", "=" * 50]
        
        for name, result in self.results.items():
            m = result["metrics"]
            lines.extend([
                f"\n{name.upper()} SET ({result['n_samples']} samples)",
                "-" * 30,
                f"  RPS:        {m['rps']:.4f} (target: < 0.22)",
                f"  Accuracy:   {m['accuracy']:.2%} (target: 52-56%)",
                f"  Log Loss:   {m['log_loss']:.4f}",
                f"  ECE:        {m['ece']:.4f}",
                f"  F1 (macro): {m['f1_macro']:.4f}",
            ])
        
        return "\n".join(lines)
    
    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        rows = []
        for name, result in self.results.items():
            row = {"dataset": name, **result["metrics"]}
            rows.append(row)
        return pd.DataFrame(rows)
