"""Evaluation and diagnostic suite for the AI detector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .model import EssayClassifier

logger = logging.getLogger(__name__)


class DetectorEvaluator:
    """Evaluates detector accuracy and analyzes false positives / false negatives."""

    def __init__(self, classifier: EssayClassifier):
        self.classifier = classifier

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        output_csv: Optional[Union[str, Path]] = None,
    ) -> Dict[str, any]:
        """Run detector inference across a dataset and compute comprehensive metrics."""
        logger.info(f"Evaluating detector on {len(df)} essays...")
        texts = df["text"].tolist()
        ground_truth = df["label"].tolist()

        predictions: List[Dict[str, any]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_preds = self.classifier.predict(batch_texts)
            if isinstance(batch_preds, dict):
                batch_preds = [batch_preds]
            predictions.extend(batch_preds)

        pred_labels = [1 if p["prediction"] == "AI" else 0 for p in predictions]
        ai_probs = [p["ai_prob"] for p in predictions]

        # Calculate metrics
        report = classification_report(
            ground_truth,
            pred_labels,
            target_names=["Human", "AI"],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(ground_truth, pred_labels)
        try:
            auc = float(roc_auc_score(ground_truth, ai_probs))
        except ValueError:
            auc = 0.5

        # Attach predictions to results dataframe
        results_df = df.copy()
        results_df["predicted_label"] = pred_labels
        results_df["predicted_name"] = [p["prediction"] for p in predictions]
        results_df["ai_prob"] = ai_probs
        results_df["human_prob"] = [p["human_prob"] for p in predictions]
        results_df["is_correct"] = results_df["label"] == results_df["predicted_label"]

        if output_csv:
            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_path, index=False)
            logger.info(f"Evaluation predictions saved to {out_path}")

        # False positives (Human essays labeled as AI - high penalty in real world)
        false_positives = results_df[(results_df["label"] == 0) & (results_df["predicted_label"] == 1)]
        # False negatives (AI essays that fooled the detector)
        evaded_ai = results_df[(results_df["label"] == 1) & (results_df["predicted_label"] == 0)]

        summary = {
            "total_essays": len(df),
            "roc_auc": round(auc, 4),
            "accuracy": round(report["accuracy"], 4),
            "human_f1": round(report["Human"]["f1-score"], 4),
            "ai_f1": round(report["AI"]["f1-score"], 4),
            "false_positive_count": len(false_positives),
            "ai_evasion_count": len(evaded_ai),
            "ai_evasion_rate": round(len(evaded_ai) / max(1, sum(df["label"] == 1)), 4),
            "confusion_matrix": cm.tolist(),
        }

        return summary
