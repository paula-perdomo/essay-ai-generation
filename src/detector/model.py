"""Sequence classification model wrapper for AI vs Human essay detection."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class EssayClassifier:
    """Discriminator model to classify essays as Human (0) or AI (1)."""

    ID2LABEL = {0: "Human", 1: "AI"}
    LABEL2ID = {"Human": 0, "AI": 1}

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name = model_name_or_path
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading classifier '{self.model_name}' on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=self.ID2LABEL,
            label2id=self.LABEL2ID,
        )
        self.model.to(self.device)

    def tokenize(self, texts: Union[str, List[str]], padding: bool = True):
        """Tokenize single text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Union[float, str]]]:
        """Predict human vs AI probability for one or multiple essays."""
        self.model.eval()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        encoded = self.tokenize(texts).to(self.device)
        outputs = self.model(**encoded)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for prob in probs:
            p_human = float(prob[0])
            p_ai = float(prob[1])
            pred_label = "AI" if p_ai >= 0.5 else "Human"

            results.append({
                "human_prob": round(p_human, 4),
                "ai_prob": round(p_ai, 4),
                "prediction": pred_label,
                "confidence": round(max(p_human, p_ai), 4),
            })

        return results[0] if is_single else results

    def save(self, save_directory: str):
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")
