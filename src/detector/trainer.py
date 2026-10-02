"""Trainer for the AI vs Human Essay Classifier using Hugging Face Trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """Compute standard classification evaluation metrics including ROC-AUC and F1."""
    logits, labels = eval_pred
    # Apply softmax to obtain probability of AI class (class 1)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    ai_probs = probs[:, 1]
    preds = np.argmax(logits, axis=1)

    try:
        auc = roc_auc_score(labels, ai_probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary")),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
        "roc_auc": float(auc),
    }


class DetectorTrainer:
    """Trains and evaluates transformer encoder models for AI essay detection."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        output_dir: str = "checkpoints/detector",
        max_length: int = 512,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        fp16: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.fp16 = fp16 and torch.cuda.is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "Human", 1: "AI"},
            label2id={"Human": 0, "AI": 1},
        )

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Tokenize dataframe and format as Hugging Face Dataset."""
        ds = Dataset.from_pandas(df[["text", "label"]])

        def tokenize_batch(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_length,
            )

        tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
        return tokenized

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        save_best_model: bool = True,
    ) -> Trainer:
        """Execute training loop with validation evaluation."""
        logger.info(f"Tokenizing {len(train_df)} training and {len(val_df)} validation examples...")
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            load_best_model_at_end=save_best_model,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=2,
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_classification_metrics,
        )

        logger.info("Commencing detector fine-tuning...")
        trainer.train()

        final_model_dir = self.output_dir / "best_model"
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        logger.info(f"Best detector model saved to {final_model_dir}")

        return trainer
