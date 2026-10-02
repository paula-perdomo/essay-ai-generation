"""Supervised Fine-Tuning (SFT) pipeline on human essays using LoRA/QLoRA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class EssaySFTTrainer:
    """Trains a base language model on authentic human essays to internalize human stylometry."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an articulate, engaging essayist. Write with natural cadence, high sentence length variance, "
        "and nuanced argumentation without resorting to formulaic transition markers."
    )

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        output_dir: str = "checkpoints/generator_sft",
        load_in_4bit: bool = True,
        max_seq_length: int = 1536,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        num_epochs: int = 2,
        fp16: bool = True,
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.fp16 = fp16 and torch.cuda.is_available()

        # Tokenizer
        logger.info(f"Loading tokenizer for {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization Config for 8GB VRAM
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading model {self.base_model_name} (4-bit: {load_in_4bit})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=torch.float16,
        )

        if load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA Configuration
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def prepare_human_dataset(self, df: pd.DataFrame) -> Dataset:
        """Filter for human essays and format into conversational chat messages."""
        human_df = df[df["label"] == 0].copy()
        logger.info(f"Preparing SFT dataset with {len(human_df)} human essays...")

        formatted_conversations = []
        for _, row in human_df.iterrows():
            formatted_conversations.append({
                "messages": [
                    {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["text"]},
                ]
            })

        return Dataset.from_list(formatted_conversations)

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Execute SFT training loop."""
        train_ds = self.prepare_human_dataset(train_df)
        val_ds = self.prepare_human_dataset(val_df) if val_df is not None else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch" if val_ds is not None else "no",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            fp16=self.fp16,
            logging_steps=25,
            save_total_limit=2,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=self.lora_config,
            max_seq_length=self.max_seq_length,
            processing_class=self.tokenizer,
        )

        logger.info("Starting SFT training on human essays...")
        trainer.train()

        final_path = self.output_dir / "final_adapter"
        trainer.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.info(f"SFT LoRA adapter saved to {final_path}")
        return trainer
