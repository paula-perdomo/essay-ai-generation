"""Direct Preference Optimization (DPO) pipeline to align essay generator away from AI patterns."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


class EssayDPOTrainer:
    """Aligns a generator model using DPO to prefer human stylometry over formulaic AI structures."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        output_dir: str = "checkpoints/generator_dpo",
        load_in_4bit: bool = True,
        beta: float = 0.1,
        max_length: int = 1536,
        max_prompt_length: int = 256,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 5e-6,
        num_epochs: int = 1,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # Tokenizer
        logger.info(f"Loading tokenizer for {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit Quantization Config for local 8GB VRAM execution
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading policy model {self.base_model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
        )

        if load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
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

    def load_preference_dataset(self, jsonl_path: str, max_samples: Optional[int] = None) -> Dataset:
        """Load prompt/chosen/rejected pairs from JSONL."""
        df = pd.read_json(jsonl_path, lines=True)
        if max_samples and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"Loaded {len(df)} preference pairs from {jsonl_path}")
        return Dataset.from_pandas(df[["prompt", "chosen", "rejected"]])

    def train(self, train_jsonl: str, val_jsonl: Optional[str] = None):
        """Run DPO alignment training."""
        train_dataset = self.load_preference_dataset(train_jsonl)
        val_dataset = self.load_preference_dataset(val_jsonl) if val_jsonl else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            beta=self.beta,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            bf16=self.bf16,
            fp16=not self.bf16 and torch.cuda.is_available(),
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            save_total_limit=2,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # PEFT handles reference model automatically with LoRA disabled
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            peft_config=self.lora_config,
        )

        logger.info("Commencing DPO alignment...")
        trainer.train()

        final_path = self.output_dir / "final_adapter"
        trainer.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.info(f"DPO-aligned adapter saved to {final_path}")
        return trainer
