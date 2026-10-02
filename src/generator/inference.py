"""Essay generation and real-time detection & stylometric verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class EssayGenerator:
    """Generates essays and analyzes them for human-like stylometrics and detection risk."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an insightful and articulate essayist. Write with an engaging, authentic voice. "
        "Vary your sentence structure naturally, blending concise, direct statements with descriptive, "
        "complex sentences. Avoid generic filler transitions like 'In conclusion', 'Moreover', or 'Furthermore'."
    )

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading generator model '{model_name_or_path}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            logger.info(f"Attaching LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if not load_in_4bit:
            self.model.to(self.device)

        self.model.eval()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 800,
        temperature: float = 0.85,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Generate an essay given an assignment prompt."""
        sys_p = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"{sys_p}\n\nUser: {prompt}\n\nAssistant:\n"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Slice off input tokens to retain only newly generated text
        new_tokens = output_tokens[0][inputs["input_ids"].shape[1] :]
        essay = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return essay
