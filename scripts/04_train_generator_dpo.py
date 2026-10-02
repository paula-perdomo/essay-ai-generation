"""Launch Direct Preference Optimization (DPO) training on the generator model.

Usage:
    python scripts/04_train_generator_dpo.py [--base_model Qwen/Qwen2.5-3B-Instruct] [--max_samples 2000]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.generator.dpo_pipeline import EssayDPOTrainer
from src.utils.logger import setup_logger

logger = setup_logger("train_generator_dpo")


def parse_args():
    parser = argparse.ArgumentParser(description="Train generator model with DPO preference alignment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generator_config.yaml",
        help="Path to generator configuration YAML",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base LLM (e.g., Qwen/Qwen2.5-3B-Instruct or meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Subsample preference pairs for quick iteration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for aligned checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model = args.base_model or cfg["model"].get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    dpo_cfg = cfg.get("dpo_training", {})
    output_dir = args.output_dir or dpo_cfg.get("output_dir", "checkpoints/generator_dpo")

    train_jsonl = "data/processed/dpo_train.jsonl"
    val_jsonl = "data/processed/dpo_val.jsonl"

    if not Path(train_jsonl).exists():
        logger.error(f"DPO dataset not found at {train_jsonl}. Run scripts/01_build_dataset.py first.")
        sys.exit(1)

    logger.info(f"Initializing DPO trainer for {base_model}...")
    trainer = EssayDPOTrainer(
        base_model_name=base_model,
        output_dir=output_dir,
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        beta=float(dpo_cfg.get("beta", 0.1)),
        max_length=dpo_cfg.get("max_length", 1536),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 256),
        batch_size=dpo_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(dpo_cfg.get("learning_rate", 5e-6)),
        num_epochs=dpo_cfg.get("num_train_epochs", 1),
        lora_r=cfg.get("lora", {}).get("r", 16),
        lora_alpha=cfg.get("lora", {}).get("lora_alpha", 32),
    )

    trainer.train(train_jsonl=train_jsonl, val_jsonl=val_jsonl)
    print("\n" + "=" * 50)
    print("DPO ALIGNMENT COMPLETED SUCCESSFULLY")
    print(f"Aligned adapter saved to: {output_dir}/final_adapter")
    print("=" * 50)


if __name__ == "__main__":
    main()
