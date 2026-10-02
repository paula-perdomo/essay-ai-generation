"""Train and evaluate the AI vs Human Essay Detector (Discriminator).

Usage:
    python scripts/03_train_detector.py [--config configs/detector_config.yaml] [--max_train 5000] [--epochs 2]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from src.detector.evaluate import DetectorEvaluator
from src.detector.model import EssayClassifier
from src.detector.trainer import DetectorTrainer
from src.utils.logger import setup_logger

logger = setup_logger("train_detector")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI vs Human Essay Detector.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detector_config.yaml",
        help="Path to detector config YAML",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Transformer model identifier (e.g. microsoft/deberta-v3-base or roberta-base)",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Limit number of training samples for fast iteration",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=None,
        help="Limit number of validation samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per-device batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"].get("name", "microsoft/deberta-v3-base")
    max_length = cfg["model"].get("max_length", 512)

    train_cfg = cfg.get("training", {})
    output_dir = train_cfg.get("output_dir", "checkpoints/detector")
    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    epochs = args.epochs or train_cfg.get("num_epochs", 3)
    learning_rate = float(train_cfg.get("learning_rate", 2e-5))
    fp16 = train_cfg.get("fp16", True)

    data_cfg = cfg.get("data", {})
    train_path = Path(data_cfg.get("train_path", "data/processed/train.parquet"))
    val_path = Path(data_cfg.get("val_path", "data/processed/val.parquet"))
    test_path = Path(data_cfg.get("test_path", "data/processed/test.parquet"))

    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}. Run scripts/01_build_dataset.py first.")
        sys.exit(1)

    logger.info(f"Loading datasets from {train_path.parent}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path) if test_path.exists() else None

    # Subsample if requested for fast benchmarking
    max_train = args.max_train or data_cfg.get("max_train_samples")
    if max_train and max_train < len(train_df):
        logger.info(f"Subsampling training set to {max_train} balanced samples.")
        human_sub = train_df[train_df["label"] == 0].sample(n=max_train // 2, random_state=42)
        ai_sub = train_df[train_df["label"] == 1].sample(n=max_train // 2, random_state=42)
        train_df = pd.concat([human_sub, ai_sub]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    max_val = args.max_val or data_cfg.get("max_val_samples")
    if max_val and max_val < len(val_df):
        val_df = val_df.sample(n=max_val, random_state=42).reset_index(drop=True)

    logger.info(f"Training records: {len(train_df)} | Validation records: {len(val_df)}")

    # Initialize Trainer
    trainer_obj = DetectorTrainer(
        model_name=model_name,
        output_dir=output_dir,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=learning_rate,
        num_epochs=epochs,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=fp16,
    )

    trainer = trainer_obj.train(train_df, val_df)

    # Evaluate on Test Set
    if test_df is not None:
        best_model_dir = Path(output_dir) / "best_model"
        logger.info(f"Running final evaluation on test set ({len(test_df)} samples) using {best_model_dir}...")
        best_classifier = EssayClassifier(model_name_or_path=str(best_model_dir), max_length=max_length)
        evaluator = DetectorEvaluator(best_classifier)

        test_sample = test_df.sample(n=min(len(test_df), 1500), random_state=42).reset_index(drop=True)
        results = evaluator.evaluate_dataset(
            test_sample,
            output_csv=Path(output_dir) / "test_predictions.csv",
        )

        results_path = Path(output_dir) / "test_evaluation.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 50)
        print("TEST SET EVALUATION RESULTS")
        print("=" * 50)
        for k, v in results.items():
            if k != "confusion_matrix":
                print(f"  {k:22s}: {v}")
        print("=" * 50)


if __name__ == "__main__":
    main()
