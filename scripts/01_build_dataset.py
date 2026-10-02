"""Executable script to load, preprocess, split, and build DPO pairs from raw essay datasets.

Usage:
    python scripts/01_build_dataset.py [--config configs/data_config.yaml] [--max_samples 5000]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.dpo_formatter import DPOPreferenceBuilder
from src.data.loader import EssayDataLoader
from src.data.preprocessor import EssayPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("build_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Build and preprocess essay dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default=None,
        help="Optional local CSV/Parquet file to load instead of HuggingFace",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of raw samples for fast testing/iteration",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        cfg = {
            "dataset": {"hf_repo": "Yunij/kaggle-comp-daigt", "output_dir": "data/processed"},
            "preprocessing": {"min_words": 80, "max_words": 2500, "test_size": 0.15, "val_size": 0.10},
            "dpo": {"pairs_per_prompt": 1500, "max_length_ratio_diff": 0.35},
        }

    # Step 1: Load Data
    loader = EssayDataLoader(cache_dir=cfg["dataset"].get("cache_dir", "data/raw"))
    if args.source_file:
        df_raw = loader.load_from_file(args.source_file)
    else:
        df_raw = loader.load_from_huggingface(
            dataset_name=cfg["dataset"].get("hf_repo", "Yunij/kaggle-comp-daigt"),
            max_samples=args.max_samples,
        )

    logger.info(f"Loaded raw dataset with {len(df_raw)} records.")
    logger.info(f"Class breakdown:\n{df_raw['label_name'].value_counts().to_string()}")

    # Step 2: Clean & Preprocess
    prep_cfg = cfg.get("preprocessing", {})
    preprocessor = EssayPreprocessor(
        min_words=prep_cfg.get("min_words", 80),
        max_words=prep_cfg.get("max_words", 2500),
        remove_duplicates=prep_cfg.get("remove_duplicates", True),
    )
    df_clean = preprocessor.process_dataframe(df_raw)

    # Step 3: Split Dataset
    train_df, val_df, test_df = preprocessor.create_splits(
        df_clean,
        test_size=prep_cfg.get("test_size", 0.15),
        val_size=prep_cfg.get("val_size", 0.10),
        split_by_prompt=prep_cfg.get("split_by_prompt", False),
        random_state=prep_cfg.get("random_state", 42),
    )

    output_dir = Path(cfg["dataset"].get("output_dir", "data/processed"))
    split_paths = preprocessor.save_splits(train_df, val_df, test_df, output_dir=output_dir)

    # Step 4: Build DPO Preference Pairs for Generator Training
    dpo_cfg = cfg.get("dpo", {})
    dpo_builder = DPOPreferenceBuilder(
        max_length_ratio_diff=dpo_cfg.get("max_length_ratio_diff", 0.35)
    )

    train_pairs = dpo_builder.build_preference_pairs(
        train_df,
        pairs_per_prompt=dpo_cfg.get("pairs_per_prompt", 1500),
    )
    val_pairs = dpo_builder.build_preference_pairs(
        val_df,
        pairs_per_prompt=max(50, dpo_cfg.get("pairs_per_prompt", 1500) // 10),
    )

    dpo_train_path = output_dir / "dpo_train.jsonl"
    dpo_val_path = output_dir / "dpo_val.jsonl"
    dpo_builder.save_preference_dataset(train_pairs, dpo_train_path)
    dpo_builder.save_preference_dataset(val_pairs, dpo_val_path)

    # Summary Statistics
    summary = {
        "total_raw": len(df_raw),
        "total_clean": len(df_clean),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_human_count": int((train_df["label"] == 0).sum()),
        "train_ai_count": int((train_df["label"] == 1).sum()),
        "val_human_count": int((val_df["label"] == 0).sum()),
        "val_ai_count": int((val_df["label"] == 1).sum()),
        "test_human_count": int((test_df["label"] == 0).sum()),
        "test_ai_count": int((test_df["label"] == 1).sum()),
        "dpo_train_pairs": len(train_pairs),
        "dpo_val_pairs": len(val_pairs),
        "unique_prompts": int(df_clean["prompt_name"].nunique()),
    }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset summary written to {summary_path}")
    print("\n" + "=" * 50)
    print("DATASET PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
