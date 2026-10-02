"""Analyze and contrast stylometric features between human and AI essays.

Usage:
    python scripts/02_analyze_stylometrics.py [--input data/processed/train.parquet] [--sample_size 1000]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.stylometrics import StylometricAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger("stylometric_analysis")


def parse_args():
    parser = argparse.ArgumentParser(description="Stylometric analysis of human vs AI essays.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/train.parquet",
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of essays per class to analyze for efficiency",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}. Please run scripts/01_build_dataset.py first.")
        sys.exit(1)

    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)

    # Sample balanced set for fair comparison
    human_df = df[df["label"] == 0]
    ai_df = df[df["label"] == 1]

    sample_n = min(len(human_df), len(ai_df), args.sample_size)
    sampled = pd.concat([
        human_df.sample(n=sample_n, random_state=42),
        ai_df.sample(n=sample_n, random_state=42),
    ]).reset_index(drop=True)

    analyzer = StylometricAnalyzer()
    df_features = analyzer.analyze_dataframe(sampled)

    summary = analyzer.summarize_contrast(df_features, label_col="label_name")
    print("\n" + "=" * 65)
    print("STYLOMETRIC PROFILE: HUMAN vs AI ESSAYS")
    print("=" * 65)
    print(summary.to_string())
    print("=" * 65)

    out_csv = input_path.parent / "stylometric_benchmark.csv"
    summary.to_csv(out_csv)
    logger.info(f"Detailed stylometric summary saved to {out_csv}")


if __name__ == "__main__":
    main()
