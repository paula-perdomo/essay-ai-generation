"""Generate essays and score them against the trained detector and stylometric benchmarks.

Usage:
    python scripts/05_generate_and_evaluate.py [--prompt "Should driverless cars be allowed?"] [--adapter checkpoints/generator_dpo/final_adapter]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stylometrics import StylometricAnalyzer
from src.detector.model import EssayClassifier
from src.generator.inference import EssayGenerator
from src.utils.logger import setup_logger

logger = setup_logger("generate_and_evaluate")

DEFAULT_PROMPT = (
    "Write an argumentative essay analyzing whether cities should implement car-free zones "
    "or restrict automobile traffic to reduce congestion, curb pollution, and promote public transit."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate humanized essays.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Essay prompt / topic",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base LLM generator",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional path to trained LoRA adapter (e.g., checkpoints/generator_dpo/final_adapter)",
    )
    parser.add_argument(
        "--detector_dir",
        type=str,
        default="checkpoints/detector/best_model",
        help="Path to trained detector model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=700,
        help="Max generated tokens",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Generate Essay
    logger.info(f"Generating essay with prompt: '{args.prompt[:80]}...'")
    generator = EssayGenerator(
        model_name_or_path=args.base_model,
        adapter_path=args.adapter,
        load_in_4bit=True,
    )
    essay_text = generator.generate(
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    print("\n" + "=" * 65)
    print("GENERATED ESSAY")
    print("=" * 65)
    print(essay_text)
    print("=" * 65)

    # Step 2: Stylometric Evaluation
    analyzer = StylometricAnalyzer()
    metrics = analyzer.extract_features(essay_text)

    # Step 3: Run Against Detector
    detector_path = Path(args.detector_dir)
    detection_result = None
    if detector_path.exists():
        logger.info(f"Scoring essay with trained detector: {detector_path}...")
        detector = EssayClassifier(model_name_or_path=str(detector_path))
        detection_result = detector.predict(essay_text)

    # Print Scorecard
    print("\n" + "=" * 65)
    print("EVALUATION & VERIFICATION SCORECARD")
    print("=" * 65)
    if detection_result:
        verdict = detection_result["prediction"]
        p_human = detection_result["human_prob"] * 100
        p_ai = detection_result["ai_prob"] * 100
        print(f"  AI Detector Verdict  : {verdict.upper()} (Human: {p_human:.1f}%, AI: {p_ai:.1f}%)")
    else:
        print("  AI Detector Verdict  : [No detector checkpoint found at specified path]")

    print(f"  Word Count           : {metrics['word_count']}")
    print(f"  Avg Sentence Length  : {metrics['avg_sentence_len']} words (Human baseline: ~28.2 words)")
    print(f"  Sentence Variance    : {metrics['std_sentence_len']} words")
    print(f"  Burstiness (std/mean): {metrics['burstiness']} (Human baseline: ~0.47, AI baseline: ~0.35)")
    print(f"  AI Cliché Density    : {metrics['ai_marker_per_1000w']} per 1k words (Human baseline: ~0.6, AI baseline: ~2.5)")
    print(f"  Comma Frequency      : {metrics['comma_frequency_per_100w']} per 100 words (Human baseline: ~3.3, AI baseline: ~6.0)")
    print(f"  Flesch Reading Ease  : {metrics['flesch_reading_ease']} (Human baseline: ~62.5)")
    print("=" * 65)


if __name__ == "__main__":
    main()
