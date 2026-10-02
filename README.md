# Humanized Essay AI Generation & Stylometric Alignment

An end-to-end framework to train language models that produce human-like argumentative essays and minimize detection by AI verification systems (e.g. GPTZero, Turnitin, DeBERTa-based classifiers).

---

## Architecture Overview

Traditional AI detectors distinguish LLM-generated text from human writing through two core mechanisms:
1. **Neural Representation Classifiers**: Transformer encoders (such as `DeBERTa-v3-base`) fine-tuned on paired human/AI corpora.
2. **Stylometric and Statistical Signals**:
   - **Perplexity & Burstiness**: Humans write with high variance in sentence length and structure (high burstiness); AI models generate predictable, uniform cadences (low burstiness).
   - **Cliché Density & Transition Markers**: LLMs disproportionately rely on formulaic transitions (`"In conclusion"`, `"Moreover"`, `"Furthermore"`, `"delve into"`, `"a testament to"`, `"rich tapestry"`).
   - **Punctuation Profiles**: AI text exhibits near-double the comma density due to mechanical subordinate clause balancing.

### The Alignment Pipeline

```
[PERSUADE & DAIGT Corpora]
          │
          ├──► Data Preprocessing & Length Filtering
          │
          ├──► 1. Binary Classifier Detector (DeBERTa-v3) ──► Objective Verification & Reward
          │
          ├──► 2. Supervised Fine-Tuning (SFT) on Human Essays (Llama-3.2-3B / Qwen2.5-3B)
          │
          └──► 3. Direct Preference Optimization (DPO)
                    • Prompt: Standardized Essay Assignment
                    • Chosen: Authentic Human Essay
                    • Rejected: Standard LLM Output
```

---

## Empirical Stylometric Benchmark

Computed across a balanced sample of human vs. AI essays from the dataset:

| Stylometric Metric | AI Essays (Mean ± Std) | Human Essays (Mean ± Std) | Distinction Factor |
| :--- | :--- | :--- | :--- |
| **Burstiness ($\sigma / \mu$)** | **0.357** ± 0.12 | **0.472** ± 0.16 | **+32% higher variance in humans** |
| **AI Cliché Density (/1000w)** | **2.49** ± 2.58 | **0.61** ± 1.29 | **>4x higher frequency in AI** |
| **Comma Frequency (/100w)** | **6.03** ± 2.36 | **3.29** ± 2.02 | **AI uses nearly double commas** |
| **Avg Sentence Length** | 20.7 words | 28.2 words | Humans write longer, more complex sentences |
| **Flesch Reading Ease** | 49.1 (Dense) | 62.5 (Standard) | AI leans toward overly elevated vocabulary |

---

## Repository Structure

```
essay-ai-generation/
├── configs/
│   ├── data_config.yaml         # Dataset loading, cleaning, & DPO pairing parameters
│   ├── detector_config.yaml     # DeBERTa-v3 classifier training settings
│   └── generator_config.yaml    # LoRA / QLoRA, SFT, and DPO hyperparameters
├── data/
│   ├── raw/                     # Cached raw datasets
│   └── processed/
│       ├── train.parquet        # 33,634 training essays (Human + AI)
│       ├── val.parquet          # 4,485 validation essays
│       ├── test.parquet         # 6,728 held-out test essays
│       ├── dpo_train.jsonl      # 19,321 prompt-matched preference pairs
│       ├── dpo_val.jsonl        # 2,192 validation preference pairs
│       ├── dataset_summary.json # Dataset statistics
│       └── stylometric_benchmark.csv # Extracted linguistic features
├── scripts/
│   ├── 01_build_dataset.py      # Automated download, clean, split, and DPO pair builder
│   ├── 02_analyze_stylometrics.py # Computes burstiness & cliché metrics
│   └── 03_train_detector.py     # Trains the DeBERTa-v3 binary discriminator
├── src/
│   ├── data/
│   │   ├── loader.py            # HuggingFace & local CSV/Parquet loader
│   │   ├── preprocessor.py      # Unicode normalization & prompt-stratified splitter
│   │   ├── stylometrics.py      # Burstiness & linguistic feature extractor
│   │   └── dpo_formatter.py     # Prompt-matched DPO pair builder
│   ├── detector/
│   │   ├── model.py             # EssayClassifier wrapper (Softmax, predict_proba)
│   │   ├── trainer.py           # HF Trainer with ROC-AUC & F1 metrics
│   │   └── evaluate.py          # Diagnostic evaluator (confusion matrix, evasion analysis)
│   ├── generator/
│   │   ├── inference.py         # Essay generator with stylometric scoring
│   └── utils/
│       └── logger.py            # Standardized logger
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Build or Update the Dataset

Downloads the benchmark corpus (44k+ human and multi-model AI essays from PERSUADE and Kaggle DAIGT), performs cleaning, creates train/val/test splits, and constructs 19k+ length-controlled DPO pairs:
```bash
python scripts/01_build_dataset.py
```

### 3. Run Stylometric Contrast Analysis

Analyze the difference in burstiness, sentence variance, and AI transition clichés:
```bash
python scripts/02_analyze_stylometrics.py --sample_size 1000
```

### 4. Train the Detector (Discriminator)

Train a local `microsoft/deberta-v3-base` classifier to act as your objective evaluation judge:
```bash
# Rapid test run (5,000 balanced samples):
python scripts/03_train_detector.py --max_train 5000 --epochs 2

# Full dataset training:
python scripts/03_train_detector.py
```

### 5. Next Phase: Generator SFT & DPO Training

Once the baseline detector is trained, use the generated `data/processed/dpo_train.jsonl` dataset to train `Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct` with DPO to actively penalize low-burstiness and formulaic AI writing patterns.