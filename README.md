# A Comparative Analysis of Machine Learning Techniques for Entity Extraction in Long Legal Documents

Systematic comparison of **LLM**, **SLM**, and **RAG** paradigms for Named Entity Recognition in legal contracts, evaluated on two datasets across two languages.

## Key Results

| Model | Dataset | P | R | F1 | Latency/doc |
|---|---|---|---|---|---|
| Gemini-2.0-Flash (LLM) | CUAD (41 types) | 59.92% | 49.72% | **54.34%** | 40.39s |
| Legal-BERT (SLM) | CUAD (41 types) | 39.75% | 49.83% | 44.22% | **1.53s** |
| GPT-4o (LLM) | CUAD (41 types) | 41.68% | 46.35% | 43.89% | 235.96s |
| RAG-GPT4o | CUAD (41 types) | 41.80% | 23.16% | 29.80% | 68.02s |
| Legal-BERTimbau (SLM) | DI2WIN (35 types) | 50.64% | 71.39% | **59.25%** | **~0.15s** |
| Gemini-2.0-Flash (LLM) | DI2WIN (35 types) | 53.36% | 66.40% | 59.17% | 23.33s |
| GPT-4-Turbo (LLM) | DI2WIN (35 types) | 56.07% | 55.85% | 55.96% | 40.54s |
| RAG-GPT4o | DI2WIN (35 types) | 34.92% | 10.01% | 15.56% | 5.08s |

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Project Structure](#project-structure)
5. [Running Experiments](#running-experiments)
   - [Step 1: LLM Evaluation on CUAD](#step-1-llm-evaluation-on-cuad)
   - [Step 2: LLM Evaluation on DI2WIN](#step-2-llm-evaluation-on-di2win)
   - [Step 3: SLM Training](#step-3-slm-training)
   - [Step 4: SLM Evaluation](#step-4-slm-evaluation)
   - [Step 5: RAG Evaluation](#step-5-rag-evaluation)
   - [Step 6: Generate Report](#step-6-generate-report)
6. [Script Reference](#script-reference)
7. [Approach Details](#approach-details)
8. [Citation](#citation)

---

## Prerequisites

- **Python** 3.10+
- **PyTorch** 2.1+ (with CUDA, MPS, or CPU)
- **GPU/MPS** recommended for SLM training (~45 min on Apple M1, ~5h on Apple M4 Pro for CUAD)
- **API keys** for LLM evaluation:
  - Azure OpenAI (GPT-4o, GPT-4-Turbo)
  - Google AI (Gemini-2.0-Flash)
- **Disk space**: ~2GB for dependencies, ~500MB per SLM checkpoint

## Installation

```bash
# Clone repository
git clone https://github.com/jorge-barros-upe/legal-entity-extraction.git
cd legal-entity-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
```

Edit `.env` with your actual API keys:

```env
# Azure OpenAI (for GPT-4o, GPT-4-Turbo)
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_VERSION=2024-06-01

# Google Gemini
GOOGLE_API_KEY=your_key_here
```

## Data Preparation

### CUAD Dataset (English)

1. Download from [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad)
2. Place files in `data/cuad/`:

```bash
mkdir -p data/cuad
# Download and place these files:
# data/cuad/test.json                      (SQuAD format, required for all CUAD scripts)
# data/cuad/train_separate_questions.json   (SQuAD format, required for SLM training)
```

The test set contains 102 documents with 2,643 annotations across 41 clause types.

### DI2WIN Dataset (Portuguese)

The DI2WIN dataset will be released under CC-BY-NC 4.0 upon publication. Contact `jbm@ecomp.poli.br` for early access.

```bash
mkdir -p data/di2win
# Place files:
# data/di2win/contratos_train_files_extractor.jsonl   (312 training contracts)
# data/di2win/contratros_test_files_extractor.jsonl    (47 test contracts)
```

The dataset contains 359 contracts with 34,674 annotations across 143 hierarchical entity types.

## Project Structure

```
.
├── .env.example                # API key template
├── .gitignore
├── README.md
├── requirements.txt
│
├── config/                     # Experiment configuration files
│   ├── base_config.yaml        # Shared settings (paths, seeds, metrics)
│   ├── llm_config.yaml         # LLM provider configs (Azure, OpenAI, Gemini)
│   ├── slm_config.yaml         # SLM training hyperparameters
│   ├── rag_config.yaml         # RAG pipeline settings (chunking, retrieval)
│   └── hybrid_config.yaml      # Hybrid approach settings
│
├── src/                        # Source code library
│   ├── approaches/
│   │   ├── llm/                # LLM extractors, prompt templates, API providers
│   │   ├── slm/                # SLM extractor (BIO tagging, sliding window)
│   │   ├── rag/                # RAG pipeline (chunking, embeddings, retrieval)
│   │   └── hybrid/             # Hybrid RAG+LLM extractor
│   ├── core/                   # Base extractor, data loader, preprocessing
│   ├── evaluation/             # Metrics (exact/partial match), analysis, visualization
│   └── utils/                  # Config management, cost tracker, text processing
│
├── scripts/                    # Executable scripts (see Script Reference below)
│
├── data/                       # Datasets (gitignored, see Data Preparation)
│   └── README.md
│
├── results/                    # Evaluation outputs (JSON, CSV)
├── checkpoints/                # Model checkpoints (gitignored)
├── logs/                       # Training logs
└── notebooks/                  # Jupyter notebooks
```

---

## Running Experiments

All commands should be run from the project root directory.

### Step 1: LLM Evaluation on CUAD

#### ContractEval Methodology (Zero-shot, recommended)

This replicates the paper's main CUAD LLM results using the ContractEval (2025) methodology with clause-type-specific prompts.

```bash
# Evaluate with Gemini-2.0-Flash (best LLM result: F1=54.34%)
python scripts/evaluate_cuad_contracteval.py \
  --data data/cuad/test.json \
  --model gemini-2.0-flash

# Evaluate with GPT-4o (F1=43.89%)
python scripts/evaluate_cuad_contracteval.py \
  --data data/cuad/test.json \
  --model gpt-4o

# Limit to first 10 docs for a quick test
python scripts/evaluate_cuad_contracteval.py \
  --data data/cuad/test.json \
  --model gemini-2.0-flash \
  --max-docs 10
```

**Output**: `results/cuad_contracteval_{model}_{timestamp}.json` with per-type P/R/F1.

**Cost estimate**: ~$5-15 per full run (102 docs) depending on model.

#### Alternative: Full Multi-Model Evaluation

Runs all LLMs + RAG on CUAD in a single pass:

```bash
python scripts/evaluate_cuad_all_models_full.py \
  --data data/cuad/test.json \
  --models gpt-4o gpt-4-turbo gemini-2.0-flash rag-gpt4o
```

**Output**: `results/cuad_full_41types_comparison_{timestamp}.csv` + per-model per-type CSVs.

#### Other CUAD Evaluation Variants

```bash
# Few-shot evaluation (uses training examples as demonstrations)
python scripts/evaluate_cuad_fewshot.py \
  --data data/cuad/test.json \
  --model gemini-2.0-flash \
  --use-groups

# Optimized evaluation (Chain-of-Thought, advanced matching)
python scripts/evaluate_cuad_optimized.py \
  --data data/cuad/test.json \
  --model gemini-2.0-flash \
  --strategy grouped
```

---

### Step 2: LLM Evaluation on DI2WIN

```bash
# Full multi-model evaluation (GPT-4o, GPT-4-Turbo, Gemini, RAG)
python scripts/evaluate_all_models_di2win.py \
  --data-path data/di2win/contratros_test_files_extractor.jsonl \
  --max-samples 47

# Optimized evaluation with different strategies
python scripts/evaluate_di2win_optimized.py
```

**Output**: `results/di2win_model_comparison_{timestamp}.csv` + per-model per-entity CSVs.

---

### Step 3: SLM Training

#### Train Legal-BERT on CUAD (English, 41 clause types)

```bash
python scripts/train_slm_cuad.py \
  --model nlpaueb/legal-bert-base-uncased \
  --train-path data/cuad/train_separate_questions.json \
  --test-path data/cuad/test.json \
  --output-dir checkpoints/cuad_slm \
  --epochs 15 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 3e-5 \
  --stride 128 \
  --patience 5
```

**Training time**: ~5 hours on Apple M4 Pro (MPS), ~2h on NVIDIA GPU.

**Output**: `checkpoints/cuad_slm/legal_bert_cuad_{timestamp}/best/` (model + label mappings).

**Expected result**: F1 ~44%, P ~40%, R ~50% on test set (102 docs).

#### Train Legal-BERTimbau on DI2WIN (Portuguese, 35 entity types)

```bash
# Optimized training with class-weighted loss (recommended)
python scripts/train_slm_optimized.py \
  --models legal-bertimbau-base \
  --train-path data/di2win/contratos_train_files_extractor.jsonl \
  --test-path data/di2win/contratros_test_files_extractor.jsonl \
  --output-dir checkpoints/slm_optimized \
  --epochs 15 \
  --patience 5

# Basic training (without class weights, for comparison)
python scripts/train_slm_di2win.py \
  --models legal-bertimbau-base \
  --train-path data/di2win/contratos_train_files_extractor.jsonl \
  --test-path data/di2win/contratros_test_files_extractor.jsonl \
  --epochs 5
```

**Training time**: ~45 min on Apple M1 (MPS).

**Expected result**: F1 ~59% (optimized) vs F1 ~6% (without class weights).

#### Train DeBERTa on CUAD (alternative architecture)

```bash
python scripts/finetune_cuad_deberta.py \
  --train data/cuad/train_separate_questions.json \
  --test data/cuad/test.json \
  --output checkpoints/cuad_deberta \
  --epochs 5 \
  --batch-size 4
```

---

### Step 4: SLM Evaluation

#### Evaluate CUAD Checkpoint

```bash
python scripts/eval_slm_cuad_checkpoint.py \
  --checkpoint checkpoints/cuad_slm/legal_bert_cuad_{timestamp}/best \
  --test-path data/cuad/test.json
```

**Output**: `checkpoints/.../best/test_eval_results.json` with overall and per-type metrics.

#### Evaluate DI2WIN Checkpoint

```bash
python scripts/evaluate_slm_di2win.py \
  --checkpoint checkpoints/slm_optimized/{timestamp}/legal-bertimbau-base/best \
  --test-path data/di2win/contratros_test_files_extractor.jsonl
```

---

### Step 5: RAG Evaluation

RAG is included in the multi-model evaluation scripts:

```bash
# CUAD RAG evaluation (uses GPT-4o + FAISS retrieval)
python scripts/evaluate_cuad_all_models_full.py \
  --data data/cuad/test.json \
  --models rag-gpt4o

# DI2WIN RAG evaluation
python scripts/evaluate_all_models_di2win.py \
  --data-path data/di2win/contratros_test_files_extractor.jsonl \
  --max-samples 47
```

The RAG pipeline:
1. Chunks documents into 512-token passages (128-token overlap)
2. Embeds with `paraphrase-multilingual-MiniLM-L12-v2`
3. Indexes with FAISS (inner product similarity)
4. Retrieves top-5 chunks per entity type
5. Extracts with GPT-4o + type-specific prompts

---

### Step 6: Generate Report

After running evaluations, generate a consolidated comparison report:

```bash
python scripts/generate_final_report.py
```

**Output**: `results/final_report_{timestamp}.md` + `results/consolidated_results_{timestamp}.csv`.

---

## Reproducing Paper Results

To reproduce the exact results from the paper, run experiments in this order:

```bash
# 1. CUAD LLM evaluation (ContractEval zero-shot)
python scripts/evaluate_cuad_contracteval.py --model gemini-2.0-flash
python scripts/evaluate_cuad_contracteval.py --model gpt-4o

# 2. CUAD full multi-model comparison (includes GPT-4-Turbo and RAG)
python scripts/evaluate_cuad_all_models_full.py \
  --models gpt-4o gpt-4-turbo gemini-2.0-flash rag-gpt4o

# 3. DI2WIN multi-model evaluation
python scripts/evaluate_all_models_di2win.py --max-samples 47

# 4. SLM training - CUAD
python scripts/train_slm_cuad.py \
  --epochs 15 --batch-size 8 --grad-accum 4 --patience 5

# 5. SLM training - DI2WIN (optimized with class-weighted loss)
python scripts/train_slm_optimized.py --epochs 15 --patience 5

# 6. SLM evaluation
python scripts/eval_slm_cuad_checkpoint.py --checkpoint checkpoints/cuad_slm/.../best
python scripts/evaluate_slm_di2win.py --checkpoint checkpoints/slm_optimized/.../best

# 7. Generate final report
python scripts/generate_final_report.py
```

**Notes**:
- LLM results use `temperature=0.0` for deterministic outputs, but commercial APIs may produce slightly different results over time due to provider updates.
- SLM results are reproducible with `seed=42` (default).
- Total API cost: ~$30-50 for all LLM evaluations.
- Total compute time: ~6-8 hours for SLM training + ~4-6 hours for LLM API calls.

---

## Script Reference

| Script | Purpose | Dataset | Requires API |
|---|---|---|---|
| `train_slm_cuad.py` | Fine-tune Legal-BERT on CUAD | CUAD | No |
| `train_slm_di2win.py` | Fine-tune BERTimbau on DI2WIN | DI2WIN | No |
| `train_slm_optimized.py` | Optimized SLM training (class-weighted) | DI2WIN | No |
| `finetune_cuad_deberta.py` | Fine-tune DeBERTa on CUAD | CUAD | No |
| `eval_slm_cuad_checkpoint.py` | Evaluate CUAD SLM checkpoint | CUAD | No |
| `evaluate_slm_di2win.py` | Evaluate DI2WIN SLM checkpoint | DI2WIN | No |
| `evaluate_cuad_contracteval.py` | ContractEval methodology (zero-shot) | CUAD | Yes |
| `evaluate_cuad_full.py` | Single-model CUAD evaluation | CUAD | Yes |
| `evaluate_cuad_fewshot.py` | Few-shot CUAD evaluation | CUAD | Yes |
| `evaluate_cuad_optimized.py` | CoT + advanced matching on CUAD | CUAD | Yes |
| `evaluate_cuad_all_models_full.py` | Multi-model CUAD comparison | CUAD | Yes |
| `evaluate_all_models_cuad.py` | Multi-model CUAD (subset types) | CUAD | Yes |
| `evaluate_all_models_di2win.py` | Multi-model DI2WIN comparison | DI2WIN | Yes |
| `evaluate_di2win_optimized.py` | Optimized DI2WIN evaluation | DI2WIN | Yes |
| `generate_final_report.py` | Generate consolidated report | Both | No |

---

## Approach Details

### LLM (Large Language Models)

| Setting | CUAD | DI2WIN |
|---|---|---|
| Prompting | Zero-shot (ContractEval) | Few-shot (2-3 demos) |
| Demo selection | N/A | Jaccard similarity over tokens |
| Models | GPT-4o, GPT-4-Turbo, Gemini-2.0-Flash | Same |
| Temperature | 0.0 | 0.0 |
| Max output tokens | 4096 | 4096 |
| Context window | 128K (GPT), 1M (Gemini) | Same |

### SLM (Small Language Models)

| Setting | CUAD | DI2WIN |
|---|---|---|
| Base model | `nlpaueb/legal-bert-base-uncased` | `rufimelo/Legal-BERTimbau-base` |
| Tagging scheme | BIO (83 labels: 41x2+1) | BIO (71 labels: 35x2+1) |
| Max length | 512 tokens | 512 tokens |
| Sliding window stride | 128 tokens | 128 tokens |
| Overlap resolution | Central token priority | Central token priority |
| Loss function | Class-weighted cross-entropy | Class-weighted cross-entropy |
| Optimizer | AdamW, lr=3e-5 | AdamW, lr=3e-5 |
| Batch size | 8 (grad_accum=4, effective=32) | 4 (grad_accum=8, effective=32) |
| Epochs | 15 (early stopping, patience=5) | 15 (early stopping, patience=5) |
| LR scheduler | Cosine | Cosine |

### RAG (Retrieval-Augmented Generation)

| Component | Configuration |
|---|---|
| Chunking | 512 tokens, 128-token overlap |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| Index | FAISS, inner product similarity |
| Retrieval | Top-5 chunks per entity type |
| Generator | GPT-4o |
| Deduplication | IoU >= 0.8 |

---

## Citation

```bibtex
@article{medeiros2026legal,
  title={A Comparative Analysis of Machine Learning Techniques for Entity Extraction in Long Legal Documents},
  author={Medeiros, Jorge Barros and Bezerra, Byron Leite Dantas},
  year={2026}
}
```

## License

Code is released under the MIT License. The DI2WIN dataset will be released under CC-BY-NC 4.0 upon publication.

## Author

Jorge Barros Medeiros -- Graduate Program in Computer Engineering, Polytechnic School of Pernambuco, University of Pernambuco.
