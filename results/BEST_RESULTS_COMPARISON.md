# Best Results Comparison: CUAD vs DI2WIN

**Generated:** 2026-01-25
**Last Updated:** 2026-01-25

---

## Executive Summary

| Dataset | Language | Best Model | Best F1 | Precision | Recall |
|---------|----------|------------|---------|-----------|--------|
| **CUAD** | English | GPT-4o (validated) | **0.7836** | 0.7177 | 0.8628 |
| **DI2WIN** | Portuguese | GPT-4-Turbo (basic) | **0.5529** | 0.5607 | 0.5585 |

### By Model Type

| Type | CUAD Best | CUAD F1 | DI2WIN Best | DI2WIN F1 |
|------|-----------|---------|-------------|-----------|
| **LLM** | GPT-4o (validated) | 0.7836 | GPT-4-Turbo (basic) | 0.5529 |
| **RAG** | RAG-GPT4o | 0.7822 | RAG-GPT4o | 0.1556 |
| **SLM** | (not evaluated) | - | Legal-BERTimbau-base | 0.0627 |

---

## CUAD Dataset (English Legal Contracts)

### Entity Types
- PARTY (parties to the contract)
- DOC_NAME (document name/title)
- AGMT_DATE (agreement date)

### Published Baselines (Hendrycks et al., 2021)

The original CUAD benchmark uses 41 clause types with QA-style span selection:

| Model | AUPR | P@80R | P@90R | Params |
|-------|------|-------|-------|--------|
| BERT-base | 32.4% | 8.2% | 0.0% | ~110M |
| RoBERTa-base | 42.6% | 31.1% | 0.0% | ~125M |
| RoBERTa-large | 48.2% | 38.1% | 0.0% | ~355M |
| **DeBERTa-xlarge** | **47.8%** | **44.0%** | **17.8%** | ~900M |

> Note: Our results use 3 entity types with direct extraction (F1 metric), not directly comparable to original QA baselines.

### Model Rankings

| Rank | Model | Strategy | Precision | Recall | F1 | Latency (s) |
|------|-------|----------|-----------|--------|-----|-------------|
| 1 | GPT-4o | validated | 0.7177 | 0.8628 | **0.7836** | 2.91 |
| 2 | GPT-4o | basic | 0.6835 | 0.8809 | 0.7697 | 3.01 |
| 3 | GPT-4o | self-consistency | 0.6750 | 0.8773 | 0.7630 | 9.06 |
| 4 | GPT-4-Turbo | basic | 0.6095 | 0.8845 | 0.7216 | 5.39 |
| 5 | Gemini-Flash | basic | 0.6280 | 0.8412 | 0.7191 | 2.84 |
| 6 | Gemini-Flash | self-consistency | 0.6193 | 0.8339 | 0.7108 | 8.73 |

### RAG Results (CUAD)

| Model | Type | Precision | Recall | F1 | Latency (s) |
|-------|------|-----------|--------|-----|-------------|
| RAG-GPT4o | rag | 0.7389 | 0.8308 | **0.7822** | 2.77 |

> RAG performs comparably to direct LLM extraction, with slightly faster latency.

### Key Insights (CUAD)
- **Validation strategy** improves precision significantly (+4% vs basic)
- **High recall** across all models (>83%)
- **Gemini-Flash** offers best speed/quality tradeoff
- All working models achieve F1 > 0.71

---

## DI2WIN Dataset (Portuguese Legal Contracts)

### Entity Types (35 high-frequency types evaluated)
- **Sociedade:** nome, CNPJ, NIRE, endereco, capital
- **Socio:** nome, CPF, RG, endereco, quotas, estado civil
- **Administracao:** administrador, poderes, vetos
- **Datas:** registro, assinatura

### LLM Model Rankings (Optimized Prompts)

| Rank | Model | Strategy | Precision | Recall | F1 | Latency (s) |
|------|-------|----------|-----------|--------|-----|-------------|
| 1 | GPT-4-Turbo | basic | 0.5607 | 0.5585 | **0.5529** | 40.54 |
| 2 | GPT-4o | validated | 0.5386 | 0.4606 | 0.4899 | 22.76 |
| 3 | GPT-4o | basic | 0.5049 | 0.4467 | 0.4684 | 22.92 |
| 4 | GPT-4o | self-consistency | 0.0892 | 0.0235 | 0.0349 | 40.52 |
| 5 | Gemini-1.5-Flash | basic | 0.0000 | 0.0000 | 0.0000 | 0.10 |

### Full Evaluation Results (All Model Types)

| Model | Type | Precision | Recall | F1 | Latency (s) |
|-------|------|-----------|--------|-----|-------------|
| GPT-4o | llm | 0.2616 | 0.2218 | 0.2400 | 19.99 |
| GPT-4-Turbo | llm | 0.2609 | 0.2112 | 0.2334 | 41.36 |
| Gemini-2.0-Flash | llm | 0.2707 | 0.1067 | 0.1531 | 13.54 |
| RAG-GPT4o | rag | 0.3492 | 0.1001 | **0.1556** | 5.08 |

> Note: Optimized prompts with strategy selection achieved better results than the full evaluation script.

### RAG Results (DI2WIN)

| Model | Type | Precision | Recall | F1 | Latency (s) |
|-------|------|-----------|--------|-----|-------------|
| RAG-GPT4o | rag | 0.3492 | 0.1001 | **0.1556** | 5.08 |

> RAG underperforms direct LLM on DI2WIN due to complexity of 143 entity types.

### SLM Results (Fine-tuned)

| Model | Type | Precision | Recall | F1 | Training Time |
|-------|------|-----------|--------|-----|---------------|
| Legal-BERTimbau-base | slm | 0.0809 | 0.0512 | **0.0627** | ~22 min (5 epochs) |

**Training Configuration:**
- Model: `rufimelo/Legal-BERTimbau-base` (Portuguese Legal BERT)
- Entity types: 35 high-frequency types
- Training samples: 1,537 (sliding window from 312 documents)
- Validation samples: 229
- Batch size: 4, Gradient accumulation: 8
- Learning rate: 2e-5
- Device: MPS (Apple M1)

**Analysis:**
- The model shows learning (loss decreased from 2.13 to 0.66)
- Low F1 due to class imbalance (mostly "O" tokens vs entity tokens)
- More epochs and hyperparameter tuning needed for better results
- Consider class weighting or focal loss for imbalanced NER

**Available Training Data:**
- `contratos_train_files_extractor.jsonl` (312 documents, 5.7MB)
- `openai_training_data.jsonl` (6.1MB OpenAI format)

### Key Insights (DI2WIN)
- **GPT-4-Turbo** outperforms GPT-4o on Portuguese contracts
- **Balanced precision/recall** (both ~56%) with GPT-4-Turbo using optimized prompts
- **Self-consistency** strategy failed (JSON parsing issues)
- **Gemini** failed with optimized prompts (needs further debugging)
- **RAG** shows high precision (0.35) but low recall (0.10)
- Task is significantly harder due to 143 entity types

---

## Cross-Dataset Analysis

### Performance Gap

| Model | CUAD F1 | DI2WIN F1 | Gap |
|-------|---------|-----------|-----|
| GPT-4o (best strategy) | 0.7836 | 0.4899 | -0.2937 |
| GPT-4-Turbo (basic) | 0.7216 | 0.5529 | -0.1687 |
| Gemini-Flash | 0.7191 | 0.0000 | -0.7191 |
| RAG-GPT4o | 0.7822 | 0.1556 | -0.6266 |

### Why DI2WIN is Harder

1. **More entity types:** 143 vs 3
2. **Hierarchical types:** `tipo->contexto` format
3. **Portuguese language:** Less training data for LLMs
4. **Complex documents:** Multiple shareholders, nested data

---

## Recommendations

### For English Contracts (CUAD-like)
- Use **GPT-4o with validation** for best quality (F1=0.78)
- Use **Gemini-Flash** for speed-sensitive applications
- **RAG** is a viable alternative with similar performance

### For Portuguese Contracts (DI2WIN-like)
- Use **GPT-4-Turbo with optimized prompts** for best quality (F1=0.55)
- Consider **fine-tuning SLMs** for higher performance
- **RAG** may help with precision but sacrifices recall

### Future Improvements
1. Fine-tune BERTimbau/Legal-BERTimbau on DI2WIN training data
2. Implement entity-type-specific prompts for hierarchical types
3. Improve RAG with Portuguese legal document embeddings
4. Debug Gemini parsing for cost-effective alternative

---

## Result Files

### CUAD
- Model comparison: `cuad_optimized_comparison_20260124_190841.csv`

### DI2WIN
- Optimized prompts evaluation: `di2win_new_prompts_20260125_145931.csv`
- Full model comparison: `di2win_model_comparison_20260125_164448.csv`
- Entity-level performance: `di2win_entity_performance_*.csv`

### Scripts
- CUAD evaluation: `scripts/evaluate_cuad_optimized.py`
- DI2WIN optimized: `scripts/evaluate_di2win_optimized.py`
- DI2WIN full: `scripts/evaluate_all_models_di2win.py`
