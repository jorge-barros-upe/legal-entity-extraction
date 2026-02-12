#!/usr/bin/env python3
"""
CUAD Full Evaluation - All 41 Clause Types with All Models

This script evaluates multiple model types on all 41 CUAD clause types:
1. LLMs: GPT-4o, GPT-4-Turbo, Gemini-1.5-Flash, Gemini-2.0-Flash
2. RAG: Retrieval-Augmented Generation with GPT-4o
3. (SLM: Would require fine-tuning on 41 types - not included yet)

This enables valid comparison with published CUAD baselines (Hendrycks et al., 2021).
"""

import os
import sys
import json
import re
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src', 'approaches', 'llm'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import pandas as pd
from tqdm import tqdm
import numpy as np

# LLM clients
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available")

# RAG components
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG dependencies not available")

# Import prompts
from cuad_full_prompts import (
    CUAD_CLAUSE_TYPES,
    CUAD_ALL_TYPES,
    create_cuad_full_extraction_prompt,
    create_cuad_focused_prompt,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelResult:
    """Results for a single model evaluation."""
    model_name: str
    model_type: str  # llm, rag
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    latency_avg: float = 0.0
    latency_std: float = 0.0
    num_documents: int = 0
    errors: int = 0
    per_type_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


# =============================================================================
# CUAD DATA LOADER
# =============================================================================

def extract_category_from_question(question: str) -> Optional[str]:
    """Extract clause type from CUAD question format."""
    match = re.search(r'related to "([^"]+)"', question)
    if match:
        category = match.group(1)
        return category.upper().replace(" ", "_").replace("-", "_").replace("/", "_")
    return None


def load_cuad_test_data(data_path: str) -> List[Dict]:
    """Load CUAD test data in QA format and convert to extraction format."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    documents = []

    for doc in data['data']:
        title = doc['title']

        for para_idx, para in enumerate(doc['paragraphs']):
            context = para['context']
            ground_truth = defaultdict(list)

            for qa in para['qas']:
                question = qa['question']
                clause_type = extract_category_from_question(question)

                if clause_type and clause_type in CUAD_ALL_TYPES:
                    for answer in qa.get('answers', []):
                        answer_text = answer.get('text', '').strip()
                        if answer_text:
                            ground_truth[clause_type].append(answer_text)

            doc_entry = {
                "doc_id": f"{title}_{para_idx}",
                "title": title,
                "context": context,
                "ground_truth": dict(ground_truth),
            }
            documents.append(doc_entry)

    return documents


# =============================================================================
# LLM EXTRACTORS
# =============================================================================

class OpenAIExtractor:
    """Azure OpenAI extractor."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        # Map to deployment names
        model_map = {
            "gpt-4o": os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            "gpt-4-turbo": "gpt-4-turbo",
        }
        self.deployment = model_map.get(model_name, model_name)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract clauses from text."""
        prompt = create_cuad_full_extraction_prompt(text)

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4000,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            print(f"OpenAI error: {e}")
            return {}

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get('entities', [])
                result = defaultdict(list)
                for ent in entities:
                    ent_type = ent.get('type', '')
                    ent_text = ent.get('text', '').strip()
                    if ent_type and ent_text:
                        result[ent_type].append(ent_text)
                return dict(result)
        except json.JSONDecodeError:
            pass
        return {}


class GeminiExtractor:
    """Google Gemini extractor."""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract clauses from text."""
        prompt = create_cuad_full_extraction_prompt(text)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=4000,
                )
            )
            return self._parse_response(response.text)
        except Exception as e:
            print(f"Gemini error: {e}")
            return {}

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get('entities', [])
                result = defaultdict(list)
                for ent in entities:
                    ent_type = ent.get('type', '')
                    ent_text = ent.get('text', '').strip()
                    if ent_type and ent_text:
                        result[ent_type].append(ent_text)
                return dict(result)
        except json.JSONDecodeError:
            pass
        return {}


class RAGExtractor:
    """RAG-based extractor using retrieval + LLM."""

    def __init__(self, llm_model: str = "gpt-4o", embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm_model = llm_model
        self.embedder = SentenceTransformer(embedding_model)
        self.llm = OpenAIExtractor(llm_model)
        self.chunk_size = 512
        self.chunk_overlap = 128

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract using RAG: retrieve relevant chunks, then extract."""
        # For short texts, just extract directly
        if len(text.split()) < 1000:
            return self.llm.extract(text)

        # Chunk and index
        chunks = self._chunk_text(text)
        embeddings = self.embedder.encode(chunks)

        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # For each clause type, retrieve relevant chunks
        all_predictions = defaultdict(list)

        # Group clause types into batches for efficiency
        batch_size = 10
        for i in range(0, len(CUAD_ALL_TYPES), batch_size):
            batch_types = CUAD_ALL_TYPES[i:i+batch_size]

            # Create query from type descriptions
            query_parts = []
            for ct in batch_types:
                if ct in CUAD_CLAUSE_TYPES:
                    query_parts.append(f"{ct}: {CUAD_CLAUSE_TYPES[ct]['description']}")

            query = " ".join(query_parts)
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)

            # Retrieve top chunks
            k = min(5, len(chunks))
            _, indices = index.search(query_embedding, k)

            # Concatenate retrieved chunks
            retrieved_text = " ".join([chunks[idx] for idx in indices[0]])

            # Extract from retrieved text
            prompt = create_cuad_focused_prompt(retrieved_text, batch_types)
            try:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2000,
                )
                batch_preds = self.llm._parse_response(response.choices[0].message.content)
                for ct, texts in batch_preds.items():
                    all_predictions[ct].extend(texts)
            except Exception as e:
                print(f"RAG batch error: {e}")

        return dict(all_predictions)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = ' '.join(text.split())
    text = re.sub(r'[.,;:!?"\'\(\)\[\]\{\}]', '', text)
    return text


def compute_overlap(pred: str, gold: str) -> float:
    """Compute token-level overlap."""
    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())
    if not gold_tokens:
        return 0.0
    intersection = pred_tokens & gold_tokens
    return len(intersection) / len(gold_tokens)


def evaluate_document(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, int]]:
    """Evaluate predictions for a single document."""
    metrics = {}
    all_types = set(list(predictions.keys()) + list(ground_truth.keys()))

    for clause_type in all_types:
        preds = predictions.get(clause_type, [])
        golds = ground_truth.get(clause_type, [])

        tp = 0
        matched_golds = set()

        for pred in preds:
            for i, gold in enumerate(golds):
                if i not in matched_golds:
                    overlap = compute_overlap(pred, gold)
                    if overlap >= threshold:
                        tp += 1
                        matched_golds.add(i)
                        break

        fp = len(preds) - tp
        fn = len(golds) - len(matched_golds)

        metrics[clause_type] = {"tp": tp, "fp": fp, "fn": fn, "num_gold": len(golds)}

    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Tuple[Dict[str, Dict], Dict]:
    """Aggregate metrics across documents."""
    aggregated = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "num_gold": 0})

    for doc_metrics in all_metrics:
        for clause_type, m in doc_metrics.items():
            aggregated[clause_type]["tp"] += m["tp"]
            aggregated[clause_type]["fp"] += m["fp"]
            aggregated[clause_type]["fn"] += m["fn"]
            aggregated[clause_type]["num_gold"] += m["num_gold"]

    per_type_results = {}
    for clause_type, m in aggregated.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_type_results[clause_type] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": m["num_gold"]
        }

    # Overall
    total_tp = sum(m["tp"] for m in aggregated.values())
    total_fp = sum(m["fp"] for m in aggregated.values())
    total_fn = sum(m["fn"] for m in aggregated.values())
    total_support = sum(m["num_gold"] for m in aggregated.values())

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    overall = {
        "precision": overall_p, "recall": overall_r, "f1": overall_f1,
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "support": total_support
    }

    return per_type_results, overall


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_model(
    model_name: str,
    model_type: str,
    documents: List[Dict],
    max_docs: Optional[int] = None,
) -> ModelResult:
    """Evaluate a single model on CUAD."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} ({model_type})")
    print(f"{'='*60}")

    # Initialize extractor
    if model_type == "llm":
        if "gpt" in model_name.lower():
            extractor = OpenAIExtractor(model_name)
        elif "gemini" in model_name.lower():
            extractor = GeminiExtractor(model_name)
        else:
            raise ValueError(f"Unknown LLM: {model_name}")
    elif model_type == "rag":
        extractor = RAGExtractor(llm_model="gpt-4o")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Limit documents if specified
    eval_docs = documents[:max_docs] if max_docs else documents

    all_metrics = []
    latencies = []
    errors = 0

    for doc in tqdm(eval_docs, desc=model_name):
        start_time = time.time()

        try:
            predictions = extractor.extract(doc["context"])
            doc_metrics = evaluate_document(predictions, doc["ground_truth"])
            all_metrics.append(doc_metrics)
        except Exception as e:
            print(f"Error processing {doc['doc_id']}: {e}")
            errors += 1

        latencies.append(time.time() - start_time)

    # Aggregate
    per_type_results, overall = aggregate_metrics(all_metrics)

    result = ModelResult(
        model_name=model_name,
        model_type=model_type,
        precision=overall["precision"],
        recall=overall["recall"],
        f1=overall["f1"],
        support=overall["support"],
        tp=overall["tp"],
        fp=overall["fp"],
        fn=overall["fn"],
        latency_avg=np.mean(latencies),
        latency_std=np.std(latencies),
        num_documents=len(eval_docs),
        errors=errors,
        per_type_metrics=per_type_results,
    )

    # Print summary
    print(f"\n{model_name} Results:")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1:        {result.f1:.4f}")
    print(f"  Support:   {result.support}")
    print(f"  Latency:   {result.latency_avg:.2f}s ± {result.latency_std:.2f}s")

    return result


def run_all_evaluations(
    data_path: str,
    output_dir: str,
    max_docs: Optional[int] = None,
    models: Optional[List[str]] = None,
):
    """Run evaluation for all models."""
    print("Loading CUAD data...")
    documents = load_cuad_test_data(data_path)
    print(f"Loaded {len(documents)} documents")

    if max_docs:
        print(f"Limiting to {max_docs} documents")

    # Define models to evaluate
    all_models = [
        ("gpt-4o", "llm"),
        ("gpt-4-turbo", "llm"),
        ("gemini-1.5-flash", "llm"),
        ("gemini-2.0-flash", "llm"),
        ("rag-gpt4o", "rag"),
    ]

    if models:
        all_models = [(m, t) for m, t in all_models if m in models]

    results = []

    for model_name, model_type in all_models:
        try:
            result = evaluate_model(model_name, model_type, documents, max_docs)
            results.append(result)
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary CSV
    summary_data = []
    for r in results:
        summary_data.append({
            "model": r.model_name,
            "type": r.model_type,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "support": r.support,
            "tp": r.tp,
            "fp": r.fp,
            "fn": r.fn,
            "latency_avg": r.latency_avg,
            "latency_std": r.latency_std,
            "num_docs": r.num_documents,
            "errors": r.errors,
        })

    summary_path = os.path.join(output_dir, f"cuad_full_41types_comparison_{timestamp}.csv")
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Per-type metrics for each model
    for r in results:
        type_data = []
        for ct, m in r.per_type_metrics.items():
            type_data.append({
                "clause_type": ct,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "support": m["support"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            })
        type_path = os.path.join(output_dir, f"cuad_full_{r.model_name}_per_type_{timestamp}.csv")
        pd.DataFrame(type_data).to_csv(type_path, index=False)

    # Full JSON results
    json_results = {
        "timestamp": timestamp,
        "num_documents": len(documents) if not max_docs else max_docs,
        "num_clause_types": len(CUAD_ALL_TYPES),
        "models": [asdict(r) for r in results],
    }
    json_path = os.path.join(output_dir, f"cuad_full_41types_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    # Print final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - CUAD Full (41 Clause Types)")
    print("="*80)
    print(f"{'Model':<25} {'Type':<6} {'P':>8} {'R':>8} {'F1':>8} {'Lat(s)':>8}")
    print("-"*80)
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"{r.model_name:<25} {r.model_type:<6} {r.precision:>8.4f} {r.recall:>8.4f} {r.f1:>8.4f} {r.latency_avg:>8.2f}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CUAD Full Evaluation - All 41 Types, All Models")
    parser.add_argument("--data", type=str, default="data/cuad/test.json",
                        help="Path to CUAD test.json")
    parser.add_argument("--output", type=str, default="experiments/results",
                        help="Output directory")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to evaluate (for testing)")
    parser.add_argument("--models", type=str, nargs="+",
                        choices=["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "gemini-2.0-flash", "rag-gpt4o"],
                        help="Specific models to evaluate")

    args = parser.parse_args()

    data_path = os.path.join(PROJECT_ROOT, args.data)
    output_dir = os.path.join(PROJECT_ROOT, args.output)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    run_all_evaluations(
        data_path=data_path,
        output_dir=output_dir,
        max_docs=args.max_docs,
        models=args.models,
    )


if __name__ == "__main__":
    main()
