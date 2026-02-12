#!/usr/bin/env python3
"""
CUAD Optimized Evaluation Script - All 41 Clause Types

This script implements optimizations based on ContractEval (2025) methodology
that achieves F1=64.1% with GPT-4.1.

Key improvements:
1. Q&A format prompts (one question per clause type or grouped)
2. Chain-of-Thought prompting
3. Advanced text normalization and matching
4. Retry with exponential backoff
5. Support for GPT-4.1 and other models
6. Parallel extraction by groups
"""

import os
import sys
import json
import re
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from functools import wraps

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src', 'approaches', 'llm'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src', 'utils'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import pandas as pd
from tqdm import tqdm

# LLM clients
try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import optimized prompts
from cuad_41_optimized_prompts import (
    CUAD_CLAUSE_TYPES_ENHANCED,
    CUAD_ALL_TYPES,
    CLAUSE_GROUPS,
    EXTRACTION_JSON_SCHEMA,
    create_qa_extraction_prompt,
    create_grouped_extraction_prompt,
    create_cot_full_extraction_prompt,
)

# Import text processing utilities
try:
    from text_processing import (
        normalize_text_advanced,
        normalize_clause_type,
        match_extraction_advanced,
        deduplicate_by_similarity,
    )
except ImportError:
    # Fallback if module not available
    def normalize_text_advanced(text, **kwargs):
        return text.lower().strip()

    def normalize_clause_type(ct):
        return ct.upper().replace(" ", "_").replace("-", "_").replace("/", "_")

    def match_extraction_advanced(pred, golds, threshold=0.5, method="combined"):
        for gold in golds:
            pred_tokens = set(pred.lower().split())
            gold_tokens = set(gold.lower().split())
            if gold_tokens and len(pred_tokens & gold_tokens) / len(gold_tokens) >= threshold:
                return True, gold, threshold
        return False, None, 0.0

    def deduplicate_by_similarity(texts, threshold=0.8):
        return list(set(texts))


# =============================================================================
# RETRY WITH EXPONENTIAL BACKOFF
# =============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"  Attempt {attempt + 1} failed: {str(e)[:50]}... Retrying in {delay:.1f}s")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# CUAD DATA LOADER
# =============================================================================

def extract_category_from_question(question: str) -> Optional[str]:
    """Extract clause type from CUAD question format."""
    match = re.search(r'related to "([^"]+)"', question)
    if match:
        category = match.group(1)
        return normalize_clause_type(category)
    return None


def load_cuad_test_data(data_path: str) -> List[Dict]:
    """Load CUAD test data and convert to extraction format."""
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
# OPTIMIZED LLM EXTRACTOR
# =============================================================================

class OptimizedLLMExtractor:
    """
    Optimized LLM extractor with:
    - Multiple extraction strategies (single, grouped, Q&A)
    - Retry with backoff
    - Support for GPT-4.1 and other models
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        strategy: str = "grouped",  # "single", "grouped", "qa"
        use_cot: bool = True,
        use_json_schema: bool = True,
    ):
        self.model_name = model_name
        self.strategy = strategy
        self.use_cot = use_cot
        self.use_json_schema = use_json_schema
        self.provider = None
        self.client = None

        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on model name."""
        if "gpt" in self.model_name.lower():
            self._init_openai()
        elif "gemini" in self.model_name.lower():
            self._init_gemini()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")

        # Check for Azure or direct OpenAI
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            self.provider = "azure"
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.provider = "openai"
        else:
            raise ValueError("No OpenAI API key found")

        # Model mapping
        model_map = {
            "gpt-4o": os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "gpt-4o-mini": "gpt-4o-mini",
        }
        self.deployment = model_map.get(self.model_name, self.model_name)

    def _init_gemini(self):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model_map = {
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-2.5-pro": "gemini-2.5-pro-preview",
        }
        model_id = model_map.get(self.model_name, self.model_name)
        self.client = genai.GenerativeModel(model_id)
        self.provider = "gemini"

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract clauses based on configured strategy.

        Args:
            text: Contract text

        Returns:
            Dict mapping clause_type -> list of extracted texts
        """
        if self.strategy == "single":
            return self._extract_single_pass(text)
        elif self.strategy == "grouped":
            return self._extract_by_groups(text)
        elif self.strategy == "qa":
            return self._extract_qa_format(text)
        else:
            return self._extract_single_pass(text)

    def _extract_single_pass(self, text: str) -> Dict[str, List[str]]:
        """Extract all 41 types in a single pass."""
        prompt = create_cot_full_extraction_prompt(text) if self.use_cot else self._create_simple_prompt(text)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _extract_by_groups(self, text: str) -> Dict[str, List[str]]:
        """Extract by semantic groups for better accuracy."""
        all_predictions = {}

        for group_name in CLAUSE_GROUPS.keys():
            prompt = create_grouped_extraction_prompt(text, group_name, include_cot=self.use_cot)
            response = self._call_llm(prompt)
            group_preds = self._parse_response(response)
            all_predictions.update(group_preds)

        return all_predictions

    def _extract_qa_format(self, text: str) -> Dict[str, List[str]]:
        """Extract using Q&A format (one question per type)."""
        all_predictions = {}

        for clause_type in CUAD_ALL_TYPES:
            prompt = create_qa_extraction_prompt(text, clause_type, include_cot=self.use_cot)
            response = self._call_llm(prompt)

            # Parse Q&A response format
            clauses = self._parse_qa_response(response)
            if clauses:
                all_predictions[clause_type] = clauses

        return all_predictions

    def _create_simple_prompt(self, text: str) -> str:
        """Create a simple prompt without CoT."""
        types_str = ", ".join(CUAD_ALL_TYPES)
        return f"""Extract all clause types from this contract.

Types: {types_str}

Contract:
{text[:50000]}

Response (JSON only):
{{"entities": [{{"text": "...", "type": "TYPE", "confidence": 0.95}}]}}
"""

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with retry."""
        if self.provider in ["azure", "openai"]:
            return self._call_openai(prompt)
        else:
            return self._call_gemini(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        kwargs = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4000,
        }

        # Add JSON schema if supported and enabled
        if self.use_json_schema and self.model_name in ["gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini"]:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        response = self.client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4000,
            )
        )
        return response.text

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response to extract entities.

        Handles multiple JSON formats:
        1. {"entities": [{"text": "...", "type": "...", "confidence": ...}]}
        2. {"DOCUMENT_NAME": ["..."], "PARTIES": ["..."]}  (simplified)
        3. {"DOCUMENT_NAME": [{"text": "..."}], ...}  (mixed)
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                result = defaultdict(list)

                # Format 1: entities array with type/text
                if 'entities' in data:
                    entities = data.get('entities', [])
                    for ent in entities:
                        if isinstance(ent, dict):
                            ent_type = normalize_clause_type(ent.get('type', ''))
                            ent_text = ent.get('text', '')
                            if isinstance(ent_text, str):
                                ent_text = ent_text.strip()
                                if ent_type and ent_text and ent_type in CUAD_ALL_TYPES:
                                    result[ent_type].append(ent_text)
                else:
                    # Format 2/3: direct type -> list mapping
                    for key, value in data.items():
                        clause_type = normalize_clause_type(key)
                        if clause_type in CUAD_ALL_TYPES:
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str) and item.strip():
                                        result[clause_type].append(item.strip())
                                    elif isinstance(item, dict):
                                        # Handle {"text": "..."} format
                                        text = item.get('text', '')
                                        if isinstance(text, str) and text.strip():
                                            result[clause_type].append(text.strip())
                            elif isinstance(value, str) and value.strip():
                                result[clause_type].append(value.strip())

                return dict(result)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"  Warning: Error parsing response: {e}")

        return {}

    def _parse_qa_response(self, response: str) -> List[str]:
        """Parse Q&A format response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                clauses = data.get('clauses', [])
                return [c.strip() for c in clauses if c and c.strip()]
        except json.JSONDecodeError:
            pass

        return []


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_document(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    threshold: float = 0.5,
    match_method: str = "combined",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions for a single document using advanced matching.

    Args:
        predictions: Predicted clauses by type
        ground_truth: Ground truth clauses by type
        threshold: Similarity threshold for matching
        match_method: Matching method ("token", "levenshtein", "sequence", "combined")

    Returns:
        Per-type metrics: TP, FP, FN
    """
    metrics = {}
    all_types = set(list(predictions.keys()) + list(ground_truth.keys()))

    for clause_type in all_types:
        preds = predictions.get(clause_type, [])
        golds = ground_truth.get(clause_type, [])

        tp = 0
        matched_golds = set()

        for pred in preds:
            is_match, matched, score = match_extraction_advanced(
                pred, golds, threshold=threshold, method=match_method
            )
            if is_match and matched:
                tp += 1
                matched_golds.add(matched)

        fp = len(preds) - tp
        fn = len(golds) - len(matched_golds)

        metrics[clause_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_pred": len(preds),
            "num_gold": len(golds),
        }

    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across all documents."""
    aggregated = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "num_pred": 0, "num_gold": 0})

    for doc_metrics in all_metrics:
        for clause_type, m in doc_metrics.items():
            aggregated[clause_type]["tp"] += m["tp"]
            aggregated[clause_type]["fp"] += m["fp"]
            aggregated[clause_type]["fn"] += m["fn"]
            aggregated[clause_type]["num_pred"] += m["num_pred"]
            aggregated[clause_type]["num_gold"] += m["num_gold"]

    results = {}
    for clause_type, m in aggregated.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[clause_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": m["num_gold"],
        }

    return results


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    data_path: str,
    model_name: str = "gpt-4o",
    strategy: str = "grouped",
    use_cot: bool = True,
    match_method: str = "combined",
    match_threshold: float = 0.5,
    max_docs: Optional[int] = None,
    output_dir: str = "results",
):
    """
    Run optimized CUAD evaluation.

    Args:
        data_path: Path to CUAD test.json
        model_name: LLM to use
        strategy: Extraction strategy ("single", "grouped", "qa")
        use_cot: Whether to use Chain-of-Thought prompting
        match_method: Matching method for evaluation
        match_threshold: Similarity threshold for matching
        max_docs: Limit number of documents
        output_dir: Directory to save results
    """
    print(f"Loading CUAD data from: {data_path}")
    documents = load_cuad_test_data(data_path)
    print(f"Loaded {len(documents)} documents")

    if max_docs:
        documents = documents[:max_docs]
        print(f"Limited to {max_docs} documents")

    print(f"\nInitializing {model_name} extractor...")
    print(f"  Strategy: {strategy}")
    print(f"  Chain-of-Thought: {use_cot}")
    print(f"  Match method: {match_method}")
    print(f"  Match threshold: {match_threshold}")

    extractor = OptimizedLLMExtractor(
        model_name=model_name,
        strategy=strategy,
        use_cot=use_cot,
    )

    all_metrics = []
    total_time = 0
    errors = 0

    print("\nRunning extraction...")
    for doc in tqdm(documents, desc=f"{model_name} ({strategy})"):
        start_time = time.time()

        try:
            predictions = extractor.extract(doc["context"])
            elapsed = time.time() - start_time
            total_time += elapsed

            doc_metrics = evaluate_document(
                predictions, doc["ground_truth"],
                threshold=match_threshold,
                match_method=match_method,
            )
            all_metrics.append(doc_metrics)

        except Exception as e:
            print(f"\n  Error processing {doc['doc_id']}: {e}")
            errors += 1
            all_metrics.append({})

    # Aggregate results
    print("\nAggregating results...")
    results = aggregate_metrics(all_metrics)

    # Overall metrics
    total_tp = sum(r["tp"] for r in results.values())
    total_fp = sum(r["fp"] for r in results.values())
    total_fn = sum(r["fn"] for r in results.values())
    total_support = sum(r["support"] for r in results.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # Print results
    print("\n" + "="*80)
    print(f"CUAD OPTIMIZED EVALUATION RESULTS")
    print("="*80)
    print(f"Model:      {model_name}")
    print(f"Strategy:   {strategy}")
    print(f"CoT:        {use_cot}")
    print(f"Match:      {match_method} @ {match_threshold}")
    print(f"Documents:  {len(documents)} ({errors} errors)")
    print(f"Time:       {total_time:.1f}s ({total_time/max(len(documents)-errors, 1):.2f}s/doc)")
    print()

    print("OVERALL METRICS:")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F1:        {overall_f1:.4f}")
    print(f"  Support:   {total_support}")
    print()

    print("TOP 10 TYPES BY F1:")
    print("-"*80)
    print(f"{'Clause Type':<40} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
    print("-"*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for clause_type, m in sorted_results[:10]:
        if m["support"] > 0:
            print(f"{clause_type:<40} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = model_name.replace("-", "_").replace(".", "_")

    # Save summary
    summary = {
        "model": model_name,
        "strategy": strategy,
        "use_cot": use_cot,
        "match_method": match_method,
        "match_threshold": match_threshold,
        "timestamp": timestamp,
        "num_documents": len(documents),
        "errors": errors,
        "total_time": total_time,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "support": total_support,
        },
        "per_type": results,
    }

    summary_path = os.path.join(output_dir, f"cuad_optimized_{model_clean}_{strategy}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Save CSV
    csv_data = []
    for clause_type, m in results.items():
        csv_data.append({
            "clause_type": clause_type,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "support": m["support"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        })

    csv_path = os.path.join(output_dir, f"cuad_optimized_{model_clean}_{strategy}_{timestamp}_per_type.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CUAD Optimized Evaluation (41 Types)")
    parser.add_argument("--data", type=str, default="data/cuad/test.json",
                        help="Path to CUAD test.json")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        choices=["gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini",
                                 "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
                        help="LLM model to use")
    parser.add_argument("--strategy", type=str, default="grouped",
                        choices=["single", "grouped", "qa"],
                        help="Extraction strategy")
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable Chain-of-Thought prompting")
    parser.add_argument("--match-method", type=str, default="combined",
                        choices=["token", "levenshtein", "sequence", "combined"],
                        help="Matching method for evaluation")
    parser.add_argument("--match-threshold", type=float, default=0.5,
                        help="Similarity threshold for matching")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum number of documents to evaluate")
    parser.add_argument("--output", type=str, default="experiments/results",
                        help="Output directory")

    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, args.data)
    output_dir = os.path.join(project_root, args.output)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    run_evaluation(
        data_path=data_path,
        model_name=args.model,
        strategy=args.strategy,
        use_cot=not args.no_cot,
        match_method=args.match_method,
        match_threshold=args.match_threshold,
        max_docs=args.max_docs,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
