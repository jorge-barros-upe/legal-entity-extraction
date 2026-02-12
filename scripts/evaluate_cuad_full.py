#!/usr/bin/env python3
"""
CUAD Full Evaluation Script - All 41 Clause Types

This script evaluates LLM-based extraction on all 41 CUAD clause types,
enabling valid comparison with published baselines (Hendrycks et al., 2021).

The original CUAD benchmark uses:
- 41 clause types (not 3)
- Question-answering format with span selection
- Metrics: AUPR, Precision@80% Recall, Precision@90% Recall

This script:
1. Loads the original CUAD test.json with all 41 question types
2. Converts QA format to entity extraction format
3. Evaluates LLM extraction performance
4. Reports F1, Precision, Recall per clause type and overall
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

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src', 'approaches', 'llm'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import pandas as pd
from tqdm import tqdm

# LLM client
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import prompts directly
from cuad_full_prompts import (
    CUAD_CLAUSE_TYPES,
    CUAD_ALL_TYPES,
    create_cuad_full_extraction_prompt,
    create_cuad_focused_prompt,
    create_cuad_batch_prompts,
)


# =============================================================================
# CUAD DATA LOADER
# =============================================================================

def extract_category_from_question(question: str) -> Optional[str]:
    """Extract clause type from CUAD question format."""
    match = re.search(r'related to "([^"]+)"', question)
    if match:
        category = match.group(1)
        # Convert to our format (uppercase, underscores)
        return category.upper().replace(" ", "_").replace("-", "_").replace("/", "_")
    return None


def load_cuad_test_data(data_path: str) -> List[Dict]:
    """
    Load CUAD test data and convert to extraction format.

    Args:
        data_path: Path to test.json

    Returns:
        List of documents with format:
        {
            "doc_id": str,
            "title": str,
            "context": str,
            "ground_truth": {
                "CLAUSE_TYPE": ["answer1", "answer2", ...]
            }
        }
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    documents = []

    for doc in data['data']:
        title = doc['title']

        for para_idx, para in enumerate(doc['paragraphs']):
            context = para['context']

            # Collect ground truth by clause type
            ground_truth = defaultdict(list)

            for qa in para['qas']:
                question = qa['question']
                clause_type = extract_category_from_question(question)

                if clause_type and clause_type in CUAD_ALL_TYPES:
                    # Get answers (multiple spans possible)
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
# LLM EXTRACTION
# =============================================================================

class LLMExtractor:
    """Unified LLM extractor for multiple providers."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = None

        if "gpt" in model_name.lower():
            self._init_openai()
        elif "gemini" in model_name.lower():
            self._init_gemini()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _init_openai(self):
        """Initialize Azure OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        # Map model names to deployment names
        model_map = {
            "gpt-4o": os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            "gpt-4-turbo": "gpt-4-turbo",
        }
        self.deployment = model_map.get(self.model_name, self.model_name)

    def _init_gemini(self):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(self.model_name)

    def extract(self, text: str, clause_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Extract clauses from text.

        Args:
            text: Contract text
            clause_types: Specific types to extract (None = all 41)

        Returns:
            Dict mapping clause_type -> list of extracted texts
        """
        if clause_types:
            prompt = create_cuad_focused_prompt(text, clause_types)
        else:
            prompt = create_cuad_full_extraction_prompt(text)

        try:
            if "gpt" in self.model_name.lower():
                response = self._call_openai(prompt)
            else:
                response = self._call_gemini(prompt)

            return self._parse_response(response)

        except Exception as e:
            print(f"Error in extraction: {e}")
            return {}

    def _call_openai(self, prompt: str) -> str:
        """Call Azure OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
        )
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
        """Parse LLM response to extract entities."""
        # Find JSON in response
        try:
            # Try to find JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get('entities', [])

                # Group by type
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


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common punctuation variations
    text = re.sub(r'[.,;:!?"\'\(\)\[\]\{\}]', '', text)
    return text


def compute_overlap(pred: str, gold: str) -> float:
    """Compute token-level overlap between predicted and gold text."""
    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())

    if not gold_tokens:
        return 0.0

    intersection = pred_tokens & gold_tokens
    return len(intersection) / len(gold_tokens)


def match_extraction(pred_text: str, gold_texts: List[str], threshold: float = 0.5) -> Tuple[bool, Optional[str]]:
    """
    Check if predicted text matches any gold text.

    Args:
        pred_text: Predicted text
        gold_texts: List of gold answers
        threshold: Minimum overlap for match

    Returns:
        (is_match, matched_gold_text)
    """
    for gold in gold_texts:
        overlap = compute_overlap(pred_text, gold)
        if overlap >= threshold:
            return True, gold
    return False, None


def evaluate_document(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions for a single document.

    Returns per-type metrics: TP, FP, FN
    """
    metrics = {}

    all_types = set(list(predictions.keys()) + list(ground_truth.keys()))

    for clause_type in all_types:
        preds = predictions.get(clause_type, [])
        golds = ground_truth.get(clause_type, [])

        tp = 0
        fp = 0
        matched_golds = set()

        # Check each prediction
        for pred in preds:
            is_match, matched = match_extraction(pred, golds, threshold)
            if is_match:
                tp += 1
                matched_golds.add(matched)
            else:
                fp += 1

        # Count unmatched golds as FN
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

    # Compute P/R/F1
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
    max_docs: Optional[int] = None,
    batch_mode: bool = False,
    output_dir: str = "results",
):
    """
    Run full CUAD evaluation.

    Args:
        data_path: Path to CUAD test.json
        model_name: LLM to use
        max_docs: Limit number of documents (for testing)
        batch_mode: Whether to extract in batches (for very long docs)
        output_dir: Directory to save results
    """
    print(f"Loading CUAD data from: {data_path}")
    documents = load_cuad_test_data(data_path)
    print(f"Loaded {len(documents)} documents")

    if max_docs:
        documents = documents[:max_docs]
        print(f"Limited to {max_docs} documents")

    print(f"\nInitializing {model_name} extractor...")
    extractor = LLMExtractor(model_name)

    all_metrics = []
    all_results = []
    total_time = 0

    print("\nRunning extraction...")
    for doc in tqdm(documents):
        start_time = time.time()

        # Extract clauses
        if batch_mode:
            # Batch extraction for long documents
            predictions = {}
            batches = create_cuad_batch_prompts(doc["context"])
            for batch in batches:
                batch_preds = extractor.extract(doc["context"], batch["clause_types"])
                predictions.update(batch_preds)
        else:
            predictions = extractor.extract(doc["context"])

        elapsed = time.time() - start_time
        total_time += elapsed

        # Evaluate
        doc_metrics = evaluate_document(predictions, doc["ground_truth"])
        all_metrics.append(doc_metrics)

        # Store detailed results
        all_results.append({
            "doc_id": doc["doc_id"],
            "predictions": predictions,
            "ground_truth": doc["ground_truth"],
            "metrics": doc_metrics,
            "time": elapsed,
        })

    # Aggregate
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
    print(f"CUAD FULL EVALUATION RESULTS ({model_name})")
    print("="*80)
    print(f"Documents: {len(documents)}")
    print(f"Clause Types: {len(CUAD_ALL_TYPES)}")
    print(f"Total Time: {total_time:.1f}s ({total_time/len(documents):.2f}s/doc)")
    print()

    print("OVERALL METRICS:")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F1:        {overall_f1:.4f}")
    print(f"  Support:   {total_support}")
    print()

    print("PER-TYPE METRICS (sorted by F1):")
    print("-"*80)
    print(f"{'Clause Type':<40} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
    print("-"*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for clause_type, m in sorted_results:
        if m["support"] > 0:  # Only show types with ground truth
            print(f"{clause_type:<40} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary = {
        "model": model_name,
        "timestamp": timestamp,
        "num_documents": len(documents),
        "num_clause_types": len(CUAD_ALL_TYPES),
        "total_time": total_time,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "support": total_support,
        },
        "per_type": results,
    }

    summary_path = os.path.join(output_dir, f"cuad_full_{model_name}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Save CSV for easy analysis
    csv_data = []
    for clause_type, m in results.items():
        csv_data.append({
            "clause_type": clause_type,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "support": m["support"],
        })

    csv_path = os.path.join(output_dir, f"cuad_full_{model_name}_{timestamp}.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CUAD Full Evaluation (41 Clause Types)")
    parser.add_argument("--data", type=str,
                        default="data/cuad/test.json",
                        help="Path to CUAD test.json")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "gemini-2.0-flash"],
                        help="LLM model to use")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum number of documents to evaluate")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch mode for extraction")
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
        max_docs=args.max_docs,
        batch_mode=args.batch,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
