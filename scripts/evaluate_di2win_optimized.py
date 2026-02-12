#!/usr/bin/env python3
"""
Evaluation script for DI2WIN dataset using the new optimized prompts.
"""

import os
import sys
import json
import time
import logging
import re
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new optimized extractor
from src.approaches.llm.di2win_optimized_extractor import (
    DI2WINOptimizedExtractor,
    create_di2win_extractor,
)

# Baseline from paper
DI2WIN_BASELINE = {"GPT-4o (paper)": {"f1": 0.9365}}


@dataclass
class ModelResult:
    """Result from model evaluation."""
    model_name: str
    strategy: str
    precision: float
    recall: float
    f1: float
    latency_avg: float
    num_documents: int


def load_di2win_data(path: str) -> List[Dict]:
    """Load di2win dataset."""
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            # Convert label format to annotations
            annotations = []
            text = doc.get('text', '')
            for label in doc.get('label', []):
                if len(label) >= 3:
                    start, end, entity_type = label[0], label[1], label[2]
                    entity_text = text[start:end] if start < len(text) and end <= len(text) else ""
                    annotations.append({
                        "start": start,
                        "end": end,
                        "type": entity_type,
                        "text": entity_text
                    })
            doc['annotations'] = annotations
            docs.append(doc)
    return docs


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Lowercase, remove extra whitespace, remove punctuation for comparison
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Keep only alphanumeric for comparison
    text = re.sub(r'[^\w\s]', '', text)
    return text


def calculate_metrics(
    predicted: List[Dict],
    ground_truth: List[Dict],
    entity_types: List[str]
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1.

    Uses fuzzy matching for text comparison.
    """
    # Create sets for comparison
    pred_set = set()
    for p in predicted:
        text = normalize_text(p.get('text', ''))
        etype = p.get('type', '')
        if text and etype in entity_types:
            pred_set.add((text, etype))

    gt_set = set()
    for g in ground_truth:
        text = normalize_text(g.get('text', ''))
        etype = g.get('type', '')
        if text and etype in entity_types:
            gt_set.add((text, etype))

    # Calculate metrics
    true_positives = len(pred_set & gt_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def evaluate_extractor(
    extractor: DI2WINOptimizedExtractor,
    docs: List[Dict],
    entity_types: List[str],
    max_docs: int = 50
) -> ModelResult:
    """Evaluate an extractor on the dataset."""
    all_precision = []
    all_recall = []
    all_f1 = []
    latencies = []

    docs_to_eval = docs[:max_docs]

    for i, doc in enumerate(docs_to_eval):
        text = doc.get('text', '')
        ground_truth = doc.get('annotations', [])

        if not text:
            continue

        try:
            start_time = time.time()
            result = extractor.extract(text, entity_types)
            latency = time.time() - start_time

            predicted = result.entities
            latencies.append(latency)

            # Calculate metrics for this document
            p, r, f = calculate_metrics(predicted, ground_truth, entity_types)
            all_precision.append(p)
            all_recall.append(r)
            all_f1.append(f)

        except Exception as e:
            logger.error(f"Error processing document {i}: {e}")
            all_precision.append(0)
            all_recall.append(0)
            all_f1.append(0)
            latencies.append(0)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(docs_to_eval)} documents")

    return ModelResult(
        model_name=extractor.model_name,
        strategy=extractor.strategy,
        precision=sum(all_precision) / len(all_precision) if all_precision else 0,
        recall=sum(all_recall) / len(all_recall) if all_recall else 0,
        f1=sum(all_f1) / len(all_f1) if all_f1 else 0,
        latency_avg=sum(latencies) / len(latencies) if latencies else 0,
        num_documents=len(docs_to_eval)
    )


def get_di2win_models() -> List[Tuple[str, DI2WINOptimizedExtractor, str]]:
    """Get list of model configurations to evaluate."""
    models = []

    # GPT-4o with different strategies
    try:
        models.append((
            "GPT-4o-DI2WIN",
            create_di2win_extractor(provider="azure", model="gpt-4o", strategy="basic"),
            "basic"
        ))
        models.append((
            "GPT-4o-DI2WIN-SC",
            create_di2win_extractor(provider="azure", model="gpt-4o", strategy="self_consistency"),
            "self_consistency"
        ))
        models.append((
            "GPT-4o-DI2WIN-Val",
            create_di2win_extractor(provider="azure", model="gpt-4o", strategy="validated"),
            "validated"
        ))
    except Exception as e:
        logger.warning(f"Could not create Azure GPT-4o extractor: {e}")

    # GPT-4-Turbo
    try:
        models.append((
            "GPT-4-Turbo-DI2WIN",
            create_di2win_extractor(provider="azure", model="gpt-4-turbo", strategy="basic"),
            "basic"
        ))
    except Exception as e:
        logger.warning(f"Could not create Azure GPT-4-Turbo extractor: {e}")

    # Gemini 2.0 Flash
    try:
        models.append((
            "Gemini-2.0-Flash-DI2WIN",
            create_di2win_extractor(provider="gemini", model="gemini-2.0-flash", strategy="basic"),
            "basic"
        ))
    except Exception as e:
        logger.warning(f"Could not create Gemini 2.0 Flash extractor: {e}")

    # Gemini 1.5 Flash (fallback)
    try:
        models.append((
            "Gemini-1.5-Flash-DI2WIN",
            create_di2win_extractor(provider="gemini", model="gemini-1.5-flash", strategy="basic"),
            "basic"
        ))
    except Exception as e:
        logger.warning(f"Could not create Gemini 1.5 Flash extractor: {e}")

    return models


def main():
    """Main evaluation function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("DI2WIN OPTIMIZED PROMPTS EVALUATION")
    print("=" * 80)
    print(f"\nBaseline to beat: F1={DI2WIN_BASELINE['GPT-4o (paper)']['f1']:.4f}")
    print()

    # Load di2win data
    di2win_path = os.path.join(script_dir, "..", "..", "data", "di2win", "contratros_test_files_extractor.jsonl")

    if not os.path.exists(di2win_path):
        print(f"di2win data not found at {di2win_path}")
        return

    di2win_docs = load_di2win_data(di2win_path)
    print(f"Loaded {len(di2win_docs)} di2win documents")

    # Get entity types from data
    entity_types = list(set(
        ann["type"] for doc in di2win_docs for ann in doc.get("annotations", [])
    ))
    print(f"Total entity types: {len(entity_types)}")

    # Count entities by type
    type_counts = Counter()
    for doc in di2win_docs:
        for ann in doc.get('annotations', []):
            type_counts[ann['type']] += 1

    print(f"Top 10 entity types:")
    for etype, count in type_counts.most_common(10):
        print(f"  {count:4d} | {etype}")
    print()

    # Get models
    models = get_di2win_models()
    print(f"Evaluating {len(models)} model configurations\n")

    if not models:
        print("No models available. Check API keys.")
        return

    results = []
    baseline_f1 = DI2WIN_BASELINE["GPT-4o (paper)"]["f1"]

    for name, extractor, strategy in models:
        print(f"\n--- Evaluating: {name} ---")
        result = evaluate_extractor(extractor, di2win_docs, entity_types, max_docs=47)
        results.append(result)

        status = "✓ BEATS BASELINE" if result.f1 > baseline_f1 else "✗ Below baseline"
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1:        {result.f1:.4f} ({result.f1 - baseline_f1:+.4f} vs baseline) {status}")
        print(f"  Latency:   {result.latency_avg:.2f}s avg")

    # Results summary
    print("\n" + "=" * 80)
    print("DI2WIN OPTIMIZED RESULTS SUMMARY")
    print("=" * 80)

    # Sort by F1
    results.sort(key=lambda x: x.f1, reverse=True)

    print(f"\n{'Model':<30} {'Strategy':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'vs Baseline':>12}")
    print("-" * 95)

    for r in results:
        diff = r.f1 - baseline_f1
        symbol = "+" if diff >= 0 else ""
        print(f"{r.model_name:<30} {r.strategy:<18} {r.precision:>10.4f} {r.recall:>10.4f} {r.f1:>10.4f} {symbol}{diff:>11.4f}")

    # Save to CSV
    csv_path = os.path.join(results_dir, f"di2win_new_prompts_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Strategy", "Precision", "Recall", "F1", "vs_Baseline", "Latency_Avg", "Num_Documents"])
        for r in results:
            diff = r.f1 - baseline_f1
            writer.writerow([r.model_name, r.strategy, f"{r.precision:.4f}", f"{r.recall:.4f}",
                           f"{r.f1:.4f}", f"{diff:+.4f}", f"{r.latency_avg:.2f}", r.num_documents])

    print(f"\nResults saved to: {csv_path}")

    # Best model
    if results:
        best = results[0]
        print(f"\n🏆 BEST MODEL: {best.model_name} ({best.strategy})")
        print(f"   F1 = {best.f1:.4f} ({'+' if best.f1 > baseline_f1 else ''}{best.f1 - baseline_f1:.4f} vs baseline)")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
