#!/usr/bin/env python3
"""Quick evaluation of a CUAD SLM checkpoint on the test set."""

import os
import sys
import json
import time
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import from training script
sys.path.insert(0, os.path.dirname(__file__))
from train_slm_cuad import (
    load_cuad_squad_data, evaluate_document_level,
    LABEL2ID, ID2LABEL, LABELS
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-path", type=str, default="data/cuad/test.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model (tokenizer from original model, weights from checkpoint)
    print(f"Loading checkpoint: {args.checkpoint}")
    # Try loading tokenizer from checkpoint first, fall back to original model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    except Exception:
        print("  Tokenizer not in checkpoint, loading from nlpaueb/legal-bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    model.to(device)

    # Load test data
    print(f"Loading test data: {args.test_path}")
    test_docs = load_cuad_squad_data(args.test_path)
    print(f"Test documents: {len(test_docs)}")
    test_entities = sum(len(d["entities"]) for d in test_docs)
    print(f"Test entities: {test_entities}")

    # Evaluate
    print("\nRunning document-level evaluation...")
    results = evaluate_document_level(
        model=model,
        tokenizer=tokenizer,
        documents=test_docs,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        max_length=args.max_length,
        stride=args.stride,
        device=device,
    )

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (document-level, entity matching)")
    print(f"{'='*60}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")
    print(f"  Avg time/doc: {results['avg_time_per_doc']:.3f}s")

    print(f"\n  Per-type results (top-20 by support):")
    per_type = results["per_type"]
    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["support"], reverse=True)
    print(f"  {'Type':<45} {'P':>6} {'R':>6} {'F1':>6} {'Supp':>6}")
    print(f"  {'-'*70}")
    for t, m in sorted_types[:20]:
        print(f"  {t:<45} {m['precision']:>6.2%} {m['recall']:>6.2%} {m['f1']:>6.2%} {m['support']:>6}")

    # Save
    out_path = os.path.join(os.path.dirname(args.checkpoint), "test_eval_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
