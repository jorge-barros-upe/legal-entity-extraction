#!/usr/bin/env python3
"""
Evaluation script for fine-tuned SLM models on DI2WIN dataset.

This script evaluates trained SLM checkpoints and compares them with LLM baselines.
"""

import os
import sys
import json
import time
import logging
import argparse
import re
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed")

# High-frequency entity types
HIGH_FREQUENCY_TYPES = [
    "nome->socio", "cpf->socio", "rg->socio", "nacionalidade->socio",
    "estado_civil->socio", "trabalho->socio", "nome_da_rua->socio",
    "numero_da_rua->socio", "bairro->socio", "municipio_(ou_cidade)->socio",
    "uf->socio", "cep->socio", "nome_(ou_razao_social)->sociedade",
    "cnpj->sociedade", "nire->sociedade", "nome_da_rua->sociedade",
    "numero_da_rua->sociedade", "bairro->sociedade", "municipio_(ou_cidade)->sociedade",
    "uf->sociedade", "cep->sociedade", "capital_social->sociedade",
    "numero_de_quotas_total->sociedade", "valor_nominal_quota->sociedade",
    "numero_de_quotas->socio", "valor_total_das_cotas->socio",
    "percentual_de_participacao->socio", "nome->adm_1", "nome->adm_2",
    "poderes->administrador", "vetos->administrador",
    "data_de_registro_do_contrato", "numero_do_registro_do_contrato",
    "data->assinatura_contrato", "quem_assina",
]


class SLMEvaluator:
    """Evaluator for fine-tuned SLM models."""

    def __init__(self, checkpoint_path: str, max_length: int = 512, stride: int = 256):
        self.checkpoint_path = checkpoint_path
        self.max_length = max_length
        self.stride = stride

        self.tokenizer = None
        self.model = None
        self.device = None
        self.id2label = None
        self.label2id = None

        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")

        logger.info(f"Loading model from {self.checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint_path)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Label mappings
        self.id2label = self.model.config.id2label
        self.label2id = {v: int(k) for k, v in self.id2label.items()}

        logger.info(f"Loaded model on {self.device}")
        logger.info(f"Number of labels: {len(self.id2label)}")

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if len(text) > self.max_length * 3:
            return self._extract_long_document(text)
        else:
            return self._extract_single(text)

    def _extract_single(self, text: str) -> List[Dict[str, Any]]:
        """Extract from a single text segment."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Convert to entities
        entities = self._predictions_to_entities(predictions, probabilities, offset_mapping, text)
        return entities

    def _extract_long_document(self, text: str) -> List[Dict[str, Any]]:
        """Extract from long document using sliding window."""
        all_entities = []
        char_stride = self.stride * 4
        char_window = self.max_length * 4

        start = 0
        while start < len(text):
            end = min(start + char_window, len(text))
            window_text = text[start:end]

            window_entities = self._extract_single(window_text)

            # Adjust offsets
            for entity in window_entities:
                entity["start"] += start
                entity["end"] += start

            all_entities.extend(window_entities)
            start += char_stride

            if start > len(text) * 2:  # Safety limit
                break

        # Deduplicate
        return self._deduplicate_entities(all_entities)

    def _predictions_to_entities(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        offset_mapping: List[Tuple[int, int]],
        text: str
    ) -> List[Dict[str, Any]]:
        """Convert BIO predictions to entity list."""
        entities = []
        current_entity = None

        for idx, pred_id in enumerate(predictions):
            if idx >= len(offset_mapping):
                break

            start, end = offset_mapping[idx]
            if start == end:  # Special token
                continue

            label = self.id2label.get(str(pred_id), self.id2label.get(pred_id, "O"))
            confidence = float(probabilities[idx][pred_id])

            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "type": entity_type,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "confidence": confidence
                }
            elif label.startswith("I-"):
                if current_entity and label[2:] == current_entity["type"]:
                    # Continue entity
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["end"] = end
                    current_entity["confidence"] = min(current_entity["confidence"], confidence)
                else:
                    # I without matching B
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = {
                        "type": entity_type,
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "confidence": confidence
                    }

        if current_entity:
            entities.append(current_entity)

        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Deduplicate overlapping entities."""
        if not entities:
            return []

        # Sort by start, then by length (longer first)
        sorted_entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))

        unique = []
        for entity in sorted_entities:
            # Check overlap
            overlaps = False
            for existing in unique:
                if entity["start"] < existing["end"] and entity["end"] > existing["start"]:
                    overlaps = True
                    # Keep higher confidence
                    if entity.get("confidence", 0) > existing.get("confidence", 0):
                        unique.remove(existing)
                        unique.append(entity)
                    break

            if not overlaps:
                unique.append(entity)

        return unique


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def calculate_metrics(
    predicted: List[Dict],
    ground_truth: List[Dict],
    entity_types: List[str]
) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1."""
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

    true_positives = len(pred_set & gt_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def load_di2win_data(path: str) -> List[Dict]:
    """Load DI2WIN dataset."""
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            text = doc.get('text', '')

            annotations = []
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


def evaluate_checkpoint(
    checkpoint_path: str,
    test_docs: List[Dict],
    entity_types: List[str],
    model_name: str = "SLM"
) -> Dict[str, Any]:
    """Evaluate a single checkpoint."""
    logger.info(f"\nEvaluating: {model_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    try:
        evaluator = SLMEvaluator(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {"model": model_name, "error": str(e)}

    all_precision = []
    all_recall = []
    all_f1 = []
    latencies = []

    for i, doc in enumerate(test_docs):
        text = doc.get('text', '')
        ground_truth = doc.get('annotations', [])

        if not text:
            continue

        try:
            start_time = time.time()
            predicted = evaluator.extract(text)
            latency = time.time() - start_time

            latencies.append(latency)

            p, r, f = calculate_metrics(predicted, ground_truth, entity_types)
            all_precision.append(p)
            all_recall.append(r)
            all_f1.append(f)

        except Exception as e:
            logger.error(f"Error on doc {i}: {e}")
            all_precision.append(0)
            all_recall.append(0)
            all_f1.append(0)
            latencies.append(0)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(test_docs)} documents")

    result = {
        "model": model_name,
        "checkpoint": checkpoint_path,
        "precision": sum(all_precision) / len(all_precision) if all_precision else 0,
        "recall": sum(all_recall) / len(all_recall) if all_recall else 0,
        "f1": sum(all_f1) / len(all_f1) if all_f1 else 0,
        "latency_avg": sum(latencies) / len(latencies) if latencies else 0,
        "num_documents": len(test_docs)
    }

    logger.info(f"  Precision: {result['precision']:.4f}")
    logger.info(f"  Recall:    {result['recall']:.4f}")
    logger.info(f"  F1:        {result['f1']:.4f}")
    logger.info(f"  Latency:   {result['latency_avg']:.2f}s")

    return result


def find_checkpoints(base_dir: str) -> List[Tuple[str, str]]:
    """Find all trained checkpoints."""
    checkpoints = []

    for root, dirs, files in os.walk(base_dir):
        if "best" in dirs:
            best_path = os.path.join(root, "best")
            if os.path.exists(os.path.join(best_path, "config.json")):
                # Extract model name from path
                model_name = os.path.basename(root)
                checkpoints.append((model_name, best_path))

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLM checkpoints on DI2WIN")
    parser.add_argument("--checkpoint-dir", default="checkpoints/slm",
                       help="Directory containing trained checkpoints")
    parser.add_argument("--checkpoint", default=None,
                       help="Specific checkpoint to evaluate")
    parser.add_argument("--test-path", default=None, help="Path to test data")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, "..", "..")
    results_dir = os.path.join(base_dir, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    test_path = args.test_path or os.path.join(base_dir, "data", "di2win", "contratros_test_files_extractor.jsonl")

    print("=" * 70)
    print("SLM EVALUATION FOR DI2WIN")
    print("=" * 70)

    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_docs = load_di2win_data(test_path)
    print(f"Loaded {len(test_docs)} test documents")

    # Entity types
    entity_types = HIGH_FREQUENCY_TYPES
    print(f"Evaluating on {len(entity_types)} entity types")

    # Find checkpoints
    if args.checkpoint:
        checkpoints = [("custom", args.checkpoint)]
    else:
        checkpoint_base = os.path.join(base_dir, "experiments", args.checkpoint_dir)
        checkpoints = find_checkpoints(checkpoint_base)

    if not checkpoints:
        print(f"\nNo checkpoints found in {checkpoint_base}")
        print("Train models first with: python train_slm_di2win.py")
        return

    print(f"\nFound {len(checkpoints)} checkpoints to evaluate")

    # Evaluate each checkpoint
    results = []
    for model_name, checkpoint_path in checkpoints:
        result = evaluate_checkpoint(checkpoint_path, test_docs, entity_types, model_name)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Latency':>10}")
    print("-" * 75)

    for r in sorted(results, key=lambda x: x.get("f1", 0), reverse=True):
        if "error" in r:
            print(f"{r['model']:<30} ERROR: {r['error']}")
        else:
            print(f"{r['model']:<30} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['latency_avg']:>9.2f}s")

    # Save results
    csv_path = os.path.join(results_dir, f"slm_evaluation_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Type", "Precision", "Recall", "F1", "Latency_Avg", "Num_Documents"])
        for r in results:
            if "error" not in r:
                writer.writerow([
                    r["model"], "slm",
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                    f"{r['f1']:.4f}", f"{r['latency_avg']:.2f}",
                    r["num_documents"]
                ])

    print(f"\nResults saved to: {csv_path}")

    # Best model
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x["f1"])
        print(f"\nBEST SLM: {best['model']}")
        print(f"  F1 = {best['f1']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
