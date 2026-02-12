#!/usr/bin/env python3
"""
Optimized SLM Training Script for DI2WIN Dataset.

Key improvements over original:
1. More epochs with early stopping
2. Class-weighted loss for imbalanced data
3. Proper train/val split
4. Learning rate scheduling
5. Better hyperparameters
6. Focal loss option for rare classes
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        get_linear_schedule_with_warmup
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/torch not installed")

try:
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False


# =============================================================================
# OPTIMIZED MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    "legal-bertimbau-base": {
        "name": "rufimelo/Legal-BERTimbau-base",
        "max_length": 512,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
    },
    "legal-bertimbau-large": {
        "name": "rufimelo/Legal-BERTimbau-large",
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
    },
    "bertimbau-base": {
        "name": "neuralmind/bert-base-portuguese-cased",
        "max_length": 512,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
    },
    "bertimbau-large": {
        "name": "neuralmind/bert-large-portuguese-cased",
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
    },
}

# High-frequency entity types (35 most common)
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_di2win_data(path: str, entity_types: Optional[List[str]] = None) -> List[Dict]:
    """Load DI2WIN dataset from JSONL file."""
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            text = doc.get('text', '')

            entities = []
            for label in doc.get('label', []):
                if len(label) >= 3:
                    start, end, entity_type = label[0], label[1], label[2]

                    if entity_types and entity_type not in entity_types:
                        continue

                    entity_text = text[start:end] if start < len(text) and end <= len(text) else ""
                    entities.append({
                        "start": start,
                        "end": end,
                        "type": entity_type,
                        "text": entity_text
                    })

            docs.append({
                "id": doc.get("id", ""),
                "text": text,
                "entities": entities
            })

    return docs


def split_train_val(data: List[Dict], val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_data = shuffled[:val_size]
    train_data = shuffled[val_size:]

    return train_data, val_data


def get_entity_types_from_data(docs: List[Dict]) -> List[str]:
    """Extract all unique entity types from data."""
    types = set()
    for doc in docs:
        for entity in doc.get("entities", []):
            types.add(entity["type"])
    return sorted(list(types))


def compute_class_weights(docs: List[Dict], label2id: Dict[str, int]) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    label_counts = Counter()

    for doc in docs:
        label_counts["O"] += len(doc["text"].split())  # Rough estimate
        for entity in doc.get("entities", []):
            entity_type = entity["type"]
            label_b = f"B-{entity_type}"
            label_i = f"I-{entity_type}"
            if label_b in label2id:
                label_counts[label_b] += 1
                # Estimate I labels based on entity length
                entity_len = max(1, len(entity.get("text", "").split()) - 1)
                label_counts[label_i] += entity_len

    # Compute inverse frequency weights
    total = sum(label_counts.values())
    num_classes = len(label2id)
    weights = torch.ones(num_classes)

    for label, idx in label2id.items():
        count = label_counts.get(label, 1)
        # Inverse frequency with smoothing
        weights[idx] = total / (num_classes * count + 1)

    # Normalize and clip
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.1, max=10.0)

    return weights


def build_label_mappings(entity_types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build BIO label mappings."""
    labels = ["O"]
    for entity_type in entity_types:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


# =============================================================================
# DATASET CLASS
# =============================================================================

class OptimizedNERDataset:
    """Optimized NER Dataset with better windowing."""

    def __init__(
        self,
        tokenizer,
        data: List[Dict],
        label2id: Dict[str, int],
        max_length: int = 512,
        stride: int = 128,
        max_windows_per_doc: int = 10
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.label2id = label2id
        self.max_length = max_length
        self.stride = stride
        self.max_windows_per_doc = max_windows_per_doc

        self.processed = self._process_data()

    def _process_data(self) -> List[Dict]:
        """Process data with optimized sliding window."""
        processed = []

        for doc in self.data:
            text = doc["text"]
            entities = doc["entities"]

            # Tokenize full document to get token count
            full_tokens = self.tokenizer(text, add_special_tokens=False)
            num_tokens = len(full_tokens["input_ids"])

            if num_tokens <= self.max_length - 2:  # Account for [CLS] and [SEP]
                tokenized = self._tokenize_and_align(text, entities)
                if tokenized:
                    processed.append(tokenized)
            else:
                # Sliding window
                windows = self._create_windows(text, entities)
                processed.extend(windows)

        return processed

    def _create_windows(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Create overlapping windows for long documents."""
        windows = []

        # Tokenize to get character offsets
        tokenized_full = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        offsets = tokenized_full["offset_mapping"]
        num_tokens = len(offsets)

        # Calculate window positions
        effective_length = self.max_length - 2  # Account for special tokens

        start_token = 0
        while start_token < num_tokens:
            end_token = min(start_token + effective_length, num_tokens)

            # Get character range for this window
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]

            window_text = text[char_start:char_end]

            # Adjust entities for this window
            window_entities = []
            for entity in entities:
                e_start, e_end = entity["start"], entity["end"]

                if e_start >= char_start and e_end <= char_end:
                    window_entities.append({
                        "start": e_start - char_start,
                        "end": e_end - char_start,
                        "type": entity["type"],
                        "text": entity.get("text", "")
                    })

            tokenized = self._tokenize_and_align(window_text, window_entities)
            if tokenized:
                windows.append(tokenized)

            if end_token >= num_tokens:
                break

            start_token += self.stride

            if len(windows) >= self.max_windows_per_doc:
                break

        return windows

    def _tokenize_and_align(self, text: str, entities: List[Dict]) -> Optional[Dict]:
        """Tokenize text and align entity labels."""
        try:
            tokenized = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True
            )

            offset_mapping = tokenized.pop("offset_mapping")
            labels = [-100] * len(offset_mapping)

            # Mark real tokens as O
            for idx, (start, end) in enumerate(offset_mapping):
                if start != end:
                    labels[idx] = self.label2id["O"]

            # Assign entity labels
            for entity in entities:
                entity_start = entity["start"]
                entity_end = entity["end"]
                entity_type = entity["type"]

                label_b = f"B-{entity_type}"
                label_i = f"I-{entity_type}"

                if label_b not in self.label2id:
                    continue

                is_first = True
                for idx, (start, end) in enumerate(offset_mapping):
                    if start == end:
                        continue

                    # Token overlaps with entity
                    if start >= entity_start and end <= entity_end:
                        if is_first:
                            labels[idx] = self.label2id[label_b]
                            is_first = False
                        else:
                            labels[idx] = self.label2id[label_i]

            tokenized["labels"] = labels
            return tokenized

        except Exception as e:
            logger.warning(f"Error tokenizing: {e}")
            return None

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        return Dataset.from_list(self.processed)


# =============================================================================
# WEIGHTED LOSS TRAINER
# =============================================================================

class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            device = logits.device
            weights = self.class_weights.to(device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred, id2label: Dict[int, str]):
    """Compute NER metrics using seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_seq.append(id2label.get(label, "O"))
                pred_seq_labels.append(id2label.get(pred, "O"))
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_labels)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels)
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model_key: str,
    train_data: List[Dict],
    val_data: List[Dict],
    entity_types: List[str],
    output_dir: str,
    num_epochs: int = 15,
    use_class_weights: bool = True,
    use_fp16: bool = True,
    patience: int = 5
) -> Dict[str, Any]:
    """Train model with optimized settings."""

    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers not installed")

    model_config = MODELS[model_key]
    model_name = model_config["name"]
    max_length = model_config["max_length"]
    batch_size = model_config["batch_size"]
    learning_rate = model_config["learning_rate"]
    weight_decay = model_config["weight_decay"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_key}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    logger.info(f"{'='*60}")

    # Build label mappings
    label2id, id2label = build_label_mappings(entity_types)
    num_labels = len(label2id)
    logger.info(f"Number of labels: {num_labels} ({len(entity_types)} entity types)")

    # Compute class weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_data, label2id)
        logger.info(f"Class weights computed (min={class_weights.min():.2f}, max={class_weights.max():.2f})")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model.to(device)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = OptimizedNERDataset(
        tokenizer, train_data, label2id, max_length, stride=128
    ).to_hf_dataset()

    val_dataset = OptimizedNERDataset(
        tokenizer, val_data, label2id, max_length, stride=128
    ).to_hf_dataset()

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training arguments
    model_output_dir = os.path.join(output_dir, model_key)
    os.makedirs(model_output_dir, exist_ok=True)

    # Calculate steps
    total_steps = (len(train_dataset) // batch_size) * num_epochs
    warmup_steps = int(total_steps * 0.1)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=use_fp16 and device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        logging_dir=os.path.join(model_output_dir, "logs"),
        report_to="none",
        dataloader_num_workers=0,
        lr_scheduler_type="cosine",
    )

    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, id2label)

    # Use weighted loss trainer
    trainer_class = WeightedLossTrainer if use_class_weights else Trainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        class_weights=class_weights if use_class_weights else None
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save best model
    trainer.save_model(os.path.join(model_output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(model_output_dir, "best"))

    # Save label mappings
    with open(os.path.join(model_output_dir, "best", "label_mappings.json"), 'w') as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f)

    # Final evaluation
    logger.info("Final evaluation...")
    eval_result = trainer.evaluate()

    logger.info(f"\n{'-'*40}")
    logger.info(f"Results for {model_key}:")
    logger.info(f"  Precision: {eval_result.get('eval_precision', 0):.4f}")
    logger.info(f"  Recall:    {eval_result.get('eval_recall', 0):.4f}")
    logger.info(f"  F1:        {eval_result.get('eval_f1', 0):.4f}")
    logger.info(f"{'-'*40}")

    return {
        "model": model_key,
        "model_name": model_name,
        "precision": eval_result.get("eval_precision", 0),
        "recall": eval_result.get("eval_recall", 0),
        "f1": eval_result.get("eval_f1", 0),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "use_class_weights": use_class_weights,
        "checkpoint_path": os.path.join(model_output_dir, "best")
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized SLM Training for DI2WIN")
    parser.add_argument("--models", nargs="+", default=["legal-bertimbau-base"],
                       choices=list(MODELS.keys()), help="Models to train")
    parser.add_argument("--train-path", default=None, help="Training data path")
    parser.add_argument("--test-path", default=None, help="Test data path")
    parser.add_argument("--output-dir", default="checkpoints/slm_optimized", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--use-all-types", action="store_true", help="Use all 143 entity types")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, "..", "..")

    train_path = args.train_path or os.path.join(base_dir, "data", "di2win", "contratos_train_files_extractor.jsonl")
    test_path = args.test_path or os.path.join(base_dir, "data", "di2win", "contratros_test_files_extractor.jsonl")
    output_dir = os.path.join(base_dir, "experiments", args.output_dir, timestamp)
    results_dir = os.path.join(base_dir, "experiments", "results")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("OPTIMIZED SLM TRAINING FOR DI2WIN")
    print("=" * 70)
    print(f"\nTrain: {train_path}")
    print(f"Test:  {test_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Class weights: {not args.no_class_weights}")
    print(f"Early stopping patience: {args.patience}")
    print()

    # Entity types
    entity_types = None if args.use_all_types else HIGH_FREQUENCY_TYPES

    # Load data
    print("Loading data...")
    train_data_full = load_di2win_data(train_path, entity_types)
    test_data = load_di2win_data(test_path, entity_types)

    print(f"Loaded {len(train_data_full)} training documents")
    print(f"Loaded {len(test_data)} test documents")

    # Split train into train/val
    train_data, val_data = split_train_val(train_data_full, val_ratio=args.val_ratio)
    print(f"Train/Val split: {len(train_data)}/{len(val_data)}")

    # Get entity types
    if entity_types is None:
        entity_types = get_entity_types_from_data(train_data_full + test_data)
    print(f"Entity types: {len(entity_types)}")

    # Count entities
    total_entities = sum(len(d["entities"]) for d in train_data_full)
    print(f"Total entities in training: {total_entities}")

    # Train models
    all_results = []

    for model_key in args.models:
        try:
            result = train_model(
                model_key=model_key,
                train_data=train_data,
                val_data=val_data,
                entity_types=entity_types,
                output_dir=output_dir,
                num_epochs=args.epochs,
                use_class_weights=not args.no_class_weights,
                use_fp16=not args.no_fp16,
                patience=args.patience
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {model_key}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'P':>10} {'R':>10} {'F1':>10}")
    print("-" * 60)

    for r in sorted(all_results, key=lambda x: x["f1"], reverse=True):
        print(f"{r['model']:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

    # Save results
    results_path = os.path.join(results_dir, f"slm_optimized_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    if all_results:
        best = max(all_results, key=lambda x: x["f1"])
        print(f"\nBEST MODEL: {best['model']}")
        print(f"  F1 = {best['f1']:.4f}")
        print(f"  Checkpoint: {best['checkpoint_path']}")


if __name__ == "__main__":
    main()
