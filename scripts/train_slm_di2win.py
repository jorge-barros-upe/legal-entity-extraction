#!/usr/bin/env python3
"""
Fine-tuning script for SLM models on DI2WIN dataset.

Supported models:
1. neuralmind/bert-base-portuguese-cased (BERTimbau)
2. neuralmind/bert-large-portuguese-cased (BERTimbau Large)
3. rufimelo/Legal-BERTimbau-base (Legal-BERTimbau)
4. rufimelo/Legal-BERTimbau-large (Legal-BERTimbau Large)
5. xlm-roberta-base (Multilingual)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add project root to path
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
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/torch not installed. Install with: pip install transformers torch datasets")

try:
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False
    logger.warning("seqeval not installed. Install with: pip install seqeval")


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    "bertimbau-base": {
        "name": "neuralmind/bert-base-portuguese-cased",
        "description": "BERTimbau Base - Portuguese BERT",
        "max_length": 512,
        "batch_size": 4,  # Reduced for MPS/CPU
        "learning_rate": 2e-5,
    },
    "bertimbau-large": {
        "name": "neuralmind/bert-large-portuguese-cased",
        "description": "BERTimbau Large - Portuguese BERT Large",
        "max_length": 512,
        "batch_size": 2,  # Reduced for MPS/CPU
        "learning_rate": 1e-5,
    },
    "legal-bertimbau-base": {
        "name": "rufimelo/Legal-BERTimbau-base",
        "description": "Legal-BERTimbau - Portuguese Legal BERT",
        "max_length": 512,
        "batch_size": 4,  # Reduced for MPS/CPU
        "learning_rate": 2e-5,
    },
    "legal-bertimbau-large": {
        "name": "rufimelo/Legal-BERTimbau-large",
        "description": "Legal-BERTimbau Large - Portuguese Legal BERT Large",
        "max_length": 512,
        "batch_size": 2,  # Reduced for MPS/CPU
        "learning_rate": 1e-5,
    },
    "xlm-roberta-base": {
        "name": "xlm-roberta-base",
        "description": "XLM-RoBERTa Base - Multilingual",
        "max_length": 512,
        "batch_size": 4,  # Reduced for MPS/CPU
        "learning_rate": 2e-5,
    },
}

# High-frequency entity types to focus on (from di2win_optimized_prompts.py)
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
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_di2win_data(path: str, entity_types: Optional[List[str]] = None) -> List[Dict]:
    """Load DI2WIN dataset from JSONL file."""
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            text = doc.get('text', '')

            # Convert label format to entities
            entities = []
            for label in doc.get('label', []):
                if len(label) >= 3:
                    start, end, entity_type = label[0], label[1], label[2]

                    # Filter by entity types if specified
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


def get_entity_types_from_data(docs: List[Dict]) -> List[str]:
    """Extract all unique entity types from data."""
    types = set()
    for doc in docs:
        for entity in doc.get("entities", []):
            types.add(entity["type"])
    return sorted(list(types))


def build_label_mappings(entity_types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build BIO label mappings."""
    labels = ["O"]  # Outside
    for entity_type in entity_types:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


# =============================================================================
# TOKENIZATION AND DATASET PREPARATION
# =============================================================================

class NERDataset:
    """Dataset class for NER fine-tuning."""

    def __init__(
        self,
        tokenizer,
        data: List[Dict],
        label2id: Dict[str, int],
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.label2id = label2id
        self.max_length = max_length
        self.stride = stride

        # Process data with sliding window for long documents
        self.processed = self._process_data()

    def _process_data(self) -> List[Dict]:
        """Process data with sliding window for long documents."""
        processed = []

        for doc in self.data:
            text = doc["text"]
            entities = doc["entities"]

            # For short documents, process directly
            if len(text) < self.max_length * 3:  # Rough char estimate
                tokenized = self._tokenize_and_align(text, entities)
                if tokenized:
                    processed.append(tokenized)
            else:
                # Sliding window for long documents
                windows = self._create_windows(text, entities)
                processed.extend(windows)

        return processed

    def _create_windows(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Create sliding windows for long documents."""
        windows = []
        char_stride = self.stride * 4  # Rough char estimate
        char_window = self.max_length * 4

        start = 0
        while start < len(text):
            end = min(start + char_window, len(text))
            window_text = text[start:end]

            # Adjust entities for this window
            window_entities = []
            for entity in entities:
                e_start, e_end = entity["start"], entity["end"]

                # Check if entity is in this window
                if e_start >= start and e_end <= end:
                    window_entities.append({
                        "start": e_start - start,
                        "end": e_end - start,
                        "type": entity["type"],
                        "text": entity.get("text", "")
                    })

            tokenized = self._tokenize_and_align(window_text, window_entities)
            if tokenized:
                windows.append(tokenized)

            start += char_stride

            # Don't create too many windows per document
            if len(windows) >= 5:
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
            labels = [-100] * len(offset_mapping)  # Default to ignore (for padding)

            # First pass: mark all real tokens as O
            for idx, (start, end) in enumerate(offset_mapping):
                if start != end:  # Real token (not special/padding)
                    labels[idx] = self.label2id["O"]

            # Second pass: assign entity labels
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
                    if start == end:  # Special/padding token
                        continue

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
# TRAINING
# =============================================================================

def compute_metrics(eval_pred, id2label: Dict[int, str]):
    """Compute NER metrics using seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Convert to label sequences
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:  # Ignore padding/special tokens
                true_seq.append(id2label.get(label, "O"))
                pred_seq_labels.append(id2label.get(pred, "O"))
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_labels)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels)
    }


def train_model(
    model_key: str,
    train_data: List[Dict],
    val_data: List[Dict],
    entity_types: List[str],
    output_dir: str,
    num_epochs: int = 10,
    use_fp16: bool = True
) -> Dict[str, Any]:
    """Train a single model."""

    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers not installed")

    model_config = MODELS[model_key]
    model_name = model_config["name"]
    max_length = model_config["max_length"]
    batch_size = model_config["batch_size"]
    learning_rate = model_config["learning_rate"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_key}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'='*60}")

    # Build label mappings
    label2id, id2label = build_label_mappings(entity_types)
    num_labels = len(label2id)
    logger.info(f"Number of labels: {num_labels} ({len(entity_types)} entity types)")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model.to(device)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = NERDataset(tokenizer, train_data, label2id, max_length).to_hf_dataset()
    val_dataset = NERDataset(tokenizer, val_data, label2id, max_length).to_hf_dataset()

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training arguments
    model_output_dir = os.path.join(output_dir, model_key)
    os.makedirs(model_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=8,  # Higher for smaller batches
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=use_fp16 and device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir=os.path.join(model_output_dir, "logs"),
        report_to="none",  # Disable wandb etc
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    # Trainer with metrics
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save best model
    trainer.save_model(os.path.join(model_output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(model_output_dir, "best"))

    # Final evaluation
    logger.info("Final evaluation...")
    eval_result = trainer.evaluate()

    # Log results
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
        "checkpoint_path": os.path.join(model_output_dir, "best")
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SLM models on DI2WIN dataset")
    parser.add_argument("--models", nargs="+", default=["legal-bertimbau-base", "bertimbau-base"],
                       help="Models to train (default: legal-bertimbau-base, bertimbau-base)")
    parser.add_argument("--train-path", default=None, help="Path to training data")
    parser.add_argument("--test-path", default=None, help="Path to test data")
    parser.add_argument("--output-dir", default="checkpoints/slm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--use-high-freq-types", action="store_true", default=True,
                       help="Use only high-frequency entity types")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 training")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed. Install with:")
        print("  pip install transformers torch datasets seqeval")
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
    print("SLM FINE-TUNING FOR DI2WIN DATASET")
    print("=" * 70)
    print(f"\nTrain data: {train_path}")
    print(f"Test data:  {test_path}")
    print(f"Output:     {output_dir}")
    print(f"Models:     {', '.join(args.models)}")
    print(f"Epochs:     {args.epochs}")
    print()

    # Validate models
    for model in args.models:
        if model not in MODELS:
            print(f"ERROR: Unknown model '{model}'. Available: {list(MODELS.keys())}")
            return

    # Load data
    print("Loading data...")

    # Entity types to use
    if args.use_high_freq_types:
        entity_types = HIGH_FREQUENCY_TYPES
        print(f"Using {len(entity_types)} high-frequency entity types")
    else:
        entity_types = None  # Will be extracted from data

    train_data = load_di2win_data(train_path, entity_types)
    test_data = load_di2win_data(test_path, entity_types)

    print(f"Loaded {len(train_data)} training documents")
    print(f"Loaded {len(test_data)} test documents")

    # Get entity types from data if not specified
    if entity_types is None:
        entity_types = get_entity_types_from_data(train_data + test_data)
        print(f"Found {len(entity_types)} entity types in data")

    # Use test data as validation (small dataset)
    # In a real scenario, you'd split train into train/val
    val_data = test_data

    # Train each model
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
                use_fp16=not args.no_fp16
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

    print(f"\n{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)

    for r in sorted(all_results, key=lambda x: x["f1"], reverse=True):
        print(f"{r['model']:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

    # Save results
    results_path = os.path.join(results_dir, f"slm_training_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Best model
    if all_results:
        best = max(all_results, key=lambda x: x["f1"])
        print(f"\nBEST MODEL: {best['model']}")
        print(f"  F1 = {best['f1']:.4f}")
        print(f"  Checkpoint: {best['checkpoint_path']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
