#!/usr/bin/env python3
"""
SLM Training for CUAD Dataset using Legal-BERT.

Fine-tunes Legal-BERT (Chalkidis et al., 2020) on the CUAD dataset
for token-level clause extraction (41 clause types).

Uses the same optimized approach as DI2WIN experiments:
- Class-weighted cross-entropy loss
- Early stopping with patience
- Cosine learning rate scheduler
- Sliding window for long documents
- seqeval entity-level metrics
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import Dataset
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


# =============================================================================
# CUAD CLAUSE TYPES (41 types from Hendrycks et al. 2021)
# =============================================================================

CUAD_CLAUSE_TYPES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "No-Solicit Of Employees", "Non-Disparagement",
    "Termination For Convenience", "Rofr/Rofo/Rofn", "Change Of Control",
    "Anti-Assignment", "Revenue/Profit Sharing", "Price Restrictions",
    "Minimum Commitment", "Volume Restriction", "Ip Ownership Assignment",
    "Joint Ip Ownership", "License Grant", "Non-Transferable License",
    "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
    "Source Code Escrow", "Post-Termination Services", "Audit Rights",
    "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
    "Warranty Duration", "Insurance", "Covenant Not To Sue",
    "Third Party Beneficiary", "Competitive Restriction Exception",
]


def normalize_type(clause_type: str) -> str:
    """Normalize clause type to label format."""
    return clause_type.upper().replace(" ", "_").replace("/", "_").replace("-", "_")


# Build label mappings
ENTITY_TYPES = [normalize_type(ct) for ct in CUAD_CLAUSE_TYPES]
LABELS = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cuad_squad_data(data_path: str) -> List[Dict]:
    """
    Load CUAD data from SQuAD-format JSON and convert to NER format.
    Each document becomes: {id, text, entities: [{start, end, type, text}]}
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    documents = []

    for entry in data.get('data', []):
        title = entry.get('title', '')

        for para in entry.get('paragraphs', []):
            context = para.get('context', '')
            entities = []

            for qa in para.get('qas', []):
                question = qa.get('question', '')
                answers = qa.get('answers', [])
                is_impossible = qa.get('is_impossible', False)

                if is_impossible or not answers:
                    continue

                # Extract clause type from question (between quotes)
                if '"' in question:
                    start = question.find('"') + 1
                    end = question.find('"', start)
                    if end > start:
                        clause_type = question[start:end]
                        normalized_type = normalize_type(clause_type)

                        if normalized_type not in ENTITY_TYPES:
                            continue

                        for ans in answers:
                            text = ans.get('text', '')
                            ans_start = ans.get('answer_start', 0)
                            if text:
                                entities.append({
                                    'text': text,
                                    'type': normalized_type,
                                    'start': ans_start,
                                    'end': ans_start + len(text)
                                })

            if context:
                documents.append({
                    'id': title,
                    'text': context,
                    'entities': entities
                })

    return documents


# =============================================================================
# DATASET CLASS WITH SLIDING WINDOW
# =============================================================================

class CUADNERDataset:
    """NER Dataset with sliding window for long CUAD documents."""

    def __init__(
        self,
        tokenizer,
        data: List[Dict],
        label2id: Dict[str, int],
        max_length: int = 512,
        stride: int = 128,
        max_windows_per_doc: int = 50,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.stride = stride
        self.max_windows_per_doc = max_windows_per_doc
        self.processed = self._process_data(data)

    def _process_data(self, data: List[Dict]) -> List[Dict]:
        """Process all documents with sliding window."""
        processed = []
        for doc in data:
            text = doc["text"]
            entities = doc["entities"]

            full_tokens = self.tokenizer(text, add_special_tokens=False)
            num_tokens = len(full_tokens["input_ids"])

            if num_tokens <= self.max_length - 2:
                tokenized = self._tokenize_and_align(text, entities)
                if tokenized:
                    processed.append(tokenized)
            else:
                windows = self._create_windows(text, entities)
                processed.extend(windows)

        return processed

    def _create_windows(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Create overlapping windows for long documents."""
        windows = []

        tokenized_full = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = tokenized_full["offset_mapping"]
        num_tokens = len(offsets)

        effective_length = self.max_length - 2
        start_token = 0

        while start_token < num_tokens:
            end_token = min(start_token + effective_length, num_tokens)
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]

            window_text = text[char_start:char_end]

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
                return_offsets_mapping=True,
            )

            offset_mapping = tokenized.pop("offset_mapping")
            labels = [-100] * len(offset_mapping)

            for idx, (start, end) in enumerate(offset_mapping):
                if start != end:
                    labels[idx] = self.label2id["O"]

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
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(docs: List[Dict], label2id: Dict[str, int]) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    label_counts = Counter()

    for doc in docs:
        label_counts["O"] += len(doc["text"].split())
        for entity in doc.get("entities", []):
            entity_type = entity["type"]
            label_b = f"B-{entity_type}"
            label_i = f"I-{entity_type}"
            if label_b in label2id:
                label_counts[label_b] += 1
                entity_len = max(1, len(entity.get("text", "").split()) - 1)
                label_counts[label_i] += entity_len

    total = sum(label_counts.values())
    num_classes = len(label2id)
    weights = torch.ones(num_classes)

    for label, idx in label2id.items():
        count = label_counts.get(label, 1)
        weights[idx] = total / (num_classes * count + 1)

    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.1, max=10.0)

    return weights


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred, id2label: Dict[int, str]):
    """Compute NER metrics using seqeval (entity-level)."""
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
        "f1": f1_score(true_labels, pred_labels),
    }


# =============================================================================
# ENTITY-LEVEL EVALUATION (document-level, not window-level)
# =============================================================================

def evaluate_document_level(
    model,
    tokenizer,
    documents: List[Dict],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    max_length: int = 512,
    stride: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate model at document level using sliding window with entity dedup.
    Returns entity-level P, R, F1 (micro-averaged).
    """
    model.eval()
    model.to(device)

    all_true_entities = []  # List of sets of (start, end, type)
    all_pred_entities = []

    total_time = 0.0

    for doc in documents:
        text = doc["text"]
        entities = doc["entities"]

        # Gold entities
        gold = set()
        for e in entities:
            gold.add((e["start"], e["end"], e["type"]))

        # Predict using sliding window
        start_time = time.time()
        pred = predict_document(
            model, tokenizer, text, id2label,
            max_length=max_length, stride=stride, device=device
        )
        total_time += time.time() - start_time

        all_true_entities.append(gold)
        all_pred_entities.append(pred)

    # Compute micro-averaged P, R, F1
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-type tracking
    type_tp = Counter()
    type_fp = Counter()
    type_fn = Counter()

    for gold, pred in zip(all_true_entities, all_pred_entities):
        tp = gold & pred
        fp = pred - gold
        fn = gold - pred

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for _, _, t in tp:
            type_tp[t] += 1
        for _, _, t in fp:
            type_fp[t] += 1
        for _, _, t in fn:
            type_fn[t] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-type results
    per_type = {}
    all_types = set(list(type_tp.keys()) + list(type_fp.keys()) + list(type_fn.keys()))
    for t in sorted(all_types):
        tp = type_tp[t]
        fp = type_fp[t]
        fn = type_fn[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp + fn
        per_type[t] = {"precision": p, "recall": r, "f1": f, "support": support}

    avg_time = total_time / len(documents) if documents else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_type": per_type,
        "avg_time_per_doc": avg_time,
        "total_docs": len(documents),
    }


def predict_document(
    model, tokenizer, text: str, id2label: Dict[int, str],
    max_length: int = 512, stride: int = 128, device: str = "cpu"
) -> set:
    """Predict entities from a document using sliding window."""
    full_tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = full_tokens["offset_mapping"]
    num_tokens = len(offsets)

    entities = set()
    effective_length = max_length - 2

    start_token = 0
    while start_token < num_tokens:
        end_token = min(start_token + effective_length, num_tokens)
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]

        window_text = text[char_start:char_end]

        tokenized = tokenizer(
            window_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = tokenized.pop("offset_mapping")[0].tolist()
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

        # Extract entities from predictions
        current_entity = None

        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                if current_entity:
                    entities.add((
                        current_entity["start"] + char_start,
                        current_entity["end"] + char_start,
                        current_entity["type"]
                    ))
                    current_entity = None
                continue

            pred_label = id2label.get(predictions[idx], "O")

            if pred_label.startswith("B-"):
                if current_entity:
                    entities.add((
                        current_entity["start"] + char_start,
                        current_entity["end"] + char_start,
                        current_entity["type"]
                    ))
                etype = pred_label[2:]
                current_entity = {"start": tok_start, "end": tok_end, "type": etype}

            elif pred_label.startswith("I-") and current_entity:
                etype = pred_label[2:]
                if etype == current_entity["type"]:
                    current_entity["end"] = tok_end
                else:
                    entities.add((
                        current_entity["start"] + char_start,
                        current_entity["end"] + char_start,
                        current_entity["type"]
                    ))
                    current_entity = None
            else:
                if current_entity:
                    entities.add((
                        current_entity["start"] + char_start,
                        current_entity["end"] + char_start,
                        current_entity["type"]
                    ))
                    current_entity = None

        if current_entity:
            entities.add((
                current_entity["start"] + char_start,
                current_entity["end"] + char_start,
                current_entity["type"]
            ))

        if end_token >= num_tokens:
            break
        start_token += stride

    return entities


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SLM on CUAD (Legal-BERT)")
    parser.add_argument("--model", type=str, default="nlpaueb/legal-bert-base-uncased",
                        help="Model to fine-tune")
    parser.add_argument("--train-path", type=str,
                        default="data/cuad/train_separate_questions.json",
                        help="Path to training data (SQuAD format)")
    parser.add_argument("--test-path", type=str,
                        default="data/cuad/test.json",
                        help="Path to test data (SQuAD format)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/checkpoints/cuad_slm",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"legal_bert_cuad_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print("SLM TRAINING FOR CUAD - Legal-BERT")
    print("=" * 70)
    print(f"Model:          {args.model}")
    print(f"Train:          {args.train_path}")
    print(f"Test:           {args.test_path}")
    print(f"Output:         {run_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Class weights:  {not args.no_class_weights}")
    print(f"Entity types:   {len(ENTITY_TYPES)} (41 CUAD clause types)")
    print(f"BIO labels:     {len(LABELS)}")
    print()

    # Load data
    print("Loading training data...")
    train_docs = load_cuad_squad_data(args.train_path)
    print(f"  {len(train_docs)} training documents")

    print("Loading test data...")
    test_docs = load_cuad_squad_data(args.test_path)
    print(f"  {len(test_docs)} test documents")

    # Count entities
    train_entities = sum(len(d["entities"]) for d in train_docs)
    test_entities = sum(len(d["entities"]) for d in test_docs)
    print(f"  Training entities: {train_entities}")
    print(f"  Test entities: {test_entities}")

    # Entity type distribution
    type_counts = Counter()
    for d in train_docs:
        for e in d["entities"]:
            type_counts[e["type"]] += 1
    print(f"\n  Top-10 entity types (train):")
    for t, c in type_counts.most_common(10):
        print(f"    {t}: {c}")

    # Split train into train/val
    shuffled = train_docs.copy()
    random.shuffle(shuffled)
    val_size = int(len(shuffled) * args.val_ratio)
    val_docs = shuffled[:val_size]
    train_split = shuffled[val_size:]
    print(f"\n  Train/Val split: {len(train_split)}/{len(val_docs)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create datasets with sliding window
    print("Creating training dataset (sliding window)...")
    train_dataset = CUADNERDataset(
        tokenizer, train_split, LABEL2ID,
        max_length=args.max_length, stride=args.stride
    ).to_hf_dataset()
    print(f"  Training windows: {len(train_dataset)}")

    print("Creating validation dataset...")
    val_dataset = CUADNERDataset(
        tokenizer, val_docs, LABEL2ID,
        max_length=args.max_length, stride=args.stride
    ).to_hf_dataset()
    print(f"  Validation windows: {len(val_dataset)}")

    # Compute class weights
    use_class_weights = not args.no_class_weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_split, LABEL2ID)
        print(f"\nClass weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Training arguments (matching DI2WIN optimized approach)
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        logging_dir=os.path.join(run_dir, "logs"),
        report_to="none",
        dataloader_num_workers=0,
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, ID2LABEL)

    # Trainer
    trainer_class = WeightedLossTrainer if use_class_weights else Trainer

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    if use_class_weights:
        trainer_kwargs["class_weights"] = class_weights

    trainer = trainer_class(**trainer_kwargs)

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start

    print(f"\nTraining time: {train_time:.1f}s ({train_time/60:.1f}min)")

    # Save best model
    best_path = os.path.join(run_dir, "best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)

    with open(os.path.join(best_path, "label_mappings.json"), 'w') as f:
        json.dump({
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
        }, f, indent=2)

    # Validation evaluation (window-level)
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS (window-level)")
    print("=" * 70)

    val_results = trainer.evaluate()
    print(f"  Precision: {val_results.get('eval_precision', 0):.4f}")
    print(f"  Recall:    {val_results.get('eval_recall', 0):.4f}")
    print(f"  F1:        {val_results.get('eval_f1', 0):.4f}")

    # Document-level evaluation on test set
    print("\n" + "=" * 70)
    print("TEST RESULTS (document-level, entity matching)")
    print("=" * 70)

    test_results = evaluate_document_level(
        model=model,
        tokenizer=tokenizer,
        documents=test_docs,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        max_length=args.max_length,
        stride=args.stride,
        device=device,
    )

    print(f"\n  Micro-averaged:")
    print(f"    Precision: {test_results['precision']:.4f}")
    print(f"    Recall:    {test_results['recall']:.4f}")
    print(f"    F1:        {test_results['f1']:.4f}")
    print(f"    TP: {test_results['tp']}, FP: {test_results['fp']}, FN: {test_results['fn']}")
    print(f"    Avg time/doc: {test_results['avg_time_per_doc']:.3f}s")

    print(f"\n  Per-type results (top-20 by support):")
    per_type = test_results["per_type"]
    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["support"], reverse=True)
    print(f"  {'Type':<45} {'P':>6} {'R':>6} {'F1':>6} {'Supp':>6}")
    print(f"  {'-'*70}")
    for t, metrics in sorted_types[:20]:
        print(f"  {t:<45} {metrics['precision']:>6.2%} {metrics['recall']:>6.2%} "
              f"{metrics['f1']:>6.2%} {metrics['support']:>6}")

    # Save full results
    results = {
        "model": args.model,
        "entity_types": len(ENTITY_TYPES),
        "train_docs": len(train_split),
        "val_docs": len(val_docs),
        "test_docs": len(test_docs),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "training_time_seconds": train_time,
        "epochs_trained": args.epochs,
        "class_weights": use_class_weights,
        "validation": {
            "precision": val_results.get("eval_precision", 0),
            "recall": val_results.get("eval_recall", 0),
            "f1": val_results.get("eval_f1", 0),
        },
        "test": {
            "precision": test_results["precision"],
            "recall": test_results["recall"],
            "f1": test_results["f1"],
            "tp": test_results["tp"],
            "fp": test_results["fp"],
            "fn": test_results["fn"],
            "avg_time_per_doc": test_results["avg_time_per_doc"],
        },
        "per_type": {t: m for t, m in test_results["per_type"].items()},
    }

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {results_path}")
    print(f"  Best model saved to: {best_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Test F1: {test_results['f1']:.4f}")
    print(f"  Test P:  {test_results['precision']:.4f}")
    print(f"  Test R:  {test_results['recall']:.4f}")
    print(f"  Time/doc: {test_results['avg_time_per_doc']:.3f}s")


if __name__ == "__main__":
    main()
