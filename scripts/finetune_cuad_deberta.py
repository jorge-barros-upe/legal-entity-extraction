#!/usr/bin/env python3
"""
Fine-tune DeBERTa for CUAD Clause Extraction.

This script fine-tunes DeBERTa-v3-base on the CUAD training set for
token classification (NER-style extraction of 41 clause types).

Based on the approach from Hendrycks et al. (2021) but adapted for
direct span extraction rather than QA.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm


# =============================================================================
# CUAD CLAUSE TYPES (41 types)
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
LABELS = ["O"] + [f"B-{normalize_type(ct)}" for ct in CUAD_CLAUSE_TYPES] + \
         [f"I-{normalize_type(ct)}" for ct in CUAD_CLAUSE_TYPES]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def load_cuad_data(data_path: str) -> List[Dict]:
    """Load CUAD data and convert to NER format."""
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

                if '"' in question and not is_impossible:
                    start = question.find('"') + 1
                    end = question.find('"', start)
                    if end > start:
                        clause_type = question[start:end]
                        normalized_type = normalize_type(clause_type)

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


class CUADDataset(Dataset):
    """Dataset for CUAD clause extraction."""

    def __init__(
        self,
        documents: List[Dict],
        tokenizer,
        max_length: int = 512,
        stride: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []

        for doc in tqdm(documents, desc="Preparing dataset"):
            windows = self._create_windows(doc)
            self.examples.extend(windows)

        print(f"Created {len(self.examples)} training examples from {len(documents)} documents")

    def _create_windows(self, doc: Dict) -> List[Dict]:
        """Create sliding windows from a document."""
        text = doc['text']
        entities = doc['entities']
        windows = []

        # Tokenize full text to get length
        full_encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        # Create sliding windows
        text_len = len(text)
        window_size = self.max_length - 2  # Account for [CLS] and [SEP]
        char_stride = int(window_size * 0.7 * 4)  # ~70% overlap in characters

        start = 0
        while start < text_len:
            end = min(start + window_size * 4, text_len)  # Approximate char window

            # Find word boundaries
            while end < text_len and text[end] not in ' \n\t':
                end += 1

            window_text = text[start:end]

            # Filter entities for this window
            window_entities = []
            for ent in entities:
                ent_start = ent['start'] - start
                ent_end = ent['end'] - start

                # Check if entity is within window
                if ent_start >= 0 and ent_end <= len(window_text):
                    window_entities.append({
                        'text': ent['text'],
                        'type': ent['type'],
                        'start': ent_start,
                        'end': ent_end
                    })

            # Tokenize and align labels
            example = self._tokenize_and_align(window_text, window_entities)
            if example is not None:
                windows.append(example)

            start += char_stride
            if start >= text_len:
                break

        return windows

    def _tokenize_and_align(self, text: str, entities: List[Dict]) -> Optional[Dict]:
        """Tokenize text and align entity labels."""
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            offset_mapping = encoding.pop('offset_mapping')[0].tolist()
            labels = [-100] * len(offset_mapping)

            # Mark all non-special tokens as O
            for idx, (start, end) in enumerate(offset_mapping):
                if start != end:  # Real token
                    labels[idx] = LABEL2ID['O']

            # Assign entity labels
            for entity in entities:
                ent_start = entity['start']
                ent_end = entity['end']
                ent_type = entity['type']

                b_label = f"B-{ent_type}"
                i_label = f"I-{ent_type}"

                if b_label not in LABEL2ID:
                    continue

                is_first = True
                for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start == tok_end:
                        continue

                    # Check overlap
                    if tok_start >= ent_start and tok_end <= ent_end:
                        if is_first:
                            labels[idx] = LABEL2ID[b_label]
                            is_first = False
                        else:
                            labels[idx] = LABEL2ID[i_label]

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(labels)
            }

        except Exception as e:
            return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred):
    """Compute precision, recall, F1 for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten and filter out -100
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_labels.append(label)
                pred_labels.append(pred)

    # Calculate metrics (excluding O label for entity-focused metrics)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )

    # Calculate strict entity-level metrics (only B- and I- labels)
    entity_true = [l for l in true_labels if l != LABEL2ID['O']]
    entity_pred = [p for p, l in zip(pred_labels, true_labels) if l != LABEL2ID['O']]

    if entity_true:
        ent_p, ent_r, ent_f1, _ = precision_recall_fscore_support(
            entity_true, entity_pred, average='micro', zero_division=0
        )
    else:
        ent_p, ent_r, ent_f1 = 0, 0, 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'entity_precision': ent_p,
        'entity_recall': ent_r,
        'entity_f1': ent_f1,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    train_path: str,
    test_path: str,
    output_dir: str,
    model_name: str = "microsoft/deberta-v3-base",
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """Train DeBERTa model on CUAD data."""

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading training data from {train_path}")
    train_docs = load_cuad_data(train_path)
    print(f"Loaded {len(train_docs)} training documents")

    print(f"Loading test data from {test_path}")
    test_docs = load_cuad_data(test_path)
    print(f"Loaded {len(test_docs)} test documents")

    print("Creating training dataset...")
    train_dataset = CUADDataset(train_docs, tokenizer, max_length)

    print("Creating test dataset...")
    test_dataset = CUADDataset(test_docs[:20], tokenizer, max_length)  # Use subset for eval

    print(f"Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"deberta_cuad_{timestamp}")

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir=os.path.join(run_output_dir, "logs"),
        report_to="none",
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    trainer.train()

    # Save best model
    best_model_path = os.path.join(run_output_dir, "best")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"\nBest model saved to: {best_model_path}")

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    results = trainer.evaluate()
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # Save results
    results_path = os.path.join(run_output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa on CUAD")
    parser.add_argument("--train", type=str, required=True, help="Path to training data")
    parser.add_argument("--test", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, default="experiments/checkpoints/cuad_deberta",
                       help="Output directory")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base",
                       help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = train_model(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
