"""
SLM Extractor - Fine-tuned Small Language Models for Entity Extraction.

Supports:
- Longformer, BigBird (sparse attention for long docs)
- Legal-BERT, BERTimbau (domain-specific)
- LoRA/QLoRA fine-tuning for efficient adaptation
"""

import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from ...core.base_extractor import (
    BaseExtractor, ExtractorRegistry, Entity, ExtractionResult
)

logger = logging.getLogger(__name__)


@ExtractorRegistry.register("slm")
class SLMExtractor(BaseExtractor):
    """
    SLM-based entity extractor using fine-tuned transformer models.

    Performs token classification (NER) for entity extraction.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        entity_types: List[str],
        name: str = "SLMExtractor",
        checkpoint_path: Optional[str] = None
    ):
        super().__init__(config, entity_types, name)

        # Get SLM-specific config
        slm_config = config.get("slm", config)

        # Model architecture
        architectures = slm_config.get("architectures", {})
        self.architecture = slm_config.get("default", {}).get("architecture", "legal_bert")
        self.model_config = architectures.get(self.architecture, {})
        self.model_name = self.model_config.get("model_name", "nlpaueb/legal-bert-base-uncased")
        self.max_length = self.model_config.get("max_length", 512)

        # Labeling scheme
        labeling_config = slm_config.get("labeling", {})
        self.label_scheme = labeling_config.get("scheme", "BIO")

        # Build label mappings
        self._build_label_mappings()

        # Long document handling
        doc_config = slm_config.get("long_document", {})
        self.doc_strategy = doc_config.get("strategy", "sliding_window")
        self.window_size = doc_config.get("sliding_window", {}).get("window_size", 512)
        self.stride = doc_config.get("sliding_window", {}).get("stride", 256)

        # Load model
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        if checkpoint_path:
            self.load_model(checkpoint_path)
        else:
            self._init_model()

        logger.info(
            f"Initialized SLMExtractor: "
            f"architecture={self.architecture}, "
            f"model={self.model_name}, "
            f"max_length={self.max_length}"
        )

    def _build_label_mappings(self):
        """Build label to ID mappings."""
        self.labels = ["O"]  # Outside

        if self.label_scheme == "BIO":
            for entity_type in self.entity_types:
                self.labels.append(f"B-{entity_type}")
                self.labels.append(f"I-{entity_type}")
        elif self.label_scheme == "BIOES":
            for entity_type in self.entity_types:
                self.labels.append(f"B-{entity_type}")
                self.labels.append(f"I-{entity_type}")
                self.labels.append(f"E-{entity_type}")
                self.labels.append(f"S-{entity_type}")

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def _init_model(self):
        """Initialize model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded model {self.model_name} on {self.device}")

        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers torch")
            raise

    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            # Update label mappings from loaded model
            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
                self.label2id = {v: int(k) for k, v in self.id2label.items()}

            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def extract(
        self,
        document: str,
        document_id: str = "unknown",
        **kwargs
    ) -> ExtractionResult:
        """
        Extract entities using fine-tuned SLM.

        Args:
            document: Document text
            document_id: Document identifier

        Returns:
            ExtractionResult with extracted entities
        """
        import torch

        start_time = time.time()

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle long documents
        if len(document) > self.max_length * 4:  # Rough char estimate
            entities = self._extract_long_document(document)
        else:
            entities = self._extract_single(document)

        extraction_time = time.time() - start_time

        self._track_extraction(extraction_time=extraction_time)

        return ExtractionResult(
            document_id=document_id,
            entities=entities,
            extraction_time=extraction_time,
            metadata={
                "approach": "slm",
                "architecture": self.architecture,
                "model": self.model_name,
                "document_length": len(document)
            }
        )

    def _extract_single(self, text: str) -> List[Entity]:
        """Extract from a single text segment."""
        import torch

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

        # Convert predictions to entities
        entities = self._predictions_to_entities(
            predictions, probabilities, offset_mapping, text
        )

        return entities

    def _extract_long_document(self, document: str) -> List[Entity]:
        """Extract from long document using sliding window."""
        all_entities = []

        # Sliding window
        start = 0
        while start < len(document):
            end = min(start + self.window_size * 4, len(document))  # Char estimate
            window_text = document[start:end]

            window_entities = self._extract_single(window_text)

            # Adjust character offsets
            for entity in window_entities:
                if entity.start_char is not None:
                    entity.start_char += start
                if entity.end_char is not None:
                    entity.end_char += start
                entity.metadata["window_start"] = start

            all_entities.extend(window_entities)

            start += self.stride * 4  # Move by stride

        # Deduplicate overlapping entities
        unique_entities = self._deduplicate_entities(all_entities)

        return unique_entities

    def _predictions_to_entities(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        offset_mapping: List[tuple],
        text: str
    ) -> List[Entity]:
        """Convert BIO predictions to Entity objects."""
        entities = []
        current_entity = None

        for idx, pred_id in enumerate(predictions):
            if idx >= len(offset_mapping):
                break

            start, end = offset_mapping[idx]
            if start == end:  # Special token
                continue

            label = self.id2label.get(int(pred_id), "O")
            confidence = float(probabilities[idx][pred_id])

            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = Entity(
                    type=entity_type,
                    value=text[start:end],
                    start_char=start,
                    end_char=end,
                    confidence=confidence,
                    source="slm_extraction"
                )
            elif label.startswith("I-"):
                if current_entity and label[2:] == current_entity.type:
                    # Continue current entity
                    current_entity.value += text[current_entity.end_char:end]
                    current_entity.end_char = end
                    current_entity.confidence = min(current_entity.confidence, confidence)
                else:
                    # I without B, start new entity
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = Entity(
                        type=entity_type,
                        value=text[start:end],
                        start_char=start,
                        end_char=end,
                        confidence=confidence,
                        source="slm_extraction"
                    )

        if current_entity:
            entities.append(current_entity)

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities, handling overlaps."""
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: (e.start_char or 0, -(e.end_char or 0)))

        unique = []
        for entity in sorted_entities:
            # Check for overlap with last unique entity
            if unique and entity.start_char is not None:
                last = unique[-1]
                if (last.start_char is not None and last.end_char is not None and
                    entity.start_char < last.end_char):
                    # Overlap - keep higher confidence
                    if entity.confidence > last.confidence:
                        unique[-1] = entity
                    continue

            unique.append(entity)

        return unique

    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
        output_dir: str = "checkpoints/slm",
        **training_args
    ):
        """
        Fine-tune the model on training data.

        Args:
            train_data: List of {"text": str, "entities": [{"type", "value", "start", "end"}]}
            val_data: Optional validation data
            output_dir: Where to save checkpoints
            **training_args: Additional training arguments
        """
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
        import torch

        # Prepare dataset
        train_dataset = self._prepare_dataset(train_data)
        val_dataset = self._prepare_dataset(val_data) if val_data else None

        # Training config
        slm_config = self.config.get("slm", {})
        train_config = slm_config.get("training", {})

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config.get("num_epochs", 10),
            per_device_train_batch_size=train_config.get("batch_size", 8),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            learning_rate=train_config.get("learning_rate", 2e-5),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.1),
            fp16=train_config.get("fp16", True),
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=train_config.get("eval_steps", 500),
            save_strategy="steps",
            save_steps=train_config.get("save_steps", 500),
            save_total_limit=train_config.get("save_total_limit", 3),
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            logging_steps=100,
            **training_args
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics if val_dataset else None
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

    def _prepare_dataset(self, data: List[Dict[str, Any]]):
        """Prepare dataset for training."""
        from datasets import Dataset

        tokenized_data = []
        for item in data:
            tokenized = self._tokenize_and_align_labels(
                item["text"],
                item.get("entities", [])
            )
            tokenized_data.append(tokenized)

        return Dataset.from_list(tokenized_data)

    def _tokenize_and_align_labels(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tokenize text and align entity labels."""
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True
        )

        offset_mapping = tokenized.pop("offset_mapping")
        labels = [self.label2id["O"]] * len(offset_mapping)

        for entity in entities:
            entity_start = entity.get("start", 0)
            entity_end = entity.get("end", 0)
            entity_type = entity.get("type", "")

            if entity_type not in self.entity_types:
                continue

            for idx, (start, end) in enumerate(offset_mapping):
                if start == end:  # Special token
                    continue

                if start >= entity_start and end <= entity_end:
                    if start == entity_start:
                        labels[idx] = self.label2id.get(f"B-{entity_type}", 0)
                    else:
                        labels[idx] = self.label2id.get(f"I-{entity_type}", 0)

        tokenized["labels"] = labels
        return tokenized

    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        from seqeval.metrics import f1_score, precision_score, recall_score

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Convert to label sequences
        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_labels = []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Ignore padding
                    true_seq.append(self.id2label.get(label, "O"))
                    pred_seq_labels.append(self.id2label.get(pred, "O"))
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels)
        }
