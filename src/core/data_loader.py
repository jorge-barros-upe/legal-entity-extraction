"""
Data Loader - Utilities for loading and preprocessing datasets.

Supports:
- CUAD dataset (English legal contracts)
- di2win dataset (Portuguese legal contracts)
- Custom datasets
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with its annotations."""
    id: str
    text: str
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)

    @property
    def num_entities(self) -> int:
        return len(self.annotations)

    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        return [a for a in self.annotations if a.get("type") == entity_type]


@dataclass
class Dataset:
    """Represents a dataset with train/val/test splits."""
    name: str
    language: str
    train: List[Document] = field(default_factory=list)
    validation: List[Document] = field(default_factory=list)
    test: List[Document] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def get_split(self, split: str) -> List[Document]:
        if split == "train":
            return self.train
        elif split in ("val", "validation"):
            return self.validation
        elif split == "test":
            return self.test
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        all_docs = self.train + self.validation + self.test
        lengths = [d.length for d in all_docs]
        num_entities = [d.num_entities for d in all_docs]

        return {
            "name": self.name,
            "language": self.language,
            "num_train": len(self.train),
            "num_val": len(self.validation),
            "num_test": len(self.test),
            "total_documents": len(all_docs),
            "entity_types": self.entity_types,
            "num_entity_types": len(self.entity_types),
            "document_lengths": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0,
            },
            "entities_per_doc": {
                "min": min(num_entities) if num_entities else 0,
                "max": max(num_entities) if num_entities else 0,
                "mean": sum(num_entities) / len(num_entities) if num_entities else 0,
            }
        }


class DataLoader:
    """
    Data loader for entity extraction datasets.

    Supports CUAD, di2win, and custom formats.
    """

    def __init__(self, base_path: str = "../data"):
        self.base_path = Path(base_path)

    def load_cuad(
        self,
        path: Optional[str] = None,
        splits: List[str] = ["train", "validation", "test"]
    ) -> Dataset:
        """
        Load CUAD dataset.

        Args:
            path: Path to CUAD data directory
            splits: Which splits to load

        Returns:
            Dataset object
        """
        if path is None:
            path = self.base_path / "cuad"
        else:
            path = Path(path)

        dataset = Dataset(
            name="CUAD",
            language="en",
            entity_types=[
                "Party", "Date", "Term", "Governing Law", "Jurisdiction",
                "Termination Clause", "Renewal Term", "Notice Period",
                "Non-Compete Term", "Exclusivity"
            ]
        )

        # Try to load from various CUAD formats
        for split in splits:
            docs = self._load_cuad_split(path, split)
            if split == "train":
                dataset.train = docs
            elif split in ("val", "validation"):
                dataset.validation = docs
            elif split == "test":
                dataset.test = docs

        logger.info(f"Loaded CUAD dataset: {dataset.get_statistics()}")
        return dataset

    def _load_cuad_split(self, path: Path, split: str) -> List[Document]:
        """Load a single CUAD split."""
        documents = []

        # Try different possible file names
        possible_files = [
            path / f"{split}.json",
            path / f"CUADv1_{split}.json",
            path / f"cuad_{split}.json",
        ]

        for file_path in possible_files:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle SQuAD-style format (CUAD uses this)
                if "data" in data:
                    for item in data["data"]:
                        for para in item.get("paragraphs", []):
                            doc = Document(
                                id=item.get("title", f"doc_{len(documents)}"),
                                text=para.get("context", ""),
                                annotations=self._parse_cuad_annotations(para.get("qas", [])),
                                metadata={"source": "cuad", "split": split}
                            )
                            documents.append(doc)
                else:
                    # Simple list format
                    for idx, item in enumerate(data):
                        doc = Document(
                            id=item.get("id", f"doc_{idx}"),
                            text=item.get("text", item.get("context", "")),
                            annotations=item.get("annotations", []),
                            metadata={"source": "cuad", "split": split}
                        )
                        documents.append(doc)

                logger.info(f"Loaded {len(documents)} documents from {file_path}")
                break

        return documents

    def _parse_cuad_annotations(self, qas: List[Dict]) -> List[Dict[str, Any]]:
        """Parse CUAD QA-style annotations to entity format."""
        annotations = []

        for qa in qas:
            question = qa.get("question", "")
            answers = qa.get("answers", [])

            # Map question to entity type
            entity_type = self._cuad_question_to_entity_type(question)

            for answer in answers:
                if answer.get("text"):
                    annotations.append({
                        "type": entity_type,
                        "value": answer["text"],
                        "start": answer.get("answer_start"),
                        "end": answer.get("answer_start", 0) + len(answer.get("text", "")),
                        "question": question
                    })

        return annotations

    def _cuad_question_to_entity_type(self, question: str) -> str:
        """Map CUAD question to entity type."""
        question_lower = question.lower()

        mappings = {
            "parties": "Party",
            "agreement date": "Date",
            "effective date": "Date",
            "expiration date": "Date",
            "renewal term": "Renewal Term",
            "notice period": "Notice Period",
            "governing law": "Governing Law",
            "jurisdiction": "Jurisdiction",
            "termination": "Termination Clause",
            "non-compete": "Non-Compete Term",
            "exclusivity": "Exclusivity",
        }

        for key, entity_type in mappings.items():
            if key in question_lower:
                return entity_type

        return "Other"

    def load_di2win(
        self,
        path: Optional[str] = None,
        splits: List[str] = ["train", "validation", "test"]
    ) -> Dataset:
        """
        Load di2win dataset (Portuguese contracts).

        Args:
            path: Path to di2win data directory
            splits: Which splits to load

        Returns:
            Dataset object
        """
        if path is None:
            path = self.base_path / "di2win"
        else:
            path = Path(path)

        dataset = Dataset(
            name="di2win",
            language="pt-br",
            entity_types=[
                "CONTRATANTE", "CONTRATADA",
                "CNPJ_CONTRATANTE", "CNPJ_CONTRATADA",
                "DATA_CONTRATO", "VALOR_CONTRATO",
                "PRAZO_VIGENCIA", "OBJETO_CONTRATO", "FORO"
            ]
        )

        for split in splits:
            docs = self._load_di2win_split(path, split)
            if split == "train":
                dataset.train = docs
            elif split in ("val", "validation"):
                dataset.validation = docs
            elif split == "test":
                dataset.test = docs

        logger.info(f"Loaded di2win dataset: {dataset.get_statistics()}")
        return dataset

    def _load_di2win_split(self, path: Path, split: str) -> List[Document]:
        """Load a single di2win split."""
        documents = []

        possible_files = [
            path / f"{split}.json",
            path / f"di2win_{split}.json",
        ]

        for file_path in possible_files:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        doc = Document(
                            id=item.get("id", f"doc_{idx}"),
                            text=item.get("text", item.get("content", "")),
                            annotations=item.get("annotations", item.get("entities", [])),
                            metadata={"source": "di2win", "split": split}
                        )
                        documents.append(doc)

                logger.info(f"Loaded {len(documents)} documents from {file_path}")
                break

        return documents

    def load_custom(
        self,
        path: str,
        name: str = "custom",
        language: str = "unknown",
        entity_types: Optional[List[str]] = None
    ) -> Dataset:
        """
        Load a custom dataset.

        Expected format: JSON with list of documents, each having:
        - id: document identifier
        - text: document text
        - annotations: list of {type, value, start?, end?}

        Args:
            path: Path to JSON file
            name: Dataset name
            language: Language code
            entity_types: List of entity types (auto-detected if None)

        Returns:
            Dataset object
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        detected_types = set()

        for idx, item in enumerate(data):
            annotations = item.get("annotations", [])
            for ann in annotations:
                if "type" in ann:
                    detected_types.add(ann["type"])

            doc = Document(
                id=item.get("id", f"doc_{idx}"),
                text=item.get("text", ""),
                annotations=annotations,
                metadata=item.get("metadata", {})
            )
            documents.append(doc)

        if entity_types is None:
            entity_types = sorted(list(detected_types))

        dataset = Dataset(
            name=name,
            language=language,
            train=documents,  # All to train by default
            entity_types=entity_types
        )

        logger.info(f"Loaded custom dataset: {dataset.get_statistics()}")
        return dataset

    def create_k_fold_splits(
        self,
        documents: List[Document],
        k: int = 5,
        seed: int = 42
    ) -> Iterator[Tuple[List[Document], List[Document]]]:
        """
        Create k-fold cross-validation splits.

        Args:
            documents: List of documents
            k: Number of folds
            seed: Random seed

        Yields:
            Tuples of (train_docs, val_docs) for each fold
        """
        import random
        random.seed(seed)

        indices = list(range(len(documents)))
        random.shuffle(indices)

        fold_size = len(documents) // k

        for i in range(k):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < k - 1 else len(documents)

            val_indices = set(indices[val_start:val_end])
            train_docs = [documents[j] for j in range(len(documents)) if j not in val_indices]
            val_docs = [documents[j] for j in val_indices]

            yield train_docs, val_docs
