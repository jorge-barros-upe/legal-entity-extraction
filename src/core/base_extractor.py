"""
Base Extractor - Abstract base class for all entity extraction approaches.

All extractors (RAG, LLM, SLM, Hybrid) inherit from this class to ensure
consistent interfaces and behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    type: str
    value: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    confidence: float = 1.0
    source: Optional[str] = None  # Which chunk/section it came from
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            type=data["type"],
            value=data["value"],
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ExtractionResult:
    """Result of entity extraction from a document."""
    document_id: str
    entities: List[Entity]
    extraction_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    api_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "entities": [e.to_dict() for e in self.entities],
            "extraction_time": self.extraction_time,
            "metadata": self.metadata,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_cost": self.api_cost
        }


class BaseExtractor(ABC):
    """
    Abstract base class for entity extractors.

    All extraction approaches must implement:
    - extract(): Extract entities from a single document
    - extract_batch(): Extract entities from multiple documents
    """

    def __init__(
        self,
        config: Dict[str, Any],
        entity_types: List[str],
        name: str = "BaseExtractor"
    ):
        """
        Initialize the extractor.

        Args:
            config: Configuration dictionary
            entity_types: List of entity types to extract
            name: Name identifier for this extractor
        """
        self.config = config
        self.entity_types = entity_types
        self.name = name

        # Statistics tracking
        self._extraction_times: List[float] = []
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._num_extractions: int = 0

        logger.info(f"Initialized {self.name} for entity types: {entity_types}")

    @abstractmethod
    def extract(
        self,
        document: str,
        document_id: str = "unknown",
        **kwargs
    ) -> ExtractionResult:
        """
        Extract entities from a single document.

        Args:
            document: The document text
            document_id: Unique identifier for the document
            **kwargs: Additional arguments specific to the extractor

        Returns:
            ExtractionResult containing extracted entities and metadata
        """
        pass

    def extract_batch(
        self,
        documents: List[Dict[str, str]],
        show_progress: bool = True,
        **kwargs
    ) -> List[ExtractionResult]:
        """
        Extract entities from multiple documents.

        Args:
            documents: List of dicts with 'id' and 'text' keys
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to extract()

        Returns:
            List of ExtractionResults
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                documents = tqdm(documents, desc=f"Extracting with {self.name}")
            except ImportError:
                pass

        for doc in documents:
            try:
                result = self.extract(
                    document=doc["text"],
                    document_id=doc["id"],
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error extracting from {doc['id']}: {e}")
                results.append(ExtractionResult(
                    document_id=doc["id"],
                    entities=[],
                    extraction_time=0,
                    metadata={"error": str(e)}
                ))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        if not self._extraction_times:
            return {}

        return {
            "extractor_name": self.name,
            "num_extractions": self._num_extractions,
            "total_time": sum(self._extraction_times),
            "mean_time": sum(self._extraction_times) / len(self._extraction_times),
            "min_time": min(self._extraction_times),
            "max_time": max(self._extraction_times),
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost
        }

    def reset_statistics(self):
        """Reset extraction statistics."""
        self._extraction_times = []
        self._total_tokens = 0
        self._total_cost = 0.0
        self._num_extractions = 0

    def _track_extraction(
        self,
        extraction_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0
    ):
        """Track extraction metrics."""
        self._extraction_times.append(extraction_time)
        self._total_tokens += input_tokens + output_tokens
        self._total_cost += cost
        self._num_extractions += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', entity_types={self.entity_types})"


class ExtractorRegistry:
    """Registry for managing available extractors."""

    _extractors: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an extractor class."""
        def decorator(extractor_class):
            cls._extractors[name] = extractor_class
            return extractor_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get an extractor class by name."""
        if name not in cls._extractors:
            raise ValueError(f"Unknown extractor: {name}. Available: {list(cls._extractors.keys())}")
        return cls._extractors[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List available extractor names."""
        return list(cls._extractors.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: Dict[str, Any],
        entity_types: List[str],
        **kwargs
    ) -> BaseExtractor:
        """Create an extractor instance by name."""
        extractor_class = cls.get(name)
        return extractor_class(config=config, entity_types=entity_types, **kwargs)
