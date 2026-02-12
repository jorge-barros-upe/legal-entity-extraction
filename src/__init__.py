"""
Entity Extraction in Long Documents - Experiment Framework

This package provides a modular framework for comparing different approaches
to entity extraction in long legal documents.

Approaches:
- RAG (Retrieval-Augmented Generation)
- LLM (Long-Context Language Models)
- SLM (Fine-tuned Small Language Models)
- Hybrid (Combinations of the above)
"""

__version__ = "1.0.0"
__author__ = "Jorge B. Medeiros"

from .core.base_extractor import BaseExtractor
from .core.data_loader import DataLoader
from .evaluation.metrics import MetricsCalculator
