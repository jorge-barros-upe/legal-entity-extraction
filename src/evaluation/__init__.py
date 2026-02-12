"""Evaluation framework for entity extraction experiments."""

from .metrics import MetricsCalculator, compute_metrics
from .analysis import ErrorAnalyzer
from .visualization import ResultsVisualizer

__all__ = [
    "MetricsCalculator",
    "compute_metrics",
    "ErrorAnalyzer",
    "ResultsVisualizer",
]
