"""
Metrics for Entity Extraction Evaluation.

Provides:
- Exact Match: Precision, Recall, F1
- Partial Match: Token overlap based
- Entity-type specific metrics
- Long document specific metrics (coverage, consistency)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric results."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support
        }


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    exact_match: MetricResult = field(default_factory=MetricResult)
    partial_match: MetricResult = field(default_factory=MetricResult)
    by_entity_type: Dict[str, MetricResult] = field(default_factory=dict)
    coverage: float = 0.0
    consistency: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_match": self.exact_match.to_dict(),
            "partial_match": self.partial_match.to_dict(),
            "by_entity_type": {k: v.to_dict() for k, v in self.by_entity_type.items()},
            "coverage": self.coverage,
            "consistency": self.consistency,
            "details": self.details
        }


class MetricsCalculator:
    """
    Calculator for entity extraction metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.partial_threshold = self.config.get("partial_match", {}).get("threshold", 0.5)

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        entity_types: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of predicted entities [{type, value, ...}]
            ground_truth: List of ground truth entities [{type, value, ...}]
            entity_types: Optional filter for entity types

        Returns:
            EvaluationResult with all metrics
        """
        # Filter by entity types if specified
        if entity_types:
            predictions = [p for p in predictions if p.get("type") in entity_types]
            ground_truth = [g for g in ground_truth if g.get("type") in entity_types]

        # Exact match
        exact_result = self._compute_exact_match(predictions, ground_truth)

        # Partial match
        partial_result = self._compute_partial_match(predictions, ground_truth)

        # By entity type
        by_type = self._compute_by_entity_type(predictions, ground_truth)

        # Coverage and consistency for long documents
        coverage = self._compute_coverage(predictions, ground_truth)
        consistency = self._compute_consistency(predictions)

        return EvaluationResult(
            exact_match=exact_result,
            partial_match=partial_result,
            by_entity_type=by_type,
            coverage=coverage,
            consistency=consistency,
            details={
                "num_predictions": len(predictions),
                "num_ground_truth": len(ground_truth),
                "entity_types": list(by_type.keys())
            }
        )

    def _compute_exact_match(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> MetricResult:
        """Compute exact match metrics."""
        # Create sets for comparison
        pred_set = {(p.get("type", ""), self._normalize(p.get("value", "")))
                    for p in predictions}
        true_set = {(g.get("type", ""), self._normalize(g.get("value", "")))
                    for g in ground_truth}

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            precision=precision,
            recall=recall,
            f1=f1,
            support=len(true_set)
        )

    def _compute_partial_match(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> MetricResult:
        """Compute partial match metrics based on token overlap."""
        if not predictions or not ground_truth:
            return MetricResult(support=len(ground_truth))

        # Group by entity type
        pred_by_type: Dict[str, List[str]] = defaultdict(list)
        true_by_type: Dict[str, List[str]] = defaultdict(list)

        for p in predictions:
            pred_by_type[p.get("type", "")].append(p.get("value", ""))
        for g in ground_truth:
            true_by_type[g.get("type", "")].append(g.get("value", ""))

        total_overlap_score = 0.0
        total_predictions = 0
        total_ground_truth = 0

        all_types = set(pred_by_type.keys()) | set(true_by_type.keys())

        for entity_type in all_types:
            pred_values = pred_by_type.get(entity_type, [])
            true_values = true_by_type.get(entity_type, [])

            # Compute best match for each prediction
            for pred in pred_values:
                best_overlap = 0.0
                for true in true_values:
                    overlap = self._token_overlap(pred, true)
                    best_overlap = max(best_overlap, overlap)

                if best_overlap >= self.partial_threshold:
                    total_overlap_score += best_overlap

            total_predictions += len(pred_values)
            total_ground_truth += len(true_values)

        precision = total_overlap_score / total_predictions if total_predictions > 0 else 0.0
        recall = total_overlap_score / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            precision=precision,
            recall=recall,
            f1=f1,
            support=total_ground_truth
        )

    def _compute_by_entity_type(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, MetricResult]:
        """Compute metrics for each entity type."""
        # Get all entity types
        all_types = set()
        for p in predictions:
            all_types.add(p.get("type", ""))
        for g in ground_truth:
            all_types.add(g.get("type", ""))

        results = {}
        for entity_type in all_types:
            type_preds = [p for p in predictions if p.get("type") == entity_type]
            type_true = [g for g in ground_truth if g.get("type") == entity_type]

            results[entity_type] = self._compute_exact_match(type_preds, type_true)

        return results

    def _compute_coverage(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> float:
        """
        Compute coverage: what proportion of document sections with
        entities were successfully covered.
        """
        if not ground_truth:
            return 1.0

        # Group by entity type
        true_types = {g.get("type") for g in ground_truth}
        pred_types = {p.get("type") for p in predictions}

        if not true_types:
            return 1.0

        return len(pred_types & true_types) / len(true_types)

    def _compute_consistency(self, predictions: List[Dict]) -> float:
        """
        Compute consistency: entities that appear multiple times
        should have consistent values.
        """
        if not predictions:
            return 1.0

        # Group predictions by type
        by_type: Dict[str, List[str]] = defaultdict(list)
        for p in predictions:
            by_type[p.get("type", "")].append(self._normalize(p.get("value", "")))

        consistencies = []
        for entity_type, values in by_type.items():
            if len(values) > 1:
                # Check if all values are the same
                unique_values = set(values)
                consistency = 1.0 / len(unique_values)
                consistencies.append(consistency)

        return sum(consistencies) / len(consistencies) if consistencies else 1.0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        return " ".join(text.lower().split())

    def _token_overlap(self, text1: str, text2: str) -> float:
        """Compute token overlap ratio."""
        tokens1 = set(self._normalize(text1).split())
        tokens2 = set(self._normalize(text2).split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0


def compute_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    entity_types: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute all metrics.

    Args:
        predictions: Predicted entities
        ground_truth: Ground truth entities
        entity_types: Optional filter
        config: Optional configuration

    Returns:
        Dictionary with all metrics
    """
    calculator = MetricsCalculator(config)
    result = calculator.evaluate(predictions, ground_truth, entity_types)
    return result.to_dict()


def compute_statistical_significance(
    results_a: List[float],
    results_b: List[float],
    test: str = "paired_t_test"
) -> Dict[str, float]:
    """
    Compute statistical significance between two sets of results.

    Args:
        results_a: Results from system A
        results_b: Results from system B
        test: Statistical test to use

    Returns:
        Dictionary with p-value and test statistic
    """
    from scipy import stats
    import numpy as np

    results_a = np.array(results_a)
    results_b = np.array(results_b)

    if test == "paired_t_test":
        statistic, p_value = stats.ttest_rel(results_a, results_b)
    elif test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(results_a, results_b)
    else:
        statistic, p_value = stats.ttest_rel(results_a, results_b)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "test": test,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01
    }
