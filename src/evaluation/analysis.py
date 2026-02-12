"""
Error Analysis utilities for entity extraction.

Provides:
- Error categorization
- Confusion matrix
- Common error patterns
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Represents a single error case."""
    error_type: str  # false_positive, false_negative, partial_match, type_error
    entity_type: str
    predicted_value: Optional[str]
    ground_truth_value: Optional[str]
    context: Optional[str] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ErrorAnalyzer:
    """
    Analyzer for entity extraction errors.
    """

    def __init__(self):
        self.errors: List[ErrorCase] = []

    def analyze(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        document: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze errors between predictions and ground truth.

        Args:
            predictions: Predicted entities
            ground_truth: Ground truth entities
            document: Original document text for context

        Returns:
            Analysis results
        """
        self.errors = []

        # Normalize for comparison
        pred_normalized = self._normalize_entities(predictions)
        true_normalized = self._normalize_entities(ground_truth)

        # Find matches and errors
        matched_pred = set()
        matched_true = set()

        # Exact matches
        for i, pred in enumerate(pred_normalized):
            for j, true in enumerate(true_normalized):
                if pred["key"] == true["key"]:
                    matched_pred.add(i)
                    matched_true.add(j)

        # False positives (predictions with no match)
        for i, pred in enumerate(pred_normalized):
            if i not in matched_pred:
                # Check for partial matches
                partial = self._find_partial_match(pred, true_normalized, matched_true)
                if partial:
                    self.errors.append(ErrorCase(
                        error_type="partial_match",
                        entity_type=pred["type"],
                        predicted_value=pred["value"],
                        ground_truth_value=partial["value"],
                        context=self._get_context(pred["value"], document),
                        details={"overlap": partial["overlap"]}
                    ))
                else:
                    # Check for type errors
                    type_error = self._find_type_error(pred, true_normalized, matched_true)
                    if type_error:
                        self.errors.append(ErrorCase(
                            error_type="type_error",
                            entity_type=pred["type"],
                            predicted_value=pred["value"],
                            ground_truth_value=type_error["value"],
                            details={"correct_type": type_error["type"]}
                        ))
                    else:
                        self.errors.append(ErrorCase(
                            error_type="false_positive",
                            entity_type=pred["type"],
                            predicted_value=pred["value"],
                            ground_truth_value=None,
                            context=self._get_context(pred["value"], document)
                        ))

        # False negatives (ground truth with no match)
        for j, true in enumerate(true_normalized):
            if j not in matched_true:
                self.errors.append(ErrorCase(
                    error_type="false_negative",
                    entity_type=true["type"],
                    predicted_value=None,
                    ground_truth_value=true["value"],
                    context=self._get_context(true["value"], document)
                ))

        return self._summarize_errors()

    def _normalize_entities(self, entities: List[Dict]) -> List[Dict]:
        """Normalize entities for comparison."""
        normalized = []
        for e in entities:
            entity_type = e.get("type", "")
            value = e.get("value", "")
            normalized_value = " ".join(value.lower().split())

            normalized.append({
                "type": entity_type,
                "value": value,
                "normalized_value": normalized_value,
                "key": (entity_type, normalized_value)
            })
        return normalized

    def _find_partial_match(
        self,
        pred: Dict,
        ground_truth: List[Dict],
        already_matched: set
    ) -> Optional[Dict]:
        """Find partial match for a prediction."""
        pred_tokens = set(pred["normalized_value"].split())

        best_match = None
        best_overlap = 0.0

        for j, true in enumerate(ground_truth):
            if j in already_matched:
                continue
            if pred["type"] != true["type"]:
                continue

            true_tokens = set(true["normalized_value"].split())
            if not pred_tokens or not true_tokens:
                continue

            intersection = len(pred_tokens & true_tokens)
            union = len(pred_tokens | true_tokens)
            overlap = intersection / union

            if overlap > 0.3 and overlap > best_overlap:
                best_overlap = overlap
                best_match = {**true, "overlap": overlap}

        return best_match

    def _find_type_error(
        self,
        pred: Dict,
        ground_truth: List[Dict],
        already_matched: set
    ) -> Optional[Dict]:
        """Find if prediction has wrong entity type."""
        for j, true in enumerate(ground_truth):
            if j in already_matched:
                continue

            if pred["normalized_value"] == true["normalized_value"]:
                return true

        return None

    def _get_context(self, value: str, document: Optional[str], window: int = 100) -> Optional[str]:
        """Get context around the entity value."""
        if not document or not value:
            return None

        idx = document.lower().find(value.lower())
        if idx == -1:
            return None

        start = max(0, idx - window)
        end = min(len(document), idx + len(value) + window)

        context = document[start:end]
        if start > 0:
            context = "..." + context
        if end < len(document):
            context = context + "..."

        return context

    def _summarize_errors(self) -> Dict[str, Any]:
        """Summarize error analysis."""
        summary = {
            "total_errors": len(self.errors),
            "by_error_type": Counter(),
            "by_entity_type": defaultdict(Counter),
            "error_samples": defaultdict(list)
        }

        for error in self.errors:
            summary["by_error_type"][error.error_type] += 1
            summary["by_entity_type"][error.entity_type][error.error_type] += 1

            # Keep sample errors
            if len(summary["error_samples"][error.error_type]) < 5:
                summary["error_samples"][error.error_type].append({
                    "entity_type": error.entity_type,
                    "predicted": error.predicted_value,
                    "ground_truth": error.ground_truth_value,
                    "context": error.context[:200] if error.context else None
                })

        # Convert to regular dicts
        summary["by_error_type"] = dict(summary["by_error_type"])
        summary["by_entity_type"] = {k: dict(v) for k, v in summary["by_entity_type"].items()}
        summary["error_samples"] = dict(summary["error_samples"])

        return summary

    def get_confusion_matrix(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, Dict[str, int]]:
        """
        Generate confusion matrix for entity types.

        Returns matrix[predicted_type][true_type] = count
        """
        matrix = defaultdict(lambda: defaultdict(int))

        pred_normalized = self._normalize_entities(predictions)
        true_normalized = self._normalize_entities(ground_truth)

        # Build value to true type mapping
        true_value_to_type = {}
        for t in true_normalized:
            true_value_to_type[t["normalized_value"]] = t["type"]

        for pred in pred_normalized:
            true_type = true_value_to_type.get(pred["normalized_value"], "NOT_FOUND")
            matrix[pred["type"]][true_type] += 1

        return {k: dict(v) for k, v in matrix.items()}

    def get_common_error_patterns(self) -> List[Dict[str, Any]]:
        """Identify common error patterns."""
        patterns = []

        # Group by error type and entity type
        error_groups = defaultdict(list)
        for error in self.errors:
            key = (error.error_type, error.entity_type)
            error_groups[key].append(error)

        for (error_type, entity_type), errors in error_groups.items():
            if len(errors) >= 2:
                patterns.append({
                    "error_type": error_type,
                    "entity_type": entity_type,
                    "count": len(errors),
                    "examples": [
                        {
                            "predicted": e.predicted_value,
                            "ground_truth": e.ground_truth_value
                        }
                        for e in errors[:3]
                    ]
                })

        return sorted(patterns, key=lambda x: x["count"], reverse=True)
