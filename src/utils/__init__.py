"""Utility modules."""

from .config import load_config, merge_configs
from .cost_tracker import CostTracker

# Text processing utilities
try:
    from .text_processing import (
        normalize_text_simple,
        normalize_text_advanced,
        normalize_clause_type,
        compute_token_overlap,
        compute_levenshtein_similarity,
        compute_sequence_similarity,
        compute_combined_similarity,
        compute_span_iou,
        match_extraction_simple,
        match_extraction_advanced,
        deduplicate_by_similarity,
        ENGLISH_LEGAL_STOPWORDS,
        PORTUGUESE_LEGAL_STOPWORDS,
    )
except ImportError:
    pass
