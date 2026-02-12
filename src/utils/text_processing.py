"""
Advanced text processing utilities for legal document analysis.

Includes improved normalization, similarity metrics, and matching algorithms
optimized for legal clause extraction tasks.
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Set
from difflib import SequenceMatcher
from collections import Counter

# Try to import Levenshtein for faster string comparison
try:
    from Levenshtein import ratio as levenshtein_ratio
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """Fallback Levenshtein using SequenceMatcher."""
        return SequenceMatcher(None, s1, s2).ratio()


# =============================================================================
# LEGAL STOPWORDS
# =============================================================================

ENGLISH_LEGAL_STOPWORDS = {
    # Articles
    'the', 'a', 'an',
    # Prepositions
    'of', 'to', 'in', 'for', 'on', 'by', 'with', 'at', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    # Conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet',
    # Common legal filler
    'such', 'any', 'all', 'each', 'every', 'other', 'shall', 'will', 'may',
    'herein', 'hereof', 'hereby', 'hereto', 'hereunder', 'therein', 'thereof',
    'thereby', 'thereto', 'thereunder', 'whereas', 'therefore', 'provided',
}

PORTUGUESE_LEGAL_STOPWORDS = {
    # Articles
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    # Prepositions
    'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos',
    'para', 'por', 'pela', 'pelo', 'pelas', 'pelos', 'com', 'sem',
    'entre', 'sobre', 'sob', 'ate', 'desde', 'contra', 'perante',
    # Conjunctions
    'e', 'ou', 'mas', 'porem', 'contudo', 'todavia', 'entretanto',
    # Common legal filler
    'que', 'qual', 'quais', 'este', 'esta', 'estes', 'estas',
    'esse', 'essa', 'esses', 'essas', 'aquele', 'aquela',
    'conforme', 'mediante', 'segundo', 'consoante',
}


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text_simple(text: str) -> str:
    """
    Simple text normalization (original behavior).

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = ' '.join(text.split())
    text = re.sub(r'[.,;:!?"\'\(\)\[\]\{\}]', '', text)
    return text


def normalize_text_advanced(
    text: str,
    remove_stopwords: bool = True,
    language: str = "en",
    preserve_numbers: bool = True,
    preserve_legal_terms: bool = True
) -> str:
    """
    Advanced text normalization for legal documents.

    Args:
        text: Input text
        remove_stopwords: Whether to remove common stopwords
        language: Language for stopwords ('en' or 'pt')
        preserve_numbers: Whether to preserve numeric values
        preserve_legal_terms: Whether to preserve common legal terms

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFKC for compatibility)
    text = unicodedata.normalize('NFKC', text)

    # 2. Lowercase
    text = text.lower()

    # 3. Preserve numbers if requested (replace temporarily)
    number_placeholders = {}
    if preserve_numbers:
        # Find all numbers (including decimals, dates, currency)
        numbers = re.findall(r'\$?[\d,]+\.?\d*%?', text)
        for i, num in enumerate(numbers):
            placeholder = f"__NUM{i}__"
            number_placeholders[placeholder] = num
            text = text.replace(num, placeholder, 1)

    # 4. Remove special characters but preserve hyphens in compound words
    # Keep alphanumeric, spaces, and hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text)

    # 5. Normalize whitespace
    text = ' '.join(text.split())

    # 6. Remove stopwords if requested
    if remove_stopwords:
        stopwords = ENGLISH_LEGAL_STOPWORDS if language == "en" else PORTUGUESE_LEGAL_STOPWORDS
        tokens = text.split()
        tokens = [t for t in tokens if t not in stopwords and not t.startswith('__num')]
        text = ' '.join(tokens)

    # 7. Restore numbers
    for placeholder, num in number_placeholders.items():
        text = text.replace(placeholder.lower(), num)

    return text.strip()


def normalize_clause_type(clause_type: str) -> str:
    """
    Normalize clause type names for consistent comparison.

    Handles variations like:
    - "Rofr/Rofo/Rofn" -> "ROFR_ROFO_ROFN"
    - "Ip Ownership Assignment" -> "IP_OWNERSHIP_ASSIGNMENT"
    - "Affiliate License-Licensor" -> "AFFILIATE_LICENSE_LICENSOR"

    Args:
        clause_type: Raw clause type name

    Returns:
        Normalized clause type
    """
    # Known aliases and corrections
    TYPE_ALIASES = {
        "ROFR/ROFO/ROFN": "ROFR_ROFO_ROFN",
        "IP OWNERSHIP": "IP_OWNERSHIP_ASSIGNMENT",
        "IP_OWNERSHIP": "IP_OWNERSHIP_ASSIGNMENT",
        "AFFILIATE LICENSE-LICENSOR": "AFFILIATE_LICENSE_LICENSOR",
        "AFFILIATE LICENSE-LICENSEE": "AFFILIATE_LICENSE_LICENSEE",
        "UNLIMITED/ALL-YOU-CAN-EAT-LICENSE": "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE",
        "NO SOLICIT OF CUSTOMERS": "NO_SOLICIT_OF_CUSTOMERS",
        "NO SOLICIT OF EMPLOYEES": "NO_SOLICIT_OF_EMPLOYEES",
        "IRREVOCABLE OR PERPETUAL LICENSE": "IRREVOCABLE_OR_PERPETUAL_LICENSE",
    }

    # Basic normalization
    normalized = clause_type.upper().strip()
    normalized = normalized.replace(" ", "_").replace("-", "_").replace("/", "_")

    # Check aliases
    if clause_type.upper() in TYPE_ALIASES:
        return TYPE_ALIASES[clause_type.upper()]

    return normalized


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def compute_token_overlap(pred: str, gold: str, use_stopwords: bool = False) -> float:
    """
    Compute token-level overlap (Jaccard-like) between texts.

    Args:
        pred: Predicted text
        gold: Gold/reference text
        use_stopwords: Whether to remove stopwords before comparison

    Returns:
        Overlap score between 0 and 1
    """
    if use_stopwords:
        pred_norm = normalize_text_advanced(pred, remove_stopwords=True)
        gold_norm = normalize_text_advanced(gold, remove_stopwords=True)
    else:
        pred_norm = normalize_text_simple(pred)
        gold_norm = normalize_text_simple(gold)

    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())

    if not gold_tokens:
        return 0.0

    intersection = pred_tokens & gold_tokens
    return len(intersection) / len(gold_tokens)


def compute_levenshtein_similarity(pred: str, gold: str) -> float:
    """
    Compute Levenshtein (edit distance) similarity.

    Args:
        pred: Predicted text
        gold: Gold/reference text

    Returns:
        Similarity score between 0 and 1
    """
    pred_norm = normalize_text_simple(pred)
    gold_norm = normalize_text_simple(gold)

    if not gold_norm:
        return 0.0

    return levenshtein_ratio(pred_norm, gold_norm)


def compute_sequence_similarity(pred: str, gold: str) -> float:
    """
    Compute sequence similarity using SequenceMatcher.

    Args:
        pred: Predicted text
        gold: Gold/reference text

    Returns:
        Similarity score between 0 and 1
    """
    pred_norm = normalize_text_simple(pred)
    gold_norm = normalize_text_simple(gold)

    if not gold_norm:
        return 0.0

    return SequenceMatcher(None, pred_norm, gold_norm).ratio()


def compute_combined_similarity(
    pred: str,
    gold: str,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> float:
    """
    Compute combined similarity using multiple methods.

    Args:
        pred: Predicted text
        gold: Gold/reference text
        weights: Weights for (token_overlap, levenshtein, sequence)

    Returns:
        Combined similarity score between 0 and 1
    """
    w_token, w_lev, w_seq = weights

    token_sim = compute_token_overlap(pred, gold)
    lev_sim = compute_levenshtein_similarity(pred, gold)
    seq_sim = compute_sequence_similarity(pred, gold)

    return w_token * token_sim + w_lev * lev_sim + w_seq * seq_sim


def compute_span_iou(
    pred_start: int, pred_end: int,
    gold_start: int, gold_end: int
) -> float:
    """
    Compute Intersection over Union for character spans.

    Args:
        pred_start: Predicted span start
        pred_end: Predicted span end
        gold_start: Gold span start
        gold_end: Gold span end

    Returns:
        IoU score between 0 and 1
    """
    intersection_start = max(pred_start, gold_start)
    intersection_end = min(pred_end, gold_end)

    if intersection_end <= intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gold_end - gold_start) - intersection

    return intersection / union if union > 0 else 0.0


# =============================================================================
# MATCHING ALGORITHMS
# =============================================================================

def match_extraction_simple(
    pred_text: str,
    gold_texts: List[str],
    threshold: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """
    Simple matching using token overlap (original behavior).

    Args:
        pred_text: Predicted text
        gold_texts: List of gold answers
        threshold: Minimum overlap for match

    Returns:
        (is_match, matched_gold_text)
    """
    for gold in gold_texts:
        overlap = compute_token_overlap(pred_text, gold)
        if overlap >= threshold:
            return True, gold
    return False, None


def match_extraction_advanced(
    pred_text: str,
    gold_texts: List[str],
    threshold: float = 0.5,
    method: str = "combined"
) -> Tuple[bool, Optional[str], float]:
    """
    Advanced matching using multiple similarity methods.

    Args:
        pred_text: Predicted text
        gold_texts: List of gold answers
        threshold: Minimum similarity for match
        method: Similarity method ("token", "levenshtein", "sequence", "combined")

    Returns:
        (is_match, matched_gold_text, best_score)
    """
    best_score = 0.0
    best_gold = None

    for gold in gold_texts:
        if method == "token":
            score = compute_token_overlap(pred_text, gold)
        elif method == "levenshtein":
            score = compute_levenshtein_similarity(pred_text, gold)
        elif method == "sequence":
            score = compute_sequence_similarity(pred_text, gold)
        elif method == "combined":
            score = compute_combined_similarity(pred_text, gold)
        else:
            score = compute_combined_similarity(pred_text, gold)

        if score > best_score:
            best_score = score
            best_gold = gold

    return best_score >= threshold, best_gold, best_score


def deduplicate_by_similarity(
    texts: List[str],
    threshold: float = 0.8
) -> List[str]:
    """
    Remove near-duplicate texts based on similarity.

    Args:
        texts: List of texts to deduplicate
        threshold: Similarity threshold for considering duplicates

    Returns:
        List of unique texts
    """
    if not texts:
        return []

    unique = [texts[0]]

    for text in texts[1:]:
        is_duplicate = False
        for existing in unique:
            if compute_combined_similarity(text, existing) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(text)

    return unique


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Normalization
    'normalize_text_simple',
    'normalize_text_advanced',
    'normalize_clause_type',

    # Similarity metrics
    'compute_token_overlap',
    'compute_levenshtein_similarity',
    'compute_sequence_similarity',
    'compute_combined_similarity',
    'compute_span_iou',

    # Matching
    'match_extraction_simple',
    'match_extraction_advanced',
    'deduplicate_by_similarity',

    # Constants
    'ENGLISH_LEGAL_STOPWORDS',
    'PORTUGUESE_LEGAL_STOPWORDS',
]
