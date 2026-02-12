"""
Optimized Entity Extractor for CUAD (Commercial Contracts in English).

Implements multiple strategies to improve F1-Score:
1. Optimized prompts with few-shot examples
2. Post-extraction validation
3. Self-consistency voting across multiple extractions

Target: Match or exceed baseline F1=0.62 with improved PARTY extraction.
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# Import optimized prompts - handle both relative and absolute imports
try:
    from .cuad_optimized_prompts import (
        create_cuad_extraction_prompt,
        create_cuad_self_consistency_prompt,
        create_cuad_compact_prompt,
        CUAD_ENTITY_TYPES
    )
except ImportError:
    from cuad_optimized_prompts import (
        create_cuad_extraction_prompt,
        create_cuad_self_consistency_prompt,
        create_cuad_compact_prompt,
        CUAD_ENTITY_TYPES
    )

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedEntity:
    """Represents an extracted entity with validation status."""
    text: str
    type: str
    confidence: float = 0.95
    is_valid: bool = True
    validation_note: str = ""
    start: int = -1  # Position in text (for CUAD format)
    end: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "confidence": self.confidence
        }

    def __hash__(self):
        return hash((self.text.lower().strip(), self.type))

    def __eq__(self, other):
        if not isinstance(other, ExtractedEntity):
            return False
        return (self.text.lower().strip() == other.text.lower().strip() and
                self.type == other.type)


# =============================================================================
# CUAD ENTITY VALIDATORS
# =============================================================================

class CUADEntityValidator:
    """Validates CUAD entity formats and patterns."""

    # Common company suffixes
    COMPANY_SUFFIXES = [
        'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation',
        'llc', 'l.l.c.', 'ltd', 'ltd.', 'limited', 'plc', 'p.l.c.',
        'gmbh', 'ag', 'sa', 's.a.', 'nv', 'n.v.', 'bv', 'b.v.',
        'co', 'co.', 'company', 'lp', 'l.p.', 'llp', 'l.l.p.'
    ]

    # Common document name patterns
    DOC_NAME_PATTERNS = [
        r'agreement', r'contract', r'license', r'amendment',
        r'addendum', r'memorandum', r'letter', r'terms'
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+)?of\s+(?:january|february|march|april|may|june|july|august|september|october|november|december),?\s+\d{4}\b',
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b'
    ]

    # Valid defined terms that represent parties in CUAD
    VALID_DEFINED_TERMS = [
        'company', 'client', 'vendor', 'licensor', 'licensee',
        'contractor', 'consultant', 'provider', 'recipient',
        'seller', 'buyer', 'lessor', 'lessee', 'landlord', 'tenant',
        'employer', 'employee', 'principal', 'agent', 'customer',
        'supplier', 'distributor', 'partner', 'member', 'shareholder',
        'investor', 'borrower', 'lender', 'guarantor', 'service provider',
        'parent', 'subsidiary', 'affiliate', 'holder', 'owner'
    ]

    @classmethod
    def validate_party(cls, text: str) -> Tuple[bool, str]:
        """
        Validate PARTY entity.
        In CUAD, defined terms like 'Company', 'Licensor' ARE valid party entities.
        """
        text_lower = text.lower().strip()

        # Check minimum length
        if len(text) < 2:
            return False, "Party name too short"

        # Accept defined terms as valid parties (CUAD includes these)
        if text_lower in cls.VALID_DEFINED_TERMS:
            return True, "OK (defined term)"

        # Only reject truly generic terms
        reject_terms = ['party', 'parties', 'the parties', 'each party']
        if text_lower in reject_terms:
            return False, "Generic term without specific reference"

        # Check for company suffix (good indicator)
        has_suffix = any(text_lower.endswith(suffix) or f" {suffix}" in text_lower
                        for suffix in cls.COMPANY_SUFFIXES)

        # Check for title case or proper capitalization (person name)
        words = text.split()
        is_proper_name = len(words) >= 1 and text[0].isupper()

        if has_suffix or is_proper_name or len(words) >= 1:
            return True, "OK"

        return False, "May not be a valid party name"

    @classmethod
    def validate_doc_name(cls, text: str) -> Tuple[bool, str]:
        """
        Validate DOC_NAME entity.
        """
        text_lower = text.lower().strip()

        # Check if contains agreement-related term
        has_doc_term = any(re.search(pattern, text_lower)
                          for pattern in cls.DOC_NAME_PATTERNS)

        if has_doc_term:
            return True, "OK"

        # Check if mostly uppercase (common for titles)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if uppercase_ratio > 0.5:
            return True, "OK (uppercase title)"

        return False, "May not be a document name"

    @classmethod
    def validate_agmt_date(cls, text: str) -> Tuple[bool, str]:
        """
        Validate AGMT_DATE entity.
        """
        text_lower = text.lower().strip()

        # Check against date patterns
        for pattern in cls.DATE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "OK"

        # Check for common date words
        date_words = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
        if any(word in text_lower for word in date_words):
            return True, "OK"

        return False, "May not be a valid date format"

    @classmethod
    def validate_entity(cls, entity: ExtractedEntity) -> ExtractedEntity:
        """Validate an entity based on its type."""
        entity_type = entity.type.upper()
        text = entity.text

        is_valid = True
        note = ""

        if entity_type == 'PARTY':
            is_valid, note = cls.validate_party(text)
        elif entity_type == 'DOC_NAME':
            is_valid, note = cls.validate_doc_name(text)
        elif entity_type == 'AGMT_DATE':
            is_valid, note = cls.validate_agmt_date(text)

        entity.is_valid = is_valid
        entity.validation_note = note

        return entity


# =============================================================================
# OPTIMIZED EXTRACTOR
# =============================================================================

class CUADOptimizedExtractor:
    """
    Optimized entity extractor for CUAD dataset.
    """

    def __init__(self, provider: str = "azure"):
        """
        Initialize extractor with specified provider.

        Args:
            provider: "azure" or "gemini"
        """
        self.provider = provider
        self.client = None
        self.model_name = None
        self._init_client()
        self.validator = CUADEntityValidator()

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-06-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
            self.name = f"azure_{self.model_name}_cuad_optimized"

        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            self.client = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.name = f"gemini_{model_name}_cuad_optimized"

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        logger.info(f"Initialized {self.name}")

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Tuple[str, int, int]:
        """
        Call LLM and return response with token counts.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        if self.provider == "azure":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return (
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

        elif self.provider == "gemini":
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json"
                }
            )
            content = response.text
            # Estimate tokens for Gemini
            input_tokens = len(prompt) // 4
            output_tokens = len(content) // 4
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', input_tokens)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', output_tokens)
            return content, input_tokens, output_tokens

    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse JSON response with multiple fallback strategies.
        """
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        # Try direct parse
        try:
            data = json.loads(content)
            return data.get("entities", [])
        except json.JSONDecodeError:
            pass

        # Try to fix truncated JSON
        if not content.endswith('}'):
            last_bracket = content.rfind('}')
            if last_bracket > 0:
                fixed = content[:last_bracket+1]
                if not fixed.endswith(']}'):
                    fixed = fixed + ']}'
                try:
                    data = json.loads(fixed)
                    logger.info("Fixed truncated JSON")
                    return data.get("entities", [])
                except json.JSONDecodeError:
                    pass

        # Fallback: regex extraction
        entities = []
        pattern = r'\{\s*"text"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"(?:\s*,\s*"confidence"\s*:\s*([\d.]+))?'

        for match in re.finditer(pattern, content):
            text, etype, conf = match.groups()
            confidence = float(conf) if conf else 0.85
            entities.append({
                "text": text,
                "type": etype,
                "confidence": confidence
            })

        if entities:
            logger.info(f"Regex fallback recovered {len(entities)} entities")

        return entities

    def _find_entity_position(self, text: str, entity_text: str) -> Tuple[int, int]:
        """Find the position of entity text in the document."""
        # Case-insensitive search
        text_lower = text.lower()
        entity_lower = entity_text.lower()

        pos = text_lower.find(entity_lower)
        if pos >= 0:
            return pos, pos + len(entity_text)

        return -1, -1

    def extract_basic(
        self,
        text: str,
        entity_types: List[str] = None
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Basic extraction with optimized prompt.

        Args:
            text: Contract text
            entity_types: List of entity types to extract (default: CUAD types)

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        if entity_types is None:
            entity_types = CUAD_ENTITY_TYPES

        prompt = create_cuad_extraction_prompt(text)

        content, input_tokens, output_tokens = self._call_llm(prompt)
        raw_entities = self._parse_json_response(content)

        # Convert to ExtractedEntity and validate
        entities = []
        for e in raw_entities:
            etype = e.get("type", "").upper()
            # Only include entities of requested types
            if etype not in entity_types:
                continue

            entity_text = e.get("text", "")
            start, end = self._find_entity_position(text, entity_text)

            entity = ExtractedEntity(
                text=entity_text,
                type=etype,
                confidence=e.get("confidence", 0.85),
                start=start,
                end=end
            )
            entity = self.validator.validate_entity(entity)
            entities.append(entity)

        return entities, input_tokens, output_tokens

    def extract_with_validation(
        self,
        text: str,
        entity_types: List[str] = None,
        filter_invalid: bool = True
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Extract entities with post-processing validation.

        Args:
            text: Contract text
            entity_types: Entity types to extract
            filter_invalid: Remove entities that fail validation

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        entities, in_tok, out_tok = self.extract_basic(text, entity_types)

        if filter_invalid:
            valid_entities = [e for e in entities if e.is_valid]
            invalid_count = len(entities) - len(valid_entities)
            if invalid_count > 0:
                logger.info(f"Filtered {invalid_count} invalid entities")
            return valid_entities, in_tok, out_tok

        return entities, in_tok, out_tok

    def extract_self_consistency(
        self,
        text: str,
        entity_types: List[str] = None,
        num_samples: int = 3,
        threshold: float = 0.5
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Self-consistency extraction with majority voting.

        Args:
            text: Contract text
            entity_types: Entity types to extract
            num_samples: Number of extraction runs
            threshold: Minimum fraction of runs an entity must appear in

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        if entity_types is None:
            entity_types = CUAD_ENTITY_TYPES

        total_input_tokens = 0
        total_output_tokens = 0

        all_extractions: List[List[ExtractedEntity]] = []

        for i in range(num_samples):
            prompt = create_cuad_self_consistency_prompt(text, entity_types, variation=i)

            # Use different temperatures for diversity
            temperatures = [0.0, 0.2, 0.4]
            temp = temperatures[i % len(temperatures)]

            content, in_tok, out_tok = self._call_llm(prompt, temperature=temp)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            raw_entities = self._parse_json_response(content)
            entities = []
            for e in raw_entities:
                etype = e.get("type", "").upper()
                if etype not in entity_types:
                    continue

                entity_text = e.get("text", "")
                start, end = self._find_entity_position(text, entity_text)

                entity = ExtractedEntity(
                    text=entity_text,
                    type=etype,
                    confidence=e.get("confidence", 0.85),
                    start=start,
                    end=end
                )
                entity = self.validator.validate_entity(entity)
                if entity.is_valid:
                    entities.append(entity)

            all_extractions.append(entities)
            logger.info(f"Self-consistency run {i+1}: {len(entities)} valid entities")

        # Voting: count occurrences of each entity
        entity_counts: Counter = Counter()
        for extraction in all_extractions:
            for entity in extraction:
                key = (entity.text.lower().strip(), entity.type)
                entity_counts[key] += 1

        # Keep entities that appear in at least threshold% of runs
        min_votes = int(num_samples * threshold)
        final_entities = []

        for (text_key, etype), count in entity_counts.items():
            if count >= min_votes:
                # Find original text and position
                original_text = text_key
                start, end = -1, -1
                confidences = []

                for extraction in all_extractions:
                    for e in extraction:
                        if e.text.lower().strip() == text_key and e.type == etype:
                            original_text = e.text
                            if e.start >= 0:
                                start, end = e.start, e.end
                            confidences.append(e.confidence)
                            break

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.85

                entity = ExtractedEntity(
                    text=original_text,
                    type=etype,
                    confidence=min(avg_confidence * (count / num_samples), 0.99),
                    start=start,
                    end=end
                )
                final_entities.append(entity)

        logger.info(f"Self-consistency voting: {len(final_entities)} entities (threshold={threshold})")

        return final_entities, total_input_tokens, total_output_tokens

    def extract(
        self,
        text: str,
        entity_types: List[str] = None,
        strategy: str = "validated"
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Main extraction method with strategy selection.

        Args:
            text: Contract text
            entity_types: Entity types to extract
            strategy: One of "basic", "validated", "self_consistency"

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        if strategy == "basic":
            return self.extract_basic(text, entity_types)
        elif strategy == "validated":
            return self.extract_with_validation(text, entity_types)
        elif strategy == "self_consistency":
            return self.extract_self_consistency(text, entity_types)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cuad_extractor(provider: str = "azure") -> CUADOptimizedExtractor:
    """
    Factory function to create CUAD optimized extractor.

    Args:
        provider: "azure" or "gemini"

    Returns:
        CUADOptimizedExtractor instance
    """
    return CUADOptimizedExtractor(provider=provider)


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ExtractedEntity',
    'CUADEntityValidator',
    'CUADOptimizedExtractor',
    'create_cuad_extractor'
]
