"""
Optimized Entity Extractor for Brazilian Legal Contracts.

Implements multiple strategies to improve F1-Score:
1. Two-phase extraction (section identification + targeted extraction)
2. Post-extraction validation with Brazilian document rules
3. Self-consistency voting across multiple extractions
4. Fallback parsing for malformed JSON responses
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
    from .optimized_prompts import (
        create_optimized_prompt,
        create_compact_prompt,
        create_self_consistency_prompt,
        create_core_extraction_prompt,
        PHASE1_SECTION_IDENTIFICATION,
        PHASE2_EXTRACT_FROM_SECTION,
        CORE_ENTITY_TYPES
    )
except ImportError:
    from optimized_prompts import (
        create_optimized_prompt,
        create_compact_prompt,
        create_self_consistency_prompt,
        create_core_extraction_prompt,
        PHASE1_SECTION_IDENTIFICATION,
        PHASE2_EXTRACT_FROM_SECTION,
        CORE_ENTITY_TYPES
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
# BRAZILIAN DOCUMENT VALIDATORS
# =============================================================================

class BrazilianDocumentValidator:
    """Validates Brazilian document formats (CPF, CNPJ, CEP, etc.)."""

    @staticmethod
    def validate_cpf(cpf: str) -> Tuple[bool, str]:
        """
        Validate CPF format (not mathematical validity).
        CPF has 11 digits: XXX.XXX.XXX-XX
        """
        # Remove formatting
        digits = re.sub(r'\D', '', cpf)

        if len(digits) != 11:
            return False, f"CPF deve ter 11 dígitos, encontrado {len(digits)}"

        # Check for known invalid patterns (all same digit)
        if digits == digits[0] * 11:
            return False, "CPF inválido (todos dígitos iguais)"

        return True, "OK"

    @staticmethod
    def validate_cnpj(cnpj: str) -> Tuple[bool, str]:
        """
        Validate CNPJ format.
        CNPJ has 14 digits: XX.XXX.XXX/XXXX-XX
        """
        digits = re.sub(r'\D', '', cnpj)

        if len(digits) != 14:
            return False, f"CNPJ deve ter 14 dígitos, encontrado {len(digits)}"

        return True, "OK"

    @staticmethod
    def validate_cep(cep: str) -> Tuple[bool, str]:
        """
        Validate CEP format.
        CEP has 8 digits: XXXXX-XXX
        """
        digits = re.sub(r'\D', '', cep)

        if len(digits) != 8:
            return False, f"CEP deve ter 8 dígitos, encontrado {len(digits)}"

        # Brazilian CEPs range from 01000-000 to 99999-999
        cep_num = int(digits)
        if cep_num < 1000000 or cep_num > 99999999:
            return False, "CEP fora do range válido"

        return True, "OK"

    @staticmethod
    def validate_rg(rg: str) -> Tuple[bool, str]:
        """
        Validate RG format (varies by state).
        Generally 7-9 digits, may have letter suffix.
        """
        # Extract digits and letters
        alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', rg)
        digits = re.sub(r'\D', '', rg)

        if len(digits) < 5 or len(digits) > 12:
            return False, f"RG deve ter 5-12 dígitos, encontrado {len(digits)}"

        return True, "OK"

    @staticmethod
    def validate_date(date: str) -> Tuple[bool, str]:
        """Validate date format (DD/MM/YYYY or variations)."""
        # Common patterns
        patterns = [
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{1,2}\s+de\s+\w+\s+de\s+\d{4}',  # 15 de março de 2020
        ]

        for pattern in patterns:
            if re.search(pattern, date, re.IGNORECASE):
                return True, "OK"

        return False, "Formato de data não reconhecido"

    @classmethod
    def validate_entity(cls, entity: ExtractedEntity) -> ExtractedEntity:
        """Validate an entity based on its type."""
        entity_type = entity.type.lower()
        text = entity.text

        is_valid = True
        note = ""

        if 'cpf' in entity_type:
            is_valid, note = cls.validate_cpf(text)
        elif 'cnpj' in entity_type:
            is_valid, note = cls.validate_cnpj(text)
        elif 'cep' in entity_type:
            is_valid, note = cls.validate_cep(text)
        elif 'rg' in entity_type:
            is_valid, note = cls.validate_rg(text)
        elif 'data' in entity_type:
            is_valid, note = cls.validate_date(text)
        elif 'nome' in entity_type:
            # Names should have at least 2 words and reasonable length
            words = text.split()
            if len(words) < 2:
                is_valid, note = False, "Nome deve ter pelo menos 2 palavras"
            elif len(text) < 5:
                is_valid, note = False, "Nome muito curto"

        entity.is_valid = is_valid
        entity.validation_note = note

        return entity


# =============================================================================
# OPTIMIZED EXTRACTOR
# =============================================================================

class OptimizedEntityExtractor:
    """
    Optimized entity extractor with multiple strategies.
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
        self.validator = BrazilianDocumentValidator()

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
            self.name = f"azure_{self.model_name}_optimized"

        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            self.client = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.name = f"gemini_{model_name}_optimized"

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
        # Clean content
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

    def extract_basic(
        self,
        text: str,
        entity_types: List[str]
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Basic extraction with optimized prompt.

        Args:
            text: Contract text
            entity_types: List of entity types to extract

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        # Check if using core entity types (10 types)
        is_core = len(entity_types) <= 12 and all(
            t in CORE_ENTITY_TYPES for t in entity_types
        )

        if is_core:
            # Use focused core prompt
            prompt = create_core_extraction_prompt(text)
        elif self.provider == "gemini":
            prompt = create_compact_prompt(text, entity_types)
        else:
            prompt = create_optimized_prompt(
                text, entity_types,
                use_cot=True,
                use_few_shot=True,
                num_examples=2
            )

        content, input_tokens, output_tokens = self._call_llm(prompt)
        raw_entities = self._parse_json_response(content)

        # Convert to ExtractedEntity and validate
        # Also filter to only requested entity types
        entities = []
        for e in raw_entities:
            etype = e.get("type", "")
            # Only include entities of requested types
            if etype in entity_types:
                entity = ExtractedEntity(
                    text=e.get("text", ""),
                    type=etype,
                    confidence=e.get("confidence", 0.85)
                )
                entity = self.validator.validate_entity(entity)
                entities.append(entity)

        return entities, input_tokens, output_tokens

    def extract_with_validation(
        self,
        text: str,
        entity_types: List[str],
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

    def extract_two_phase(
        self,
        text: str,
        entity_types: List[str]
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Two-phase extraction: identify sections, then extract from each.

        Phase 1: Identify relevant sections of the contract
        Phase 2: Extract entities from each identified section

        This approach helps with long documents and reduces hallucination.
        """
        total_input_tokens = 0
        total_output_tokens = 0

        # Phase 1: Section identification
        phase1_prompt = PHASE1_SECTION_IDENTIFICATION.format(text=text[:20000])
        content1, in_tok, out_tok = self._call_llm(phase1_prompt)
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        try:
            sections_data = json.loads(content1)
        except json.JSONDecodeError:
            logger.warning("Phase 1 failed, falling back to basic extraction")
            return self.extract_with_validation(text, entity_types)

        # Phase 2: Extract from each section
        all_entities = []
        sections = sections_data.get("sections", {})

        # Extract from sócios sections
        socio_types = [t for t in entity_types if '->socio' in t or 'socio' in t.lower()]
        for section_text in sections.get("socios", []):
            if section_text:
                phase2_prompt = PHASE2_EXTRACT_FROM_SECTION.format(
                    section_type="SÓCIO",
                    expected_types=", ".join(socio_types),
                    section_text=section_text
                )
                content2, in_tok, out_tok = self._call_llm(phase2_prompt, max_tokens=1024)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                raw_entities = self._parse_json_response(content2)
                for e in raw_entities:
                    entity = ExtractedEntity(
                        text=e.get("text", ""),
                        type=e.get("type", ""),
                        confidence=e.get("confidence", 0.85)
                    )
                    entity = self.validator.validate_entity(entity)
                    if entity.is_valid:
                        all_entities.append(entity)

        # Extract from sociedade sections
        sociedade_types = [t for t in entity_types if '->sociedade' in t]
        for section_text in sections.get("sociedade", []):
            if section_text:
                phase2_prompt = PHASE2_EXTRACT_FROM_SECTION.format(
                    section_type="SOCIEDADE",
                    expected_types=", ".join(sociedade_types),
                    section_text=section_text
                )
                content2, in_tok, out_tok = self._call_llm(phase2_prompt, max_tokens=1024)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                raw_entities = self._parse_json_response(content2)
                for e in raw_entities:
                    entity = ExtractedEntity(
                        text=e.get("text", ""),
                        type=e.get("type", ""),
                        confidence=e.get("confidence", 0.85)
                    )
                    entity = self.validator.validate_entity(entity)
                    if entity.is_valid:
                        all_entities.append(entity)

        # Extract dates from registro sections
        date_types = [t for t in entity_types if 'data' in t.lower()]
        for section_text in sections.get("registro", []):
            if section_text:
                phase2_prompt = PHASE2_EXTRACT_FROM_SECTION.format(
                    section_type="REGISTRO/DATA",
                    expected_types=", ".join(date_types),
                    section_text=section_text
                )
                content2, in_tok, out_tok = self._call_llm(phase2_prompt, max_tokens=512)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                raw_entities = self._parse_json_response(content2)
                for e in raw_entities:
                    entity = ExtractedEntity(
                        text=e.get("text", ""),
                        type=e.get("type", ""),
                        confidence=e.get("confidence", 0.85)
                    )
                    all_entities.append(entity)

        # Deduplicate
        unique_entities = list(set(all_entities))
        logger.info(f"Two-phase extraction: {len(unique_entities)} unique entities")

        return unique_entities, total_input_tokens, total_output_tokens

    def extract_self_consistency(
        self,
        text: str,
        entity_types: List[str],
        num_samples: int = 3,
        threshold: float = 0.5
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Self-consistency extraction with majority voting.

        Runs multiple extractions with different prompt variations
        and keeps entities that appear in at least threshold% of runs.

        Args:
            text: Contract text
            entity_types: Entity types to extract
            num_samples: Number of extraction runs
            threshold: Minimum fraction of runs an entity must appear in

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        total_input_tokens = 0
        total_output_tokens = 0

        all_extractions: List[List[ExtractedEntity]] = []

        for i in range(num_samples):
            prompt = create_self_consistency_prompt(text, entity_types, variation=i)

            # Use different temperatures for diversity
            temperatures = [0.0, 0.3, 0.5]
            temp = temperatures[i % len(temperatures)]

            content, in_tok, out_tok = self._call_llm(prompt, temperature=temp)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            raw_entities = self._parse_json_response(content)
            entities = []
            for e in raw_entities:
                etype = e.get("type", "")
                # Only include entities of requested types
                if etype not in entity_types:
                    continue
                entity = ExtractedEntity(
                    text=e.get("text", ""),
                    type=etype,
                    confidence=e.get("confidence", 0.85)
                )
                entity = self.validator.validate_entity(entity)
                if entity.is_valid:
                    entities.append(entity)

            all_extractions.append(entities)
            logger.info(f"Self-consistency run {i+1}: {len(entities)} valid entities (filtered to {len(entity_types)} types)")

        # Voting: count occurrences of each entity
        entity_counts: Counter = Counter()
        for extraction in all_extractions:
            for entity in extraction:
                # Use (text, type) tuple as key
                key = (entity.text.lower().strip(), entity.type)
                entity_counts[key] += 1

        # Keep entities that appear in at least threshold% of runs
        min_votes = int(num_samples * threshold)
        final_entities = []

        for (text, etype), count in entity_counts.items():
            if count >= min_votes:
                # Average confidence from all runs
                confidences = []
                for extraction in all_extractions:
                    for e in extraction:
                        if e.text.lower().strip() == text and e.type == etype:
                            confidences.append(e.confidence)
                            break

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.85

                # Find original text (preserve case from first occurrence)
                original_text = text
                for extraction in all_extractions:
                    for e in extraction:
                        if e.text.lower().strip() == text and e.type == etype:
                            original_text = e.text
                            break
                    if original_text != text:
                        break

                entity = ExtractedEntity(
                    text=original_text,
                    type=etype,
                    confidence=min(avg_confidence * (count / num_samples), 0.99)
                )
                final_entities.append(entity)

        logger.info(f"Self-consistency voting: {len(final_entities)} entities (threshold={threshold})")

        return final_entities, total_input_tokens, total_output_tokens

    def extract(
        self,
        text: str,
        entity_types: List[str],
        strategy: str = "validated"
    ) -> Tuple[List[ExtractedEntity], int, int]:
        """
        Main extraction method with strategy selection.

        Args:
            text: Contract text
            entity_types: Entity types to extract
            strategy: One of "basic", "validated", "two_phase", "self_consistency"

        Returns:
            Tuple of (entities, input_tokens, output_tokens)
        """
        if strategy == "basic":
            return self.extract_basic(text, entity_types)
        elif strategy == "validated":
            return self.extract_with_validation(text, entity_types)
        elif strategy == "two_phase":
            return self.extract_two_phase(text, entity_types)
        elif strategy == "self_consistency":
            return self.extract_self_consistency(text, entity_types)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_optimized_extractor(provider: str = "azure") -> OptimizedEntityExtractor:
    """
    Factory function to create an optimized extractor.

    Args:
        provider: "azure" or "gemini"

    Returns:
        OptimizedEntityExtractor instance
    """
    return OptimizedEntityExtractor(provider=provider)


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ExtractedEntity',
    'BrazilianDocumentValidator',
    'OptimizedEntityExtractor',
    'create_optimized_extractor'
]
