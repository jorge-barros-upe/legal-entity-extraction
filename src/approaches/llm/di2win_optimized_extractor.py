"""
Optimized Entity Extractor for DI2WIN Dataset.

This extractor uses specialized prompts designed for Brazilian social contracts
(contratos sociais) with 143+ entity types.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from .di2win_optimized_prompts import (
    create_di2win_prompt,
    create_di2win_simple_prompt,
    create_di2win_self_consistency_prompt,
    HIGH_FREQUENCY_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[Dict[str, Any]]
    raw_response: str
    latency: float
    model: str
    strategy: str


class DI2WINOptimizedExtractor:
    """
    Optimized extractor for DI2WIN Brazilian social contract dataset.

    Supports multiple strategies:
    - basic: Simple extraction with optimized prompt
    - self_consistency: Multiple extractions with voting
    - validated: Basic extraction with post-processing validation
    """

    def __init__(
        self,
        llm_client,
        model_name: str = "gpt-4o",
        strategy: str = "basic",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        """
        Initialize the extractor.

        Args:
            llm_client: LLM client (OpenAI, Azure, Gemini, etc.)
            model_name: Name of the model
            strategy: Extraction strategy (basic, self_consistency, validated)
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.strategy = strategy
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(
        self,
        text: str,
        entity_types: List[str],
        max_text_length: int = 30000,
    ) -> ExtractionResult:
        """
        Extract entities from text.

        Args:
            text: Contract text
            entity_types: List of entity types to extract
            max_text_length: Maximum text length (truncate if longer)

        Returns:
            ExtractionResult with extracted entities
        """
        import time
        start_time = time.time()

        # Truncate text if too long
        if len(text) > max_text_length:
            text = text[:max_text_length] + "\n[...TEXTO TRUNCADO...]"

        # Select strategy
        if self.strategy == "self_consistency":
            entities = self._extract_with_self_consistency(text, entity_types)
        elif self.strategy == "validated":
            entities = self._extract_with_validation(text, entity_types)
        else:
            entities = self._extract_basic(text, entity_types)

        latency = time.time() - start_time

        return ExtractionResult(
            entities=entities,
            raw_response="",
            latency=latency,
            model=self.model_name,
            strategy=self.strategy,
        )

    def _extract_basic(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Basic extraction with optimized prompt."""
        # Filter to high-frequency types for better focus
        filtered_types = [t for t in entity_types if t in HIGH_FREQUENCY_TYPES]
        if len(filtered_types) < 10:
            filtered_types = entity_types[:40]

        prompt = create_di2win_prompt(
            text=text,
            entity_types=filtered_types,
            include_examples=True,
            max_examples=1
        )

        response = self._call_llm(prompt)
        entities = self._parse_response(response)

        # Filter to only requested types
        entities = [e for e in entities if e.get("type") in entity_types]

        return entities

    def _extract_with_self_consistency(
        self,
        text: str,
        entity_types: List[str],
        num_samples: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract with self-consistency voting."""
        all_entities = []

        for i in range(num_samples):
            prompt = create_di2win_self_consistency_prompt(
                text=text,
                entity_types=entity_types,
                variation=i
            )

            try:
                response = self._call_llm(prompt, temperature=0.3)
                entities = self._parse_response(response)
                logger.info(f"Self-consistency sample {i}: extracted {len(entities)} entities")
                all_entities.extend(entities)
            except Exception as e:
                logger.warning(f"Self-consistency sample {i} failed: {e}")

        logger.info(f"Self-consistency total: {len(all_entities)} entities before voting")
        # Vote on entities - use threshold=1 for union (any entity that appears at least once)
        # This improves recall at the cost of some precision
        return self._vote_on_entities(all_entities, threshold=1)

    def _extract_with_validation(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract with post-processing validation."""
        entities = self._extract_basic(text, entity_types)
        return self._validate_entities(entities)

    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """Call the LLM and return response text."""
        temp = temperature if temperature is not None else self.temperature

        try:
            # Handle different client types
            if hasattr(self.llm_client, 'chat'):
                # OpenAI/Azure style
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

            elif hasattr(self.llm_client, 'generate_content'):
                # Gemini style
                response = self.llm_client.generate_content(prompt)
                return response.text

            elif hasattr(self.llm_client, 'messages'):
                # Anthropic style
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            else:
                raise ValueError(f"Unknown client type: {type(self.llm_client)}")

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract entities."""
        if not response:
            return []

        # Clean response - remove markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Try direct parse first (most common case)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "entities" in data:
                return data["entities"]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response with patterns
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # JSON in markdown
            r'```\s*([\s\S]*?)\s*```',       # Any code block
            r'\{[\s\S]*"entities"[\s\S]*\}', # Object with entities
            r'\[[\s\S]*\{[\s\S]*"text"[\s\S]*\}[\s\S]*\]',  # Array of entities
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    # Use group(1) for patterns with capture groups, group(0) otherwise
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "entities" in data:
                        return data["entities"]
                    elif isinstance(data, list):
                        return data
                except (json.JSONDecodeError, IndexError):
                    continue

        logger.warning(f"Could not parse response: {response[:200]}...")
        return []

        return []

    def _vote_on_entities(
        self,
        all_entities: List[Dict[str, Any]],
        threshold: int = 2
    ) -> List[Dict[str, Any]]:
        """Vote on entities from multiple extractions."""
        # Create canonical keys
        entity_votes = Counter()
        entity_map = {}

        for entity in all_entities:
            text = entity.get("text", "").strip().lower()
            etype = entity.get("type", "")
            key = (text, etype)

            entity_votes[key] += 1
            if key not in entity_map:
                entity_map[key] = entity

        # Return entities that meet threshold
        return [
            entity_map[key]
            for key, count in entity_votes.items()
            if count >= threshold
        ]

    def _validate_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate entities against business rules."""
        validated = []

        for entity in entities:
            text = entity.get("text", "").strip()
            etype = entity.get("type", "")

            # Skip empty
            if not text:
                continue

            # Validate by type
            if "cpf" in etype.lower():
                # CPF must have 11 digits
                digits = re.sub(r'\D', '', text)
                if len(digits) != 11:
                    continue

            elif "cnpj" in etype.lower():
                # CNPJ must have 14 digits
                digits = re.sub(r'\D', '', text)
                if len(digits) != 14:
                    continue

            elif "cep" in etype.lower():
                # CEP must have 8 digits
                digits = re.sub(r'\D', '', text)
                if len(digits) != 8:
                    continue

            elif "nome" in etype.lower() and "->socio" in etype:
                # Name must have at least 2 words
                if len(text.split()) < 2:
                    continue

            validated.append(entity)

        return validated


def create_di2win_extractor(
    provider: str = "azure",
    model: str = "gpt-4o",
    strategy: str = "basic",
    **kwargs
) -> DI2WINOptimizedExtractor:
    """
    Factory function to create DI2WIN extractor.

    Args:
        provider: LLM provider (azure, openai, gemini, anthropic)
        model: Model name
        strategy: Extraction strategy
        **kwargs: Additional arguments

    Returns:
        Configured DI2WINOptimizedExtractor
    """
    from dotenv import load_dotenv
    load_dotenv()

    if provider == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        return DI2WINOptimizedExtractor(
            llm_client=client,
            model_name=model,
            strategy=strategy,
            **kwargs
        )

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return DI2WINOptimizedExtractor(
            llm_client=client,
            model_name=model,
            strategy=strategy,
            **kwargs
        )

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        client = genai.GenerativeModel(model)
        return DI2WINOptimizedExtractor(
            llm_client=client,
            model_name=model,
            strategy=strategy,
            **kwargs
        )

    elif provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return DI2WINOptimizedExtractor(
            llm_client=client,
            model_name=model,
            strategy=strategy,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = [
    'DI2WINOptimizedExtractor',
    'ExtractionResult',
    'create_di2win_extractor',
]
