"""
LLM Extractor - Long-Context Language Models for Entity Extraction.

Uses LLMs with large context windows (128k-1M tokens) to process
entire documents without chunking, or with minimal splitting.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional

from ...core.base_extractor import (
    BaseExtractor, ExtractorRegistry, Entity, ExtractionResult
)
from .providers import LLMProvider, create_provider
from .prompting import PromptStrategy, create_prompt_strategy

logger = logging.getLogger(__name__)


@ExtractorRegistry.register("llm")
class LLMExtractor(BaseExtractor):
    """
    LLM-based entity extractor using long-context models.

    Supports:
    - GPT-4 Turbo (128k tokens)
    - Claude 3 (200k tokens)
    - Gemini 1.5 Pro (1M tokens)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        entity_types: List[str],
        name: str = "LLMExtractor"
    ):
        super().__init__(config, entity_types, name)

        # Get LLM-specific config
        llm_config = config.get("llm", config)

        # Default settings
        default_config = llm_config.get("default", {})
        self.provider_name = default_config.get("provider", "openai")
        self.model_name = default_config.get("model", "gpt-4o")
        self.temperature = default_config.get("temperature", 0.0)
        self.max_retries = default_config.get("max_retries", 3)

        # Initialize provider
        providers_config = llm_config.get("providers", {})
        self.provider = create_provider(
            self.provider_name,
            providers_config.get(self.provider_name, {}),
            self.model_name
        )

        # Initialize prompting strategy
        prompting_config = llm_config.get("prompting", {})
        self.prompt_strategy_name = prompting_config.get("strategy", "few_shot")
        self.prompt_strategy = create_prompt_strategy(
            self.prompt_strategy_name,
            prompting_config,
            entity_types
        )

        # Document handling for very long documents
        self.doc_handling = llm_config.get("document_handling", {})
        self.doc_strategy = self.doc_handling.get("strategy", "direct")

        logger.info(
            f"Initialized LLMExtractor: "
            f"provider={self.provider_name}, "
            f"model={self.model_name}, "
            f"prompting={self.prompt_strategy_name}"
        )

    def extract(
        self,
        document: str,
        document_id: str = "unknown",
        **kwargs
    ) -> ExtractionResult:
        """
        Extract entities using long-context LLM.

        Args:
            document: Document text
            document_id: Document identifier

        Returns:
            ExtractionResult with extracted entities
        """
        start_time = time.time()

        # Check document length vs context window
        estimated_tokens = len(document) // 4  # Rough estimate
        context_window = self.provider.get_context_window()

        if estimated_tokens > context_window * 0.9:
            # Document too long, use handling strategy
            return self._extract_long_document(document, document_id, **kwargs)

        # Direct extraction
        return self._extract_direct(document, document_id, start_time, **kwargs)

    def _extract_direct(
        self,
        document: str,
        document_id: str,
        start_time: float,
        **kwargs
    ) -> ExtractionResult:
        """Extract from document that fits in context."""
        # Build prompt
        prompt = self.prompt_strategy.build_prompt(document, self.entity_types)

        # Call LLM
        response, input_tokens, output_tokens = self.provider.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_retries=self.max_retries
        )

        # Parse response
        entities = self._parse_response(response)

        extraction_time = time.time() - start_time
        api_cost = self.provider.calculate_cost(input_tokens, output_tokens)

        # Track statistics
        self._track_extraction(
            extraction_time=extraction_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=api_cost
        )

        return ExtractionResult(
            document_id=document_id,
            entities=entities,
            extraction_time=extraction_time,
            metadata={
                "approach": "llm",
                "provider": self.provider_name,
                "model": self.model_name,
                "prompting": self.prompt_strategy_name,
                "document_length": len(document)
            },
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_cost=api_cost
        )

    def _extract_long_document(
        self,
        document: str,
        document_id: str,
        **kwargs
    ) -> ExtractionResult:
        """Handle documents that exceed context window."""
        start_time = time.time()
        strategy = self.doc_strategy

        if strategy == "truncate":
            return self._extract_truncate(document, document_id, start_time)
        elif strategy == "chunk_aggregate":
            return self._extract_chunk_aggregate(document, document_id, start_time)
        elif strategy == "hierarchical":
            return self._extract_hierarchical(document, document_id, start_time)
        else:
            # Default: truncate
            return self._extract_truncate(document, document_id, start_time)

    def _extract_truncate(
        self,
        document: str,
        document_id: str,
        start_time: float
    ) -> ExtractionResult:
        """Extract by truncating document to fit context."""
        context_window = self.provider.get_context_window()
        max_chars = int(context_window * 4 * 0.8)  # 80% of estimated capacity

        truncate_config = self.doc_handling.get("strategies", {}).get("truncate", {})
        keep = truncate_config.get("keep", "start")

        if keep == "start":
            truncated = document[:max_chars]
        elif keep == "end":
            truncated = document[-max_chars:]
        elif keep == "both":
            half = max_chars // 2
            truncated = document[:half] + "\n\n[...TRUNCATED...]\n\n" + document[-half:]
        else:
            truncated = document[:max_chars]

        logger.warning(
            f"Document {document_id} truncated from {len(document)} to {len(truncated)} chars"
        )

        return self._extract_direct(truncated, document_id, start_time)

    def _extract_chunk_aggregate(
        self,
        document: str,
        document_id: str,
        start_time: float
    ) -> ExtractionResult:
        """Extract by processing chunks and aggregating results."""
        chunk_config = self.doc_handling.get("strategies", {}).get("chunk_aggregate", {})
        chunk_size = chunk_config.get("chunk_size", 50000) * 4  # tokens to chars
        overlap = chunk_config.get("overlap", 1000) * 4

        # Split document into chunks
        chunks = []
        pos = 0
        while pos < len(document):
            end = min(pos + chunk_size, len(document))
            chunks.append(document[pos:end])
            pos = end - overlap if end < len(document) else len(document)

        # Process each chunk
        all_entities = []
        total_input_tokens = 0
        total_output_tokens = 0

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {document_id}")

            prompt = self.prompt_strategy.build_prompt(chunk, self.entity_types)
            response, in_tokens, out_tokens = self.provider.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_retries=self.max_retries
            )

            entities = self._parse_response(response)
            for entity in entities:
                entity.metadata["chunk"] = i
            all_entities.extend(entities)

            total_input_tokens += in_tokens
            total_output_tokens += out_tokens

        # Deduplicate
        unique_entities = self._deduplicate_entities(all_entities)

        extraction_time = time.time() - start_time
        api_cost = self.provider.calculate_cost(total_input_tokens, total_output_tokens)

        self._track_extraction(
            extraction_time=extraction_time,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost=api_cost
        )

        return ExtractionResult(
            document_id=document_id,
            entities=unique_entities,
            extraction_time=extraction_time,
            metadata={
                "approach": "llm",
                "strategy": "chunk_aggregate",
                "num_chunks": len(chunks),
                "provider": self.provider_name,
                "model": self.model_name
            },
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            api_cost=api_cost
        )

    def _extract_hierarchical(
        self,
        document: str,
        document_id: str,
        start_time: float
    ) -> ExtractionResult:
        """Two-pass extraction: identify sections, then extract from relevant ones."""
        hier_config = self.doc_handling.get("strategies", {}).get("hierarchical", {})

        # First pass: identify relevant sections
        first_pass_prompt = f"""Analyze this legal document and identify sections that likely contain the following information:
{', '.join(self.entity_types)}

For each entity type, list the section numbers or headers where this information is likely found.

DOCUMENT:
{document[:50000]}  # Use first portion for section analysis

OUTPUT FORMAT (JSON):
{{
  "sections": [
    {{"entity_type": "TYPE", "likely_locations": ["section 1", "clause 2"]}}
  ]
}}"""

        # Second pass will be a focused extraction
        # For now, fall back to chunk_aggregate for simplicity
        logger.info(f"Hierarchical extraction for {document_id} - using chunk_aggregate")
        return self._extract_chunk_aggregate(document, document_id, start_time)

    def _parse_response(self, response: str) -> List[Entity]:
        """Parse LLM response to extract entities."""
        entities = []

        try:
            # Clean response (handle markdown code blocks)
            cleaned = response
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]

            data = json.loads(cleaned)

            # Handle various response formats
            entity_list = (
                data.get("entities") or
                data.get("extracted_entities") or
                data.get("results") or
                []
            )

            for item in entity_list:
                if isinstance(item, dict):
                    entity_type = item.get("type") or item.get("entity_type") or ""
                    value = item.get("value") or item.get("text") or item.get("extracted_text") or ""

                    if entity_type and value:
                        # Skip NOT_FOUND markers
                        if value.upper() in ("NOT_FOUND", "N/A", "NONE", "NULL"):
                            continue

                        entities.append(Entity(
                            type=entity_type,
                            value=value.strip(),
                            confidence=item.get("confidence", 0.9),
                            source=f"llm_{self.provider_name}",
                            metadata={
                                "location": item.get("location"),
                                "context": item.get("context")
                            }
                        ))

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, attempting regex extraction")
            entities = self._regex_fallback(response)

        return entities

    def _regex_fallback(self, response: str) -> List[Entity]:
        """Fallback extraction using regex patterns."""
        import re
        entities = []

        for entity_type in self.entity_types:
            # Look for patterns like "Entity_Type: value" or "- Entity_Type: value"
            patterns = [
                rf'{re.escape(entity_type)}[:\s]+["\']?([^"\'\n,]+)["\']?',
                rf'\*\*{re.escape(entity_type)}\*\*[:\s]+([^\n]+)',
                rf'- {re.escape(entity_type)}[:\s]+([^\n]+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    value = match.strip().strip('"\'')
                    if value and value.upper() not in ("NOT_FOUND", "N/A", "NONE"):
                        entities.append(Entity(
                            type=entity_type,
                            value=value,
                            confidence=0.7,
                            source="llm_regex_fallback"
                        ))

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen: Dict[tuple, Entity] = {}

        for entity in entities:
            key = (entity.type, entity.value.lower().strip())
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())
