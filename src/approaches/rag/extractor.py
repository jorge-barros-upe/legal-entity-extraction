"""
RAG Extractor - Retrieval-Augmented Generation for Entity Extraction.

This approach:
1. Chunks the document
2. Embeds and indexes chunks
3. Retrieves relevant chunks for each query
4. Uses LLM to extract entities from retrieved context
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional

from ...core.base_extractor import (
    BaseExtractor, ExtractorRegistry, Entity, ExtractionResult
)
from ...core.preprocessing import TextPreprocessor, TextChunk
from .chunking import ChunkingStrategy, create_chunker
from .embeddings import EmbeddingModel, create_embedding_model
from .retrieval import Retriever, create_retriever

logger = logging.getLogger(__name__)


@ExtractorRegistry.register("rag")
class RAGExtractor(BaseExtractor):
    """
    RAG-based entity extractor.

    Combines retrieval (chunking + embedding + search) with
    generation (LLM-based extraction from retrieved context).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        entity_types: List[str],
        name: str = "RAGExtractor"
    ):
        super().__init__(config, entity_types, name)

        # Get RAG-specific config
        rag_config = config.get("rag", config)

        # Initialize components
        self.preprocessor = TextPreprocessor(config.get("preprocessing", {}))

        # Chunking
        chunking_config = rag_config.get("chunking", {})
        self.chunking_strategy = chunking_config.get("strategy", "sliding_window")
        self.chunker = create_chunker(self.chunking_strategy, chunking_config)

        # Embeddings
        embedding_config = rag_config.get("embeddings", {})
        self.embedding_model_name = embedding_config.get("model", "openai")
        self.embedding_model = create_embedding_model(
            self.embedding_model_name, embedding_config
        )

        # Retrieval
        retrieval_config = rag_config.get("retrieval", {})
        self.retrieval_strategy = retrieval_config.get("strategy", "dense")
        self.retriever = create_retriever(
            self.retrieval_strategy,
            retrieval_config,
            self.embedding_model
        )

        # Generation (LLM for final extraction)
        generation_config = rag_config.get("generation", {})
        self.llm_provider = generation_config.get("provider", "openai")
        self.llm_model = generation_config.get("model", "gpt-4o")
        self.llm_params = generation_config.get("parameters", {})
        self.prompt_template = generation_config.get("prompt_template", self._default_prompt())

        # Initialize LLM client
        self._init_llm_client()

        logger.info(
            f"Initialized RAGExtractor: "
            f"chunking={self.chunking_strategy}, "
            f"embeddings={self.embedding_model_name}, "
            f"retrieval={self.retrieval_strategy}, "
            f"llm={self.llm_provider}/{self.llm_model}"
        )

    def _init_llm_client(self):
        """Initialize the LLM client based on provider."""
        import os

        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("OpenAI not installed. Install with: pip install openai")
                self.llm_client = None

        elif self.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                logger.warning("Anthropic not installed. Install with: pip install anthropic")
                self.llm_client = None

        elif self.llm_provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.llm_client = genai
            except ImportError:
                logger.warning("Google AI not installed. Install with: pip install google-generativeai")
                self.llm_client = None

    def extract(
        self,
        document: str,
        document_id: str = "unknown",
        **kwargs
    ) -> ExtractionResult:
        """
        Extract entities using RAG approach.

        Steps:
        1. Preprocess and chunk document
        2. Embed and index chunks
        3. For each entity type, retrieve relevant chunks
        4. Use LLM to extract entities from context

        Args:
            document: Document text
            document_id: Document identifier

        Returns:
            ExtractionResult with extracted entities
        """
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        # Step 1: Preprocess
        normalized_text = self.preprocessor.normalize_text(document)

        # Step 2: Chunk document
        chunks = self.chunker.chunk(normalized_text)
        logger.debug(f"Created {len(chunks)} chunks for document {document_id}")

        # Step 3: Index chunks (embed and store)
        self.retriever.index(chunks)

        # Step 4: Extract entities
        all_entities = []

        # Create queries for entity types
        queries = self._create_entity_queries()

        for entity_type, query in queries.items():
            # Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(query, top_k=self.config.get("top_k", 5))

            if not retrieved_chunks:
                continue

            # Build context from retrieved chunks
            context = self._build_context(retrieved_chunks)

            # Extract entities using LLM
            entities, input_tokens, output_tokens = self._extract_with_llm(
                context=context,
                entity_types=[entity_type],
                document_id=document_id
            )

            all_entities.extend(entities)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Deduplicate entities
        unique_entities = self._deduplicate_entities(all_entities)

        extraction_time = time.time() - start_time

        # Calculate cost
        api_cost = self._calculate_cost(total_input_tokens, total_output_tokens)

        # Track statistics
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
                "approach": "rag",
                "chunking_strategy": self.chunking_strategy,
                "num_chunks": len(chunks),
                "retrieval_strategy": self.retrieval_strategy,
                "llm_model": f"{self.llm_provider}/{self.llm_model}"
            },
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            api_cost=api_cost
        )

    def _create_entity_queries(self) -> Dict[str, str]:
        """Create search queries for each entity type."""
        # Query templates optimized for legal documents
        query_templates = {
            # English
            "Party": "contracting parties names companies buyer seller",
            "Date": "agreement date effective date execution date signed",
            "Governing Law": "governing law applicable law jurisdiction governed by",
            "Jurisdiction": "jurisdiction courts disputes resolution venue",
            "Termination Clause": "termination terminate end agreement cancellation",
            "Renewal Term": "renewal auto-renewal extension term",
            "Notice Period": "notice days prior written notice notification",
            "Non-Compete Term": "non-compete competition restriction covenant",
            "Exclusivity": "exclusive exclusivity sole rights",

            # Portuguese
            "CONTRATANTE": "contratante parte contratação empresa",
            "CONTRATADA": "contratada prestadora fornecedor empresa",
            "CNPJ_CONTRATANTE": "CNPJ inscrita contratante número",
            "CNPJ_CONTRATADA": "CNPJ inscrita contratada prestadora",
            "DATA_CONTRATO": "data assinatura celebrado firmado",
            "VALOR_CONTRATO": "valor preço remuneração pagamento R$",
            "PRAZO_VIGENCIA": "prazo vigência duração período meses",
            "OBJETO_CONTRATO": "objeto contratação prestação serviço",
            "FORO": "foro comarca eleito jurisdição",
        }

        return {
            entity_type: query_templates.get(entity_type, entity_type)
            for entity_type in self.entity_types
        }

    def _build_context(self, chunks: List[TextChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Trecho {i}]\n{chunk.text}")
        return "\n\n".join(context_parts)

    def _extract_with_llm(
        self,
        context: str,
        entity_types: List[str],
        document_id: str
    ) -> tuple[List[Entity], int, int]:
        """Use LLM to extract entities from context."""
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return [], 0, 0

        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            entity_types=", ".join(entity_types)
        )

        input_tokens = 0
        output_tokens = 0
        entities = []

        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_params.get("temperature", 0.0),
                    max_tokens=self.llm_params.get("max_tokens", 2048),
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=self.llm_params.get("max_tokens", 2048),
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            elif self.llm_provider == "google":
                model = self.llm_client.GenerativeModel(self.llm_model)
                response = model.generate_content(prompt)
                content = response.text
                # Google doesn't always provide token counts
                input_tokens = len(prompt) // 4  # Rough estimate
                output_tokens = len(content) // 4

            # Parse response
            entities = self._parse_llm_response(content, entity_types)

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")

        return entities, input_tokens, output_tokens

    def _parse_llm_response(
        self,
        response: str,
        entity_types: List[str]
    ) -> List[Entity]:
        """Parse LLM response to extract entities."""
        entities = []

        try:
            # Try to parse as JSON
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)

            # Handle different response formats
            entity_list = data.get("entities", data.get("extracted_entities", []))

            for item in entity_list:
                if isinstance(item, dict):
                    entity_type = item.get("type", item.get("entity_type", ""))
                    value = item.get("value", item.get("text", ""))

                    if entity_type and value and value.upper() != "NOT_FOUND":
                        entities.append(Entity(
                            type=entity_type,
                            value=value,
                            confidence=item.get("confidence", 0.9),
                            source="rag_extraction"
                        ))

        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON")
            # Try regex fallback for simple cases
            entities = self._regex_fallback(response, entity_types)

        return entities

    def _regex_fallback(
        self,
        response: str,
        entity_types: List[str]
    ) -> List[Entity]:
        """Fallback extraction using regex patterns."""
        import re
        entities = []

        for entity_type in entity_types:
            # Look for patterns like "ENTITY_TYPE: value" or "ENTITY_TYPE = value"
            pattern = rf"{re.escape(entity_type)}[:\s=]+([^\n,]+)"
            matches = re.findall(pattern, response, re.IGNORECASE)

            for match in matches:
                value = match.strip().strip('"\'')
                if value and value.upper() != "NOT_FOUND":
                    entities.append(Entity(
                        type=entity_type,
                        value=value,
                        confidence=0.7,  # Lower confidence for regex
                        source="rag_regex_fallback"
                    ))

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []

        for entity in entities:
            key = (entity.type, entity.value.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost based on token usage."""
        # Cost per 1M tokens (as of 2024)
        costs = {
            "openai": {
                "gpt-4-turbo": {"input": 10.0, "output": 30.0},
                "gpt-4o": {"input": 5.0, "output": 15.0},
                "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            },
            "anthropic": {
                "claude-3-opus": {"input": 15.0, "output": 75.0},
                "claude-3-sonnet": {"input": 3.0, "output": 15.0},
                "claude-3-haiku": {"input": 0.25, "output": 1.25},
            },
            "google": {
                "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
                "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
            }
        }

        provider_costs = costs.get(self.llm_provider, {})
        model_costs = provider_costs.get(self.llm_model, {"input": 0, "output": 0})

        cost = (
            (input_tokens / 1_000_000) * model_costs["input"] +
            (output_tokens / 1_000_000) * model_costs["output"]
        )

        return cost

    def _default_prompt(self) -> str:
        """Default prompt template for entity extraction."""
        return """You are an expert legal document analyst specializing in contract entity extraction.

Given the following context from a legal contract, extract the specified entities.

CONTEXT:
{context}

ENTITY TYPES TO EXTRACT:
{entity_types}

INSTRUCTIONS:
1. Extract only entities that are explicitly mentioned in the context
2. For each entity, provide the exact text as it appears in the document
3. If an entity is not found, do NOT include it in the output
4. Be precise - extract the complete entity value without truncation

OUTPUT FORMAT (JSON):
{{
  "entities": [
    {{"type": "ENTITY_TYPE", "value": "extracted text", "confidence": 0.95}}
  ]
}}

EXTRACTED ENTITIES:"""
