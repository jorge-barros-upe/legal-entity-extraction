"""
Hybrid Extractor - Combines multiple approaches for entity extraction.

Supports:
- RAG + LLM validation
- Ensemble of SLMs
- Multi-stage pipelines
"""

import time
import logging
from typing import Dict, List, Any, Optional
from collections import Counter

from ...core.base_extractor import (
    BaseExtractor, ExtractorRegistry, Entity, ExtractionResult
)

logger = logging.getLogger(__name__)


@ExtractorRegistry.register("hybrid")
class HybridExtractor(BaseExtractor):
    """
    Hybrid entity extractor combining multiple approaches.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        entity_types: List[str],
        name: str = "HybridExtractor"
    ):
        super().__init__(config, entity_types, name)

        hybrid_config = config.get("hybrid", config)
        self.approach = self._determine_approach(hybrid_config)

        logger.info(f"Initialized HybridExtractor with approach: {self.approach}")

    def _determine_approach(self, config: Dict[str, Any]) -> str:
        """Determine which hybrid approach to use."""
        if config.get("rag_llm_validation", {}).get("enabled"):
            return "rag_llm_validation"
        elif config.get("ensemble_slm", {}).get("enabled"):
            return "ensemble_slm"
        elif config.get("pipeline", {}).get("enabled"):
            return "pipeline"
        else:
            return "rag_llm_validation"  # Default

    def extract(
        self,
        document: str,
        document_id: str = "unknown",
        **kwargs
    ) -> ExtractionResult:
        """Extract entities using hybrid approach."""
        start_time = time.time()

        if self.approach == "rag_llm_validation":
            entities = self._extract_rag_llm_validation(document, **kwargs)
        elif self.approach == "ensemble_slm":
            entities = self._extract_ensemble(document, **kwargs)
        elif self.approach == "pipeline":
            entities = self._extract_pipeline(document, **kwargs)
        else:
            entities = []

        extraction_time = time.time() - start_time
        self._track_extraction(extraction_time=extraction_time)

        return ExtractionResult(
            document_id=document_id,
            entities=entities,
            extraction_time=extraction_time,
            metadata={
                "approach": "hybrid",
                "hybrid_method": self.approach,
                "document_length": len(document)
            }
        )

    def _extract_rag_llm_validation(
        self,
        document: str,
        **kwargs
    ) -> List[Entity]:
        """
        Two-stage extraction: RAG extracts, LLM validates.
        """
        from ..rag import RAGExtractor
        from ..llm import LLMExtractor

        hybrid_config = self.config.get("hybrid", {})
        rag_llm_config = hybrid_config.get("rag_llm_validation", {})

        # Stage 1: RAG extraction
        rag_config = self.config.copy()
        rag_extractor = RAGExtractor(rag_config, self.entity_types, name="RAG_Stage1")
        rag_result = rag_extractor.extract(document, "temp")
        candidates = rag_result.entities

        if not candidates:
            return []

        # Stage 2: LLM validation
        validation_config = rag_llm_config.get("validation", {})
        validation_prompt = self._build_validation_prompt(
            candidates, document, validation_config
        )

        llm_config = self.config.copy()
        llm_config["llm"] = llm_config.get("llm", {})
        llm_config["llm"]["default"] = {
            "provider": validation_config.get("provider", "anthropic"),
            "model": validation_config.get("model", "claude-3-sonnet")
        }

        llm_extractor = LLMExtractor(llm_config, self.entity_types, name="LLM_Validator")

        # Use custom prompt for validation
        from ..llm.prompting import ZeroShotPrompt
        llm_extractor.prompt_strategy = ZeroShotPrompt(
            {"strategies": {"zero_shot": {"user_template": "{document}"}}},
            self.entity_types
        )

        validation_result = llm_extractor.extract(validation_prompt, "validation")

        # Merge results
        validated = self._merge_validation_results(candidates, validation_result.entities)

        return validated

    def _build_validation_prompt(
        self,
        candidates: List[Entity],
        document: str,
        config: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM validation."""
        candidates_text = "\n".join([
            f"- {e.type}: \"{e.value}\" (confidence: {e.confidence:.2f})"
            for e in candidates
        ])

        # Truncate document if too long
        max_doc_len = 50000
        doc_excerpt = document[:max_doc_len] if len(document) > max_doc_len else document

        template = config.get("prompt_template", """
You are validating entity extractions from a legal contract.

EXTRACTED ENTITIES (candidates):
{candidates}

DOCUMENT EXCERPT:
{document}

For each candidate, determine if it's correct. Output JSON:
{{
  "entities": [
    {{"type": "TYPE", "value": "value", "valid": true/false, "corrected_value": "if needed"}}
  ]
}}
""")

        return template.format(candidates=candidates_text, document=doc_excerpt)

    def _merge_validation_results(
        self,
        candidates: List[Entity],
        validated: List[Entity]
    ) -> List[Entity]:
        """Merge original candidates with validation results."""
        # Build lookup from validated results
        validated_lookup = {}
        for v in validated:
            key = (v.type, v.value.lower().strip())
            validated_lookup[key] = v

        merged = []
        for candidate in candidates:
            key = (candidate.type, candidate.value.lower().strip())
            if key in validated_lookup:
                # Update confidence based on validation
                candidate.confidence = max(candidate.confidence, 0.9)
                candidate.metadata["validated"] = True
                merged.append(candidate)
            elif candidate.confidence >= 0.8:
                # Keep high-confidence even if not validated
                candidate.metadata["validated"] = False
                merged.append(candidate)

        return merged

    def _extract_ensemble(self, document: str, **kwargs) -> List[Entity]:
        """Ensemble of multiple SLM extractors."""
        from ..slm import SLMExtractor

        hybrid_config = self.config.get("hybrid", {})
        ensemble_config = hybrid_config.get("ensemble_slm", {})
        models_config = ensemble_config.get("models", {})

        all_results = []

        for model_name, model_config in models_config.items():
            try:
                # Create SLM extractor for this model
                slm_config = self.config.copy()
                slm_config["slm"] = slm_config.get("slm", {})
                slm_config["slm"]["default"] = {"architecture": model_config.get("base_model")}

                entity_types = model_config.get("entity_types", self.entity_types)
                checkpoint = model_config.get("checkpoint")

                extractor = SLMExtractor(
                    slm_config,
                    entity_types,
                    name=model_name,
                    checkpoint_path=checkpoint
                )

                result = extractor.extract(document, "temp")
                weight = model_config.get("weight", 1.0)

                for entity in result.entities:
                    entity.confidence *= weight
                    entity.metadata["model"] = model_name

                all_results.extend(result.entities)

            except Exception as e:
                logger.warning(f"Ensemble model {model_name} failed: {e}")

        # Aggregate results
        aggregation_method = ensemble_config.get("aggregation", {}).get("method", "weighted_vote")
        aggregated = self._aggregate_ensemble(all_results, aggregation_method)

        return aggregated

    def _aggregate_ensemble(
        self,
        entities: List[Entity],
        method: str
    ) -> List[Entity]:
        """Aggregate ensemble results."""
        if method == "vote":
            return self._majority_vote(entities, threshold=2)
        elif method == "weighted_vote":
            return self._weighted_vote(entities)
        elif method == "confidence_max":
            return self._confidence_max(entities)
        elif method == "union":
            return self._union_aggregate(entities)
        else:
            return self._weighted_vote(entities)

    def _majority_vote(self, entities: List[Entity], threshold: int = 2) -> List[Entity]:
        """Majority voting aggregation."""
        counts: Counter = Counter()
        best_entity: Dict[tuple, Entity] = {}

        for entity in entities:
            key = (entity.type, entity.value.lower().strip())
            counts[key] += 1
            if key not in best_entity or entity.confidence > best_entity[key].confidence:
                best_entity[key] = entity

        results = []
        for key, count in counts.items():
            if count >= threshold:
                entity = best_entity[key]
                entity.confidence = count / len(set(e.metadata.get("model") for e in entities))
                entity.metadata["vote_count"] = count
                results.append(entity)

        return results

    def _weighted_vote(self, entities: List[Entity]) -> List[Entity]:
        """Weighted voting based on confidence."""
        scores: Dict[tuple, float] = {}
        best_entity: Dict[tuple, Entity] = {}

        for entity in entities:
            key = (entity.type, entity.value.lower().strip())
            scores[key] = scores.get(key, 0) + entity.confidence
            if key not in best_entity or entity.confidence > best_entity[key].confidence:
                best_entity[key] = entity

        results = []
        for key, score in scores.items():
            entity = best_entity[key]
            entity.confidence = min(score, 1.0)
            entity.metadata["weighted_score"] = score
            results.append(entity)

        return sorted(results, key=lambda e: e.confidence, reverse=True)

    def _confidence_max(self, entities: List[Entity]) -> List[Entity]:
        """Keep highest confidence for each entity."""
        best: Dict[tuple, Entity] = {}

        for entity in entities:
            key = (entity.type, entity.value.lower().strip())
            if key not in best or entity.confidence > best[key].confidence:
                best[key] = entity

        return list(best.values())

    def _union_aggregate(self, entities: List[Entity]) -> List[Entity]:
        """Union of all entities, deduplicated."""
        seen = set()
        results = []

        for entity in sorted(entities, key=lambda e: e.confidence, reverse=True):
            key = (entity.type, entity.value.lower().strip())
            if key not in seen:
                seen.add(key)
                results.append(entity)

        return results

    def _extract_pipeline(self, document: str, **kwargs) -> List[Entity]:
        """
        Multi-stage pipeline: Segmentation -> Classification -> Extraction.
        """
        hybrid_config = self.config.get("hybrid", {})
        pipeline_config = hybrid_config.get("pipeline", {})
        stages_config = pipeline_config.get("stages", {})

        # Stage 1: Segmentation
        segments = self._segment_document(document, stages_config.get("segmentation", {}))

        # Stage 2: Classification (which segments have which entities)
        classified = self._classify_segments(segments, stages_config.get("classification", {}))

        # Stage 3: Extraction from relevant segments
        entities = self._extract_from_segments(classified, stages_config.get("extraction", {}))

        return entities

    def _segment_document(
        self,
        document: str,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Segment document into sections."""
        from ...core.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor(config)
        method = config.get("method", "section")

        if method == "section":
            sections = preprocessor.detect_sections(document)
            return [{"text": s.text, "start": s.start_char, "end": s.end_char, "type": s.section_type}
                    for s in sections]
        else:
            chunks = preprocessor.chunk_text(document, strategy="sliding_window")
            return [{"text": c.text, "start": c.start_char, "end": c.end_char, "type": "chunk"}
                    for c in chunks]

    def _classify_segments(
        self,
        segments: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Classify which entity types each segment likely contains."""
        # Simple keyword-based classification as fallback
        keywords = {
            "Party": ["contratante", "contratada", "parte", "party", "buyer", "seller"],
            "CONTRATANTE": ["contratante", "parte", "empresa"],
            "CONTRATADA": ["contratada", "prestadora", "fornecedor"],
            "Date": ["data", "date", "assinatura", "signed"],
            "DATA_CONTRATO": ["data", "celebrado", "firmado"],
            "CNPJ": ["cnpj", "inscrita", "cadastro"],
            "VALOR": ["valor", "preço", "r$", "reais"],
            "PRAZO": ["prazo", "vigência", "meses", "anos"],
            "FORO": ["foro", "comarca", "jurisdição"],
        }

        for segment in segments:
            text_lower = segment["text"].lower()
            relevant_types = []

            for entity_type, kws in keywords.items():
                if any(kw in text_lower for kw in kws):
                    relevant_types.append(entity_type)

            segment["relevant_entity_types"] = relevant_types

        return segments

    def _extract_from_segments(
        self,
        segments: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Entity]:
        """Extract entities from classified segments."""
        from ..llm import LLMExtractor

        all_entities = []

        for segment in segments:
            relevant_types = segment.get("relevant_entity_types", [])
            if not relevant_types:
                continue

            # Filter to types we care about
            types_to_extract = [t for t in relevant_types if t in self.entity_types]
            if not types_to_extract:
                continue

            # Use LLM for extraction
            llm_config = self.config.copy()
            extractor = LLMExtractor(llm_config, types_to_extract, name="Pipeline_LLM")

            result = extractor.extract(segment["text"], "segment")

            # Adjust offsets
            for entity in result.entities:
                if entity.start_char is not None:
                    entity.start_char += segment["start"]
                if entity.end_char is not None:
                    entity.end_char += segment["start"]
                entity.metadata["segment_type"] = segment["type"]

            all_entities.extend(result.entities)

        # Deduplicate
        return self._deduplicate(all_entities)

    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities."""
        seen = set()
        unique = []

        for entity in sorted(entities, key=lambda e: e.confidence, reverse=True):
            key = (entity.type, entity.value.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique
