"""
Prompting Strategies for LLM-based Entity Extraction.

Supports:
- Zero-shot: Direct extraction without examples
- Few-shot: Extraction with examples
- Chain-of-thought: Step-by-step reasoning
- Self-consistency: Multiple samples with voting
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptStrategy(ABC):
    """Abstract base class for prompting strategies."""

    def __init__(self, config: Dict[str, Any], entity_types: List[str]):
        self.config = config
        self.entity_types = entity_types

    @abstractmethod
    def build_prompt(self, document: str, entity_types: List[str]) -> str:
        """Build the prompt for entity extraction."""
        pass


class ZeroShotPrompt(PromptStrategy):
    """Zero-shot prompting - no examples provided."""

    def __init__(self, config: Dict[str, Any], entity_types: List[str]):
        super().__init__(config, entity_types)
        strategies = config.get("strategies", {})
        zero_shot = strategies.get("zero_shot", {})
        self.system_prompt = zero_shot.get("system_prompt", self._default_system_prompt())
        self.user_template = zero_shot.get("user_template", self._default_user_template())

    def build_prompt(self, document: str, entity_types: List[str]) -> str:
        """Build zero-shot prompt."""
        entity_list = "\n".join(f"- {et}" for et in entity_types)

        prompt = f"""{self.system_prompt}

{self.user_template.format(
    entity_types=entity_list,
    document=document
)}"""
        return prompt

    def _default_system_prompt(self) -> str:
        return """You are an expert legal document analyst specializing in extracting specific entities from contracts.
You must extract entities precisely as they appear in the document.
Always respond in valid JSON format."""

    def _default_user_template(self) -> str:
        return """Extract the following entity types from this legal contract:
{entity_types}

CONTRACT TEXT:
{document}

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
  "entities": [
    {{"type": "ENTITY_TYPE", "value": "exact text from document", "confidence": 0.95}}
  ]
}}

Only include entities that are explicitly found in the document. Do not guess or infer."""


class FewShotPrompt(PromptStrategy):
    """Few-shot prompting with examples."""

    def __init__(self, config: Dict[str, Any], entity_types: List[str]):
        super().__init__(config, entity_types)
        strategies = config.get("strategies", {})
        few_shot = strategies.get("few_shot", {})
        self.system_prompt = few_shot.get("system_prompt", self._default_system_prompt())
        self.examples = few_shot.get("examples", self._default_examples())
        self.num_examples = few_shot.get("num_examples", 2)
        self.user_template = few_shot.get("user_template", self._default_user_template())

    def build_prompt(self, document: str, entity_types: List[str]) -> str:
        """Build few-shot prompt with examples."""
        # Build examples section
        examples_text = ""
        for i, example in enumerate(self.examples[:self.num_examples], 1):
            examples_text += f"""
EXAMPLE {i}:
Input: {example['input']}

Output: {example['output']}
"""

        entity_list = ", ".join(entity_types)

        prompt = f"""{self.system_prompt}

Here are some examples of entity extraction:
{examples_text}

Now extract entities from the following contract:

{self.user_template.format(
    entity_types=entity_list,
    document=document
)}"""
        return prompt

    def _default_system_prompt(self) -> str:
        return """You are an expert legal document analyst. Your task is to extract specific entities from contracts.
Learn from the examples provided and apply the same extraction patterns.
Always respond with valid JSON only."""

    def _default_examples(self) -> List[Dict[str, str]]:
        return [
            {
                "input": """Contract between ABC Corporation ("Buyer") and XYZ Inc. ("Seller")
dated January 15, 2024. This agreement shall be governed by the laws
of the State of Delaware. The term of this agreement is 24 months.""",
                "output": """{
  "entities": [
    {"type": "Party", "value": "ABC Corporation", "confidence": 0.95},
    {"type": "Party", "value": "XYZ Inc.", "confidence": 0.95},
    {"type": "Date", "value": "January 15, 2024", "confidence": 0.95},
    {"type": "Governing Law", "value": "laws of the State of Delaware", "confidence": 0.90},
    {"type": "Term", "value": "24 months", "confidence": 0.90}
  ]
}"""
            },
            {
                "input": """CONTRATO DE PRESTAÇÃO DE SERVIÇOS celebrado entre EMPRESA ALPHA LTDA,
inscrita no CNPJ sob nº 12.345.678/0001-90 (CONTRATANTE), e BETA TECNOLOGIA S.A.,
CNPJ 98.765.432/0001-10 (CONTRATADA), com vigência de 12 meses a partir de 01/03/2024.
Valor total: R$ 150.000,00. Foro: Comarca de São Paulo.""",
                "output": """{
  "entities": [
    {"type": "CONTRATANTE", "value": "EMPRESA ALPHA LTDA", "confidence": 0.95},
    {"type": "CNPJ_CONTRATANTE", "value": "12.345.678/0001-90", "confidence": 0.98},
    {"type": "CONTRATADA", "value": "BETA TECNOLOGIA S.A.", "confidence": 0.95},
    {"type": "CNPJ_CONTRATADA", "value": "98.765.432/0001-10", "confidence": 0.98},
    {"type": "PRAZO_VIGENCIA", "value": "12 meses", "confidence": 0.90},
    {"type": "DATA_CONTRATO", "value": "01/03/2024", "confidence": 0.95},
    {"type": "VALOR_CONTRATO", "value": "R$ 150.000,00", "confidence": 0.95},
    {"type": "FORO", "value": "Comarca de São Paulo", "confidence": 0.90}
  ]
}"""
            }
        ]

    def _default_user_template(self) -> str:
        return """Extract these entity types: {entity_types}

CONTRACT:
{document}

EXTRACTED ENTITIES (JSON only):"""


class ChainOfThoughtPrompt(PromptStrategy):
    """Chain-of-thought prompting with step-by-step reasoning."""

    def __init__(self, config: Dict[str, Any], entity_types: List[str]):
        super().__init__(config, entity_types)
        strategies = config.get("strategies", {})
        cot = strategies.get("chain_of_thought", {})
        self.system_prompt = cot.get("system_prompt", self._default_system_prompt())
        self.user_template = cot.get("user_template", self._default_user_template())

    def build_prompt(self, document: str, entity_types: List[str]) -> str:
        """Build chain-of-thought prompt."""
        entity_list = "\n".join(f"- {et}" for et in entity_types)

        prompt = f"""{self.system_prompt}

{self.user_template.format(
    entity_types=entity_list,
    document=document
)}"""
        return prompt

    def _default_system_prompt(self) -> str:
        return """You are an expert legal document analyst. Think step-by-step when extracting entities.

Your approach:
1. First, identify the document type and structure
2. Locate the sections most likely to contain each entity type
3. Extract the exact text for each entity
4. Verify each extraction is complete and accurate
5. Provide your final answer in JSON format"""

    def _default_user_template(self) -> str:
        return """Extract the following entities from this legal contract:
{entity_types}

CONTRACT TEXT:
{document}

Let's work through this step by step:

Step 1: Document Analysis
- Identify the type of contract and its main sections

Step 2: Entity Location
- For each entity type, identify where it might appear

Step 3: Extraction
- Extract the exact text for each entity found

Step 4: Verification
- Ensure extractions are complete and accurate

Step 5: Final Output
Provide ONLY a JSON object (no other text):
{{
  "reasoning": "Brief summary of analysis",
  "entities": [
    {{"type": "ENTITY_TYPE", "value": "extracted text", "confidence": 0.95, "location": "section/clause"}}
  ]
}}"""


class SelfConsistencyPrompt(PromptStrategy):
    """Self-consistency prompting with multiple samples."""

    def __init__(self, config: Dict[str, Any], entity_types: List[str]):
        super().__init__(config, entity_types)
        strategies = config.get("strategies", {})
        sc = strategies.get("self_consistency", {})
        self.num_samples = sc.get("num_samples", 3)
        self.temperature = sc.get("temperature", 0.7)
        self.aggregation = sc.get("aggregation", "majority_vote")

        # Use few-shot as base
        self.base_strategy = FewShotPrompt(config, entity_types)

    def build_prompt(self, document: str, entity_types: List[str]) -> str:
        """Build prompt for self-consistency (returns base prompt, sampling handled by extractor)."""
        return self.base_strategy.build_prompt(document, entity_types)

    def get_num_samples(self) -> int:
        """Get number of samples to generate."""
        return self.num_samples

    def get_temperature(self) -> float:
        """Get temperature for sampling."""
        return self.temperature

    def aggregate_results(self, results: List[List[dict]]) -> List[dict]:
        """Aggregate multiple extraction results."""
        if self.aggregation == "majority_vote":
            return self._majority_vote(results)
        elif self.aggregation == "union":
            return self._union(results)
        else:
            return self._majority_vote(results)

    def _majority_vote(self, results: List[List[dict]]) -> List[dict]:
        """Majority voting for entity aggregation."""
        from collections import Counter

        # Count occurrences of each (type, value) pair
        entity_counts: Counter = Counter()
        entity_examples: Dict[tuple, dict] = {}

        for result in results:
            for entity in result:
                key = (entity.get("type", ""), entity.get("value", "").lower().strip())
                entity_counts[key] += 1
                entity_examples[key] = entity

        # Keep entities that appear in majority of samples
        threshold = len(results) // 2 + 1
        aggregated = []

        for key, count in entity_counts.items():
            if count >= threshold:
                entity = entity_examples[key].copy()
                entity["confidence"] = count / len(results)
                entity["vote_count"] = count
                aggregated.append(entity)

        return aggregated

    def _union(self, results: List[List[dict]]) -> List[dict]:
        """Union of all entities with confidence based on frequency."""
        entity_counts: Counter = Counter()
        entity_examples: Dict[tuple, dict] = {}

        for result in results:
            for entity in result:
                key = (entity.get("type", ""), entity.get("value", "").lower().strip())
                entity_counts[key] += 1
                entity_examples[key] = entity

        aggregated = []
        for key, count in entity_counts.items():
            entity = entity_examples[key].copy()
            entity["confidence"] = count / len(results)
            aggregated.append(entity)

        return aggregated


def create_prompt_strategy(
    strategy_name: str,
    config: Dict[str, Any],
    entity_types: List[str]
) -> PromptStrategy:
    """Factory function to create a prompting strategy."""
    strategies = {
        "zero_shot": ZeroShotPrompt,
        "few_shot": FewShotPrompt,
        "chain_of_thought": ChainOfThoughtPrompt,
        "self_consistency": SelfConsistencyPrompt,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown prompting strategy: {strategy_name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy_name](config, entity_types)
