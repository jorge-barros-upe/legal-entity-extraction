"""LLM (Long-Context Language Models) approach for entity extraction."""

from .extractor import LLMExtractor
from .providers import LLMProvider, OpenAIProvider, AnthropicProvider, GoogleProvider
from .prompting import PromptStrategy, ZeroShotPrompt, FewShotPrompt, ChainOfThoughtPrompt
from .optimized_extractor import OptimizedEntityExtractor, create_optimized_extractor
from .optimized_prompts import create_optimized_prompt, create_compact_prompt

__all__ = [
    "LLMExtractor",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "PromptStrategy",
    "ZeroShotPrompt",
    "FewShotPrompt",
    "ChainOfThoughtPrompt",
    "OptimizedEntityExtractor",
    "create_optimized_extractor",
    "create_optimized_prompt",
    "create_compact_prompt",
]
