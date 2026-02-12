"""
LLM Providers for entity extraction.

Supports:
- Azure OpenAI (GPT-4 Turbo) - PRIMARY
- OpenAI (GPT-4 Turbo, GPT-4o)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini 2.0 Flash, Gemini 1.5 Pro)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import os
import time
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.model_config = self._get_model_config(model_name)

    @abstractmethod
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for the specific model."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """
        Generate response from LLM.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        pass

    def get_context_window(self) -> int:
        """Get context window size for current model."""
        return self.model_config.get("context_window", 128000)

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost for token usage."""
        pass


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider for GPT-4 models."""

    # Cost per 1M tokens (Azure pricing)
    COSTS = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
    }

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)

        try:
            from openai import AzureOpenAI

            # Get Azure credentials from environment
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-06-01")

            if not api_key or not endpoint:
                raise ValueError(
                    "Azure OpenAI credentials not found. "
                    "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
                )

            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )

            # Azure uses deployment names, not model names
            self.deployment_name = os.getenv("AZURE_OPENAI_MODEL", model_name)

            logger.info(f"Initialized Azure OpenAI provider with deployment: {self.deployment_name}")

        except ImportError:
            raise ImportError("OpenAI not installed. Install with: pip install openai")

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get Azure OpenAI model configuration."""
        models = self.config.get("models", {})
        if model_name in models:
            return models[model_name]

        # Defaults for Azure
        defaults = {
            "gpt-4-turbo": {"context_window": 128000, "max_output_tokens": 4096},
            "gpt-4": {"context_window": 8192, "max_output_tokens": 4096},
            "gpt-4o": {"context_window": 128000, "max_output_tokens": 16384},
        }
        return defaults.get(model_name, {"context_window": 128000, "max_output_tokens": 4096})

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Generate response using Azure OpenAI API."""
        max_output = max_tokens or self.model_config.get("max_output_tokens", 4096)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,  # Azure uses deployment name
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                return content, input_tokens, output_tokens

            except Exception as e:
                logger.warning(f"Azure OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "", 0, 0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Azure OpenAI API usage."""
        costs = self.COSTS.get(self.model_name, {"input": 10.0, "output": 30.0})
        return (
            (input_tokens / 1_000_000) * costs["input"] +
            (output_tokens / 1_000_000) * costs["output"]
        )


class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT-4 models (direct API, not Azure)."""

    COSTS = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    }

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)

        try:
            from openai import OpenAI
            api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI provider with model: {model_name}")
        except ImportError:
            raise ImportError("OpenAI not installed. Install with: pip install openai")

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get OpenAI model configuration."""
        models = self.config.get("models", {})
        if model_name in models:
            return models[model_name]

        defaults = {
            "gpt-4-turbo": {"model_id": "gpt-4-turbo-preview", "context_window": 128000},
            "gpt-4o": {"model_id": "gpt-4o", "context_window": 128000},
            "gpt-4o-mini": {"model_id": "gpt-4o-mini", "context_window": 128000},
        }
        return defaults.get(model_name, {"model_id": model_name, "context_window": 128000})

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Generate response using OpenAI API."""
        model_id = self.model_config.get("model_id", self.model_name)
        max_output = max_tokens or self.model_config.get("max_output_tokens", 4096)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                return content, input_tokens, output_tokens

            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "", 0, 0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI API usage."""
        costs = self.COSTS.get(self.model_name, {"input": 5.0, "output": 15.0})
        return (
            (input_tokens / 1_000_000) * costs["input"] +
            (output_tokens / 1_000_000) * costs["output"]
        )


class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models."""

    COSTS = {
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)

        try:
            from anthropic import Anthropic
            api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
            self.client = Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic provider with model: {model_name}")
        except ImportError:
            raise ImportError("Anthropic not installed. Install with: pip install anthropic")

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get Anthropic model configuration."""
        models = self.config.get("models", {})
        if model_name in models:
            return models[model_name]

        defaults = {
            "claude-3-opus": {"model_id": "claude-3-opus-20240229", "context_window": 200000},
            "claude-3-sonnet": {"model_id": "claude-3-5-sonnet-20241022", "context_window": 200000},
            "claude-3-haiku": {"model_id": "claude-3-haiku-20240307", "context_window": 200000},
        }
        return defaults.get(model_name, {"model_id": model_name, "context_window": 200000})

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Generate response using Anthropic API."""
        model_id = self.model_config.get("model_id", self.model_name)
        max_output = max_tokens or self.model_config.get("max_output_tokens", 4096)

        json_instruction = """
IMPORTANT: You must respond with valid JSON only. No other text or explanation.
"""

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=model_id,
                    max_tokens=max_output,
                    messages=[{
                        "role": "user",
                        "content": json_instruction + prompt
                    }]
                )

                content = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                return content, input_tokens, output_tokens

            except Exception as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "", 0, 0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic API usage."""
        costs = self.COSTS.get(self.model_name, {"input": 3.0, "output": 15.0})
        return (
            (input_tokens / 1_000_000) * costs["input"] +
            (output_tokens / 1_000_000) * costs["output"]
        )


class GoogleProvider(LLMProvider):
    """Google provider for Gemini models."""

    COSTS = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
        "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
        "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
    }

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)

        try:
            import google.generativeai as genai

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            self.genai = genai

            # Use model from env or config
            model_id = os.getenv("GOOGLE_MODEL", self.model_config.get("model_id", model_name))
            self.model = genai.GenerativeModel(model_id)
            self.model_id = model_id

            logger.info(f"Initialized Google provider with model: {model_id}")

        except ImportError:
            raise ImportError(
                "Google AI not installed. Install with: pip install google-generativeai"
            )

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get Google model configuration."""
        models = self.config.get("models", {})
        if model_name in models:
            return models[model_name]

        defaults = {
            "gemini-2.0-flash": {"model_id": "gemini-2.0-flash", "context_window": 1000000},
            "gemini-2.0-flash-exp": {"model_id": "gemini-2.0-flash-exp", "context_window": 1000000},
            "gemini-1.5-pro": {"model_id": "gemini-1.5-pro", "context_window": 1000000},
            "gemini-1.5-flash": {"model_id": "gemini-1.5-flash", "context_window": 1000000},
        }
        return defaults.get(model_name, {"model_id": model_name, "context_window": 1000000})

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Generate response using Google AI API."""
        max_output = max_tokens or self.model_config.get("max_output_tokens", 8192)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output,
            "response_mime_type": "application/json"
        }

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                content = response.text

                # Get token counts from usage metadata
                input_tokens = len(prompt) // 4  # Fallback estimate
                output_tokens = len(content) // 4

                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', input_tokens)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', output_tokens)

                return content, input_tokens, output_tokens

            except Exception as e:
                logger.warning(f"Google API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "", 0, 0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Google API usage."""
        costs = self.COSTS.get(self.model_name, {"input": 0.10, "output": 0.40})
        return (
            (input_tokens / 1_000_000) * costs["input"] +
            (output_tokens / 1_000_000) * costs["output"]
        )


def create_provider(
    provider_name: str,
    config: Dict[str, Any],
    model_name: str
) -> LLMProvider:
    """Factory function to create an LLM provider."""
    providers = {
        "azure_openai": AzureOpenAIProvider,
        "azure": AzureOpenAIProvider,  # Alias
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,  # Alias
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {list(providers.keys())}"
        )

    return providers[provider_name](config, model_name)
