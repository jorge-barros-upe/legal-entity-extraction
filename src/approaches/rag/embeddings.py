"""
Embedding Models for RAG-based Entity Extraction.

Provides different embedding models:
- OpenAI: text-embedding-3-large/small
- HuggingFace: sentence-transformers, legal-bert, etc.
- BGE: High-performance open embeddings
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimensions: int = 0

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text(s) into vectors.

        Args:
            texts: Single text or list of texts

        Returns:
            numpy array of shape (n_texts, dimensions)
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (may use different instruction)."""
        return self.embed([query])[0]

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed documents."""
        return self.embed(documents)


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI embedding models (text-embedding-3-large, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        models_config = config.get("models", {}).get("openai", {})
        self.model_name = models_config.get("model_name", "text-embedding-3-large")
        self.dimensions = models_config.get("dimensions", 3072)
        self.batch_size = models_config.get("batch_size", 100)
        self.max_retries = models_config.get("max_retries", 3)

        # Initialize client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info(f"Initialized OpenAI embeddings: {self.model_name}")
        except ImportError:
            raise ImportError("OpenAI not installed. Install with: pip install openai")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Clean texts (OpenAI has issues with empty strings)
            batch = [t.replace("\n", " ").strip() or " " for t in batch]

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )

                    batch_embeddings = [e.embedding for e in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to embed batch after {self.max_retries} attempts: {e}")
                        # Return zeros for failed batch
                        all_embeddings.extend([[0.0] * self.dimensions] * len(batch))
                    else:
                        import time
                        time.sleep(2 ** attempt)

        return np.array(all_embeddings)


class HuggingFaceEmbeddings(EmbeddingModel):
    """HuggingFace/sentence-transformers embedding models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get model config (check multiple possible locations)
        models_config = config.get("models", {})
        model_key = config.get("model", "multilingual")

        if model_key in models_config:
            model_config = models_config[model_key]
        else:
            model_config = models_config.get("multilingual", {})

        self.model_name = model_config.get(
            "model_name",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.dimensions = model_config.get("dimensions", 768)
        self.max_length = model_config.get("max_length", 512)
        self.pooling = model_config.get("pooling", "mean")
        self.instruction = model_config.get("instruction", "")

        # Load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Update dimensions from model
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized HuggingFace embeddings: {self.model_name} (dim={self.dimensions})")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        if isinstance(texts, str):
            texts = [texts]

        # Add instruction prefix if specified (for models like E5)
        if self.instruction:
            texts = [self.instruction + t for t in texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings


class LegalBERTEmbeddings(EmbeddingModel):
    """Legal-domain BERT embeddings."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        models_config = config.get("models", {}).get("legal_bert", {})
        self.model_name = models_config.get("model_name", "nlpaueb/legal-bert-base-uncased")
        self.dimensions = models_config.get("dimensions", 768)
        self.max_length = models_config.get("max_length", 512)
        self.pooling = models_config.get("pooling", "mean")

        # Load model and tokenizer
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Initialized Legal-BERT embeddings: {self.model_name} on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers/torch not installed. "
                "Install with: pip install transformers torch"
            )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using Legal-BERT."""
        import torch

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)

                outputs = self.model(**inputs)

                # Pool embeddings
                if self.pooling == "cls":
                    embedding = outputs.last_hidden_state[:, 0, :]
                elif self.pooling == "mean":
                    attention_mask = inputs["attention_mask"]
                    hidden_state = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
                    sum_mask = mask_expanded.sum(1)
                    embedding = sum_embeddings / sum_mask
                else:
                    embedding = outputs.last_hidden_state[:, 0, :]

                embeddings.append(embedding.cpu().numpy()[0])

        return np.array(embeddings)


class BGEEmbeddings(EmbeddingModel):
    """BGE (BAAI General Embedding) models - high performance open embeddings."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        models_config = config.get("models", {}).get("bge", {})
        self.model_name = models_config.get("model_name", "BAAI/bge-large-en-v1.5")
        self.dimensions = models_config.get("dimensions", 1024)
        self.max_length = models_config.get("max_length", 512)
        self.query_instruction = models_config.get(
            "instruction",
            "Represent this legal document clause for retrieval:"
        )

        # Load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized BGE embeddings: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using BGE model."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with instruction prefix."""
        query_with_instruction = self.query_instruction + " " + query
        return self.embed([query_with_instruction])[0]


class E5Embeddings(EmbeddingModel):
    """E5 (EmbEddings from bidirEctional Encoder rEpresentations) models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        models_config = config.get("models", {}).get("e5", {})
        self.model_name = models_config.get("model_name", "intfloat/multilingual-e5-large")
        self.dimensions = models_config.get("dimensions", 1024)
        self.max_length = models_config.get("max_length", 512)

        # Load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized E5 embeddings: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using E5 model."""
        if isinstance(texts, str):
            texts = [texts]

        # E5 uses "passage: " prefix for documents
        texts = ["passage: " + t for t in texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with query prefix."""
        query_with_prefix = "query: " + query
        return self.model.encode(
            [query_with_prefix],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]


def create_embedding_model(model_type: str, config: Dict[str, Any]) -> EmbeddingModel:
    """Factory function to create an embedding model."""
    models = {
        "openai": OpenAIEmbeddings,
        "huggingface": HuggingFaceEmbeddings,
        "multilingual": HuggingFaceEmbeddings,
        "legal_bert": LegalBERTEmbeddings,
        "bge": BGEEmbeddings,
        "e5": E5Embeddings,
    }

    if model_type not in models:
        raise ValueError(f"Unknown embedding model: {model_type}. Available: {list(models.keys())}")

    return models[model_type](config)
