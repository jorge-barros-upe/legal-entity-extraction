"""
Retrieval Strategies for RAG-based Entity Extraction.

Provides different retrieval methods:
- Dense: Semantic similarity using embeddings
- Sparse: BM25/TF-IDF based retrieval
- Hybrid: Combination of dense and sparse
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging

from .chunking import Chunk
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, config: Dict[str, Any], embedding_model: Optional[EmbeddingModel] = None):
        self.config = config
        self.embedding_model = embedding_model
        self.indexed_chunks: List[Chunk] = []

    @abstractmethod
    def index(self, chunks: List[Chunk]):
        """Index chunks for retrieval."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve relevant chunks for a query."""
        pass

    def clear_index(self):
        """Clear the index."""
        self.indexed_chunks = []


class DenseRetriever(Retriever):
    """Dense retrieval using semantic embeddings."""

    def __init__(self, config: Dict[str, Any], embedding_model: EmbeddingModel):
        super().__init__(config, embedding_model)
        strategy_config = config.get("strategies", {}).get("dense", {})
        self.top_k = strategy_config.get("top_k", 10)
        self.similarity_metric = strategy_config.get("similarity_metric", "cosine")
        self.min_similarity = strategy_config.get("min_similarity", 0.3)

        self.chunk_embeddings: Optional[np.ndarray] = None

    def index(self, chunks: List[Chunk]):
        """Index chunks by computing embeddings."""
        self.indexed_chunks = chunks
        texts = [c.text for c in chunks]
        self.chunk_embeddings = self.embedding_model.embed_documents(texts)
        logger.debug(f"Indexed {len(chunks)} chunks with dense embeddings")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Retrieve chunks using semantic similarity."""
        if not self.indexed_chunks or self.chunk_embeddings is None:
            logger.warning("No chunks indexed")
            return []

        top_k = top_k or self.top_k

        # Embed query
        query_embedding = self.embedding_model.embed_query(query)

        # Compute similarities
        if self.similarity_metric == "cosine":
            similarities = self._cosine_similarity(query_embedding, self.chunk_embeddings)
        elif self.similarity_metric == "dot":
            similarities = np.dot(self.chunk_embeddings, query_embedding)
        elif self.similarity_metric == "l2":
            distances = np.linalg.norm(self.chunk_embeddings - query_embedding, axis=1)
            similarities = 1 / (1 + distances)
        else:
            similarities = self._cosine_similarity(query_embedding, self.chunk_embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by minimum similarity
        retrieved = []
        for idx in top_indices:
            if similarities[idx] >= self.min_similarity:
                chunk = self.indexed_chunks[idx]
                chunk.metadata["similarity"] = float(similarities[idx])
                retrieved.append(chunk)

        return retrieved

    def _cosine_similarity(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        query_norm = np.linalg.norm(query)
        doc_norms = np.linalg.norm(documents, axis=1)

        # Avoid division by zero
        query_norm = max(query_norm, 1e-10)
        doc_norms = np.maximum(doc_norms, 1e-10)

        return np.dot(documents, query) / (doc_norms * query_norm)


class SparseRetriever(Retriever):
    """Sparse retrieval using BM25 or TF-IDF."""

    def __init__(self, config: Dict[str, Any], embedding_model: Optional[EmbeddingModel] = None):
        super().__init__(config, embedding_model)
        strategy_config = config.get("strategies", {}).get("sparse", {})
        self.algorithm = strategy_config.get("algorithm", "bm25")
        self.top_k = strategy_config.get("top_k", 10)
        self.k1 = strategy_config.get("k1", 1.5)
        self.b = strategy_config.get("b", 0.75)

        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def index(self, chunks: List[Chunk]):
        """Index chunks for sparse retrieval."""
        self.indexed_chunks = chunks
        texts = [c.text for c in chunks]

        if self.algorithm == "bm25":
            self._index_bm25(texts)
        else:
            self._index_tfidf(texts)

    def _index_bm25(self, texts: List[str]):
        """Index using BM25."""
        try:
            from rank_bm25 import BM25Okapi
            # Tokenize
            tokenized = [self._tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
            logger.debug(f"Indexed {len(texts)} chunks with BM25")
        except ImportError:
            logger.warning("rank_bm25 not installed. Falling back to TF-IDF.")
            self._index_tfidf(texts)

    def _index_tfidf(self, texts: List[str]):
        """Index using TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.debug(f"Indexed {len(texts)} chunks with TF-IDF")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Retrieve chunks using sparse retrieval."""
        if not self.indexed_chunks:
            return []

        top_k = top_k or self.top_k

        if self.bm25:
            return self._retrieve_bm25(query, top_k)
        elif self.tfidf_matrix is not None:
            return self._retrieve_tfidf(query, top_k)
        else:
            return []

    def _retrieve_bm25(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve using BM25."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.indexed_chunks[idx]
                chunk.metadata["bm25_score"] = float(scores[idx])
                retrieved.append(chunk)

        return retrieved

    def _retrieve_tfidf(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve using TF-IDF."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved = []
        for idx in top_indices:
            if similarities[idx] > 0:
                chunk = self.indexed_chunks[idx]
                chunk.metadata["tfidf_score"] = float(similarities[idx])
                retrieved.append(chunk)

        return retrieved

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]


class HybridRetriever(Retriever):
    """Hybrid retrieval combining dense and sparse."""

    def __init__(self, config: Dict[str, Any], embedding_model: EmbeddingModel):
        super().__init__(config, embedding_model)
        strategy_config = config.get("strategies", {}).get("hybrid", {})
        self.dense_weight = strategy_config.get("dense_weight", 0.7)
        self.sparse_weight = strategy_config.get("sparse_weight", 0.3)
        self.top_k = strategy_config.get("top_k", 15)
        self.use_reranking = strategy_config.get("reranking", True)

        # Initialize sub-retrievers
        self.dense_retriever = DenseRetriever(config, embedding_model)
        self.sparse_retriever = SparseRetriever(config)

        # Reranker
        self.reranker = None
        if self.use_reranking:
            self._init_reranker(config)

    def _init_reranker(self, config: Dict[str, Any]):
        """Initialize cross-encoder reranker."""
        rerank_config = config.get("reranking", {})
        rerank_model = rerank_config.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2")

        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(rerank_model)
            logger.info(f"Initialized reranker: {rerank_model}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
            self.use_reranking = False

    def index(self, chunks: List[Chunk]):
        """Index chunks in both retrievers."""
        self.indexed_chunks = chunks
        self.dense_retriever.index(chunks)
        self.sparse_retriever.index(chunks)
        logger.debug(f"Indexed {len(chunks)} chunks in hybrid retriever")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Retrieve using hybrid approach."""
        if not self.indexed_chunks:
            return []

        top_k = top_k or self.top_k

        # Get candidates from both retrievers
        dense_k = int(top_k * 1.5)  # Get more candidates for fusion
        sparse_k = int(top_k * 1.5)

        dense_results = self.dense_retriever.retrieve(query, dense_k)
        sparse_results = self.sparse_retriever.retrieve(query, sparse_k)

        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Rerank if enabled
        if self.use_reranking and self.reranker and len(fused_results) > 0:
            fused_results = self._rerank(query, fused_results, top_k)
        else:
            fused_results = fused_results[:top_k]

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Chunk],
        sparse_results: List[Chunk],
        k: int = 60
    ) -> List[Chunk]:
        """Combine results using Reciprocal Rank Fusion."""
        chunk_scores: Dict[int, float] = {}
        chunk_map: Dict[int, Chunk] = {}

        # Score dense results
        for rank, chunk in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + self.dense_weight / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        # Score sparse results
        for rank, chunk in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + self.sparse_weight / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        # Sort by combined score
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids:
            chunk = chunk_map[chunk_id]
            chunk.metadata["rrf_score"] = chunk_scores[chunk_id]
            results.append(chunk)

        return results

    def _rerank(self, query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        """Rerank chunks using cross-encoder."""
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.reranker.predict(pairs)

        # Combine with chunk
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Add rerank score to metadata
        results = []
        for chunk, score in scored_chunks[:top_k]:
            chunk.metadata["rerank_score"] = float(score)
            results.append(chunk)

        return results


def create_retriever(
    strategy: str,
    config: Dict[str, Any],
    embedding_model: Optional[EmbeddingModel] = None
) -> Retriever:
    """Factory function to create a retriever."""
    retrievers = {
        "dense": DenseRetriever,
        "sparse": SparseRetriever,
        "hybrid": HybridRetriever,
    }

    if strategy not in retrievers:
        raise ValueError(f"Unknown retrieval strategy: {strategy}. Available: {list(retrievers.keys())}")

    if strategy in ("dense", "hybrid") and embedding_model is None:
        raise ValueError(f"{strategy} retriever requires an embedding model")

    return retrievers[strategy](config, embedding_model)
