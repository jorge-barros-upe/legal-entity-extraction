"""
Chunking Strategies for RAG-based Entity Extraction.

Provides different strategies for splitting documents:
- Sliding Window: Fixed-size chunks with overlap
- Semantic: Based on content coherence
- Section: Based on document structure
- Recursive: Hierarchical splitting
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def length(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, len={self.length}, start={self.start_char})"


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


class SlidingWindowChunker(ChunkingStrategy):
    """
    Sliding window chunking with overlap.

    Simple but effective approach that ensures no information is lost
    at chunk boundaries.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        strategy_config = config.get("strategies", {}).get("sliding_window", {})
        self.chunk_size = strategy_config.get("chunk_size", 512)
        self.overlap = strategy_config.get("overlap", 128)
        self.respect_sentences = strategy_config.get("respect_sentences", True)

    def chunk(self, text: str) -> List[Chunk]:
        """Create sliding window chunks."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Adjust end to sentence boundary if requested
            if self.respect_sentences and end < len(text):
                # Find last sentence ending before chunk_size
                chunk_text = text[start:end]
                for punct in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    last_idx = chunk_text.rfind(punct)
                    if last_idx > self.chunk_size * 0.5:
                        end = start + last_idx + len(punct)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "strategy": "sliding_window",
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap
                    }
                ))
                chunk_id += 1

            # Move to next position with overlap
            start = end - self.overlap if end < len(text) else len(text)

        logger.debug(f"Created {len(chunks)} sliding window chunks")
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking based on content coherence.

    Groups sentences that are semantically related together.
    Falls back to sentence-based splitting if embeddings unavailable.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        strategy_config = config.get("strategies", {}).get("semantic", {})
        self.min_chunk_size = strategy_config.get("min_chunk_size", 256)
        self.max_chunk_size = strategy_config.get("max_chunk_size", 1024)
        self.similarity_threshold = strategy_config.get("similarity_threshold", 0.7)

        # Try to load embedding model for semantic similarity
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            model_name = strategy_config.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model for semantic chunking: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic chunking will fall back to sentence-based. "
                "Install with: pip install sentence-transformers"
            )

    def chunk(self, text: str) -> List[Chunk]:
        """Create semantic chunks."""
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return [Chunk(text=text, chunk_id=0, start_char=0, end_char=len(text))]

        if self.embedding_model:
            return self._semantic_chunk(sentences)
        else:
            return self._sentence_based_chunk(sentences)

    def _semantic_chunk(self, sentences: List[tuple]) -> List[Chunk]:
        """Chunk based on semantic similarity."""
        import numpy as np

        # Get embeddings for all sentences
        sentence_texts = [s[0] for s in sentences]
        embeddings = self.embedding_model.encode(sentence_texts, show_progress_bar=False)

        chunks = []
        current_chunk_sentences = []
        current_chunk_start = sentences[0][1]
        chunk_id = 0

        for i, (sentence, start, end) in enumerate(sentences):
            current_length = sum(len(s[0]) for s in current_chunk_sentences)

            # Always add first sentence
            if not current_chunk_sentences:
                current_chunk_sentences.append((sentence, start, end))
                continue

            # Check if chunk is getting too large
            if current_length + len(sentence) > self.max_chunk_size:
                # Save current chunk
                chunk_text = " ".join(s[0] for s in current_chunk_sentences)
                chunk_end = current_chunk_sentences[-1][2]

                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=current_chunk_start,
                    end_char=chunk_end,
                    metadata={"strategy": "semantic"}
                ))
                chunk_id += 1

                current_chunk_sentences = [(sentence, start, end)]
                current_chunk_start = start
                continue

            # Check semantic similarity with previous sentence
            if i > 0:
                similarity = np.dot(embeddings[i], embeddings[i-1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
                )

                if similarity < self.similarity_threshold and current_length >= self.min_chunk_size:
                    # Low similarity and chunk is large enough - start new chunk
                    chunk_text = " ".join(s[0] for s in current_chunk_sentences)
                    chunk_end = current_chunk_sentences[-1][2]

                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        start_char=current_chunk_start,
                        end_char=chunk_end,
                        metadata={"strategy": "semantic", "split_similarity": float(similarity)}
                    ))
                    chunk_id += 1

                    current_chunk_sentences = [(sentence, start, end)]
                    current_chunk_start = start
                    continue

            current_chunk_sentences.append((sentence, start, end))

        # Handle last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(s[0] for s in current_chunk_sentences)
            chunk_end = current_chunk_sentences[-1][2]

            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_char=current_chunk_start,
                end_char=chunk_end,
                metadata={"strategy": "semantic"}
            ))

        logger.debug(f"Created {len(chunks)} semantic chunks")
        return chunks

    def _sentence_based_chunk(self, sentences: List[tuple]) -> List[Chunk]:
        """Fallback: chunk by sentence count."""
        chunks = []
        current_chunk_sentences = []
        current_chunk_start = 0
        current_length = 0
        chunk_id = 0

        for sentence, start, end in sentences:
            if not current_chunk_sentences:
                current_chunk_start = start

            if current_length + len(sentence) > self.max_chunk_size and current_chunk_sentences:
                chunk_text = " ".join(s[0] for s in current_chunk_sentences)
                chunk_end = current_chunk_sentences[-1][2]

                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=current_chunk_start,
                    end_char=chunk_end,
                    metadata={"strategy": "sentence_based"}
                ))
                chunk_id += 1

                current_chunk_sentences = []
                current_length = 0
                current_chunk_start = start

            current_chunk_sentences.append((sentence, start, end))
            current_length += len(sentence) + 1

        if current_chunk_sentences:
            chunk_text = " ".join(s[0] for s in current_chunk_sentences)
            chunk_end = current_chunk_sentences[-1][2]

            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_char=current_chunk_start,
                end_char=chunk_end,
                metadata={"strategy": "sentence_based"}
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[tuple]:
        """Split text into sentences with positions."""
        sentence_endings = re.compile(r'([.!?])\s+')
        sentences = []
        last_end = 0

        for match in sentence_endings.finditer(text):
            sent_end = match.end()
            sentence = text[last_end:sent_end].strip()
            if sentence:
                sentences.append((sentence, last_end, sent_end))
            last_end = sent_end

        if last_end < len(text):
            sentence = text[last_end:].strip()
            if sentence:
                sentences.append((sentence, last_end, len(text)))

        return sentences


class SectionChunker(ChunkingStrategy):
    """
    Section-based chunking for legal documents.

    Detects document sections (clauses, articles, etc.) and uses
    them as natural chunk boundaries.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        strategy_config = config.get("strategies", {}).get("section", {})
        self.min_section_size = strategy_config.get("min_section_size", 128)
        self.max_section_size = strategy_config.get("max_section_size", 2048)
        self.merge_small_sections = strategy_config.get("merge_small_sections", True)

        # Section detection patterns
        self.section_patterns = strategy_config.get("section_patterns", [
            r"^(CLÁUSULA|Cláusula)\s+(\w+)",
            r"^(ARTIGO|Artigo)\s+(\d+)",
            r"^(\d+)\.\s+",
            r"^([IVXLCDM]+)\s*[.\-]",
            r"^(ARTICLE|Article)\s+(\w+)",
            r"^(SECTION|Section)\s+(\d+)",
        ])

    def chunk(self, text: str) -> List[Chunk]:
        """Create section-based chunks."""
        sections = self._detect_sections(text)

        if not sections:
            # No sections found, fall back to sliding window
            logger.debug("No sections detected, falling back to sliding window")
            fallback = SlidingWindowChunker({
                "strategies": {
                    "sliding_window": {
                        "chunk_size": self.max_section_size,
                        "overlap": self.min_section_size // 2
                    }
                }
            })
            return fallback.chunk(text)

        chunks = []
        chunk_id = 0

        # Process sections
        current_merged = ""
        merged_start = None

        for section in sections:
            section_text = section["text"].strip()
            section_start = section["start"]
            section_end = section["end"]

            if len(section_text) > self.max_section_size:
                # Section too large - split it
                if current_merged:
                    chunks.append(Chunk(
                        text=current_merged,
                        chunk_id=chunk_id,
                        start_char=merged_start,
                        end_char=section_start,
                        metadata={"strategy": "section", "type": "merged"}
                    ))
                    chunk_id += 1
                    current_merged = ""
                    merged_start = None

                # Split large section
                sub_chunker = SlidingWindowChunker({
                    "strategies": {
                        "sliding_window": {
                            "chunk_size": self.max_section_size,
                            "overlap": self.min_section_size // 2
                        }
                    }
                })
                sub_chunks = sub_chunker.chunk(section_text)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = chunk_id
                    sub_chunk.start_char += section_start
                    sub_chunk.end_char += section_start
                    sub_chunk.metadata["type"] = section.get("type", "large_section")
                    chunks.append(sub_chunk)
                    chunk_id += 1

            elif len(section_text) < self.min_section_size and self.merge_small_sections:
                # Section too small - merge
                if merged_start is None:
                    merged_start = section_start

                if current_merged:
                    current_merged += "\n\n" + section_text
                else:
                    current_merged = section_text

                # Check if merged is large enough
                if len(current_merged) >= self.min_section_size:
                    chunks.append(Chunk(
                        text=current_merged,
                        chunk_id=chunk_id,
                        start_char=merged_start,
                        end_char=section_end,
                        metadata={"strategy": "section", "type": "merged"}
                    ))
                    chunk_id += 1
                    current_merged = ""
                    merged_start = None

            else:
                # Section is good size
                if current_merged:
                    chunks.append(Chunk(
                        text=current_merged,
                        chunk_id=chunk_id,
                        start_char=merged_start,
                        end_char=section_start,
                        metadata={"strategy": "section", "type": "merged"}
                    ))
                    chunk_id += 1
                    current_merged = ""
                    merged_start = None

                chunks.append(Chunk(
                    text=section_text,
                    chunk_id=chunk_id,
                    start_char=section_start,
                    end_char=section_end,
                    metadata={"strategy": "section", "type": section.get("type", "section")}
                ))
                chunk_id += 1

        # Handle remaining merged text
        if current_merged:
            chunks.append(Chunk(
                text=current_merged,
                chunk_id=chunk_id,
                start_char=merged_start,
                end_char=len(text),
                metadata={"strategy": "section", "type": "merged"}
            ))

        logger.debug(f"Created {len(chunks)} section-based chunks")
        return chunks

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections in document."""
        sections = []
        lines = text.split("\n")
        current_pos = 0
        current_section = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check if line matches any section pattern
            for pattern in self.section_patterns:
                match = re.match(pattern, line_stripped, re.MULTILINE)
                if match:
                    # Save previous section
                    if current_section is not None:
                        current_section["end"] = current_pos
                        sections.append(current_section)

                    # Start new section
                    current_section = {
                        "title": match.group(0),
                        "text": line,
                        "start": current_pos,
                        "type": self._get_section_type(pattern)
                    }
                    break
            else:
                # No match - add to current section
                if current_section is not None:
                    current_section["text"] += "\n" + line

            current_pos += len(line) + 1  # +1 for newline

        # Don't forget last section
        if current_section is not None:
            current_section["end"] = len(text)
            sections.append(current_section)

        return sections

    def _get_section_type(self, pattern: str) -> str:
        """Determine section type from pattern."""
        pattern_lower = pattern.lower()
        if "cláusula" in pattern_lower or "clausula" in pattern_lower:
            return "clause"
        elif "artigo" in pattern_lower or "article" in pattern_lower:
            return "article"
        elif "section" in pattern_lower:
            return "section"
        elif r"\d+\." in pattern:
            return "numbered"
        elif "[ivxlcdm]" in pattern_lower:
            return "roman"
        return "other"


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive text splitting (LangChain-style).

    Tries to split by larger separators first, falling back to
    smaller ones if chunks are still too large.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        strategy_config = config.get("strategies", {}).get("recursive", {})
        self.chunk_size = strategy_config.get("chunk_size", 1000)
        self.chunk_overlap = strategy_config.get("chunk_overlap", 200)
        self.separators = strategy_config.get("separators", [
            "\n\n", "\n", ". ", " ", ""
        ])

    def chunk(self, text: str) -> List[Chunk]:
        """Create recursive chunks."""
        chunks = []
        self._recursive_split(text, 0, 0, chunks)
        logger.debug(f"Created {len(chunks)} recursive chunks")
        return chunks

    def _recursive_split(
        self,
        text: str,
        separator_idx: int,
        offset: int,
        chunks: List[Chunk]
    ):
        """Recursively split text."""
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(Chunk(
                    text=text.strip(),
                    chunk_id=len(chunks),
                    start_char=offset,
                    end_char=offset + len(text),
                    metadata={"strategy": "recursive"}
                ))
            return

        if separator_idx >= len(self.separators):
            # No more separators - force split
            chunks.append(Chunk(
                text=text[:self.chunk_size].strip(),
                chunk_id=len(chunks),
                start_char=offset,
                end_char=offset + self.chunk_size,
                metadata={"strategy": "recursive", "forced": True}
            ))
            remaining = text[self.chunk_size - self.chunk_overlap:]
            if remaining.strip():
                self._recursive_split(
                    remaining, separator_idx,
                    offset + self.chunk_size - self.chunk_overlap, chunks
                )
            return

        separator = self.separators[separator_idx]
        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        current_chunk = ""
        current_offset = offset

        for i, part in enumerate(parts):
            part_with_sep = part + separator if i < len(parts) - 1 and separator else part

            if len(current_chunk) + len(part_with_sep) > self.chunk_size:
                if current_chunk.strip():
                    if len(current_chunk) > self.chunk_size:
                        self._recursive_split(
                            current_chunk, separator_idx + 1,
                            current_offset, chunks
                        )
                    else:
                        chunks.append(Chunk(
                            text=current_chunk.strip(),
                            chunk_id=len(chunks),
                            start_char=current_offset,
                            end_char=current_offset + len(current_chunk),
                            metadata={"strategy": "recursive"}
                        ))

                current_offset = offset + text.find(part, current_offset - offset)
                current_chunk = part_with_sep
            else:
                current_chunk += part_with_sep

        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size:
                self._recursive_split(
                    current_chunk, separator_idx + 1,
                    current_offset, chunks
                )
            else:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    chunk_id=len(chunks),
                    start_char=current_offset,
                    end_char=current_offset + len(current_chunk),
                    metadata={"strategy": "recursive"}
                ))


def create_chunker(strategy: str, config: Dict[str, Any]) -> ChunkingStrategy:
    """Factory function to create a chunker."""
    chunkers = {
        "sliding_window": SlidingWindowChunker,
        "semantic": SemanticChunker,
        "section": SectionChunker,
        "recursive": RecursiveChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Available: {list(chunkers.keys())}")

    return chunkers[strategy](config)
