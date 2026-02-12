"""
Text Preprocessing - Utilities for preprocessing legal documents.

Handles:
- Text normalization
- Section detection
- Document segmentation
- Chunking for different strategies
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text."""
    text: str
    start_char: int
    end_char: int
    chunk_id: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class DocumentSection:
    """Represents a document section (clause, article, etc.)."""
    title: str
    text: str
    start_char: int
    end_char: int
    section_type: str
    level: int = 0
    children: List["DocumentSection"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TextPreprocessor:
    """
    Text preprocessing utilities for legal documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Default section patterns
        self.section_patterns = self.config.get("section_patterns", [
            # Portuguese patterns
            (r"^(CLÁUSULA|Cláusula)\s+(\w+)[:\s\-]*(.*?)$", "clause"),
            (r"^(ARTIGO|Artigo)\s+(\d+)[:\s\-]*(.*?)$", "article"),
            (r"^(\d+)\.\s+(.*?)$", "numbered"),
            (r"^([IVXLCDM]+)\s*[.\-]\s*(.*?)$", "roman"),
            # English patterns
            (r"^(ARTICLE|Article)\s+(\w+)[:\s\-]*(.*?)$", "article"),
            (r"^(SECTION|Section)\s+(\d+)[:\s\-]*(.*?)$", "section"),
            (r"^(\d+\.\d+)\s+(.*?)$", "subsection"),
            # All caps headers
            (r"^([A-Z][A-Z\s]{2,}):?\s*$", "header"),
        ])

    def normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving structure.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")

        # Normalize dashes
        text = text.replace("–", "-").replace("—", "-")

        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r"\n\s*Page\s+\d+\s+of\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", text)

        return text.strip()

    def detect_sections(self, text: str) -> List[DocumentSection]:
        """
        Detect document sections using pattern matching.

        Args:
            text: Document text

        Returns:
            List of DocumentSection objects
        """
        sections = []
        lines = text.split("\n")
        current_pos = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            for pattern, section_type in self.section_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    # Find section content (until next section)
                    section_start = current_pos
                    section_lines = [line]

                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Check if next section starts
                        is_new_section = any(
                            re.match(p, next_line, re.MULTILINE)
                            for p, _ in self.section_patterns
                        )
                        if is_new_section:
                            break
                        section_lines.append(lines[j])
                        j += 1

                    section_text = "\n".join(section_lines)
                    section_end = section_start + len(section_text)

                    sections.append(DocumentSection(
                        title=match.group(0),
                        text=section_text,
                        start_char=section_start,
                        end_char=section_end,
                        section_type=section_type
                    ))

                    i = j - 1
                    break

            current_pos += len(lines[i]) + 1  # +1 for newline
            i += 1

        # If no sections detected, treat whole document as one section
        if not sections:
            sections.append(DocumentSection(
                title="Document",
                text=text,
                start_char=0,
                end_char=len(text),
                section_type="document"
            ))

        return sections

    def chunk_text(
        self,
        text: str,
        strategy: str = "sliding_window",
        **kwargs
    ) -> List[TextChunk]:
        """
        Chunk text using specified strategy.

        Args:
            text: Input text
            strategy: Chunking strategy
            **kwargs: Strategy-specific parameters

        Returns:
            List of TextChunk objects
        """
        if strategy == "sliding_window":
            return self._chunk_sliding_window(text, **kwargs)
        elif strategy == "semantic":
            return self._chunk_semantic(text, **kwargs)
        elif strategy == "section":
            return self._chunk_by_section(text, **kwargs)
        elif strategy == "recursive":
            return self._chunk_recursive(text, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_sliding_window(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 128,
        respect_sentences: bool = True
    ) -> List[TextChunk]:
        """Sliding window chunking with optional sentence boundaries."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to end at sentence boundary
            if respect_sentences and end < len(text):
                # Look for sentence-ending punctuation
                for punct in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > chunk_size * 0.5:  # At least half the chunk
                        end = start + last_punct + len(punct)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_id=chunk_id,
                    metadata={"strategy": "sliding_window"}
                ))
                chunk_id += 1

            start = end - overlap if end < len(text) else len(text)

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        min_chunk_size: int = 256,
        max_chunk_size: int = 1024,
        similarity_threshold: float = 0.7
    ) -> List[TextChunk]:
        """
        Semantic chunking based on content coherence.
        Falls back to sentence-based if no embeddings available.
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        chunk_id = 0

        for sentence, sent_start, sent_end in sentences:
            sent_length = len(sentence)

            # If adding this sentence exceeds max, save current chunk
            if current_length + sent_length > max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=sent_start,
                    chunk_id=chunk_id,
                    metadata={"strategy": "semantic"}
                ))
                chunk_id += 1
                current_chunk = []
                current_length = 0
                chunk_start = sent_start

            current_chunk.append(sentence)
            current_length += sent_length + 1  # +1 for space

        # Don't forget last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=chunk_start,
                end_char=len(text),
                chunk_id=chunk_id,
                metadata={"strategy": "semantic"}
            ))

        return chunks

    def _chunk_by_section(
        self,
        text: str,
        min_section_size: int = 128,
        max_section_size: int = 2048,
        merge_small: bool = True
    ) -> List[TextChunk]:
        """Chunk by document sections."""
        sections = self.detect_sections(text)
        chunks = []
        chunk_id = 0

        current_chunk_text = ""
        current_start = 0

        for section in sections:
            section_text = section.text.strip()

            if len(section_text) > max_section_size:
                # Section too large, split it
                if current_chunk_text:
                    chunks.append(TextChunk(
                        text=current_chunk_text,
                        start_char=current_start,
                        end_char=section.start_char,
                        chunk_id=chunk_id,
                        metadata={"strategy": "section", "section_type": "merged"}
                    ))
                    chunk_id += 1
                    current_chunk_text = ""

                # Split large section with sliding window
                sub_chunks = self._chunk_sliding_window(
                    section_text,
                    chunk_size=max_section_size,
                    overlap=min_section_size // 2
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.start_char += section.start_char
                    sub_chunk.end_char += section.start_char
                    sub_chunk.chunk_id = chunk_id
                    sub_chunk.metadata["section_type"] = section.section_type
                    chunks.append(sub_chunk)
                    chunk_id += 1

                current_start = section.end_char

            elif len(section_text) < min_section_size and merge_small:
                # Section too small, merge with previous
                if current_chunk_text:
                    current_chunk_text += "\n\n" + section_text
                else:
                    current_chunk_text = section_text
                    current_start = section.start_char

                # Check if merged chunk is large enough
                if len(current_chunk_text) >= min_section_size:
                    chunks.append(TextChunk(
                        text=current_chunk_text,
                        start_char=current_start,
                        end_char=section.end_char,
                        chunk_id=chunk_id,
                        metadata={"strategy": "section", "section_type": "merged"}
                    ))
                    chunk_id += 1
                    current_chunk_text = ""

            else:
                # Section is good size
                if current_chunk_text:
                    chunks.append(TextChunk(
                        text=current_chunk_text,
                        start_char=current_start,
                        end_char=section.start_char,
                        chunk_id=chunk_id,
                        metadata={"strategy": "section", "section_type": "merged"}
                    ))
                    chunk_id += 1
                    current_chunk_text = ""

                chunks.append(TextChunk(
                    text=section_text,
                    start_char=section.start_char,
                    end_char=section.end_char,
                    chunk_id=chunk_id,
                    metadata={"strategy": "section", "section_type": section.section_type}
                ))
                chunk_id += 1
                current_start = section.end_char

        # Handle remaining text
        if current_chunk_text:
            chunks.append(TextChunk(
                text=current_chunk_text,
                start_char=current_start,
                end_char=len(text),
                chunk_id=chunk_id,
                metadata={"strategy": "section", "section_type": "merged"}
            ))

        return chunks

    def _chunk_recursive(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ) -> List[TextChunk]:
        """Recursive chunking (LangChain-style)."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        chunks = []
        self._recursive_split(
            text, chunk_size, chunk_overlap, separators, 0, 0, chunks
        )
        return chunks

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
        separator_idx: int,
        offset: int,
        chunks: List[TextChunk]
    ):
        """Recursive helper for chunking."""
        if len(text) <= chunk_size:
            chunks.append(TextChunk(
                text=text,
                start_char=offset,
                end_char=offset + len(text),
                chunk_id=len(chunks),
                metadata={"strategy": "recursive"}
            ))
            return

        if separator_idx >= len(separators):
            # No more separators, force split
            chunks.append(TextChunk(
                text=text[:chunk_size],
                start_char=offset,
                end_char=offset + chunk_size,
                chunk_id=len(chunks),
                metadata={"strategy": "recursive"}
            ))
            if len(text) > chunk_size:
                self._recursive_split(
                    text[chunk_size - chunk_overlap:],
                    chunk_size, chunk_overlap, separators,
                    separator_idx, offset + chunk_size - chunk_overlap, chunks
                )
            return

        separator = separators[separator_idx]
        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        current_chunk = ""
        current_offset = offset

        for i, part in enumerate(parts):
            if separator:
                part_with_sep = part + separator if i < len(parts) - 1 else part
            else:
                part_with_sep = part

            if len(current_chunk) + len(part_with_sep) > chunk_size:
                if current_chunk:
                    if len(current_chunk) > chunk_size:
                        # Chunk still too large, recurse with next separator
                        self._recursive_split(
                            current_chunk, chunk_size, chunk_overlap,
                            separators, separator_idx + 1, current_offset, chunks
                        )
                    else:
                        chunks.append(TextChunk(
                            text=current_chunk.strip(),
                            start_char=current_offset,
                            end_char=current_offset + len(current_chunk),
                            chunk_id=len(chunks),
                            metadata={"strategy": "recursive"}
                        ))

                current_offset = offset + text.find(part)
                current_chunk = part_with_sep
            else:
                current_chunk += part_with_sep

        if current_chunk:
            if len(current_chunk) > chunk_size:
                self._recursive_split(
                    current_chunk, chunk_size, chunk_overlap,
                    separators, separator_idx + 1, current_offset, chunks
                )
            else:
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    start_char=current_offset,
                    end_char=current_offset + len(current_chunk),
                    chunk_id=len(chunks),
                    metadata={"strategy": "recursive"}
                ))

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with positions."""
        # Simple sentence splitter
        sentence_endings = re.compile(r"([.!?])\s+")
        sentences = []
        last_end = 0

        for match in sentence_endings.finditer(text):
            sent_end = match.end()
            sentence = text[last_end:sent_end].strip()
            if sentence:
                sentences.append((sentence, last_end, sent_end))
            last_end = sent_end

        # Last sentence
        if last_end < len(text):
            sentence = text[last_end:].strip()
            if sentence:
                sentences.append((sentence, last_end, len(text)))

        return sentences
