import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------- Chunking Configs --------------------------

class ChunkConfig(ABC):
    @abstractmethod
    def validate(self):
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        pass

@dataclass
class SectionRecursiveConfig(ChunkConfig):
    """Configuration for section-based chunking with recursive splitting."""
    recursive_chunk_size: int
    recursive_overlap: int
    
    def to_string(self) -> str:
        return f"chunk_mode=sections+recursive, chunk_size={self.recursive_chunk_size}, overlap={self.recursive_overlap}"

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"

@dataclass
class ParagraphConfig(ChunkConfig):
    # """Configuration for section-based chunking with recursive splitting."""
    min_chars: int

    def to_string(self) -> str:
        return f"chunk_mode=paragraph, chunk_size={self.min_chars}"

    def validate(self):
        assert self.min_chars >= 0, "min_chars must be >= 0"
        
@dataclass
class FixedSizeConfig(ChunkConfig):
    # """Configuration for section-based chunking with recursive splitting."""
    chunk_size: int
    chunk_overlap: int
    
    def to_string(self) -> str:
        return f"chunk_mode=fixed_size, chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"

    def validate(self):
        assert self.chunk_size > 0, "chunk_size must be > 0"
        assert self.chunk_overlap >= 0, "chunk_overlap must be >= 0"
        assert self.chunk_overlap < self.chunk_size, "chunk_overlap must be < chunk_size"

@dataclass
class ContextAwareConfig(ChunkConfig):
    # """Configuration for section-based chunking with recursive splitting."""
    window_size: int
    overlap_sentences: int
    
    def to_string(self) -> str:
        return f"chunk_mode=context_aware, window_size={self.window_size}, overlap_sentences={self.overlap_sentences}"

    def validate(self):
        assert self.window_size > 0, "chunk_size must be > 0"
        assert self.overlap_sentences >= 0, "overlap_sentences must be >= 0"
        assert self.overlap_sentences < self.window_size, "overlap_sentences must be < window_size"


@dataclass
class HybridParagraphFixedConfig(ChunkConfig):
    # """Configuration for section-based chunking with recursive splitting."""
    max_para_chars: int
    chunk_size: int
    chunk_overlap: int
    
    def to_string(self) -> str:
        return f"chunk_mode=hybrid_para_fixed, max_para_chars={self.max_para_chars}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"

    def validate(self):
        assert self.max_para_chars > 0, "max_para_chars must be > 0"
        assert self.chunk_size > 0, "chunk_size must be > 0"
        assert self.chunk_overlap >= 0, "chunk_overlap must be >= 0"
        assert self.chunk_overlap < self.chunk_size, "chunk_overlap must be < chunk_size"

@dataclass
class HybridParagraphContextAwareConfig(ChunkConfig):
    # """Configuration for section-based chunking with recursive splitting."""
    max_para_chars: int
    window_size: int
    overlap_sentences: int
    
    def to_string(self) -> str:
        return f"chunk_mode=hybrid_para_context_aware, max_para_chars={self.max_para_chars}, window_size={self.window_size}, overlap_sentences={self.overlap_sentences}"

    def validate(self):
        assert self.max_para_chars > 0, "max_para_chars must be > 0"
        assert self.window_size > 0, "window_size must be > 0"
        assert self.overlap_sentences >= 0, "overlap_sentences must be >= 0"
        assert self.overlap_sentences < self.window_size, "overlap_sentences must be < window_size"

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass
    
    @abstractmethod
    def artifact_folder_name(self) -> str:
        pass

class SectionRecursiveStrategy(ChunkStrategy):
    """
    Applies recursive character-based splitting to text.
    This is meant to be used on already-extracted sections.
    """

    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        self.recursive_chunk_size = config.recursive_chunk_size
        self.recursive_overlap = config.recursive_overlap

    def name(self) -> str:
        return f"sections+recursive({self.recursive_chunk_size},{self.recursive_overlap})"

    def artifact_folder_name(self) -> str:
        return "sections"

    def chunk(self, text: str) -> List[str]:
        """
        Recursively splits text into smaller chunks based on sentence boundaries.
        If a chunk exceeds recursive_chunk_size, it is further split.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.recursive_chunk_size,
            chunk_overlap=self.recursive_overlap,
            separators=[". "]
        )
        return splitter.split_text(text)
    
class ParagraphStrategy(ChunkStrategy):
    # """
    # Applies recursive character-based splitting to text.
    # This is meant to be used on already-extracted sections.
    # """

    def __init__(self, config: ParagraphConfig):
        self.config = config

    def name(self) -> str:
        return f"paragraph({self.config.min_chars})"

    def artifact_folder_name(self) -> str:
        return "paragraphs"

    def chunk(self, text: str) -> List[str]:
        # """
        # Recursively splits text into smaller chunks based on sentence boundaries.
        # If a chunk exceeds recursive_chunk_size, it is further split.
        # """
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        merged = []
        buffer = paragraphs[0]
        for para in paragraphs[1:]:
            if len(buffer) < self.config.min_chars:
                buffer = buffer + "\n\n" + para
            else:
                merged.append(buffer)
                buffer = para
        merged.append(buffer)
        return merged

class FixedSizeStrategy(ChunkStrategy):
    # """
    # Applies recursive character-based splitting to text.
    # This is meant to be used on already-extracted sections.
    # """

    def __init__(self, config: FixedSizeConfig):
        self.config = config

    def name(self) -> str:
        return f"fixed_size({self.config.chunk_size}, {self.config.chunk_overlap})"

    def artifact_folder_name(self) -> str:
        return "fixed"

    def chunk(self, text: str) -> List[str]:
        # """
        # Recursively splits text into smaller chunks based on sentence boundaries.
        # If a chunk exceeds recursive_chunk_size, it is further split.
        # """
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = size - overlap
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += step
        return chunks
    
class ContextAwareStrategy(ChunkStrategy):
    # """
    # Applies recursive character-based splitting to text.
    # This is meant to be used on already-extracted sections.
    # """

    def __init__(self, config: ContextAwareConfig):
        self.config = config

    def name(self) -> str:
        return f"context_aware({self.config.window_size}, {self.config.overlap_sentences})"

    def artifact_folder_name(self) -> str:
        return "context_aware"

    def chunk(self, text: str) -> List[str]:
        # """
        # Recursively splits text into smaller chunks based on sentence boundaries.
        # If a chunk exceeds recursive_chunk_size, it is further split.
        # """
        work_text = text.strip()
        for punct in [".", "?", "!"]:
            for space in [" ", "\n", "\t"]:
                work_text = work_text.replace(punct + space, punct + "<SPLIT>")

        sentences = [s.strip() for s in work_text.split("<SPLIT>") if s.strip()]
        if not sentences:
            return []
        
        window = self.config.window_size
        overlap = self.config.overlap_sentences
        step = window - overlap
        chunks = []
        start = 0
        while start < len(sentences):
            end = min(start + window, len(sentences))
            chunks.append(" ".join(sentences[start:end]))
            if end == len(sentences):
                break
            start += step
        return chunks

class HybridParagraphFixedStrategy(ChunkStrategy):
    # """
    # Applies recursive character-based splitting to text.
    # This is meant to be used on already-extracted sections.
    # """

    def __init__(self, config: HybridParagraphFixedConfig):
        self.config = config

        # TODO: Do I need
        self.para = ParagraphStrategy(ParagraphConfig(min_chars=0))
        self.fixed = FixedSizeStrategy(
            FixedSizeConfig(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        )


    def name(self) -> str:
        return f"hybrid_para_fixed({self.config.max_para_chars}, {self.config.max_para_chars}, {self.config.chunk_size})"

    def artifact_folder_name(self) -> str:
        return "hybrid_para_fixed"

    def chunk(self, text: str) -> List[str]:
        # """
        # Recursively splits text into smaller chunks based on sentence boundaries.
        # If a chunk exceeds recursive_chunk_size, it is further split.
        # """
        paragraphs = self.para.chunk(text)
        result = []
        for para in paragraphs:
            if len(para) > self.config.max_para_chars:
                result.extend(self.fixed.chunk(para))
            else:
                result.append(para)
        return result        
    
class HybridParagraphContextAwareStrategy(ChunkStrategy):
    # """
    # Applies recursive character-based splitting to text.
    # This is meant to be used on already-extracted sections.
    # """

    def __init__(self, config: HybridParagraphContextAwareConfig):
        self.config = config

        # TODO: Do I need
        self.para = ParagraphStrategy(ParagraphConfig(min_chars=0))
        self.context_aware = ContextAwareStrategy(
            ContextAwareConfig(window_size=config.window_size, overlap_sentences=config.overlap_sentences)
        )


    def name(self) -> str:
        return f"hybrid_para_context_aware({self.config.max_para_chars}, {self.config.window_size}, {self.config.overlap_sentences})"

    def artifact_folder_name(self) -> str:
        return "hybrid_para_context_aware"

    def chunk(self, text: str) -> List[str]:
        # """
        # Recursively splits text into smaller chunks based on sentence boundaries.
        # If a chunk exceeds recursive_chunk_size, it is further split.
        # """
        paragraphs = self.para.chunk(text)
        result = []
        for para in paragraphs:
            if len(para) > self.config.max_para_chars:
                result.extend(self.context_aware.chunk(para))
            else:
                result.append(para)
        return result  
# ----------------------------- Document Chunker ---------------------------------

class DocumentChunker:
    """
    Chunk text via a provided strategy.
    Table blocks (<table>...</table>) are preserved within chunks.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True
    ):
        self.strategy = strategy
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, t in enumerate(tables):
            text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, t in enumerate(tables):
            ph = f"[TABLE_PLACEHOLDER_{i}]"
            if ph in chunk:
                chunk = chunk.replace(ph, t)
        return chunk

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.strategy is None:
            raise ValueError("No chunk strategy provided")
        else:
            chunks = self.strategy.chunk(work)

        if self.keep_tables and tables:
            chunks = [self._restore_tables(c, tables) for c in chunks]
        return chunks
