"""
Hierarchical (Parent-Child) Chunker

Creates a two-level chunk hierarchy:
- **Parent chunks** (large, ~1000 chars): provide broad context to the LLM
- **Child chunks** (small, ~300 chars): used for precise retrieval

When a child chunk is retrieved, the corresponding parent chunk is sent
to the LLM so it has sufficient context to generate accurate answers.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


@dataclass
class ParentChunk:
    """A large parent chunk that provides context"""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    children: List[str] = field(default_factory=list)


@dataclass
class ChildChunk:
    """A small child chunk used for retrieval"""
    id: str
    text: str
    embedding: List[float] = field(default_factory=list)
    parent_id: str = ""
    metadata: dict = field(default_factory=dict)


class HierarchicalChunker:
    """
    Creates parent-child chunk hierarchy.

    Algorithm:
    1. Split text into sentences
    2. Group sentences into large parent chunks
    3. Split each parent into small child chunks
    4. Embed child chunks for retrieval
    5. Store mapping from child → parent
    """

    def __init__(
        self,
        embedding_model: str,
        base_url: str,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedder = OllamaTextEmbedder(
            model=embedding_model,
            url=base_url,
        )
        # self._embedder.warm_up()

    def _embed_text(self, text: str) -> List[float]:
        return self._embedder.run(text=text)["embedding"]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_parent_chunks(
        self, text: str, source_id: str
    ) -> List[ParentChunk]:
        sentences = self._split_into_sentences(text)
        parents: List[ParentChunk] = []
        current_sentences: List[str] = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > self.parent_chunk_size and current_sentences:
                parent_text = " ".join(current_sentences)
                parent_id = f"{source_id}_parent_{len(parents)}"
                parents.append(ParentChunk(id=parent_id, text=parent_text))
                current_sentences = []
                current_size = 0

            current_sentences.append(sentence)
            current_size += len(sentence) + 1

        if current_sentences:
            parent_text = " ".join(current_sentences)
            parent_id = f"{source_id}_parent_{len(parents)}"
            parents.append(ParentChunk(id=parent_id, text=parent_text))

        return parents

    def _create_child_chunks(self, parent: ParentChunk) -> List[ChildChunk]:
        text = parent.text
        children: List[ChildChunk] = []
        start = 0
        child_idx = 0

        while start < len(text):
            end = min(start + self.child_chunk_size, len(text))

            if end < len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start + self.child_chunk_size // 2:
                    end = last_period + 1

            child_text = text[start:end].strip()
            if child_text:
                child_id = f"{parent.id}_child_{child_idx}"
                children.append(
                    ChildChunk(
                        id=child_id,
                        text=child_text,
                        parent_id=parent.id,
                    )
                )
                child_idx += 1

            start = end - self.chunk_overlap if end < len(text) else len(text)

        return children

    def chunk_hierarchical(
        self,
        text: str,
        metadata: dict = None,
        source_id: str = "doc",
    ) -> tuple[List[ParentChunk], List[ChildChunk]]:
        """Create parent-child chunk hierarchy."""
        metadata = metadata or {}

        parents = self._create_parent_chunks(text, source_id)

        all_children: List[ChildChunk] = []

        for parent in parents:
            parent.metadata = {**metadata}
            children = self._create_child_chunks(parent)

            child_texts = [c.text for c in children]
            if child_texts:
                embeddings = self._embed_texts(child_texts)
                for child, emb in zip(children, embeddings):
                    child.embedding = emb
                    child.metadata = {
                        **metadata,
                        "parent_id": parent.id,
                        "is_child_chunk": True,
                    }

            parent.children = [c.id for c in children]
            all_children.extend(children)

        return parents, all_children

    def format_for_storage(
        self,
        parents: List[ParentChunk],
        children: List[ChildChunk],
    ) -> List[Dict[str, Any]]:
        """Format child chunks for vector store ingestion."""
        parent_map = {p.id: p.text for p in parents}

        storage_chunks = []
        for child in children:
            storage_chunks.append(
                {
                    "text": child.text,
                    "embedding": child.embedding,
                    "metadata": {
                        **child.metadata,
                        "parent_text": parent_map.get(child.parent_id, ""),
                    },
                }
            )

        return storage_chunks
