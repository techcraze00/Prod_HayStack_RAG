import re
from typing import List
import numpy as np
from dataclasses import dataclass

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk"""

    text: str
    embedding: List[float]
    metadata: dict


class SemanticChunker:
    """
    Performs semantic chunking based on embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Generate embeddings for each sentence
    3. Group consecutive sentences with similarity > threshold
    4. Ensure chunks respect min/max size constraints
    """

    def __init__(
        self,
        embedding_model: str,
        base_url: str,
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
    ):
        self._embedder = OllamaTextEmbedder(
            model=embedding_model,
            url=base_url,
        )
        # self._embedder.warm_up()
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        result = self._embedder.run(text=text)
        return result["embedding"]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (one at a time via OllamaTextEmbedder)."""
        return [self._embed_text(t) for t in texts]

    def chunk(self, text: str, metadata: dict = None) -> List[SemanticChunk]:
        """Create semantic chunks from text."""
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        sentence_embeddings = self._embed_texts(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_size = len(sentences[0])

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(
                sentence_embeddings[i - 1], sentence_embeddings[i]
            )

            would_exceed_max = (
                current_chunk_size + len(sentences[i]) > self.max_chunk_size
            )
            is_semantically_different = similarity < self.similarity_threshold

            if would_exceed_max or is_semantically_different:
                if current_chunk_size >= self.min_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunk_embedding = self._embed_text(chunk_text)

                    chunks.append(
                        SemanticChunk(
                            text=chunk_text,
                            embedding=chunk_embedding,
                            metadata=metadata or {},
                        )
                    )

                current_chunk = [sentences[i]]
                current_chunk_size = len(sentences[i])
            else:
                current_chunk.append(sentences[i])
                current_chunk_size += len(sentences[i])

        # Add final chunk
        if current_chunk and current_chunk_size >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk_embedding = self._embed_text(chunk_text)
            chunks.append(
                SemanticChunk(
                    text=chunk_text,
                    embedding=chunk_embedding,
                    metadata=metadata or {},
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
