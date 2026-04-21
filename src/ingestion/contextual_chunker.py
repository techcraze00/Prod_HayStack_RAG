"""
Contextual Retrieval Chunker

Enhances standard semantic chunks by prepending document-level context.
Uses LLM to generate:
1. A document summary
2. Per-chunk contextual descriptions

This helps embeddings capture broader document meaning, improving
retrieval accuracy — especially for ambiguous or short chunks.
"""

from typing import List
from dataclasses import dataclass, field

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

from .semantic_chunker import SemanticChunker, SemanticChunk
from ..utils.llm import chat_sync


@dataclass
class ContextualChunk:
    """A chunk enriched with document-level context"""
    text: str
    original_text: str
    context: str
    embedding: List[float]
    metadata: dict = field(default_factory=dict)


class ContextualChunker:
    """
    Extends SemanticChunker by prepending LLM-generated context to each chunk.

    Pipeline:
    1. Chunk text semantically (via SemanticChunker)
    2. Generate a full-document summary
    3. For each chunk, ask the LLM for a brief context sentence
    4. Prepend context to chunk text
    5. Re-embed the enriched chunk
    """

    CONTEXT_PROMPT = """You are an AI assistant helping process documents for retrieval.
Here is a summary of the full document:
<summary>
{doc_summary}
</summary>

Here is a specific chunk from that document:
<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context (1-2 sentences) to situate this chunk within the overall document.
The context should help a search engine understand what this chunk is about.
Only output the context sentence(s), nothing else."""

    SUMMARY_PROMPT = """Summarize the following document in 3-5 sentences. Focus on the key topics, arguments, and conclusions.

Document:
{text}

Summary:"""

    def __init__(
        self,
        generator,
        embedding_model: str,
        base_url: str,
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        context_batch_size: int = 5,
    ):
        """
        Args:
            generator: A Haystack ChatGenerator for context generation.
            embedding_model: Ollama embedding model name.
            base_url: Ollama base URL.
            similarity_threshold: Semantic similarity threshold for chunking.
            max_chunk_size: Maximum characters per chunk.
            min_chunk_size: Minimum characters per chunk.
            context_batch_size: Number of chunks to process in each batch.
        """
        self.generator = generator
        self.context_batch_size = context_batch_size

        self.semantic_chunker = SemanticChunker(
            embedding_model=embedding_model,
            base_url=base_url,
            similarity_threshold=similarity_threshold,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )

        self._embedder = OllamaTextEmbedder(
            model=embedding_model,
            url=base_url,
        )
        # self._embedder.warm_up()

    def _generate_document_summary(self, text: str) -> str:
        """Generate a concise summary of the full document via LLM."""
        try:
            return chat_sync(
                self.generator,
                system="",
                user=self.SUMMARY_PROMPT.format(text=text[:8000]),
            )
        except Exception:
            return text[:500]

    def _generate_chunk_context(self, chunk_text: str, doc_summary: str) -> str:
        """Generate a context sentence for a single chunk."""
        prompt = self.CONTEXT_PROMPT.format(
            doc_summary=doc_summary,
            chunk_text=chunk_text,
        )
        try:
            return chat_sync(self.generator, system="", user=prompt)
        except Exception:
            return ""

    def _add_context_to_chunk(self, chunk_text: str, context: str) -> str:
        """Prepend context to chunk text."""
        if not context:
            return chunk_text
        return f"[Context: {context}]\n\n{chunk_text}"

    def chunk_with_context(
        self,
        text: str,
        metadata: dict = None,
    ) -> List[ContextualChunk]:
        """
        Create contextualised semantic chunks.

        1. Split into semantic chunks
        2. Generate document summary
        3. Add per-chunk context
        4. Re-embed enriched chunks
        """
        metadata = metadata or {}

        raw_chunks: List[SemanticChunk] = self.semantic_chunker.chunk(text, metadata)

        if not raw_chunks:
            return []

        doc_summary = self._generate_document_summary(text)

        contextual_chunks: List[ContextualChunk] = []

        for i in range(0, len(raw_chunks), self.context_batch_size):
            batch = raw_chunks[i : i + self.context_batch_size]

            for chunk in batch:
                context = self._generate_chunk_context(chunk.text, doc_summary)
                enriched_text = self._add_context_to_chunk(chunk.text, context)

                enriched_embedding = self._embedder.run(text=enriched_text)["embedding"]

                contextual_chunks.append(
                    ContextualChunk(
                        text=enriched_text,
                        original_text=chunk.text,
                        context=context,
                        embedding=enriched_embedding,
                        metadata={
                            **metadata,
                            "has_context": True,
                            "doc_summary": doc_summary[:200],
                        },
                    )
                )

        return contextual_chunks

    def chunk_with_optional_context(
        self,
        text: str,
        metadata: dict = None,
        enable_context: bool = True,
    ) -> List[ContextualChunk]:
        """Chunk with optional context enrichment."""
        if enable_context:
            return self.chunk_with_context(text, metadata)

        raw_chunks = self.semantic_chunker.chunk(text, metadata)
        return [
            ContextualChunk(
                text=c.text,
                original_text=c.text,
                context="",
                embedding=c.embedding,
                metadata=c.metadata,
            )
            for c in raw_chunks
        ]
