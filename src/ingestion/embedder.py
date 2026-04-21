from typing import List

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


class Embedder:
    """
    Wrapper for Ollama embedding model using Haystack.

    Provides sync methods for generating embeddings for text content.
    Supports optional caching to avoid redundant embedding calls.
    """

    def __init__(self, model: str, base_url: str, cache=None):
        self.model = model
        self.base_url = base_url
        self._embedder = OllamaTextEmbedder(model=model, url=base_url)
        # self._embedder.warm_up()
        self._cache = cache  # Optional RAGCache instance

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Check cache
        if self._cache:
            cache_key = self._cache.make_key("embed", text)
            cached = self._cache.embeddings.get(cache_key)
            if cached is not None:
                return cached

        embedding = self._embedder.run(text=text)["embedding"]

        # Store in cache
        if self._cache:
            self._cache.embeddings.put(cache_key, embedding)

        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(t) for t in texts]
