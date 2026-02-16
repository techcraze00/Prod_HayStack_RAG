from typing import List, Dict, Any, Optional

from ...storage.base import VectorStoreInterface
from ...config import settings


class VectorSearchTool:
    """
    Standalone vector search tool with support for:
    - Hybrid search (Vector + BM25)
    - Metadata filtering
    - Caching
    - Parent-child context retrieval
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        embedder,
        cache=None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.cache = cache  # Optional RAGCache instance

    def search(
        self,
        query: str,
        top_k: int = None,
        use_hybrid: bool = None,
        filters: Optional[Dict[str, Any]] = None,
        return_parent: bool = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic vector search with optional enhancements.

        Args:
            query: Search query.
            top_k: Number of results.
            use_hybrid: Use hybrid search (None = use config).
            filters: Metadata filters.
            return_parent: Replace child text with parent text (None = use config).

        Returns:
            List of search results with text and metadata.
        """
        top_k = top_k or settings.vector_top_k
        if use_hybrid is None:
            use_hybrid = settings.hybrid_search_enabled
        if return_parent is None:
            return_parent = (
                settings.use_hierarchical_chunking
                and settings.return_parent_context
            )

        # ── Check cache ──────────────────────────────────────────────
        cache_key = None
        if self.cache and settings.cache_enabled:
            cache_key = self.cache.make_key(query, top_k=top_k, filters=filters)
            cached = self.cache.retrieval.get(cache_key)
            if cached is not None:
                return cached

        # ── Search ───────────────────────────────────────────────────
        results = []

        if use_hybrid and hasattr(self.vector_store, "hybrid_search"):
            results = self.vector_store.hybrid_search(
                query,
                top_k=top_k,
                vector_weight=settings.hybrid_vector_weight,
                bm25_weight=settings.hybrid_bm25_weight,
                retrieval_k=settings.hybrid_retrieval_k,
            )
        else:
            query_embedding = self.embedder.embed_text(query)
            results = self.vector_store.search(query_embedding, top_k)

        # ── Parent context swap ──────────────────────────────────────
        if return_parent and results:
            for r in results:
                parent_text = r.get("metadata", {}).get("parent_text", "")
                if parent_text:
                    r["child_text"] = r["text"]
                    r["text"] = parent_text

        # ── Cache results ────────────────────────────────────────────
        if cache_key and self.cache:
            self.cache.retrieval.put(cache_key, results)

        return results

    def as_haystack_tool(self):
        """Convert to Haystack Tool for the Agent."""
        from haystack.tools import Tool

        _self = self

        def vector_search(query: str, top_k: int = 5) -> str:
            """Perform semantic search across document chunks.

            Use this tool when:
            - Looking for general information
            - Seeking definitions or explanations
            - Finding content similar to a concept

            Args:
                query: The search query.
                top_k: Number of results to return.
            """
            results = _self.search(query, top_k)

            if not results:
                return "No results found."

            formatted = []
            for i, r in enumerate(results, 1):
                text = r.get("text", "")[:500]
                score = r.get("score", 0)
                formatted.append(f"[{i}] (score: {score:.3f})\n{text}...")
            return "\n\n".join(formatted)

        return Tool(
            name="vector_search",
            description=(
                "Perform semantic search across document chunks. "
                "Use this tool when looking for general information, "
                "seeking definitions or explanations, or finding content "
                "similar to a concept."
            ),
            function=vector_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
