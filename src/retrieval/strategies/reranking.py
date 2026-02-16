"""
OllamaRanker — Custom Haystack component for reranking via Ollama.

Uses the Ollama reranker model (e.g. qllama/bge-reranker-v2-m3)
to score query-document pairs and return top_k documents sorted by relevance.
"""

from typing import List, Optional

import requests
from haystack import Document, component

from ...config import settings


@component
class OllamaRanker:
    """
    Reranks documents using an Ollama-hosted reranker model.

    Uses the Ollama API to score query-document pairs,
    replacing sentence-transformers CrossEncoder.
    """

    def __init__(
        self,
        model: str = None,
        url: str = None,
        top_k: int = 5,
    ):
        self.model = model or settings.reranker_model
        self.url = (url or settings.ollama_base_url).rstrip("/")
        self.default_top_k = top_k

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> dict:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Dict with "documents" key containing reranked documents.
        """
        if not documents:
            return {"documents": []}

        top_k = top_k or self.default_top_k

        scored_docs = []
        for doc in documents:
            score = self._score_pair(query, doc.content or "")
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        result_docs = []
        for doc, score in scored_docs[:top_k]:
            doc.score = score
            result_docs.append(doc)

        return {"documents": result_docs}

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair using Ollama."""
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"query: {query}\ndocument: {document}",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # The reranker model outputs a relevance score.
            # Parse the response text as a float score.
            response_text = data.get("response", "0").strip()
            try:
                return float(response_text)
            except ValueError:
                # If the model outputs text instead of a score,
                # use response length as a rough proxy (longer = more relevant content found)
                return len(response_text) / 1000.0

        except Exception as e:
            # On failure, return 0 so the document sorts to the bottom
            return 0.0


class RerankingStrategy:
    """
    Re-ranks retrieved results using OllamaRanker.

    Process:
    1. Initial retrieval (fast, broad)
    2. Re-rank with Ollama reranker (precise)
    3. Return top_k after re-ranking
    """

    def __init__(self, model: str = None, url: str = None):
        self.ranker = OllamaRanker(
            model=model or settings.reranker_model,
            url=url,
        )

    def rerank(
        self, query: str, results: List[dict], top_k: int = 5
    ) -> List[dict]:
        """Re-rank results using Ollama reranker.

        Args:
            query: Original query.
            results: Initial retrieval results (list of dicts with 'text').
            top_k: Number of results to return.

        Returns:
            Re-ranked results.
        """
        if not results:
            return []

        # Convert dicts to Haystack Documents for the ranker
        documents = [
            Document(content=r.get("text", ""), meta=r.get("metadata", {}))
            for r in results
        ]

        ranked = self.ranker.run(query=query, documents=documents, top_k=top_k)

        # Convert back to dicts
        reranked = []
        for doc in ranked["documents"]:
            reranked.append({
                "text": doc.content,
                "score": doc.score or 0.0,
                "rerank_score": doc.score or 0.0,
                "metadata": doc.meta,
            })

        return reranked
