"""
Failed Retrieval Handling — Progressive Fallback Strategies

When retrieval quality is low (poor scores, low diversity, empty results),
this module applies a cascade of progressively broader strategies to
recover useful context.

Fallback cascade:
1. Query expansion (rephrase and retry)
2. Query decomposition (split into sub-queries)
3. Summary index search
4. Relax metadata filters
5. Broader search (increased top_k, lower threshold)
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from ..utils.llm import chat_sync


class FallbackStrategy(Enum):
    QUERY_EXPANSION = "query_expansion"
    QUERY_DECOMPOSITION = "query_decomposition"
    SUMMARY_SEARCH = "summary_search"
    RELAX_FILTERS = "relax_filters"
    BROADER_SEARCH = "broader_search"


@dataclass
class FallbackResult:
    strategy: FallbackStrategy
    results: List[Dict[str, Any]]
    success: bool
    message: str = ""


class RetrievalFallbackHandler:
    """
    Progressive fallback handler for failed retrievals.

    Uses synchronous Haystack generator for query rewriting.
    """

    def __init__(
        self,
        generator,
        vector_search_fn: Callable[..., List[Dict[str, Any]]],
        summary_search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        min_score_threshold: float = 0.3,
        min_results: int = 2,
    ):
        self.generator = generator
        self.vector_search_fn = vector_search_fn
        self.summary_search_fn = summary_search_fn
        self.min_score_threshold = min_score_threshold
        self.min_results = min_results

    def is_low_quality(self, results: List[Dict[str, Any]]) -> bool:
        if not results or len(results) < self.min_results:
            return True

        scores = [r.get("score", 0) for r in results]
        if all(s < self.min_score_threshold for s in scores):
            return True

        texts = [r.get("text", "")[:100] for r in results]
        unique_prefixes = len(set(texts))
        if len(results) > 2 and unique_prefixes <= 1:
            return True

        return False

    def apply_fallback(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[FallbackStrategy]] = None,
    ) -> FallbackResult:
        if strategies is None:
            strategies = [
                FallbackStrategy.QUERY_EXPANSION,
                FallbackStrategy.QUERY_DECOMPOSITION,
                FallbackStrategy.SUMMARY_SEARCH,
                FallbackStrategy.RELAX_FILTERS,
                FallbackStrategy.BROADER_SEARCH,
            ]

        best_results = initial_results

        for strategy in strategies:
            try:
                result = self._execute_strategy(strategy, query, best_results, filters)
                if result.success and len(result.results) >= self.min_results:
                    return result
                if len(result.results) > len(best_results):
                    best_results = result.results
            except Exception:
                continue

        return FallbackResult(
            strategy=strategies[-1] if strategies else FallbackStrategy.BROADER_SEARCH,
            results=best_results,
            success=len(best_results) >= self.min_results,
            message="Exhausted all fallback strategies",
        )

    def _execute_strategy(
        self,
        strategy: FallbackStrategy,
        query: str,
        current_results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> FallbackResult:
        if strategy == FallbackStrategy.QUERY_EXPANSION:
            expanded = self._expand_query(query)
            results = self.vector_search_fn(expanded, 10)
            combined = self._deduplicate_results(current_results + results)
            return FallbackResult(
                strategy=strategy,
                results=combined,
                success=not self.is_low_quality(combined),
                message=f"Expanded query: {expanded}",
            )

        elif strategy == FallbackStrategy.QUERY_DECOMPOSITION:
            sub_queries = self._decompose_query(query)
            all_results = list(current_results)
            for sq in sub_queries:
                sub_results = self.vector_search_fn(sq, 5)
                all_results.extend(sub_results)
            combined = self._deduplicate_results(all_results)
            return FallbackResult(
                strategy=strategy,
                results=combined,
                success=not self.is_low_quality(combined),
                message=f"Decomposed into {len(sub_queries)} sub-queries",
            )

        elif strategy == FallbackStrategy.SUMMARY_SEARCH:
            if self.summary_search_fn:
                results = self.summary_search_fn(query, 5)
                combined = self._deduplicate_results(current_results + results)
                return FallbackResult(
                    strategy=strategy,
                    results=combined,
                    success=not self.is_low_quality(combined),
                    message="Searched summary index",
                )
            return FallbackResult(
                strategy=strategy,
                results=current_results,
                success=False,
                message="No summary search available",
            )

        elif strategy == FallbackStrategy.RELAX_FILTERS:
            if filters:
                results = self.vector_search_fn(query, 10)
                combined = self._deduplicate_results(current_results + results)
                return FallbackResult(
                    strategy=strategy,
                    results=combined,
                    success=not self.is_low_quality(combined),
                    message="Relaxed metadata filters",
                )
            return FallbackResult(
                strategy=strategy,
                results=current_results,
                success=False,
                message="No filters to relax",
            )

        elif strategy == FallbackStrategy.BROADER_SEARCH:
            results = self.vector_search_fn(query, 20)
            return FallbackResult(
                strategy=strategy,
                results=results,
                success=not self.is_low_quality(results),
                message="Broadened search with top_k=20",
            )

        return FallbackResult(
            strategy=strategy,
            results=current_results,
            success=False,
            message="Unknown strategy",
        )

    def _expand_query(self, query: str) -> str:
        try:
            return chat_sync(
                self.generator,
                system="",
                user=(
                    "Rephrase the following query to be more specific and searchable. "
                    "Only output the rephrased query, nothing else.\n\n"
                    f"Query: {query}\n\nRephrased:"
                ),
            ).strip()
        except Exception:
            return query

    def _decompose_query(self, query: str) -> List[str]:
        try:
            response = chat_sync(
                self.generator,
                system="",
                user=(
                    "Break down the following complex query into 2-3 simpler sub-queries "
                    "that together answer the original question. "
                    "Output each sub-query on a separate line, nothing else.\n\n"
                    f"Query: {query}\n\nSub-queries:"
                ),
            )
            lines = [
                line.strip().lstrip("- \u20220123456789.")
                for line in response.strip().split("\n")
                if line.strip()
            ]
            return lines[:3] if lines else [query]
        except Exception:
            return [query]

    @staticmethod
    def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for r in results:
            text = r.get("text", "")[:200]
            if text not in seen:
                seen.add(text)
                unique.append(r)
        return unique
