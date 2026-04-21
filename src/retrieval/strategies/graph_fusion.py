"""
GRAPH VECTOR FUSION — Hybrid Retrieval Strategy for RAG3

Combines vector search results with graph search results for richer context.

Strategy:
1. Run vector search in parallel with graph search
2. Deduplicate entities across results
3. Score and rank combined context
4. Return fused context for synthesis

Usage:
- Dispatched by Orchestrator when intent is HybridRetrieval
- Combines VectorSearchTool + GraphSearchTool results
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class GraphVectorFusion:
    """
    Combines vector search results with graph context for hybrid retrieval.

    Runs both pipelines in parallel, deduplicates overlapping content,
    scores and ranks the combined results, then returns fused context
    ready for LLM consumption.
    """

    def __init__(self, vector_tool, graph_search_tool):
        """
        Args:
            vector_tool: VectorSearchTool instance for semantic chunk search.
            graph_search_tool: GraphSearchTool instance for graph-based search.
        """
        self.vector_tool = vector_tool
        self.graph_tool = graph_search_tool

    def fused_search(
        self,
        query: str,
        vector_top_k: int = 5,
        graph_num_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Run parallel vector + graph search and return fused results.

        Args:
            query: User's search query.
            vector_top_k: Number of vector search results.
            graph_num_results: Number of graph search results.

        Returns:
            Dict with:
              - fused_context: merged context string
              - vector_results: raw vector results
              - graph_results: raw graph results
              - vector_count: number of vector hits used
              - graph_count: number of graph facts used
              - total_count: combined unique results
        """
        vector_results = []
        graph_results = []

        # Parallel execution of both search pipelines
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    self._safe_vector_search, query, vector_top_k
                ): "vector",
                executor.submit(
                    self._safe_graph_search, query, graph_num_results
                ): "graph",
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    if source == "vector":
                        vector_results = result
                    else:
                        graph_results = result
                except Exception as e:
                    logger.error(f"Fusion {source} search failed: {e}")

        # Deduplicate overlapping content
        vector_clean, graph_clean = self._deduplicate(vector_results, graph_results)

        # Rank and merge into fused context
        fused_context = self._rank_and_merge(vector_clean, graph_clean)

        total = len(vector_clean) + len(graph_clean)
        logger.info(
            f"Fusion complete: {len(vector_clean)} vector + {len(graph_clean)} graph = {total} results"
        )

        return {
            "fused_context": fused_context,
            "vector_results": vector_clean,
            "graph_results": graph_clean,
            "vector_count": len(vector_clean),
            "graph_count": len(graph_clean),
            "total_count": total,
        }

    def _safe_vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector search with error handling."""
        try:
            return self.vector_tool.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Vector search in fusion failed: {e}")
            return []

    def _safe_graph_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Graph search with error handling."""
        try:
            return self.graph_tool.search(query, num_results=num_results)
        except Exception as e:
            logger.error(f"Graph search in fusion failed: {e}")
            return []

    def _deduplicate(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Remove overlapping content between vector and graph results.

        Uses substring matching on content to detect duplicates.
        When overlap is detected, the vector result is kept (higher precision),
        and the graph result is removed.

        Returns:
            Tuple of (deduplicated_vector, deduplicated_graph).
        """
        if not vector_results or not graph_results:
            return vector_results, graph_results

        # Build set of vector content fingerprints (lowered, trimmed)
        vector_fingerprints = set()
        for vr in vector_results:
            text = vr.get("text", "").lower().strip()[:200]
            if text:
                vector_fingerprints.add(text)

        # Filter graph results that substantially overlap with vector content
        filtered_graph = []
        for gr in graph_results:
            graph_content = gr.get("content", "").lower().strip()
            if not graph_content:
                continue

            is_duplicate = False
            for vfp in vector_fingerprints:
                # Check if either is a substring of the other (fuzzy overlap)
                if (
                    graph_content[:100] in vfp
                    or vfp[:100] in graph_content
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_graph.append(gr)

        removed = len(graph_results) - len(filtered_graph)
        if removed > 0:
            logger.info(f"Deduplication removed {removed} overlapping graph results")

        return vector_results, filtered_graph

    def _rank_and_merge(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> str:
        """
        Score, rank, and merge results into a single context string.

        Scoring:
        - Vector results: use native similarity score (0-1)
        - Graph results: use position-based scoring (1.0 → 0.5 descending)
        - Apply configurable weight to graph results

        Returns:
            Formatted context string for LLM consumption.
        """
        graph_weight = getattr(settings, "hybrid_graph_weight", 0.4)
        vector_weight = 1.0 - graph_weight

        scored_items = []

        # Score vector results
        for i, vr in enumerate(vector_results):
            native_score = vr.get("score", 0.5)
            weighted_score = native_score * vector_weight
            text = vr.get("text", "")[:500]
            source = vr.get("metadata", {}).get("source", "unknown")

            scored_items.append({
                "type": "vector",
                "score": weighted_score,
                "content": text,
                "source": source,
            })

        # Score graph results (position-based since graph results don't have numeric scores)
        total_graph = max(len(graph_results), 1)
        for i, gr in enumerate(graph_results):
            position_score = 1.0 - (i / total_graph) * 0.5  # 1.0 → 0.5
            weighted_score = position_score * graph_weight
            content = gr.get("content", "")
            source_node = gr.get("source_node", "")
            target_node = gr.get("target_node", "")

            scored_items.append({
                "type": "graph",
                "score": weighted_score,
                "content": content,
                "source_node": source_node,
                "target_node": target_node,
            })

        # Sort by score descending
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        # Format into context string
        parts = []

        # Vector section
        vector_items = [s for s in scored_items if s["type"] == "vector"]
        if vector_items:
            parts.append("=== Document Context (Vector Search) ===")
            for i, item in enumerate(vector_items, 1):
                parts.append(f"[V{i}] (score: {item['score']:.3f}) {item['content']}")

        # Graph section
        graph_items = [s for s in scored_items if s["type"] == "graph"]
        if graph_items:
            parts.append("\n=== Knowledge Graph Facts ===")
            for i, item in enumerate(graph_items, 1):
                src = item.get("source_node", "")
                tgt = item.get("target_node", "")
                if src and tgt:
                    parts.append(f"[G{i}] {src} → {tgt}: {item['content']}")
                else:
                    parts.append(f"[G{i}] {item['content']}")

        if not parts:
            return "No results found from either vector or graph search."

        return "\n".join(parts)
