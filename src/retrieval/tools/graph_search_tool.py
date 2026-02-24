"""
GRAPH SEARCH TOOL — Knowledge Graph Search for RAG3

Wraps Neo4jGraphStore for graph-based retrieval.
Provides both direct Python API and Haystack Tool interface.

Features:
- Hybrid graph search (semantic + keyword + graph traversal via Graphiti)
- Entity neighborhood expansion
- Formatted context output for LLM consumption
- Haystack Tool wrapper for Agent compatibility
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from ...storage.base import GraphStoreInterface
from ...config import settings

logger = logging.getLogger(__name__)


class GraphSearchTool:
    """
    Graph search tool wrapping Neo4jGraphStore.

    Supports:
    - Semantic + structural graph search via Graphiti
    - Entity neighborhood expansion via APOC
    - Context formatting for LLM consumption
    """

    def __init__(self, graph_store: GraphStoreInterface):
        self.graph_store = graph_store

    def search(
        self,
        query: str,
        num_results: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph using Graphiti's hybrid retrieval.

        Combines semantic similarity, keyword matching, and graph structure.

        Args:
            query: Search query string.
            num_results: Number of results to return.

        Returns:
            List of result dicts with content, source/target nodes, etc.
        """
        num_results = num_results or settings.graph_search_results

        try:
            results = asyncio.run(
                self.graph_store.search(query=query, num_results=num_results)
            )
            logger.info(f"Graph search returned {len(results)} results for: {query[:60]}")
            return results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def get_neighborhood(
        self,
        entity_name: str,
        depth: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Expand context around a specific entity.

        Uses APOC's subgraph traversal to find connected entities.

        Args:
            entity_name: Name of the entity to explore.
            depth: Number of hops to traverse.

        Returns:
            List of connected entity dicts.
        """
        depth = depth or settings.graph_search_depth

        try:
            results = asyncio.run(
                self.graph_store.get_entity_neighborhood(
                    entity_name=entity_name,
                    depth=depth,
                )
            )
            logger.info(
                f"Neighborhood for '{entity_name}': {len(results)} connected entities"
            )
            return results
        except Exception as e:
            logger.error(f"Neighborhood query failed: {e}")
            return []

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format graph search results into a context string for LLM consumption.

        Args:
            results: List of graph search result dicts.

        Returns:
            Formatted string with facts and relationships.
        """
        if not results:
            return "No graph results found."

        formatted = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            source = r.get("source_node", "")
            target = r.get("target_node", "")

            if source and target:
                formatted.append(f"[{i}] {source} → {target}: {content}")
            else:
                formatted.append(f"[{i}] {content}")

        return "\n".join(formatted)

    def as_haystack_tool(self):
        """Convert to Haystack Tool for the Agent framework."""
        from haystack.tools import Tool

        _self = self

        def graph_search(query: str, num_results: int = 10) -> str:
            """Search the knowledge graph for relationships and entities.

            Use this tool when:
            - The question asks about relationships between concepts
            - Multi-hop reasoning is needed
            - Temporal or structural queries about how things connect
            - Finding all entities related to a topic

            Args:
                query: The search query about relationships or entities.
                num_results: Number of results to return.
            """
            results = _self.search(query, num_results)
            return _self.format_context(results)

        return Tool(
            name="graph_search",
            description=(
                "Search the knowledge graph for relationships and entities. "
                "Use when the question involves relationships between concepts, "
                "multi-hop reasoning, temporal connections, or entity exploration."
            ),
            function=graph_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about relationships or entities",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        )
