"""
NEO4J GRAPH STORE — Graphiti Wrapper for RAG3

Wraps graphiti-core for temporal knowledge graph construction and search.
Uses Neo4j as the backing graph database.

Features:
- Episode ingestion (text → entity extraction → graph construction)
- Hybrid search (semantic + keyword + graph traversal)
- Document ingestion tracking (avoid re-processing)
- Ollama LLM support via OpenAI-compatible API
- Optional Groq LLM override for faster extraction

Usage:
- Called by main.py during document ingestion (add_episode)
- Called by GraphAgent during query time (search)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from uuid import uuid4

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient as GraphitiLLMClient
from graphiti_core.nodes import EpisodeType
from neo4j import GraphDatabase

from src.config import settings
from src.storage.base import GraphStoreInterface

logger = logging.getLogger(__name__)


def _build_llm_client(
    model: str = None,
    base_url: str = None,
    use_groq: bool = False,
    groq_api_key: str = None,
):
    """
    Build a Graphiti-compatible LLM client.

    Default: Ollama via OpenAI-compatible API.
    Override: Groq API when --groq flag is passed during ingestion.
    """
    if use_groq and groq_api_key:
        # Groq is OpenAI-compatible
        llm_config = LLMConfig(
            model=model or "llama-3.3-70b-versatile",
            small_model=model or "llama-3.3-70b-versatile",
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info(f"Graph LLM: Groq API ({llm_config.model})")
        return GraphitiLLMClient(config=llm_config)

    # Default: Ollama via OpenAI-compatible endpoint
    ollama_base = base_url or settings.ollama_base_url
    ollama_model = model or settings.graph_builder_model

    llm_config = LLMConfig(
        model=ollama_model,
        small_model=ollama_model,
        api_key="ollama",  # Dummy key required by graphiti
        base_url=f"{ollama_base}/v1",
    )
    logger.info(f"Graph LLM: Ollama ({ollama_model}) at {ollama_base}")
    return GraphitiLLMClient(config=llm_config)


class Neo4jGraphStore(GraphStoreInterface):
    """
    Wraps Graphiti for graph construction and Neo4j for querying.

    Responsibilities:
    - Initialize Graphiti client with Neo4j connection
    - Add episodes (text chunks → graph entities + edges)
    - Search graph (semantic, keyword, hybrid traversal)
    - Track which documents have been graph-ingested
    - Retrieve subgraphs for context injection
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        model: str = None,
        use_groq: bool = False,
        groq_api_key: str = None,
    ):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        # Build LLM client for entity extraction
        self.llm_client = _build_llm_client(
            model=model,
            use_groq=use_groq,
            groq_api_key=groq_api_key,
        )

        # Graphiti client (lazy-initialized)
        self._graphiti: Optional[Graphiti] = None
        self._initialized = False

        # Neo4j driver for direct queries (ingestion tracking, etc.)
        self._driver = None

        logger.info(f"Neo4jGraphStore configured: {self.uri}")

    def _get_driver(self):
        """Get or create Neo4j driver for direct queries."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    async def initialize(self):
        """Initialize Graphiti client and build indices."""
        if self._initialized:
            return

        # Set dummy OPENAI_API_KEY if not set (required by graphiti internals)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "ollama-dummy-key"

        self._graphiti = Graphiti(
            self.uri,
            self.user,
            self.password,
            llm_client=self.llm_client,
        )

        await self._graphiti.build_indices_and_constraints()
        self._initialized = True
        logger.info("Graphiti initialized with indices and constraints.")

    def _ensure_initialized(self):
        """Ensure Graphiti is initialized (sync wrapper)."""
        if not self._initialized:
            asyncio.run(self.initialize())

    async def add_episode(
        self,
        text: str,
        source_metadata: Dict[str, Any],
        reference_time: str = None,
    ) -> Dict[str, Any]:
        """
        Add a text episode to the knowledge graph.

        Graphiti extracts entities and relationships from the text,
        creates/merges nodes and edges in Neo4j.

        Args:
            text: The text content to process.
            source_metadata: Metadata about the source (source path, page, type, etc.)
            reference_time: ISO timestamp for temporal awareness. Defaults to now.

        Returns:
            Dict with episode details (uuid, entities extracted, etc.)
        """
        if not self._initialized:
            await self.initialize()

        ref_time = reference_time or datetime.now(timezone.utc).isoformat()
        source = source_metadata.get("source", "unknown")
        group_id = source_metadata.get("group_id", str(uuid4()))

        try:
            await self._graphiti.add_episode(
                name=f"doc:{source}:{source_metadata.get('page', '?')}",
                episode_body=text,
                reference_time=datetime.fromisoformat(ref_time),
                source=EpisodeType.text,
                source_description=f"Document: {source}",
                group_id=group_id,
            )

            logger.info(
                f"Graph episode added: {source} (page {source_metadata.get('page', '?')})"
            )

            return {
                "status": "success",
                "source": source,
                "group_id": group_id,
                "reference_time": ref_time,
            }

        except Exception as e:
            logger.error(f"Failed to add graph episode: {e}")
            return {"status": "error", "error": str(e), "source": source}

    async def search(
        self, query: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph using Graphiti's hybrid search.

        Combines semantic similarity, keyword matching, and graph traversal.

        Args:
            query: Search query string.
            num_results: Number of results to return.

        Returns:
            List of result dicts with content, metadata, and scores.
        """
        if not self._initialized:
            await self.initialize()

        try:
            results = await self._graphiti._search(
                query=query,
                num_results=num_results,
            )

            formatted = []
            for edge in results:
                formatted.append({
                    "content": edge.fact if hasattr(edge, "fact") else str(edge),
                    "source_node": edge.source_node_name if hasattr(edge, "source_node_name") else "",
                    "target_node": edge.target_node_name if hasattr(edge, "target_node_name") else "",
                    "created_at": str(edge.created_at) if hasattr(edge, "created_at") else "",
                    "metadata": {
                        "uuid": edge.uuid if hasattr(edge, "uuid") else "",
                        "group_id": edge.group_id if hasattr(edge, "group_id") else "",
                    },
                })

            logger.info(f"Graph search returned {len(formatted)} results for: {query[:50]}")
            return formatted

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def is_document_ingested(self, source: str) -> bool:
        """
        Check if a document has already been ingested into the graph.

        Queries Neo4j for EpisodicNode entries with matching source_description.
        """
        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Episodic)
                    WHERE e.source_description CONTAINS $source
                    RETURN count(e) AS count
                    """,
                    source=source,
                )
                record = result.single()
                count = record["count"] if record else 0
                return count > 0
        except Exception as e:
            logger.error(f"Failed to check graph ingestion state: {e}")
            return False

    async def get_entity_neighborhood(
        self, entity_name: str, depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the neighborhood of an entity in the graph.

        Useful for expanding context around discovered entities.

        Args:
            entity_name: Name of the entity to explore.
            depth: How many hops to traverse from the entity.

        Returns:
            List of connected entity dicts.
        """
        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(
                    f"""
                    MATCH (n:Entity {{name: $name}})
                    CALL apoc.path.subgraphAll(n, {{maxLevel: $depth}})
                    YIELD nodes, relationships
                    RETURN nodes, relationships
                    """,
                    name=entity_name,
                    depth=depth,
                )
                record = result.single()
                if not record:
                    return []

                nodes = []
                for node in record["nodes"]:
                    nodes.append({
                        "name": node.get("name", ""),
                        "labels": list(node.labels),
                        "properties": dict(node),
                    })
                return nodes
        except Exception as e:
            logger.error(f"Entity neighborhood query failed: {e}")
            return []

    def verify_connectivity(self) -> bool:
        """Test Neo4j connectivity. Returns True if connected."""
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            logger.info("Neo4j connectivity verified.")
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    async def close(self):
        """Close all connections."""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
        if self._driver:
            self._driver.close()
            self._driver = None
        self._initialized = False
        logger.info("Neo4jGraphStore closed.")
