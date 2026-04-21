from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage"""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add chunks with embeddings"""
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        """Delete chunks from a source"""
        pass


class SummaryStoreInterface(ABC):
    """Abstract interface for summary index"""

    @abstractmethod
    def add_summary(
        self, doc_id: str, summary: str, metadata: Dict[str, Any]
    ) -> str:
        """Add document summary"""
        pass

    @abstractmethod
    def search_summaries(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search summaries"""
        pass


class GraphStoreInterface(ABC):
    """Abstract interface for graph-based knowledge storage"""

    @abstractmethod
    async def add_episode(
        self, text: str, source_metadata: Dict[str, Any], reference_time: str = None
    ) -> Dict[str, Any]:
        """Add a text episode to the knowledge graph (entity extraction + edge creation)"""
        pass

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""
        pass

    @abstractmethod
    async def get_entity_neighborhood(
        self, entity_name: str, depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Retrieve the local neighborhood of an entity up to the specified depth."""
        pass

    @abstractmethod
    def is_document_ingested(self, source: str) -> bool:
        """Check if a document has already been ingested into the graph"""
        pass

    @abstractmethod
    async def close(self):
        """Close the graph store connection"""
        pass

