from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage"""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add chunks with embeddings"""
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int
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
