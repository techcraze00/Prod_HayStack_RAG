"""
VECTOR STORE MANAGER
FAISS wrapper for Episodic and Archival memory.

Functionality:
- Manages two distinct FAISS indices:
  - Index A: Episodic (Summarized conversation history).
  - Index B: Archival (Gold nuggets, successful scripts/queries).
- Handles embedding, storage, and similarity search.

Usage:
- Called by the Orchestrator for RAG context retrieval.
- Updated by Workers (Archival) and the Summarizer (Episodic).
"""

import os
import logging
import faiss
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)

class FAISSManager:
    def __init__(self, index_name: str, base_path: str = None, embedding_model: str = None, session_id: str = None):
        """
        Initializes the FAISS manager for a specific index.
        index_name should be 'episodic', 'archival', or a schema-specific name.
        """
        self.index_name = index_name
        self.embeddings = OllamaEmbeddings(
            model=embedding_model or settings.ollama_embedding_model,
            base_url=settings.ollama_base_url
        )
        
        # Determine index path
        if base_path:
            self.index_path = os.path.join(base_path, f"faiss_{index_name}")
        else:
            # For episodic/archival, use session-specific subdirs in temp if session_id is provided
            data_temp = os.path.join(settings.parsed_docs_dir, "temp_memory")
            if session_id:
                session_dir = os.path.join(data_temp, session_id)
                os.makedirs(session_dir, exist_ok=True)
                self.index_path = os.path.join(session_dir, f"faiss_{index_name}")
            else:
                os.makedirs(data_temp, exist_ok=True)
                self.index_path = os.path.join(data_temp, f"faiss_{index_name}")
            
        self.vector_store = self._load_or_create_index()

    def _load_or_create_index(self):
        """Loads index from disk or creates a new one."""
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading existing FAISS index: {self.index_name}")
                return FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=False  # Security: prevent pickle RCE
                )
            except Exception as e:
                logger.warning(
                    f"Could not load index {self.index_name} with safe deserialization: {e}. "
                    f"Creating a fresh index. (Old index may use unsafe pickle format.)"
                )
        
        # Create new index
        logger.info(f"Creating new FAISS index: {self.index_name}")
        # Initialize with a dummy document to set dimensions if needed, 
        # but LangChain handles this gracefully on first add.
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("dummy")))
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store

    def add_to_index(self, text: str, metadata: Dict[str, Any] = None):
        """Embeds text and adds it to the index."""
        try:
            self.vector_store.add_texts([text], metadatas=[metadata or {}])
            self.vector_store.save_local(self.index_path)
            
            # Explicit logging for memory ingestion
            log_content = text[:200] + "..." if len(text) > 200 else text
            print(f"\n[MEMORY INGESTION] ({self.index_name.upper()}) Saved: {log_content}")
            if metadata:
                print(f"[MEMORY META] {metadata}")
                
            logger.info(f"Added entry to {self.index_name} memory.")
        except Exception as e:
            logger.error(f"Failed to add to index {self.index_name}: {e}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Returns relevant snippets from the index."""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        except Exception as e:
            logger.error(f"Search failed in {self.index_name}: {e}")
            return []

    def search_with_scores(self, query: str, k: int = 3, max_distance: float = None) -> List[Dict[str, Any]]:
        """
        Returns relevant snippets with their L2 distance scores.
        Lower distance = more relevant.
        
        Args:
            query: Search query string.
            k: Number of results to return.
            max_distance: If set, only return results with distance ≤ this value.
                          Useful for filtering out low-relevance noise.
        """
        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            results = []
            for doc, score in docs_and_scores:
                if max_distance is not None and score > max_distance:
                    continue
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": float(score)
                })
            return results
        except Exception as e:
            logger.error(f"Scored search failed in {self.index_name}: {e}")
            return []

    def batch_add(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Efficiently adds multiple texts to the index at once."""
        try:
            if not texts:
                return
            self.vector_store.add_texts(texts, metadatas=metadatas)
            self.vector_store.save_local(self.index_path)
            logger.info(f"Batch added {len(texts)} entries to {self.index_name} memory.")
        except Exception as e:
            logger.error(f"Batch add failed for {self.index_name}: {e}")

    def clear(self):
        """Clears the index (session-scoped cleanup)."""
        if os.path.exists(self.index_path):
            import shutil
            shutil.rmtree(self.index_path)
        self.vector_store = self._load_or_create_index()


class GraphMemoryManager:
    """
    Graph-backed episodic memory manager.

    Wraps Neo4jGraphStore for memory-specific retrieval:
    - Searches the knowledge graph for entities/facts related to the current query
    - Returns formatted memory context for injection into agent prompts
    - Optionally stores conversation episodes into the graph for long-term recall

    Usage:
    - Initialized in RAGSession when graph_rag_enabled and a graph_store is available
    - Called by session.get_memory_context() to augment FAISS episodic memory
    """

    def __init__(self, graph_store):
        """
        Args:
            graph_store: Neo4jGraphStore instance (must implement GraphStoreInterface).
        """
        self.graph_store = graph_store

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for facts/entities related to the query.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of result dicts with 'content' and 'metadata' keys
            (same format as FAISSManager.search for compatibility).
        """
        import asyncio

        try:
            results = asyncio.run(self.graph_store.search(query=query, num_results=k))

            # Convert to FAISSManager-compatible format
            formatted = []
            for r in results:
                content = r.get("content", "")
                source_node = r.get("source_node", "")
                target_node = r.get("target_node", "")

                # Build a readable content string
                if source_node and target_node:
                    display = f"{source_node} → {target_node}: {content}"
                else:
                    display = content

                formatted.append({
                    "content": display,
                    "metadata": {
                        "source": "knowledge_graph",
                        "source_node": source_node,
                        "target_node": target_node,
                        **r.get("metadata", {}),
                    }
                })

            return formatted

        except Exception as e:
            logger.error(f"Graph memory search failed: {e}")
            return []

    def add_to_graph(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """
        Store a conversation episode into the knowledge graph.

        This allows the graph to learn from conversations over time,
        building a temporal knowledge base of user interactions.

        Args:
            text: The text content to store (e.g. conversation summary).
            metadata: Additional metadata (session_id, type, etc.).
        """
        import asyncio

        try:
            source_metadata = {
                "source": "conversation_memory",
                "type": "episodic",
                **(metadata or {}),
            }
            asyncio.run(
                self.graph_store.add_episode(
                    text=text,
                    source_metadata=source_metadata,
                )
            )
            logger.info(f"Added conversation episode to graph memory.")
        except Exception as e:
            logger.error(f"Failed to add episode to graph memory: {e}")