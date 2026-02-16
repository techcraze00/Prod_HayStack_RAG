"""
PostgreSQL Vector Store using Haystack pgvector integration.

Uses haystack-pgvector for native vector similarity search.

Enhanced with:
- Hybrid Search (Vector + BM25) via PostgreSQL full-text search
- Reciprocal Rank Fusion (RRF) for combining results
- Metadata filtering
"""

from typing import List, Dict, Any, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from haystack import Document
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

from ..base import VectorStoreInterface


class PostgresVectorStore(VectorStoreInterface):
    """
    PostgreSQL-based vector storage using pgvector via Haystack.

    Features:
    - Native vector similarity search (HNSW)
    - Full-text (BM25) keyword search via PostgreSQL tsvector
    - Hybrid search with Reciprocal Rank Fusion
    - Metadata filtering
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "chunks",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.connection_string = connection_string
        self.table_name = table_name

        # Embedder for query-time embedding
        self._embedder = OllamaTextEmbedder(
            model=embedding_model,
            url=ollama_base_url,
        )
        # self._embedder.warm_up()

        # Document store
        self._store = PgvectorDocumentStore(
            connection_string=Secret.from_token(connection_string),
            table_name=table_name,
            embedding_dimension=768,
            vector_function="cosine_similarity",
            search_strategy="hnsw",
            recreate_table=False,
        )

        # Retrievers
        self._embedding_retriever = PgvectorEmbeddingRetriever(
            document_store=self._store,
        )
        self._keyword_retriever = PgvectorKeywordRetriever(
            document_store=self._store,
        )

        self._fts_initialized = False

    # ─── Full-Text Search Index Setup ────────────────────────────────

    def _ensure_fts_index(self):
        """Ensure full-text search index exists for hybrid search."""
        if self._fts_initialized:
            return

        try:
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                cur.execute(f"""
                    ALTER TABLE {self.table_name}
                    ADD COLUMN IF NOT EXISTS content_tsv tsvector;
                """)

                cur.execute(f"""
                    CREATE OR REPLACE FUNCTION {self.table_name}_tsv_trigger()
                    RETURNS trigger AS $$
                    BEGIN
                        NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """)

                cur.execute(f"""
                    DROP TRIGGER IF EXISTS tsv_update ON {self.table_name};

                    CREATE TRIGGER tsv_update
                    BEFORE INSERT OR UPDATE ON {self.table_name}
                    FOR EACH ROW EXECUTE FUNCTION {self.table_name}_tsv_trigger();
                """)

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_content_tsv
                    ON {self.table_name} USING GIN(content_tsv);
                """)

                cur.execute(f"""
                    UPDATE {self.table_name}
                    SET content_tsv = to_tsvector('english', COALESCE(content, ''))
                    WHERE content_tsv IS NULL;
                """)

                conn.commit()
            conn.close()
            self._fts_initialized = True
        except Exception as e:
            print(f"Warning: Could not create FTS index: {e}")

    # ─── BM25 Search ─────────────────────────────────────────────────

    def bm25_search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """Pure BM25/full-text search using PostgreSQL ts_rank_cd."""
        conn = psycopg2.connect(self.connection_string)
        results = []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT
                        content,
                        ts_rank_cd(content_tsv, query, 32) as score,
                        meta as metadata
                    FROM {self.table_name},
                         plainto_tsquery('english', %s) query
                    WHERE content_tsv @@ query
                    ORDER BY score DESC
                    LIMIT %s
                """, (query, top_k))

                rows = cur.fetchall()
                results = [
                    (row['content'], float(row['score']), row['metadata'] or {})
                    for row in rows
                ]
        finally:
            conn.close()

        return results

    # ─── Reciprocal Rank Fusion ───────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Tuple[str, float, Dict]],
        top_k: int,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion to combine vector and BM25 results."""
        scores = {}

        for rank, result in enumerate(vector_results, 1):
            content = result['text']
            rrf_score = vector_weight / (k + rank)
            scores[content] = {
                'rrf_score': rrf_score,
                'vector_rank': rank,
                'vector_score': result.get('score', 0),
                'metadata': result.get('metadata', {}),
            }

        for rank, (content, bm25_score, metadata) in enumerate(bm25_results, 1):
            rrf_score = bm25_weight / (k + rank)
            if content in scores:
                scores[content]['rrf_score'] += rrf_score
                scores[content]['bm25_rank'] = rank
                scores[content]['bm25_score'] = bm25_score
            else:
                scores[content] = {
                    'rrf_score': rrf_score,
                    'bm25_rank': rank,
                    'bm25_score': bm25_score,
                    'metadata': metadata,
                }

        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True,
        )[:top_k]

        return [
            {
                'text': content,
                'score': data['rrf_score'],
                'vector_score': data.get('vector_score', 0),
                'bm25_score': data.get('bm25_score', 0),
                'vector_rank': data.get('vector_rank', None),
                'bm25_rank': data.get('bm25_rank', None),
                'metadata': data['metadata'],
            }
            for content, data in sorted_results
        ]

    # ─── Hybrid Search ────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        retrieval_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and BM25 keyword search."""
        self._ensure_fts_index()

        # Vector search
        vector_results = self.search_with_text(query, retrieval_k)

        # BM25 search
        bm25_results = self.bm25_search(query, retrieval_k)

        return self._reciprocal_rank_fusion(
            vector_results, bm25_results, top_k, vector_weight, bm25_weight
        )

    # ─── Core Methods ─────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, int]:
        """Add chunks with embeddings in batches.

        Writes in small batches so partial progress is committed on failure.
        """
        if not chunks:
            return {"added": 0}

        total_added = 0
        total_failed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            documents = [
                Document(
                    content=chunk.get("text", ""),
                    embedding=chunk.get("embedding"),
                    meta=chunk.get("metadata", {}),
                )
                for chunk in batch
            ]

            try:
                self._store.write_documents(documents)
                total_added += len(documents)
            except Exception as e:
                total_failed += len(documents)
                print(
                    f"Warning: Failed to write batch {i // batch_size + 1} "
                    f"({len(documents)} docs): {e}"
                )

        return {"added": total_added, "failed": total_failed}

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        result = self._embedding_retriever.run(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            {
                "text": doc.content,
                "score": doc.score if doc.score is not None else 1.0,
                "metadata": doc.meta,
            }
            for doc in result["documents"]
        ]

    def search_with_text(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search using text query (embedding generated automatically)."""
        query_embedding = self._embedder.run(text=query)["embedding"]
        return self.search(query_embedding, top_k)

    def delete_by_source(self, source: str) -> int:
        """Delete chunks from a source."""
        try:
            self._store.delete_documents(
                filters={"field": "meta.source", "operator": "==", "value": source}
            )
            return 1
        except Exception:
            return 0

    def count(self) -> int:
        """Count total documents."""
        return self._store.count_documents()
