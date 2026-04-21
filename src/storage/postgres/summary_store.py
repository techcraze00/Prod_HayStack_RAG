from typing import List, Dict, Any, Optional
import json
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

from ..base import SummaryStoreInterface


class PostgresSummaryStore(SummaryStoreInterface):
    """
    PostgreSQL-based Universal Summary Store.

    Supports hierarchical summaries (Full, Topic, Section) using JSONB metadata
    and pgvector for semantic search.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        table_name: str = "universal_summaries",
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self._embedder = OllamaTextEmbedder(
            model=embedding_model,
            url=ollama_base_url,
        )
        # self._embedder.warm_up()
        self._conn = None
        self._ensure_table()

    def _get_connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.connection_string)
        return self._conn

    def _ensure_table(self):
        """Create table with vector extension."""
        dummy_vec = self._embedder.run(text="test")["embedding"]
        dim = len(dummy_vec)

        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    embedding vector({dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            conn.commit()

    def add_summary(
        self, doc_id: str, summary: str, metadata: Dict[str, Any]
    ) -> str:
        """Add a summary to the store."""
        embedding = self._embedder.run(text=summary)["embedding"]
        summary_type = metadata.get("summary_type", "full")

        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.table_name}
                (doc_id, summary_text, summary_type, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                doc_id,
                summary,
                summary_type,
                embedding,
                json.dumps(metadata),
            ))
            conn.commit()
            new_id = cur.fetchone()[0]

        return str(new_id)

    def search_summaries(
        self, query: str, top_k: int = 3, filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search summaries with optional type filtering."""
        query_embedding = self._embedder.run(text=query)["embedding"]

        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = f"""
                SELECT doc_id, summary_text, summary_type, metadata,
                       1 - (embedding <=> %s::vector) as score
                FROM {self.table_name}
                WHERE 1=1
            """
            params = [query_embedding]

            if filter_type:
                sql += " AND summary_type = %s"
                params.append(filter_type)

            sql += " ORDER BY score DESC LIMIT %s"
            params.append(top_k)

            cur.execute(sql, tuple(params))
            results = cur.fetchall()

        return [
            {
                "doc_id": row["doc_id"],
                "summary": row["summary_text"],
                "type": row["summary_type"],
                "score": float(row["score"]),
                "metadata": row["metadata"],
            }
            for row in results
        ]

    def close(self):
        if self._conn:
            self._conn.close()
