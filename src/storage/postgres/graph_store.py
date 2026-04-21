"""
PostgreSQL Graph Store — Lightweight Knowledge Graph fallback.

Stores entities and relationships in standard Postgres tables using JSONB.
Implements GraphStoreInterface so it can be swapped in for Neo4j when
Neo4j is unavailable or undesired.

Uses LLM-based entity extraction via the shared chat_sync helper.
"""

import json
import logging
import re
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor

from ..base import GraphStoreInterface
from ...utils.llm import chat_sync

logger = logging.getLogger(__name__)


class PostgresGraphStore(GraphStoreInterface):
    """
    Lightweight graph store backed by PostgreSQL.

    Schema:
    - graph_entities: name, entity_type, description, attributes (JSONB), sources
    - graph_relationships: source_name, target_name, relationship_type, attributes (JSONB)

    Uses LLM (via chat_sync) for entity/relationship extraction.
    """

    def __init__(self, connection_string: str, generator):
        self.connection_string = connection_string
        self.generator = generator
        self._conn = None
        self._ensure_tables()

    def _get_connection(self):
        """Get or create a persistent Postgres connection with ping check."""
        try:
            if self._conn is not None and not self._conn.closed:
                # Ping the connection to ensure it's alive
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self._conn
        except psycopg2.OperationalError:
            logger.warning("PostgresGraphStore connection dropped. Reconnecting...")
            self._conn = None
            
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.connection_string)
        return self._conn

    def _ensure_tables(self):
        """Create graph tables and indexes if they don't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS graph_entities (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        entity_type TEXT,
                        description TEXT,
                        attributes JSONB DEFAULT '{}',
                        sources TEXT[],
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(name, entity_type)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS graph_relationships (
                        id SERIAL PRIMARY KEY,
                        source_name TEXT NOT NULL,
                        target_name TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        attributes JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(source_name, target_name, relationship_type)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_entities_name ON graph_entities(name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_rels_source ON graph_relationships(source_name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_rels_target ON graph_relationships(target_name)")
                conn.commit()
        except Exception as e:
            logger.error(f"PostgresGraphStore table setup failed: {e}")

    async def add_episode(
        self, text: str, source_metadata: Dict[str, Any], reference_time: str = None
    ) -> Dict[str, Any]:
        """Extracts entities and relationships via LLM and stores in Postgres."""
        prompt = f"""Extract entities and relationships from this text.
        Text: {text[:4000]}
        Return ONLY valid JSON:
        {{"entities": [{{"name": "EntityName", "type": "EntityType", "description": "Brief description"}}],
         "relationships": [{{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE"}}]}}"""

        try:
            response = chat_sync(
                self.generator,
                system="You are an exact data extraction system.",
                user=prompt,
            )
            # Use regex to robustly find JSON object
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response.")
            data = json.loads(match.group(0))
        except Exception as e:
            logger.error(f"PostgresGraph extraction failed: {e}")
            return {"status": "error", "error": str(e)}

        conn = self._get_connection()
        source = source_metadata.get("source", "unknown")

        try:
            with conn.cursor() as cur:
                # Insert Entities
                for entity in data.get("entities", []):
                    if not entity.get("name"):
                        continue
                    cur.execute("""
                        INSERT INTO graph_entities (name, entity_type, description, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (name, entity_type) DO UPDATE SET 
                            description = COALESCE(EXCLUDED.description, graph_entities.description),
                            sources = array_cat(graph_entities.sources, EXCLUDED.sources)
                    """, (
                        entity['name'],
                        entity.get('type', 'Unknown'),
                        entity.get('description', ''),
                        [source],
                    ))

                # Insert Relationships
                for rel in data.get("relationships", []):
                    if not rel.get("source") or not rel.get("target"):
                        continue
                    cur.execute("""
                        INSERT INTO graph_relationships (source_name, target_name, relationship_type, attributes)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_name, target_name, relationship_type) DO NOTHING
                    """, (
                        rel['source'],
                        rel['target'],
                        rel.get('type', 'RELATED_TO'),
                        json.dumps({"source": source}),
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"PostgresGraph insert failed: {e}")
            return {"status": "error", "error": str(e)}

        return {"status": "success"}

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Basic keyword/regex graph retrieval."""
        # Ask LLM to extract keywords from query
        kwd_prompt = (
            f"Extract 2 main entity names from this query as a comma separated list. "
            f"Query: {query}"
        )
        try:
            keywords = [
                k.strip()
                for k in chat_sync(self.generator, "", kwd_prompt).split(",")
            ]
        except Exception:
            keywords = query.split()[:2]

        conn = self._get_connection()
        results = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for kwd in keywords:
                    cur.execute("""
                        SELECT source_name, relationship_type, target_name 
                        FROM graph_relationships 
                        WHERE source_name ILIKE %s OR target_name ILIKE %s LIMIT %s
                    """, (f"%{kwd}%", f"%{kwd}%", num_results))

                    for row in cur.fetchall():
                        results.append({
                            "content": f"{row['source_name']} -> {row['relationship_type']} -> {row['target_name']}",
                            "source_node": row['source_name'],
                            "target_node": row['target_name'],
                        })
        except Exception as e:
            logger.error(f"PostgresGraph search failed: {e}")

        return results[:num_results]

    async def get_entity_neighborhood(
        self, entity_name: str, depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Retrieve structural neighborhood for a given entity."""
        conn = self._get_connection()
        results = []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Using a basic 1-hop query for this fallback implementation.
                cur.execute("""
                    SELECT source_name, relationship_type, target_name 
                    FROM graph_relationships 
                    WHERE source_name = %s OR target_name = %s LIMIT 20
                """, (entity_name, entity_name))

                for row in cur.fetchall():
                    results.append({
                        "content": f"{row['source_name']} -> {row['relationship_type']} -> {row['target_name']}",
                        "source_node": row['source_name'],
                        "target_node": row['target_name'],
                    })
        except Exception as e:
            logger.error(f"PostgresGraph get_entity_neighborhood failed: {e}")
            
        return results

    def is_document_ingested(self, source: str) -> bool:
        """Check if a document has already been ingested by looking at entity sources."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM graph_entities WHERE %s = ANY(sources) LIMIT 1",
                    (source,),
                )
                return bool(cur.fetchone())
        except Exception as e:
            logger.error(f"PostgresGraph ingestion check failed: {e}")
            return False

    async def close(self):
        """Close the persistent Postgres connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None
        logger.info("PostgresGraphStore closed.")
