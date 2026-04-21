# RAG3 Architecture Upgrade: Feature Porting & Enterprise Hardening Plan

**To the AI Coding Agent:**
This document is your standalone, comprehensive blueprint for upgrading the target repository (Project 1: RAG3 / Agentic Haystack RAG). You do not need the legacy codebase. This document contains the exact code snippets, SQL queries, prompt templates, and architectural patterns required to graft 5 missing enterprise features back into the RAG3 repository.

Execute these updates methodically, ensuring you respect the existing Agentic Workflow and Haystack v2 architectural paradigms present in the repository.

---

## Feature 1: Metadata Filtering in Postgres Hybrid Search

### Objective
The current `PostgresVectorStore.hybrid_search` in RAG3 lacks metadata filtering. You must implement a dynamic SQL `WHERE` clause builder for the `meta` (JSONB) column and apply it to both the BM25 and Vector sub-queries before executing Reciprocal Rank Fusion (RRF).

### Target Files
*   `src/storage/postgres/vector_store.py`
*   `src/storage/base.py`

### Implementation Steps

1. **Update the Interface (`src/storage/base.py`)**:
   Modify `VectorStoreInterface.search` and any hybrid search signatures to accept an optional `filters: Dict[str, Any] = None` parameter.

2. **Implement the SQL Filter Builder (`src/storage/postgres/vector_store.py`)**:
   Add this exact helper method to the `PostgresVectorStore` class to handle Haystack's `meta` JSONB column:

   ```python
   def _build_metadata_filter(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
       """Builds a SQL WHERE clause and parameters for JSONB meta column."""
       if not filters:
           return "1=1", []

       where_conditions = []
       params = []

       for key, value in filters.items():
           if isinstance(value, dict):
               if "$gte" in value:
                   where_conditions.append(f"(meta->>'{key}')::text >= %s")
                   params.append(str(value["$gte"]))
               if "$lte" in value:
                   where_conditions.append(f"(meta->>'{key}')::text <= %s")
                   params.append(str(value["$lte"]))
               if "$in" in value:
                   placeholders = ",".join(["%s"] * len(value["$in"]))
                   where_conditions.append(f"(meta->>'{key}') IN ({placeholders})")
                   params.extend(value["$in"])
           elif isinstance(value, list):
               placeholders = ",".join(["%s"] * len(value))
               where_conditions.append(f"(meta->>'{key}') IN ({placeholders})")
               params.extend(value)
           else:
               where_conditions.append(f"(meta->>'{key}') = %s")
               params.append(str(value))

       where_clause = " AND ".join(where_conditions)
       return where_clause, params
   ```

3. **Refactor Hybrid Search SQL**:
   Add a `hybrid_search_with_filter` method. Notice that Haystack's `pgvector` implementation uses `content` for the text and `meta` for the JSONB metadata.

   ```python
   def hybrid_search_with_filter(
       self,
       query: str,
       top_k: int = 5,
       filters: Dict[str, Any] = None,
       vector_weight: float = 0.7,
       bm25_weight: float = 0.3,
       retrieval_k: int = 20,
   ) -> List[Dict[str, Any]]:
       
       self._ensure_fts_index()
       where_clause, params = self._build_metadata_filter(filters)
       
       # Generate Query Embedding
       query_embedding = self._embedder.run(text=query)["embedding"]
       
       conn = psycopg2.connect(self.connection_string)
       try:
           with conn.cursor(cursor_factory=RealDictCursor) as cur:
               # 1. Filtered Vector Search
               cur.execute(f"""
                   SELECT 
                       content,
                       1 - (embedding <=> %s::vector) as vector_score,
                       meta as metadata
                   FROM {self.table_name}
                   WHERE {where_clause}
                   ORDER BY vector_score DESC
                   LIMIT %s
               """, [query_embedding] + params + [retrieval_k])
               
               vector_results = [
                   {
                       "text": row['content'],
                       "score": float(row['vector_score']),
                       "metadata": row['metadata'] or {}
                   }
                   for row in cur.fetchall()
               ]

               # 2. Filtered BM25 Search
               cur.execute(f"""
                   SELECT 
                       content,
                       ts_rank_cd(content_tsv, plainto_tsquery('english', %s), 32) as bm25_score,
                       meta as metadata
                   FROM {self.table_name}
                   WHERE content_tsv @@ plainto_tsquery('english', %s)
                     AND {where_clause}
                   ORDER BY bm25_score DESC
                   LIMIT %s
               """, [query, query] + params + [retrieval_k])
               
               bm25_results = [
                   (row['content'], float(row['bm25_score']), row['metadata'] or {})
                   for row in cur.fetchall()
               ]
       finally:
           conn.close()

       # 3. Fuse Results
       return self._reciprocal_rank_fusion(
           vector_results, bm25_results, top_k, vector_weight, bm25_weight
       )
   ```

4. **Update `VectorSearchTool`**:
   In `src/retrieval/tools/vector_tool.py`, update `search()` to pass the `filters` dict down to `hybrid_search_with_filter` instead of ignoring it.

---

## Feature 2: Lightweight Postgres Graph Store

### Objective
Neo4j is heavy. Implement a fallback `PostgresGraphStore` that adheres to the `GraphStoreInterface` and stores Knowledge Graph entities/relationships in standard Postgres tables using `JSONB`.

### Target Files
*   `src/storage/postgres/graph_store.py` (Create new)
*   `src/storage/postgres/__init__.py`
*   `src/main.py` (Fallback logic)

### Implementation Steps

1. **Create `src/storage/postgres/graph_store.py`**:
   Implement `PostgresGraphStore` using the following exact logic and schema:

   ```python
   import json
   from typing import List, Dict, Any
   import psycopg2
   from psycopg2.extras import RealDictCursor
   from ..base import GraphStoreInterface
   from ...utils.llm import chat_sync
   import logging

   logger = logging.getLogger(__name__)

   class PostgresGraphStore(GraphStoreInterface):
       def __init__(self, connection_string: str, generator):
           self.connection_string = connection_string
           self.generator = generator
           self._conn = None
           self._ensure_tables()

       def _get_connection(self):
           if self._conn is None or self._conn.closed:
               self._conn = psycopg2.connect(self.connection_string)
           return self._conn

       def _ensure_tables(self):
           conn = self._get_connection()
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

       async def add_episode(self, text: str, source_metadata: Dict[str, Any], reference_time: str = None) -> Dict[str, Any]:
           """Extracts entities and relationships via LLM and stores in Postgres."""
           prompt = f"""Extract entities and relationships from this text.
           Text: {text[:4000]}
           Return ONLY valid JSON:
           {{"entities": [{{"name": "EntityName", "type": "EntityType", "description": "Brief description"}}],
            "relationships": [{{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE"}}]}}"""
           
           try:
               response = chat_sync(self.generator, system="You are an exact data extraction system.", user=prompt)
               # Strip markdown blocks
               if "```" in response:
                   response = response.split("```")[1]
                   if response.startswith("json"): response = response[4:]
               data = json.loads(response.strip())
           except Exception as e:
               logger.error(f"PostgresGraph extraction failed: {e}")
               return {"status": "error"}

           conn = self._get_connection()
           source = source_metadata.get("source", "unknown")
           
           with conn.cursor() as cur:
               # Insert Entities
               for e in data.get("entities", []):
                   if not e.get("name"): continue
                   cur.execute("""
                       INSERT INTO graph_entities (name, entity_type, description, sources)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (name, entity_type) DO UPDATE SET 
                           description = COALESCE(EXCLUDED.description, graph_entities.description),
                           sources = array_cat(graph_entities.sources, EXCLUDED.sources)
                   """, (e['name'], e.get('type', 'Unknown'), e.get('description', ''), [source]))
               
               # Insert Relationships
               for r in data.get("relationships", []):
                   if not r.get("source") or not r.get("target"): continue
                   cur.execute("""
                       INSERT INTO graph_relationships (source_name, target_name, relationship_type, attributes)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (source_name, target_name, relationship_type) DO NOTHING
                   """, (r['source'], r['target'], r.get('type', 'RELATED_TO'), json.dumps({"source": source})))
               conn.commit()

           return {"status": "success"}

       async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
           """Basic Keyword/Regex graph retrieval."""
           # Ask LLM to extract keywords from query
           kwd_prompt = f"Extract 2 main entity names from this query as a comma separated list. Query: {query}"
           try:
               keywords = [k.strip() for k in chat_sync(self.generator, "", kwd_prompt).split(",")]
           except:
               keywords = query.split()[:2]

           conn = self._get_connection()
           results = []
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
                           "target_node": row['target_name']
                       })
           return results[:num_results]
           
       def is_document_ingested(self, source: str) -> bool:
           conn = self._get_connection()
           with conn.cursor() as cur:
               cur.execute("SELECT 1 FROM graph_entities WHERE %s = ANY(sources) LIMIT 1", (source,))
               return bool(cur.fetchone())

       async def close(self):
           if self._conn:
               self._conn.close()
   ```

2. **Update `src/main.py` Initialization**:
   In `RAGSystem._initialize_storage()`, if `GRAPH_RAG_ENABLED` is true, attempt to initialize `Neo4jGraphStore`. If it fails or Neo4j credentials are empty, fallback to the new `PostgresGraphStore`.

---

## Feature 3: Explicit LangSmith Tracing Decorators

### Objective
Restore full code-level observability by injecting `@traceable` decorators onto key orchestration methods.

### Target Files
*   `src/main.py`
*   `src/agents/orchestrator.py`
*   `src/retrieval/tools/vector_tool.py`
*   `src/agents/synthesizer.py`

### Implementation Steps

1. **Add Safe Import Block** to the top of all target files:
   ```python
   try:
       from langsmith import traceable
   except ImportError:
       def traceable(*args, **kwargs):
           def decorator(fn): return fn
           return args[0] if args and callable(args[0]) else decorator
   ```

2. **Apply Decorators**:
   *   `src/main.py`: Add `@traceable(name="RAGSystem.ingest_document", run_type="chain")` above the `ingest_document` method.
   *   `src/agents/orchestrator.py`: 
       *   Add `@traceable(name="Orchestrator.run", run_type="chain")` above `run()`.
       *   Add `@traceable(name="Orchestrator.hybrid_fusion", run_type="retriever")` above `_run_hybrid()`.
   *   `src/retrieval/tools/vector_tool.py`: Add `@traceable(name="VectorSearchTool.search", run_type="retriever")` above the `search()` method.
   *   `src/agents/synthesizer.py`: Add `@traceable(name="Synthesizer.format", run_type="prompt")` above the `synthesize()` method.

---

## Feature 4: Temporal Search Logic

### Objective
Improve Graph RAG context retrieval by giving the LLM the ability to resolve relative time ("last year", "in Q2") to absolute timestamps before querying the Graph Database.

### Target Files
*   `src/retrieval/tools/graph_search_tool.py`

### Implementation Steps

1. **Inject Temporal Resolution** into `GraphSearchTool`:
   Modify `src/retrieval/tools/graph_search_tool.py` to accept a `generator` (LLM) in its `__init__` (you will need to update `src/main.py` to pass `self.generator` into `GraphSearchTool()`).

2. **Add `_resolve_temporal_context`**:
   ```python
   import json
   from datetime import datetime
   from ...utils.llm import chat_sync

   def _resolve_temporal_context(self, query: str) -> str:
       """Identifies temporal references and appends absolute dates to the query."""
       current_date = datetime.now().strftime("%Y-%m-%d")
       prompt = f"""Current date: {current_date}
       Extract temporal references from this question and convert to specific values.
       Question: {query}
       Examples: "last year" -> ["2023"], "in 2020" -> ["2020"]
       Return ONLY a JSON array of strings: ["term1", "term2"]"""
       
       try:
           response = chat_sync(self.generator, system="You are a temporal extraction expert.", user=prompt)
           if "```" in response:
               response = response.split("```")[1]
               if response.startswith("json"): response = response[4:]
           terms = json.loads(response.strip())
           
           if isinstance(terms, list) and terms:
               return query + " | Time context: " + ", ".join(terms)
       except Exception as e:
           logger.debug(f"Temporal resolution failed: {e}")
       return query
   ```

3. **Update `search()` method**:
   In `GraphSearchTool.search()`, pass the query through `_resolve_temporal_context(query)` *before* sending it to `self.graph_store.search()`.

---

## Feature 5: Local CrossEncoder Fallback for Reranking

### Objective
Use `sentence-transformers` for zero-latency, in-memory reranking as a high-performance alternative to hitting the Ollama API for scoring document chunks.

### Target Files
*   `src/config.py`
*   `src/retrieval/strategies/reranking.py`

### Implementation Steps

1. **Update `Settings`** in `src/config.py`:
   Add `use_local_reranker: bool = Field(default=False, alias="USE_LOCAL_RERANKER")`

2. **Implement `LocalCrossEncoderRanker`** in `src/retrieval/strategies/reranking.py`:

   ```python
   from haystack import Document, component
   from typing import List, Optional
   from ...config import settings

   @component
   class LocalCrossEncoderRanker:
       """In-memory reranking using sentence-transformers."""
       
       def __init__(self, model: str = None, top_k: int = 5):
           self.model_name = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
           self.default_top_k = top_k
           self._model = None

       def warm_up(self):
           if self._model is None:
               from sentence_transformers import CrossEncoder
               self._model = CrossEncoder(self.model_name)

       @component.output_types(documents=List[Document])
       def run(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> dict:
           if not documents: return {"documents": []}
           self.warm_up()
           
           top_k = top_k or self.default_top_k
           pairs = [[query, doc.content or ""] for doc in documents]
           scores = self._model.predict(pairs)
           
           for doc, score in zip(documents, scores):
               doc.score = float(score)
               
           documents.sort(key=lambda x: x.score, reverse=True)
           return {"documents": documents[:top_k]}
   ```

3. **Update `RerankingStrategy`** to toggle based on settings:
   In `src/retrieval/strategies/reranking.py`, modify the `__init__` of `RerankingStrategy`:

   ```python
   class RerankingStrategy:
       def __init__(self, model: str = None, url: str = None):
           if getattr(settings, "use_local_reranker", False):
               self.ranker = LocalCrossEncoderRanker(model=model)
           else:
               self.ranker = OllamaRanker(model=model, url=url)
   ```

---

## Integration Checklist for the Coding Agent

1. **Dependencies**: Note that `sentence-transformers` is already in the `pyproject.toml` from the legacy version, so it will resolve perfectly when adding Feature 5.
2. **Haystack Parity**: Ensure that any modified interfaces continue to pass Haystack `Document` objects properly, especially around the reranker where Haystack expects `doc.content` and `doc.meta`.
3. **LLM `chat_sync` utility**: Utilize `src/utils/llm.py`'s `chat_sync(generator, system, user)` function for all LLM calls in the new Graph Store and Temporal tools. This guarantees that Groq rate-limiting and Haystack integration work seamlessly without breaking the abstraction.

End of Document. Execute the code changes exactly as specified.