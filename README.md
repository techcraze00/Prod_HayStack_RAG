# RAG3

Production Retrieval-Augmented Generation system built with an **Agentic Workflow Architecture** on [Haystack](https://haystack.deepset.ai/) (v2.x), with a full **Graph RAG** pipeline powered by Neo4j + Graphiti.

Features:
- **Multi-Tier Memory**: Episodic vector persistence, conversational sliding-window RAM, and graph-backed episodic memory via Neo4j.
- **Advanced Semantic Router**: 4-intent classification (General, Vector, Graph, Hybrid) via parallel LLM and Vector-grounding.
- **Agentic Supervisor**: Orchestrator-managed Worker nodes (`VectorAgent`, `GeneralAgent`, `GraphAgent`, `HybridFusion`) for optimized inference.
- **PostgreSQL-native**: pgvector vector RAG with semantic chunking, hybrid search, and Ollama reranking.
- **Graph RAG**: Neo4j knowledge graph with entity extraction via Graphiti, relationship-aware retrieval, and hybrid vector+graph fusion.

## Prerequisites

- **Python** >= 3.11
- **[uv](https://docs.astral.sh/uv/)** (package manager)
- **[Ollama](https://ollama.com/)** running locally with the following models pulled:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ollama pull llava
  ollama pull qllama/bge-reranker-v2-m3
  ```
- **PostgreSQL** with the [pgvector](https://github.com/pgvector/pgvector) extension enabled:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- **Neo4j 5.21+** *(optional, for Graph RAG)* — install via [Neo4j Desktop](https://neo4j.com/download/) or Docker:
  ```bash
  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/rag3password neo4j:5
  ```

## Installation

```bash
cd RAG3

# Install dependencies
uv sync

# With evaluation support (RAGAS)
uv pip install rag3[eval]
```

## Configuration

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

At minimum, set your PostgreSQL connection string:

```env
POSTGRES_URI=postgresql://postgres:password@localhost:5432/rag_system
```

Ollama defaults (`http://localhost:11434`, `llama3.2`, `nomic-embed-text`) work out of the box if Ollama is running locally.

### Optional: Groq API (cloud LLM)

For faster inference via Groq, add your API key(s):

```env
GROQ_API_KEYS=["gsk_abc123...", "gsk_def456..."]
GROQ_MODEL=llama-3.3-70b-versatile
```

Multiple keys enable automatic rotation when rate limits are hit.

### Optional: Graph RAG (Neo4j + Graphiti)

Enable the Graph RAG pipeline by setting these in `.env`:

```env
# ── Graph RAG ──────────────────────────────────────────
GRAPH_RAG_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=rag3password
GRAPH_BUILDER_MODEL=llama3.2
GRAPH_SEARCH_DEPTH=2
GRAPH_SEARCH_RESULTS=10
HYBRID_GRAPH_WEIGHT=0.4
```

When `GRAPH_RAG_ENABLED=false` (default), the graph pipeline is completely inactive — no Neo4j connection is attempted.

## Usage

### Ingest Documents

```bash
# Ingest a single file (vector + graph if GRAPH_RAG_ENABLED)
python -m src.main ingest path/to/document.pdf

# Ingest all files in a directory
python -m src.main ingest path/to/docs/

# Vector store only (skip graph)
python -m src.main ingest path/to/document.pdf --vector

# Graph store only (skip vector)
python -m src.main ingest path/to/document.pdf --graph

# Force re-parse (ignore cached JSON)
python -m src.main ingest path/to/document.pdf --force-reparse

# Force regenerate summaries
python -m src.main ingest path/to/document.pdf --force-regenerate-summary

# Use Groq instead of local Ollama
python -m src.main ingest path/to/document.pdf --groq
```

Supported file types: `.pdf`, `.docx`, `.doc`, `.txt`, `.md`

Parsed documents are cached as JSON in `parsed_docs/` to avoid re-parsing on subsequent runs.

**Ingestion state tracking**: The system auto-detects which stores already have a document and skips them unless `--force-reparse` is used.

### Resuming Failed Ingestion

During ingestion, chunks (with embeddings) are automatically saved as a JSON checkpoint **before** being written to PostgreSQL. If the database write fails mid-way, you can resume without re-parsing or re-embedding:

```bash
# Resume from saved chunks (skips parsing, chunking, and embedding)
python -m src.main ingest path/to/document.pdf --chunks-only

# Also works with Groq
python -m src.main ingest path/to/document.pdf --groq --chunks-only
```

Checkpoint files are saved at `parsed_docs/{filename}_chunks.json`. The `--chunks-only` flag loads the saved chunks and writes them directly to PostgreSQL in batches of 50, with per-batch error handling so partial progress is preserved even if some batches fail.

### Query

```bash
# Single question
python -m src.main query "What are the key provisions?"

# Interactive mode
python -m src.main query -i

# Use Groq for faster responses
python -m src.main query "What is X?" --groq
```

## Core Architecture & Agentic Workflow

RAG3 replaces the monolithic generation loop with a multi-agent **Agentic Workflow Architecture** supporting 4 intent types:

<div align="center" style="padding: 30px;">
  <img src="./public/architecture-neon.svg" alt="RAG Architecture Diagram" width="100%">
</div>

### Intent Routing

The `IntentRouter` uses a 3-tier hierarchy (Regex → Weighted Keywords → LLM Arbiter) to classify queries:

| Intent | Trigger | Worker |
|---|---|---|
| `GeneralChat` | Greetings, trivia, small talk | `GeneralAgent` |
| `VectorRetrieval` | Factual queries answerable from documents | `VectorAgent` |
| `GraphRetrieval` | Relationship / dependency / multi-hop queries | `GraphAgent` |
| `HybridRetrieval` | Queries needing both factual context AND relational reasoning | `GraphVectorFusion` |

### Memory System
RAG3 integrates deep conversational memory:
- **Sliding Window:** Temporarily stores the last N messages in RAM.
- **Episodic Persistence (FAISS):** Once the sliding window fills up, older segments are summarized by an LLM and stored as vectors in a local FAISS database for long-term historical recall.
- **Graph Episodic Memory (Neo4j):** When Graph RAG is enabled, the knowledge graph is also searched for entity-rich context related to the current query, augmenting FAISS-based recall.

## Optional Enhancements

All enhancements are disabled by default. Enable them in `.env`:

| Enhancement | Flag | Description |
|---|---|---|
| **Graph RAG** | `GRAPH_RAG_ENABLED=true` | Neo4j knowledge graph with entity extraction, relationship-aware retrieval, and hybrid fusion |
| Hybrid Search | `HYBRID_SEARCH_ENABLED=true` | Combines vector similarity + BM25 keyword search via RRF |
| Contextual Retrieval | `CONTEXTUAL_RETRIEVAL_ENABLED=true` | Prepends LLM-generated context to each chunk before embedding |
| Hierarchical Chunking | `USE_HIERARCHICAL_CHUNKING=true` | Parent/child chunks — retrieve small, send large context to LLM |
| Query Routing | `QUERY_ROUTING_ENABLED=true` | LLM classifies query type and tunes retrieval strategy |
| Caching | `CACHE_ENABLED=true` | LRU cache with TTL for retrieval results and embeddings |
| Fallback Handling | `FALLBACK_ENABLED=true` | Progressive fallback cascade when retrieval quality is low |
| Monitoring | `MONITORING_ENABLED=true` | Structured logging and latency metrics (P50/P90/P99) |

## Project Structure

```
src/
├── config.py                        # Pydantic Settings (all config via .env)
├── main.py                          # CLI entry point + RAGSystem orchestrator
├── agents/                          # Agentic Workflow Components
│   ├── orchestrator.py              # Central supervisor (Vector + Graph + Hybrid dispatch)
│   ├── router.py                    # 4-Intent Semantic Router (General/Vector/Graph/Hybrid)
│   ├── synthesizer.py               # Output formatter with intent-specific metadata
│   └── workers/
│       ├── general_agent.py         # Fast conversational LLM wrapper
│       ├── vector_agent.py          # Vector query wrapper for AdvancedRAGAgent
│       └── graph_agent.py           # LangGraph-based graph retrieval agent
├── memory/                          # Conversational State
│   ├── memory_tools.py              # Context builders (FAISS + Graph memory injection)
│   ├── summarizer.py                # Turn-based historic summarizer
│   └── vector_store.py              # FAISS Episodic/Archival + GraphMemoryManager
├── utils/
│   ├── llm.py                       # chat_sync() helper for Haystack generators
│   └── groq_client.py               # RotatableGroqGenerator (key rotation + rate limits)
├── ingestion/
│   ├── parser.py                    # DocumentParser (unstructured hi_res)
│   ├── vision.py                    # VisionProcessor (OllamaChatGenerator + LLaVA)
│   ├── tables.py                    # TableReformatter (HTML → natural language)
│   ├── semantic_chunker.py          # Embedding-based semantic chunking
│   ├── contextual_chunker.py        # Context-enriched chunks
│   ├── hierarchical_chunker.py      # Parent/child chunk hierarchy
│   └── embedder.py                  # OllamaTextEmbedder wrapper with caching
├── storage/
│   ├── base.py                      # Abstract interfaces (Vector, Summary, Graph)
│   ├── postgres/
│   │   ├── vector_store.py          # PgvectorDocumentStore + hybrid search + RRF
│   │   └── summary_store.py         # Summary index with pgvector
│   └── graph/
│       └── neo4j_store.py           # Neo4j + Graphiti wrapper (episodes, search, traversal)
├── retrieval/
│   ├── agent.py                     # Haystack Agent with ReAct pattern
│   ├── cache.py                     # Multi-level LRU cache
│   ├── fallback.py                  # Progressive fallback strategies
│   ├── session.py                   # RAGSession (sliding window + FAISS + graph memory)
│   ├── tools/
│   │   ├── vector_tool.py           # Vector/hybrid search as Haystack Tool
│   │   └── graph_search_tool.py     # Graph search tool wrapping Neo4jGraphStore
│   └── strategies/
│       ├── reranking.py             # OllamaRanker (bge-reranker-v2-m3)
│       ├── query_expansion.py       # LLM-based query reformulation
│       ├── self_reflection.py       # Retrieval quality grading + refinement
│       ├── query_router.py          # Query classification and strategy routing
│       ├── summary_index.py         # Full/topic/section summary generation
│       └── graph_fusion.py          # Hybrid Vector+Graph fusion strategy
├── evaluation/
│   ├── metrics.py                   # RAGAS + simple fallback metrics
│   └── datasets.py                  # Evaluation dataset management
└── monitoring/
    ├── metrics.py                   # MetricsCollector (P50/P90/P99 latencies)
    └── logger.py                    # Structured JSON logging
```
