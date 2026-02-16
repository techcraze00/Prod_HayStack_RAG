# RAG3

Production Retrieval-Augmented Generation system built on [Haystack](https://haystack.deepset.ai/) (v0.1.0).

PostgreSQL-only vector RAG with semantic chunking, hybrid search, agentic retrieval, and Ollama-native reranking.

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

## Usage

### Ingest Documents

```bash
# Ingest a single file
python -m src.main ingest path/to/document.pdf

# Ingest all files in a directory
python -m src.main ingest path/to/docs/

# Force re-parse (ignore cached JSON)
python -m src.main ingest path/to/document.pdf --force-reparse

# Force regenerate summaries
python -m src.main ingest path/to/document.pdf --force-regenerate-summary

# Use Groq instead of local Ollama
python -m src.main ingest path/to/document.pdf --groq
```

Supported file types: `.pdf`, `.docx`, `.doc`, `.txt`, `.md`

Parsed documents are cached as JSON in `parsed_docs/` to avoid re-parsing on subsequent runs.

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

## Optional Enhancements

All enhancements are disabled by default. Enable them in `.env`:

| Enhancement | Flag | Description |
|---|---|---|
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
│   ├── base.py                      # Abstract interfaces
│   └── postgres/
│       ├── vector_store.py          # PgvectorDocumentStore + hybrid search + RRF
│       └── summary_store.py         # Summary index with pgvector
├── retrieval/
│   ├── agent.py                     # Haystack Agent with ReAct pattern
│   ├── cache.py                     # Multi-level LRU cache
│   ├── fallback.py                  # Progressive fallback strategies
│   ├── tools/
│   │   └── vector_tool.py           # Vector/hybrid search as Haystack Tool
│   └── strategies/
│       ├── reranking.py             # OllamaRanker (bge-reranker-v2-m3)
│       ├── query_expansion.py       # LLM-based query reformulation
│       ├── self_reflection.py       # Retrieval quality grading + refinement
│       ├── query_router.py          # Query classification and strategy routing
│       └── summary_index.py         # Full/topic/section summary generation
├── evaluation/
│   ├── metrics.py                   # RAGAS + simple fallback metrics
│   └── datasets.py                  # Evaluation dataset management
└── monitoring/
    ├── metrics.py                   # MetricsCollector (P50/P90/P99 latencies)
    └── logger.py                    # Structured JSON logging
```
