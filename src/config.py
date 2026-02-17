from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    """Application settings — PostgreSQL + Vector RAG only"""

    # PostgreSQL Configuration
    postgres_uri: str = Field(
        default="postgresql://postgres:password@localhost:5432/rag_system",
        alias="POSTGRES_URI",
    )
    postgres_vector_table: str = Field(
        default="chunks", alias="POSTGRES_VECTOR_TABLE"
    )
    postgres_summary_table: str = Field(
        default="universal_summaries", alias="POSTGRES_SUMMARY_TABLE"
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    ollama_vision_model: str = Field(default="llava", alias="OLLAMA_VISION_MODEL")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="qllama/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )
    ollama_num_ctx: int = Field(default=8192, alias="OLLAMA_NUM_CTX")

    # Groq Configuration
    groq_api_keys: List[str] = Field(default=[], alias="GROQ_API_KEYS")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_batch_buffer: float = Field(default=1.5, alias="GROQ_BATCH_BUFFER")
    groq_batch_size: int = Field(default=10, alias="GROQ_BATCH_SIZE")
    groq_batch_cooldown: float = Field(default=30.0, alias="GROQ_BATCH_COOLDOWN")

    # Processing Configuration
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    semantic_threshold: float = Field(default=0.7, alias="SEMANTIC_THRESHOLD")

    # Retrieval Configuration
    vector_top_k: int = Field(default=10, alias="VECTOR_TOP_K")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    max_agent_iterations: int = Field(default=5, alias="MAX_AGENT_ITERATIONS")

    # File Paths
    parsed_docs_dir: str = Field(default="./parsed_docs", alias="PARSED_DOCS_DIR")

    # Docling Configuration
    docling_model_path: str = Field(default="", alias="DOCLING_MODEL_PATH")

    # ─── Enhancement 1: Hybrid Search ────────────────────────────────
    hybrid_search_enabled: bool = Field(default=False, alias="HYBRID_SEARCH_ENABLED")
    hybrid_vector_weight: float = Field(default=0.7, alias="HYBRID_VECTOR_WEIGHT")
    hybrid_bm25_weight: float = Field(default=0.3, alias="HYBRID_BM25_WEIGHT")
    hybrid_retrieval_k: int = Field(default=20, alias="HYBRID_RETRIEVAL_K")

    # ─── Enhancement 2: Contextual Retrieval ─────────────────────────
    contextual_retrieval_enabled: bool = Field(
        default=False, alias="CONTEXTUAL_RETRIEVAL_ENABLED"
    )
    context_batch_size: int = Field(default=5, alias="CONTEXT_BATCH_SIZE")

    # ─── Enhancement 3: Hierarchical Chunking ────────────────────────
    use_hierarchical_chunking: bool = Field(
        default=False, alias="USE_HIERARCHICAL_CHUNKING"
    )
    parent_chunk_size: int = Field(default=1000, alias="PARENT_CHUNK_SIZE")
    child_chunk_size: int = Field(default=300, alias="CHILD_CHUNK_SIZE")
    return_parent_context: bool = Field(
        default=True, alias="RETURN_PARENT_CONTEXT"
    )

    # ─── Enhancement 4: Query Routing ────────────────────────────────
    query_routing_enabled: bool = Field(
        default=False, alias="QUERY_ROUTING_ENABLED"
    )

    # ─── Enhancement 7: Caching ──────────────────────────────────────
    cache_enabled: bool = Field(default=False, alias="CACHE_ENABLED")
    cache_retrieval_capacity: int = Field(
        default=200, alias="CACHE_RETRIEVAL_CAPACITY"
    )
    cache_retrieval_ttl: float = Field(default=1800, alias="CACHE_RETRIEVAL_TTL")
    cache_embedding_capacity: int = Field(
        default=500, alias="CACHE_EMBEDDING_CAPACITY"
    )
    cache_embedding_ttl: float = Field(default=7200, alias="CACHE_EMBEDDING_TTL")

    # ─── Enhancement 8: Fallback Handling ────────────────────────────
    fallback_enabled: bool = Field(default=False, alias="FALLBACK_ENABLED")
    fallback_min_score: float = Field(default=0.3, alias="FALLBACK_MIN_SCORE")
    fallback_min_results: int = Field(default=2, alias="FALLBACK_MIN_RESULTS")

    # ─── Enhancement 9: Monitoring ───────────────────────────────────
    monitoring_enabled: bool = Field(default=False, alias="MONITORING_ENABLED")
    metrics_max_history: int = Field(default=1000, alias="METRICS_MAX_HISTORY")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
