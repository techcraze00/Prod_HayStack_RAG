"""
RAG3 — Production RAG System built on Haystack.

CLI entry point and RAGSystem orchestrator class.
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

from src.config import settings
from src.ingestion.unstructured_parser import DocumentParser, ParsedElement
from src.ingestion.vision import VisionProcessor
from src.ingestion.tables import TableReformatter
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.embedder import Embedder
from src.retrieval.tools import VectorSearchTool
from src.retrieval.agent import AdvancedRAGAgent
from src.retrieval.strategies.summary_index import SummaryIndexStrategy
from src.storage.postgres import PostgresVectorStore, PostgresSummaryStore

console = Console()


class RAGSystem:
    """Main RAG system orchestrator"""

    def __init__(self, use_groq: bool = False, use_docling: bool = False):
        self.settings = settings
        self.use_groq = use_groq
        self.use_docling = use_docling
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components based on config."""
        # Initialize LLM (Haystack ChatGenerator)
        if self.use_groq:
            if not settings.groq_api_keys:
                console.print("[bold red]Error:[/bold red] --groq flag used but GROQ_API_KEYS not found in .env")
                exit(1)

            console.print(f"[bold green]Using Groq API[/bold green] (Model: {settings.groq_model})")
            from src.utils.groq_client import RotatableGroqGenerator

            self.generator = RotatableGroqGenerator(
                api_keys=settings.groq_api_keys,
                model=settings.groq_model,
                buffer_time=settings.groq_batch_buffer,
                batch_size=settings.groq_batch_size,
                batch_cooldown=settings.groq_batch_cooldown,
            )
        else:
            console.print(f"[bold blue]Using Local Ollama[/bold blue] (Model: {settings.ollama_model})")
            from haystack_integrations.components.generators.ollama import OllamaChatGenerator

            self.generator = OllamaChatGenerator(
                model=settings.ollama_model,
                url=settings.ollama_base_url,
                generation_kwargs={
                    "temperature": 0.1,
                    "num_ctx": settings.ollama_num_ctx,
                },
            )
            # self.generator.warm_up()

        # Initialize cache (Enhancement 7)
        self.cache = None
        if settings.cache_enabled:
            from src.retrieval.cache import RAGCache
            self.cache = RAGCache(
                retrieval_capacity=settings.cache_retrieval_capacity,
                retrieval_ttl=settings.cache_retrieval_ttl,
                embedding_capacity=settings.cache_embedding_capacity,
                embedding_ttl=settings.cache_embedding_ttl,
            )

        # Initialize embedder
        self.embedder = Embedder(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            cache=self.cache,
        )

        # Initialize vision processor
        self.vision_processor = VisionProcessor(
            model=settings.ollama_vision_model,
            base_url=settings.ollama_base_url,
        )

        # Initialize table reformatter
        self.table_reformatter = TableReformatter(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

        # Initialize parser
        if self.use_docling:
            from src.ingestion.docling_parser import DoclingParser

            console.print("[bold green]Using Docling parser[/bold green]")
            self.parser = DoclingParser()
        else:
            self.parser = DocumentParser(
                vision_processor=self.vision_processor,
                table_reformatter=self.table_reformatter,
            )

        # Initialize chunker
        self.chunker = SemanticChunker(
            embedding_model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            similarity_threshold=settings.semantic_threshold,
            max_chunk_size=settings.chunk_size,
        )

        # Initialize contextual chunker (Enhancement 2)
        self.contextual_chunker = None
        if settings.contextual_retrieval_enabled:
            from src.ingestion.contextual_chunker import ContextualChunker
            self.contextual_chunker = ContextualChunker(
                generator=self.generator,
                embedding_model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
                similarity_threshold=settings.semantic_threshold,
                max_chunk_size=settings.chunk_size,
                context_batch_size=settings.context_batch_size,
            )

        # Initialize hierarchical chunker (Enhancement 3)
        self.hierarchical_chunker = None
        if settings.use_hierarchical_chunking:
            from src.ingestion.hierarchical_chunker import HierarchicalChunker
            self.hierarchical_chunker = HierarchicalChunker(
                embedding_model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
                parent_chunk_size=settings.parent_chunk_size,
                child_chunk_size=settings.child_chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

        # Initialize storage (PostgreSQL only)
        self._initialize_storage()

        self.summary_strategy = None
        if hasattr(self, 'summary_store') and self.summary_store:
            self.summary_strategy = SummaryIndexStrategy(self.generator, self.summary_store)

        # Initialize tools
        self.vector_tool = VectorSearchTool(
            vector_store=self.vector_store,
            embedder=self.embedder,
            cache=self.cache,
        )

    def _initialize_storage(self):
        """Initialize PostgreSQL storage."""
        console.print("[dim]Storage backend: postgres[/dim]")

        self.vector_store = PostgresVectorStore(
            connection_string=settings.postgres_uri,
            table_name=settings.postgres_vector_table,
            embedding_model=settings.ollama_embedding_model,
            ollama_base_url=settings.ollama_base_url,
        )

        self.summary_store = PostgresSummaryStore(
            connection_string=settings.postgres_uri,
            embedding_model=settings.ollama_embedding_model,
            ollama_base_url=settings.ollama_base_url,
            table_name=settings.postgres_summary_table,
        )

    def _create_agent(self) -> AdvancedRAGAgent:
        """Create agent with tool configuration."""
        return AdvancedRAGAgent(
            generator=self.generator,
            vector_tool=self.vector_tool,
            max_iterations=settings.max_agent_iterations,
        )

    def _check_summaries_exist(self, doc_id: str) -> bool:
        """Check if summaries already exist for a document."""
        try:
            results = self.summary_store.search_summaries(doc_id, top_k=1)
            return len(results) > 0
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check summary cache: {e}[/yellow]")
            return False

    def ingest_document(
        self,
        file_path: str,
        force_reparse: bool = False,
        force_regenerate_summary: bool = False,
        chunks_only: bool = False,
    ):
        """Ingest a single document with intelligent caching.

        Pipeline: Parse → Consolidate → Chunk → Save chunks JSON → Summarize → Store to PG

        If any step after chunking fails, re-run with --chunks-only to resume
        from the saved chunks JSON without re-parsing or re-embedding.
        """
        file_path = Path(file_path)

        console.print(
            f"[bold blue]Ingesting:[/bold blue] {file_path} "
            f"[dim](Vector)[/dim]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Checking cache...", total=None)

            output_dir = Path(settings.parsed_docs_dir)
            output_dir.mkdir(exist_ok=True)
            json_path = output_dir / f"{file_path.stem}_parsed.json"
            chunks_path = output_dir / f"{file_path.stem}_chunks.json"

            # ── Resume path: load pre-saved chunks ───────────────────
            if chunks_only:
                if not chunks_path.exists():
                    console.print(
                        f"[red]Error:[/red] No saved chunks found at {chunks_path}\n"
                        "[dim]Run without --chunks-only first to parse and chunk the document.[/dim]"
                    )
                    return

                progress.update(task, description="Loading saved chunks...")
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunk_dicts = json.load(f)

                console.print(
                    f"  [green]\u2713[/green] [dim]Loaded {len(chunk_dicts)} saved chunks from: {chunks_path.name}[/dim]"
                )

                progress.update(task, description="Storing in vector database...")
                result = self.vector_store.add_chunks(chunk_dicts)
                console.print(
                    f"  [green]\u2713[/green] [dim]Added {result['added']} chunks "
                    f"({result.get('failed', 0)} failed)[/dim]"
                )

                progress.update(task, completed=True, description="Done!")

                console.print(
                    f"[green]\u2713[/green] Resumed ingestion for {file_path.name}: "
                    f"{result['added']} chunks stored"
                )
                return

            # ── Full pipeline ────────────────────────────────────────
            parsed = None
            if json_path.exists() and not force_reparse:
                progress.update(task, description="Loading cached parsed JSON...")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    parsed = [
                        ParsedElement(
                            element_type=elem["element_type"],
                            content=elem["content"],
                            metadata=elem["metadata"],
                            page_number=elem["page_number"],
                        )
                        for elem in data
                    ]
                    console.print(
                        f"  [green]\u2713[/green] [dim]Loaded {len(parsed)} cached elements from: {json_path.name}[/dim]"
                    )
                except Exception as e:
                    console.print(f"  [yellow]\u26a0[/yellow] [dim]Failed to load cache, re-parsing: {e}[/dim]")
                    parsed = None

            if parsed is None:
                progress.update(task, description="Parsing document...")
                parsed = self.parser.parse(str(file_path))
                progress.update(task, description=f"Parsed {len(parsed)} elements")

                progress.update(task, description="Saving parsed JSON...")
                json_path = self.parser.save_parsed_json(parsed, str(file_path))
                console.print(f"  [dim]Saved to: {json_path}[/dim]")

            # Consolidate elements
            progress.update(task, description="Consolidating elements into sections...")
            consolidated = self.parser.consolidate_elements(parsed)
            console.print(
                f"  [green]\u2713[/green] [dim]Consolidated {len(parsed)} raw elements \u2192 "
                f"{len(consolidated)} sections[/dim]"
            )

            # Create chunks from consolidated sections
            progress.update(task, description="Creating semantic chunks...")
            chunks = []
            for element in consolidated:
                if element.content.strip():
                    element_chunks = self.chunker.chunk(
                        element.content,
                        {
                            "source": str(file_path),
                            "page": element.page_number,
                            "type": element.element_type,
                            "section": element.metadata.get("section", ""),
                        },
                    )
                    chunks.extend(element_chunks)

            progress.update(task, description=f"Created {len(chunks)} semantic chunks")

            # Save chunks as JSON checkpoint (before DB write)
            progress.update(task, description="Saving chunks checkpoint...")
            chunk_dicts = [
                {
                    "text": c.text,
                    "embedding": c.embedding,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunk_dicts, f, ensure_ascii=False)
            console.print(
                f"  [green]\u2713[/green] [dim]Chunks saved to: {chunks_path.name} "
                f"(re-run with --chunks-only to resume if storage fails)[/dim]"
            )

            # Generate summaries (with caching)
            if self.summary_strategy:
                doc_id = str(file_path)
                summaries_exist = self._check_summaries_exist(doc_id)

                if summaries_exist and not force_regenerate_summary:
                    console.print(
                        f"  [green]\u2713[/green] [dim]Summaries already exist for {file_path.name}, skipping[/dim]"
                    )
                else:
                    progress.update(
                        task,
                        description="Generating universal summaries (Full, Topic, Section)...",
                    )

                    full_text = "\n".join(
                        [e.content for e in parsed if e.element_type == "text"]
                    )
                    if full_text:
                        try:
                            self.summary_strategy.build_universal_index(
                                doc_id=doc_id,
                                full_text=full_text,
                                metadata={"source": str(file_path)},
                            )
                            console.print(
                                f"  [green]\u2713[/green] [dim]Summaries generated successfully[/dim]"
                            )
                        except Exception as e:
                            console.print(
                                f"  [yellow]\u26a0[/yellow] [dim]Summary generation failed: {e}[/dim]"
                            )

            # Vector ingestion (batched)
            progress.update(task, description="Storing in vector database...")
            result = self.vector_store.add_chunks(chunk_dicts)
            console.print(
                f"  [green]\u2713[/green] [dim]Added {result['added']} chunks "
                f"({result.get('failed', 0)} failed)[/dim]"
            )

            progress.update(task, completed=True, description="Done!")

        console.print(
            f"[green]\u2713[/green] Ingested from {file_path.name}: {result['added']} chunks"
        )

    def ingest_directory(
        self,
        dir_path: str,
        force_reparse: bool = False,
        force_regenerate_summary: bool = False,
        chunks_only: bool = False,
    ):
        """Ingest all documents in a directory."""
        dir_path = Path(dir_path)
        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}

        files = [
            f for f in dir_path.iterdir() if f.suffix.lower() in supported_extensions
        ]

        if not files:
            console.print("[yellow]No supported documents found.[/yellow]")
            return

        console.print(f"[bold]Found {len(files)} documents to ingest[/bold]")

        for file_path in files:
            self.ingest_document(
                str(file_path),
                force_reparse=force_reparse,
                force_regenerate_summary=force_regenerate_summary,
                chunks_only=chunks_only,
            )

    def query_interactive(self):
        """Interactive query mode."""
        agent = self._create_agent()

        console.print(
            Panel(
                "[bold]RAG3 System[/bold]\n\n"
                "Vector Search: \u2713\n\n"
                "Type 'exit' to quit.",
                title="Interactive Mode",
                border_style="blue",
            )
        )

        while True:
            try:
                question = console.input("\n[bold cyan]You:[/bold cyan] ")

                if question.lower() in ["exit", "quit"]:
                    break

                if not question.strip():
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Thinking...", total=None)
                    result = agent.query(
                        question,
                        use_expansion=True,
                        use_reranking=True,
                        use_reflection=True,
                    )
                    progress.update(task, completed=True)

                console.print(
                    Panel(
                        result["answer"],
                        title=f"[green]Answer[/green] ({result['iterations']} iterations)",
                        border_style="green",
                    )
                )

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Error processing query:[/red] {e}\n\n"
                        "[dim]The model may have failed to call tools correctly. "
                        "Try rephrasing your question or using a different model.[/dim]",
                        title="[red]\u26a0 Error[/red]",
                        border_style="red",
                    )
                )

        console.print("\n[dim]Goodbye![/dim]")

    def query_single(self, question: str):
        """Single query mode."""
        agent = self._create_agent()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing query...", total=None)
            result = agent.query(question)
            progress.update(task, completed=True)

        console.print(
            Panel(
                result["answer"],
                title=f"[green]Answer[/green] ({result['iterations']} iterations)",
                border_style="green",
            )
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG3 — Production RAG System built on Haystack"
    )
    subparsers = parser.add_subparsers(dest="command")

    # GROQ arg
    groq_args = argparse.ArgumentParser(add_help=False)
    groq_args.add_argument(
        "--groq", "-k",
        action="store_true",
        help="Use Groq API (with rotation) instead of local Ollama",
    )

    # Docling arg (for ingest only)
    docling_args = argparse.ArgumentParser(add_help=False)
    docling_args.add_argument(
        "--docling",
        action="store_true",
        help="Use Docling parser instead of unstructured (requires DOCLING_MODEL_PATH in .env)",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents",
        parents=[groq_args, docling_args],
    )
    ingest_parser.add_argument("path", help="File or directory to ingest")
    ingest_parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Force re-parsing even if cached JSON exists",
    )
    ingest_parser.add_argument(
        "--force-regenerate-summary",
        action="store_true",
        help="Force summary regeneration even if summaries exist",
    )
    ingest_parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Resume from saved chunks JSON (skip parsing, chunking, summaries)",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the system",
        parents=[groq_args],
    )
    query_parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    query_parser.add_argument("question", nargs="?", help="Question to ask")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize system with flags
    use_groq = getattr(args, "groq", False)
    use_docling = getattr(args, "docling", False)
    system = RAGSystem(use_groq=use_groq, use_docling=use_docling)

    if args.command == "ingest":
        path = Path(args.path)
        if path.is_dir():
            system.ingest_directory(
                str(path),
                force_reparse=args.force_reparse,
                force_regenerate_summary=args.force_regenerate_summary,
                chunks_only=args.chunks_only,
            )
        else:
            system.ingest_document(
                str(path),
                force_reparse=args.force_reparse,
                force_regenerate_summary=args.force_regenerate_summary,
                chunks_only=args.chunks_only,
            )

    elif args.command == "query":
        if args.interactive:
            system.query_interactive()
        else:
            if args.question:
                system.query_single(args.question)
            else:
                console.print(
                    "[red]Please provide a question or use -i for interactive mode[/red]"
                )


if __name__ == "__main__":
    main()
