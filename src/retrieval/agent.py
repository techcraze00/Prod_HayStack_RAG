import logging
from typing import List, Dict, Any

from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage

from .tools.vector_tool import VectorSearchTool
from .strategies.query_expansion import QueryExpansionStrategy
from .strategies.reranking import RerankingStrategy
from .strategies.self_reflection import SelfReflectionStrategy
from ..utils.llm import chat_sync
from ..config import settings

logger = logging.getLogger(__name__)


class AdvancedRAGAgent:
    """
    Autonomous RAG agent with advanced strategies.

    Features:
    - Dynamic tool selection via Haystack Agent
    - Query refinement loop
    - Multi-strategy retrieval
    - Self-correction
    - Query routing (Enhancement 4)
    - Fallback handling (Enhancement 8)
    - Monitoring & observability (Enhancement 9)
    """

    AGENT_SYSTEM_PROMPT = (
        "You are a document retrieval assistant. Your job is to find information "
        "from the ingested documents using ONLY the tools provided to you.\n\n"
        "STRICT RULES:\n"
        "1. ONLY use the tools listed below. NEVER invent or call tools that are not provided.\n"
        "2. Use the vector_search tool to find relevant document chunks.\n"
        "3. If the tools return no relevant results, respond with: "
        "'I could not find the answer in the available documents.'\n"
        "4. NEVER guess, speculate, or make up information.\n"
        "5. For regulatory or compliance documents, accuracy is critical — "
        "if you are unsure, say so explicitly.\n"
        "6. Always cite which document sections your answer comes from."
    )

    SYNTHESIS_SYSTEM = (
        "You are a precise document analysis assistant. "
        "You answer questions strictly based on provided sources. "
        "You never fabricate information. If uncertain, you say so."
    )

    def __init__(
        self,
        generator,
        vector_tool: VectorSearchTool,
        max_iterations: int = 5,
    ):
        self.generator = generator
        self.vector_tool = vector_tool
        self.max_iterations = max_iterations

        # Core strategies
        self.query_expander = QueryExpansionStrategy(generator)
        self.reranker = RerankingStrategy()
        self.reflector = SelfReflectionStrategy(generator)

        # ── Enhancement 4: Query Routing ─────────────────────────────
        self.query_router = None
        if settings.query_routing_enabled:
            from .strategies.query_router import QueryRouter
            self.query_router = QueryRouter(generator)

        # ── Enhancement 8: Fallback Handling ─────────────────────────
        self.fallback_handler = None
        if settings.fallback_enabled:
            from .fallback import RetrievalFallbackHandler
            self.fallback_handler = RetrievalFallbackHandler(
                generator=generator,
                vector_search_fn=self._vector_search_for_fallback,
                min_score_threshold=settings.fallback_min_score,
                min_results=settings.fallback_min_results,
            )

        # ── Enhancement 9: Monitoring ────────────────────────────────
        self.metrics_collector = None
        self.rag_logger = None
        if settings.monitoring_enabled:
            from ..monitoring.metrics import MetricsCollector
            from ..monitoring.logger import RAGLogger
            self.metrics_collector = MetricsCollector(
                max_history=settings.metrics_max_history
            )
            self.rag_logger = RAGLogger("rag_agent")

        # Build Haystack Agent
        self._agent = self._build_agent()

    def _vector_search_for_fallback(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        return self.vector_tool.search(query, top_k)

    def _build_agent(self):
        """Build Haystack Agent with tools."""
        try:
            from haystack.components.agents import Agent

            tools = [self.vector_tool.as_haystack_tool()]

            agent = Agent(
                chat_generator=self.generator,
                tools=tools,
                system_prompt=self.AGENT_SYSTEM_PROMPT,
                max_steps=self.max_iterations,
            )
            return agent
        except ImportError:
            # Fallback: if Agent is not available, use direct tool calling
            logger.warning(
                "Haystack Agent not available (requires haystack-ai >= 2.12). "
                "Falling back to direct tool invocation."
            )
            return None

    def query(
        self,
        question: str,
        use_expansion: bool = True,
        use_reranking: bool = True,
        use_reflection: bool = True,
        use_routing: bool = None,
    ) -> Dict[str, Any]:
        """
        Process query with advanced strategies.

        Args:
            question: User question.
            use_expansion: Enable query expansion.
            use_reranking: Enable re-ranking.
            use_reflection: Enable self-reflection.
            use_routing: Enable query routing (None = use config).

        Returns:
            Answer with metadata.
        """
        original_query = question
        current_query = question
        iteration = 0
        all_results = []
        query_metadata = {}

        # Start monitoring timer
        overall_timer = None
        if self.metrics_collector:
            overall_timer = self.metrics_collector.timer("query")
            overall_timer.__enter__()

        try:
            # ── Step 0: Query Routing ────────────────────────────────
            if use_routing is None:
                use_routing = settings.query_routing_enabled

            if use_routing and self.query_router:
                classification = self.query_router.classify(question)
                strategy = self.query_router.determine_strategy(classification)
                query_metadata["routing"] = {
                    "classification": classification,
                    "strategy": strategy,
                }

                if not strategy.get("use_vector", True):
                    use_expansion = False
                if strategy.get("use_reranking", False):
                    use_reranking = True

                if self.rag_logger:
                    self.rag_logger.log_routing(question, classification, strategy)

            if self.rag_logger:
                self.rag_logger.log_query(
                    question,
                    strategy=query_metadata.get("routing", {}).get(
                        "classification", {}
                    ).get("type", "default"),
                )

            while iteration < self.max_iterations:
                iteration += 1

                # Step 1: Query Expansion
                if use_expansion and iteration == 1:
                    queries = self.query_expander.expand(current_query)
                else:
                    queries = [current_query]

                # Step 2: Execute search for each query
                retrieval_timer = None
                if self.metrics_collector:
                    retrieval_timer = self.metrics_collector.timer("retrieval")
                    retrieval_timer.__enter__()

                iteration_results = []

                if self._agent:
                    # Use Haystack Agent
                    try:
                        for q in queries:
                            result = self._agent.run(
                                messages=[ChatMessage.from_user(q)]
                            )
                            if "messages" in result:
                                last_msg = result["messages"][-1]
                                iteration_results.append(
                                    {"query": q, "text": last_msg.text}
                                )
                    except Exception as e:
                        logger.error(f"Agent error: {e}")
                        # Fallback to direct search
                        for q in queries:
                            results = self.vector_tool.search(q, settings.vector_top_k)
                            for r in results:
                                iteration_results.append(
                                    {"query": q, "text": r.get("text", ""), "score": r.get("score", 0)}
                                )
                else:
                    # Direct tool invocation (no Agent)
                    for q in queries:
                        try:
                            results = self.vector_tool.search(q, settings.vector_top_k)
                            for r in results:
                                iteration_results.append(
                                    {"query": q, "text": r.get("text", ""), "score": r.get("score", 0)}
                                )
                        except Exception as e:
                            logger.error(f"Search error: {e}")

                if retrieval_timer:
                    retrieval_timer.__exit__(None, None, None)

                if self.rag_logger:
                    self.rag_logger.log_retrieval(
                        iteration_results,
                        latency_ms=retrieval_timer.elapsed_ms if retrieval_timer else 0,
                        method="agent" if self._agent else "direct",
                    )

                all_results.extend(iteration_results)

                # ── Step 2.5: Fallback Handling ──────────────────────
                if (
                    self.fallback_handler
                    and self.fallback_handler.is_low_quality(iteration_results)
                ):
                    fallback_result = self.fallback_handler.apply_fallback(
                        current_query, iteration_results
                    )
                    if fallback_result.success:
                        iteration_results = fallback_result.results
                        all_results = iteration_results
                        query_metadata["fallback"] = {
                            "strategy": fallback_result.strategy.value,
                            "message": fallback_result.message,
                        }
                    if self.rag_logger:
                        self.rag_logger.log_fallback(
                            fallback_result.strategy.value,
                            fallback_result.success,
                            fallback_result.message,
                        )

                # Step 3: Re-ranking
                if use_reranking and iteration_results:
                    reranked = self.reranker.rerank(
                        original_query,
                        iteration_results,
                        top_k=settings.rerank_top_k,
                    )
                    iteration_results = reranked

                # Step 4: Self-Reflection
                if use_reflection and iteration < self.max_iterations:
                    grade = self.reflector.should_refine(
                        original_query, iteration_results
                    )

                    if not grade.get("needs_refinement", False):
                        break
                    else:
                        current_query = self.reflector.refine_query(
                            current_query, grade.get("reason", "")
                        )
                else:
                    break

            # Step 5: Synthesize final answer
            gen_timer = None
            if self.metrics_collector:
                gen_timer = self.metrics_collector.timer("generation")
                gen_timer.__enter__()

            final_answer = self._synthesize_answer(original_query, all_results)

            if gen_timer:
                gen_timer.__exit__(None, None, None)
                if self.rag_logger:
                    self.rag_logger.log_generation(latency_ms=gen_timer.elapsed_ms)

        except Exception as e:
            if self.rag_logger:
                self.rag_logger.log_error("query_failed", {"error": str(e)})
            if self.metrics_collector:
                self.metrics_collector.record_error("query_failed")
            raise
        finally:
            if overall_timer:
                overall_timer.__exit__(None, None, None)

            if self.metrics_collector:
                from ..monitoring.metrics import QueryMetrics

                qm = QueryMetrics(
                    query=original_query,
                    strategy=query_metadata.get("routing", {}).get(
                        "classification", {}
                    ).get("type", "default"),
                    latency_ms=overall_timer.elapsed_ms if overall_timer else 0,
                    num_results=len(all_results),
                    fallback_used="fallback" in query_metadata,
                    fallback_strategy=query_metadata.get("fallback", {}).get(
                        "strategy", ""
                    ),
                )
                self.metrics_collector.record_query(qm)

        return {
            "question": original_query,
            "answer": final_answer,
            "iterations": iteration,
            "queries_used": queries,
            "total_results": len(all_results),
            **query_metadata,
        }

    def _synthesize_answer(
        self, question: str, results: List[Dict[str, Any]]
    ) -> str:
        """Synthesize final answer from all results."""
        if not results:
            return "I could not find relevant information to answer your question."

        context = "\n\n".join(
            [
                f"Source {i+1} (Query: {r.get('query', 'N/A')}): {r.get('text', '')}"
                for i, r in enumerate(results[:10])
            ]
        )

        prompt = f"""Answer the following question based ONLY on the search results provided below.

Question: {question}

Search Results:
{context}

STRICT RULES:
1. Base your answer ONLY on the information in the search results above.
2. Cite your sources (e.g., [Source 1], [Source 2]).
3. If the search results do NOT contain enough information, explicitly state what is missing.
4. NEVER speculate, guess, or infer beyond what the source text states.
5. For regulatory or compliance content, accuracy is critical — err on the side of saying "the document does not specify" rather than guessing.
6. If multiple interpretations are possible, list them all rather than choosing one.
"""

        return chat_sync(self.generator, system=self.SYNTHESIS_SYSTEM, user=prompt)
