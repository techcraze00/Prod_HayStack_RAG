import logging
from typing import Optional
from .router import classify_intent
from .workers.general_agent import GeneralAgent
from .workers.vector_agent import VectorAgent
from .synthesizer import Synthesizer
from src.retrieval.session import RAGSession
from src.retrieval.agent import AdvancedRAGAgent
from src.config import settings

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return args[0] if args and callable(args[0]) else decorator

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Supervisor layer for RAG3.
    Manages state context mapping, routes intents, dispatches workers, and synthesizes output.

    Workers:
    - GeneralAgent: greetings, small talk, trivia
    - VectorAgent: factual queries from vector store chunks
    - GraphAgent: relationship and structural queries from knowledge graph
    - HybridFusion: combined vector + graph retrieval for complex queries
    """
    def __init__(self, rag_agent: AdvancedRAGAgent, graph_search_tool=None):
        self.general_agent = GeneralAgent()
        self.vector_agent = VectorAgent(rag_agent=rag_agent)
        self.synthesizer = Synthesizer()

        # Graph agent (lazy init, behind feature flag)
        self.graph_agent = None
        if settings.graph_rag_enabled and graph_search_tool is not None:
            try:
                from .workers.graph_agent import GraphAgent
                self.graph_agent = GraphAgent(graph_search_tool=graph_search_tool)
                logger.info("GraphAgent initialized in Orchestrator.")
            except Exception as e:
                logger.warning(f"GraphAgent init failed: {e}")

        # Hybrid fusion (requires both vector + graph tools)
        self.fusion = None
        if settings.graph_rag_enabled and graph_search_tool is not None:
            try:
                from src.retrieval.strategies.graph_fusion import GraphVectorFusion
                self.fusion = GraphVectorFusion(
                    vector_tool=rag_agent.vector_tool,
                    graph_search_tool=graph_search_tool,
                )
                logger.info("GraphVectorFusion initialized in Orchestrator.")
            except Exception as e:
                logger.warning(f"GraphVectorFusion init failed: {e}")

    @traceable(name="Orchestrator.run", run_type="chain")
    def run(self, query: str, session: RAGSession) -> str:
        """
        Executes the agentic workflow: Route -> Dispatch -> Synthesize
        """
        logger.info(f"Orchestrator received query: {query}")

        # 1. Gather Context
        chat_history = session.chat_history
        memory_context = session.get_memory_context(query)

        # 2. Route Intent (Advanced Semantic Engine)
        # Extract a global syllabus if we have one defined in settings/memory, else use a default.
        global_syllabus = getattr(session, "syllabus", None)

        intent = classify_intent(
            query,
            history=chat_history,
            vector_tool=self.vector_agent.rag_agent.vector_tool,
            syllabus=global_syllabus
        )
        logger.info(f"Orchestrator decided intent: {intent}")

        # 3. Dispatch to Worker
        worker_result = {}
        if intent == "GeneralChat":
            worker_result = self.general_agent.query(
                question=query,
                chat_history=chat_history,
                memory_context=memory_context
            )
        elif intent == "HybridRetrieval" and self.fusion:
            worker_result = self._run_hybrid(
                query=query,
                chat_history=chat_history,
                memory_context=memory_context
            )
        elif intent == "GraphRetrieval" and self.graph_agent:
            worker_result = self.graph_agent.query(
                question=query,
                chat_history=chat_history,
                memory_context=memory_context
            )
        else:  # Default to VectorRetrieval
            worker_result = self.vector_agent.query(
                question=query,
                chat_history=chat_history,
                memory_context=memory_context
            )

        # 4. Synthesize Final Response
        final_answer = self.synthesizer.synthesize(worker_result, intent)

        return final_answer

    @traceable(name="Orchestrator.hybrid_fusion", run_type="retriever")
    def _run_hybrid(self, query: str, chat_history=None, memory_context: str = "") -> dict:
        """
        Run hybrid vector + graph retrieval and generate an answer from fused context.
        """
        from haystack.dataclasses import ChatMessage
        from haystack_integrations.components.generators.ollama import OllamaChatGenerator

        # 1. Fused search
        fusion_result = self.fusion.fused_search(query)
        fused_context = fusion_result["fused_context"]

        # 2. Generate answer using fused context
        llm = OllamaChatGenerator(
            model=settings.ollama_model,
            url=settings.ollama_base_url,
            generation_kwargs={"temperature": 0.3, "num_ctx": settings.ollama_num_ctx},
        )

        messages = []
        system = (
            "You are an AI assistant with access to both document chunks and a knowledge graph. "
            "Use the combined context below to provide a comprehensive, well-structured answer. "
            "Cite both document sources and graph relationships where relevant."
        )
        if memory_context:
            system += f"\n\nMemory Context:\n{memory_context}"
        messages.append(ChatMessage.from_system(system))

        if chat_history:
            for m in chat_history[-4:]:
                msg_type = getattr(m, "type", "human")
                content = m.text if hasattr(m, "text") else m.content
                if type(m).__name__ == "HumanMessage" or msg_type == "human":
                    messages.append(ChatMessage.from_user(content))
                else:
                    messages.append(ChatMessage.from_assistant(content))

        user_prompt = (
            f"Using the following combined context from both document search and knowledge graph, "
            f"answer the question.\n\n{fused_context}\n\nQuestion: {query}\n\n"
            f"Provide a comprehensive answer that draws from both sources."
        )
        messages.append(ChatMessage.from_user(user_prompt))

        try:
            response = llm.run(messages=messages)
            answer = response["replies"][0].text
        except Exception as e:
            logger.error(f"Hybrid answer generation failed: {e}")
            answer = f"Hybrid retrieval found {fusion_result['total_count']} results but answer generation failed: {e}"

        return {
            "question": query,
            "answer": answer,
            "vector_search_used": True,
            "graph_search_used": True,
            "total_results": fusion_result["total_count"],
            "vector_count": fusion_result["vector_count"],
            "graph_facts_count": fusion_result["graph_count"],
            "graph_entities": [],  # Entities not individually tracked in fusion
        }

