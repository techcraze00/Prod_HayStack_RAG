"""
GRAPH AGENT — LangGraph-based Graph Retrieval Worker for RAG3

Implements a stateful graph query workflow using LangGraph:
1. Search the knowledge graph for relevant facts/relationships
2. Optionally expand entity neighborhoods for deeper context
3. Synthesize an answer using the graph context

Follows the same interface as VectorAgent/GeneralAgent:
- query(question, chat_history, memory_context) -> Dict[str, Any]
- Returns {"question", "answer", "total_results", "graph_search_used"}
"""

import logging
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from src.config import settings
from src.retrieval.tools.graph_search_tool import GraphSearchTool

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for the LangGraph graph query workflow."""
    question: str
    graph_results: List[Dict[str, Any]]
    graph_context: str
    entity_names: List[str]
    expanded_context: str
    answer: str
    iteration: int


class GraphAgent:
    """
    Worker agent for graph-based retrieval.

    Uses a multi-step workflow:
    1. Graph Search — find relevant facts and relationships
    2. Entity Expansion — expand neighborhoods of key entities (optional)
    3. Answer Generation — synthesize answer from graph context

    Compatible with the Orchestrator's worker dispatch pattern.
    """

    def __init__(
        self,
        graph_search_tool: GraphSearchTool,
        model: str = None,
        max_iterations: int = 2,
    ):
        self.graph_tool = graph_search_tool
        self.max_iterations = max_iterations

        # LLM for answer synthesis
        self.llm = OllamaChatGenerator(
            model=model or settings.ollama_model,
            url=settings.ollama_base_url,
            generation_kwargs={"temperature": 0.3, "num_ctx": settings.ollama_num_ctx},
        )

        self.system_prompt = (
            "You are a graph-augmented AI assistant. You answer questions using "
            "knowledge extracted from a graph database containing entities and their "
            "relationships. Your answers should be precise and cite the relationships "
            "you are drawing from. If the graph context does not contain relevant "
            "information, say so clearly rather than making up facts."
        )

    def _search_graph(self, state: GraphState) -> GraphState:
        """Step 1: Search the knowledge graph."""
        results = self.graph_tool.search(
            query=state["question"],
            num_results=settings.graph_search_results,
        )

        # Extract entity names for potential expansion
        entity_names = set()
        for r in results:
            src = r.get("source_node", "")
            tgt = r.get("target_node", "")
            if src:
                entity_names.add(src)
            if tgt:
                entity_names.add(tgt)

        graph_context = self.graph_tool.format_context(results)

        state["graph_results"] = results
        state["graph_context"] = graph_context
        state["entity_names"] = list(entity_names)[:5]  # Limit expansion candidates
        return state

    def _expand_entities(self, state: GraphState) -> GraphState:
        """Step 2: Expand neighborhoods of key entities for deeper context."""
        if not state["entity_names"] or not state["graph_results"]:
            state["expanded_context"] = ""
            return state

        expansions = []
        for entity_name in state["entity_names"][:3]:  # Expand top 3 entities
            neighbors = self.graph_tool.get_neighborhood(
                entity_name=entity_name,
                depth=settings.graph_search_depth,
            )
            if neighbors:
                neighbor_names = [n.get("name", "unknown") for n in neighbors[:5]]
                expansions.append(
                    f"  {entity_name} connects to: {', '.join(neighbor_names)}"
                )

        state["expanded_context"] = "\n".join(expansions) if expansions else ""
        return state

    def _generate_answer(
        self,
        state: GraphState,
        chat_history: List = None,
        memory_context: str = "",
    ) -> GraphState:
        """Step 3: Generate answer using graph context."""
        messages = []

        # System prompt with memory context
        system = self.system_prompt
        if memory_context:
            system += f"\n\nMemory Context:\n{memory_context}"
        messages.append(ChatMessage.from_system(system))

        # Chat history
        if chat_history:
            for m in chat_history:
                msg_type = getattr(m, "type", "human")
                content = m.text if hasattr(m, "text") else m.content
                if type(m).__name__ == "HumanMessage" or msg_type == "human":
                    messages.append(ChatMessage.from_user(content))
                else:
                    messages.append(ChatMessage.from_assistant(content))

        # Build context-enriched prompt
        context_parts = []
        if state["graph_context"]:
            context_parts.append(
                f"=== Knowledge Graph Facts ===\n{state['graph_context']}"
            )
        if state["expanded_context"]:
            context_parts.append(
                f"\n=== Entity Connections ===\n{state['expanded_context']}"
            )

        full_context = "\n".join(context_parts) if context_parts else "No graph context found."

        user_prompt = (
            f"Using the following knowledge graph context, answer the question.\n\n"
            f"{full_context}\n\n"
            f"Question: {state['question']}\n\n"
            f"Provide a clear, well-structured answer based on the graph relationships above. "
            f"If the context doesn't contain enough information, say so."
        )
        messages.append(ChatMessage.from_user(user_prompt))

        try:
            response = self.llm.run(messages=messages)
            state["answer"] = response["replies"][0].text
        except Exception as e:
            logger.error(f"GraphAgent LLM generation failed: {e}")
            state["answer"] = (
                f"I found {len(state['graph_results'])} graph results but "
                f"encountered an error generating the answer: {e}"
            )

        return state

    def query(
        self,
        question: str,
        chat_history: List[BaseMessage] = None,
        memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        Run graph retrieval workflow.

        Follows the same interface as VectorAgent and GeneralAgent.

        Args:
            question: User's query.
            chat_history: Previous conversation messages.
            memory_context: Memory injection context string.

        Returns:
            Dict with question, answer, total_results, graph_search_used.
        """
        logger.info("GraphAgent processing query via graph retrieval workflow.")

        # Initialize state
        state: GraphState = {
            "question": question,
            "graph_results": [],
            "graph_context": "",
            "entity_names": [],
            "expanded_context": "",
            "answer": "",
            "iteration": 0,
        }

        try:
            # Step 1: Search graph
            state = self._search_graph(state)

            # Step 2: Expand entity neighborhoods (if results found)
            if state["graph_results"]:
                state = self._expand_entities(state)

            # Step 3: Generate answer
            state = self._generate_answer(
                state,
                chat_history=chat_history,
                memory_context=memory_context,
            )

            return {
                "question": question,
                "answer": state["answer"],
                "total_results": len(state["graph_results"]),
                "graph_search_used": True,
                "vector_search_used": False,
                "graph_entities": state["entity_names"],
                "graph_facts_count": len(state["graph_results"]),
            }

        except Exception as e:
            logger.error(f"GraphAgent workflow failed: {e}")
            return {
                "question": question,
                "answer": f"Graph retrieval failed: {e}",
                "total_results": 0,
                "graph_search_used": True,
                "vector_search_used": False,
            }
