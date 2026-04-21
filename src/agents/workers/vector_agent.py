import logging
from typing import List, Dict, Any
from src.retrieval.agent import AdvancedRAGAgent
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

class VectorAgent:
    """
    Worker wrapper around the AdvancedRAGAgent to fit into the Orchestrator pattern.
    """
    def __init__(self, rag_agent: AdvancedRAGAgent):
        self.rag_agent = rag_agent
        
    def query(self, question: str, chat_history: List[BaseMessage] = None, memory_context: str = "") -> Dict[str, Any]:
        logger.info("VectorAgent processing query via AdvancedRAGAgent.")
        
        # AdvancedRAGAgent executes search, reranking, and self-reflection
        result = self.rag_agent.query(
            question=question,
            chat_history=chat_history,
            memory_context=memory_context
        )
        
        result["vector_search_used"] = True
        return result
