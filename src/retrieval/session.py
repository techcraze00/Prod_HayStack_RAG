import logging
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..config import settings
from src.memory.summarizer import summarize_history
from src.memory.vector_store import FAISSManager

logger = logging.getLogger(__name__)

class RAGSession:
    """
    Session manager containing Sliding Window history and memory integration points.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chat_history: List[BaseMessage] = []
        self.max_window = settings.memory_window_size if hasattr(settings, "memory_window_size") else 10
        
        # Load Long-Term Memories
        self.episodic_memory = FAISSManager("episodic", session_id=session_id)
        self.archival_memory = FAISSManager("archival", session_id=session_id)
        
        # Short Term Facts
        self.core_memory = ""
        self.negative_constraints = []

    def add_user_message(self, content: str):
        self.chat_history.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.chat_history.append(AIMessage(content=content))
        self._check_window_size()

    def _check_window_size(self):
        """Trim history and trigger progressive summarization if it's too long."""
        if len(self.chat_history) > self.max_window:
            # Take the oldest 4 messages (2 turns typically)
            overflow_msgs = self.chat_history[:4]
            # Retain the rest
            self.chat_history = self.chat_history[4:]
            
            # Summarize and embed into FAISS
            summarize_history(overflow_msgs, session_id=self.session_id)

    def retrieve_episodic(self, query: str, top_k: int = 3) -> str:
        """Find past conversation summaries related to current query"""
        results = self.episodic_memory.search(query, k=top_k)
        if not results:
            return ""
        
        context = "Past Conversation Summaries:\n"
        for r in results:
            context += f"- {r['content']}\n"
        return context

    def get_memory_context(self, current_query: str) -> str:
        """Return formatted memory injections for the agent's prompt"""
        from src.memory.memory_tools import get_context_injection
        
        episodic_context = self.retrieve_episodic(current_query)
        # Note: archival retrieval could be added here if needed
        
        return get_context_injection(
            core_memory=self.core_memory,
            negative_constraints=self.negative_constraints,
            episodic_context=episodic_context,
            archival_context=""
        )
