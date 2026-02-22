"""
PROGRESSIVE SUMMARIZER
Converts conversation history into condensed memory.

Functionality:
- Monitors the message window size.
- When limits are reached, condenses the oldest messages into a summary.
- Triggers embedding of summaries into the Episodic Memory (FAISS Index A).

Usage:
- Integrated into the BaseWorker memory management pipeline or Orchestrator.
"""

import logging
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import settings
from src.memory.vector_store import FAISSManager

logger = logging.getLogger(__name__)

def summarize_history(messages: List[BaseMessage], model: str = None, session_id: str = None) -> str:
    """
    Calls LLM to condense a list of messages into a single descriptive summary.
    """
    if not messages:
        return ""

    try:
        llm = ChatOllama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0
        )

        # Format messages for the summarizer
        history_text = ""
        for m in messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            history_text += f"{role}: {m.content}\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a memory consolidation agent. Summarize the conversation below into a concise, entity-rich paragraph.

CRITICAL — you MUST preserve these details exactly as they appear:
1. **Table and column names** (e.g. "OrderDetails", "financial_year") — keep the exact identifiers.
2. **SQL queries or Python code** that succeeded — include the key clauses (SELECT, FROM, WHERE).
3. **Specific numeric results** (e.g. "revenue was ₹12,50,000 in 2023").
4. **The user's original question** — paraphrase it but keep the intent clear.
5. **Any corrections the user made** (e.g. "user clarified the column is 'fiscal_year' not 'year'").

Do NOT write vague summaries like "the user asked about sales." Write factual, searchable summaries."""),
            ("human", "Summarize this conversation:\n\n{history}")
        ])

        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"history": history_text})
        
        # Add to Episodic Memory
        episodic_mem = FAISSManager("episodic", session_id=session_id)
        episodic_mem.add_to_index(summary, metadata={"type": "episodic_summary"})

        return summary

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return "History summarization failed."