"""
3-TIER INTENT ROUTER for RAG3
Hierarchy-based classification for Document Retrieval and General Chat.

Tiers:
1. Regex (Zero Latency)
2. Weighted Keywords (Low Latency)
3. LLM Fallback (High Accuracy)
"""

import re
import logging
from typing import List
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from src.config import settings

logger = logging.getLogger(__name__)

class IntentRouter:
    def __init__(self, model: str = None, vector_tool=None, syllabus: str = None):
        self.llm = OllamaChatGenerator(
            model=model or settings.ollama_model,
            url=settings.ollama_base_url,
            generation_kwargs={"temperature": 0.0}
        )
        self.vector_tool = vector_tool
        
        # Default global context if not provided.
        self.syllabus = syllabus or (
            "This database contains highly specific business documents, reports, manuals, "
            "and factual records related to the organization's internal workings."
        )

    def classify(self, query: str, history: List[ChatMessage] = None) -> str:
        """
        Executes the Advanced Semantic Route classification hierarchy. Returns primary intent.
        """
        query_clean = query.lower().strip()
        logger.info(f"Classifying intent for query: {query[:50]}...")

        # --- Tier 1: Strict Regex (Zero Latency) ---
        if self._match_regex_general(query_clean):
            logger.info("Matched Tier 1: Regex (GeneralChat)")
            return "GeneralChat"

        # --- Tier 2: Parallel Grounding Check (Vector Probe) ---
        context_snippets = ""
        if self.vector_tool:
            try:
                # Fast semantic search for top 3 hits
                results = self.vector_tool.search(query, top_k=3)
                if results:
                    snippets = []
                    for i, r in enumerate(results):
                        text = r.get("text", "").strip()[:200]
                        score = r.get("score", 0.0)
                        snippets.append(f"Snippet {i+1} [Score: {score:.3f}]: {text}...")
                    context_snippets = "\n".join(snippets)
                else:
                    context_snippets = "No semantic matches found in database."
            except Exception as e:
                logger.error(f"Tier 2 Vector Probe failed: {e}")
                context_snippets = "Database semantic search failed or is unavailable."
        else:
            context_snippets = "Vector Tool not provided for semantic probing."

        # --- Tier 3: LLM Intent Arbiter ---
        logger.info(f"Matched Tier 3: LLM Intent Arbiter")
        llm_intent = self._llm_fallback(query, history=history, snippets=context_snippets)
        logger.info(f"Final classified intent from LLM Arbiter: {llm_intent}")
        return llm_intent

    def _match_regex_general(self, query: str) -> bool:
        """Greetings and system commands."""
        greetings = r"\b(hi|hello|hey|good morning|good evening|greetings)\b"
        system = r"\b(exit|quit|bye|clear|help|reset|restart)\b"
        gibberish = r"^([asdfjkl;]{4,}|[0-9]{10,})$"
        
        return any(re.search(p, query) for p in [greetings, system, gibberish])

    def _llm_fallback(self, query: str, history: List[ChatMessage] = None, snippets: str = "") -> str:
        """Intelligent fallback utilizing Semantic Probes and Global Syllabus."""
        try:
            # Format history for context
            history_context = ""
            if history:
                recent = history[-4:]
                history_context = "\n### RECENT CONVERSATION HISTORY:\n"
                for m in recent:
                    msg_type = getattr(m, "type", "human")
                    role = "User" if type(m).__name__ == "HumanMessage" or msg_type == "human" else "Assistant"
                    msg_text = m.text if hasattr(m, 'text') else m.content
                    content = msg_text[:200] if msg_text else ""
                    history_context += f"{role}: {content}\n"

            system_prompt = f"""You are the Advanced Semantic Router for an Enterprise RAG System.
Your job is to objectively evaluate the user's query against the known database contents and route it appropriately.

You have exactly two categories. You must respond with ONLY the category name. No punctuation or explanation.
{history_context}

### GLOBAL DATABASE SYLLABUS
This RAG dataset contains: {self.syllabus}

### LIVE VECTOR PROBE RESULTS
We performed a live test search against the database for the user's query. Here are the Top 3 results:
{snippets}

### TASK
Based on the syllabus and the vector probe results, does the user's query `{query}` attempt to extract factual knowledge embedded in this specific database, or is it a general conversational/trivia query? 

1. **'VectorRetrieval'**
- **Trigger:** The query requests facts and the vector probe returned highly relevant snippets, OR the topic heavily aligns with the Syllabus constraints.
- **Example:** "How do I configure the server?", "Who is the CEO?", "Summarize the findings."

2. **'GeneralChat'**
- **Trigger:** The query is general trivia ("What is the capital of France?"), small talk, or abstract logic. Even if they use the word "What", if the Vector Probe results are irrelevant and it ignores the Syllabus, it is GeneralChat.
- **Example:** "What is the capital of France?", "Hi", "Tell me a joke."
"""

            messages = [
                ChatMessage.from_system(system_prompt),
                ChatMessage.from_user(query)
            ]

            response = self.llm.run(messages=messages)
            raw_response = response["replies"][0].text
            
            # Sanitize output
            intent = raw_response.strip().replace("'", "").replace('"', "").replace(".", "")
            
            if intent.lower() == "generalchat":
                return "GeneralChat"
            return "VectorRetrieval" # Safer default
            
        except Exception as e:
            logger.error(f"Router Tier 3 Error: {e}")
            return "VectorRetrieval" # Safe default

def classify_intent(query: str, history: list = None, model: str = None, vector_tool=None, syllabus: str = None) -> str:
    """Functional wrapper. Returns the primary intent."""
    router = IntentRouter(model=model, vector_tool=vector_tool, syllabus=syllabus)
    return router.classify(query, history=history)
