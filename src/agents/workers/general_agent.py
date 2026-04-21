import logging
from typing import List, Dict, Any
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from src.config import settings

logger = logging.getLogger(__name__)

class GeneralAgent:
    def __init__(self, model: str = None):
        self.llm = OllamaChatGenerator(
            model=model or settings.ollama_model,
            url=settings.ollama_base_url,
            generation_kwargs={"temperature": 0.7}
        )
        self.system_prompt = (
            "You are a helpful and polite conversational AI assistant for a document retrieval system. "
            "You assist users with general queries, greetings, and small talk. "
            "If the user asks about specific factual data, politely explain "
            "that you are the general chat agent and they should ask their specific question "
            "using standard keywords to trigger the document search agent."
        )

    def query(self, question: str, chat_history: List[ChatMessage] = None, memory_context: str = "") -> Dict[str, Any]:
        logger.info("GeneralAgent processing query.")
        
        messages = []
        
        # Build prompt: System -> Memory Context -> Chat History -> Question
        contextual_prompt = self.system_prompt
        if memory_context:
            contextual_prompt += f"\n\nMemory Context:\n{memory_context}"
            
        messages.append(ChatMessage.from_system(contextual_prompt))
            
            
        if chat_history:
            for m in chat_history:
                msg_type = getattr(m, "type", "human")
                if type(m).__name__ == "HumanMessage" or msg_type == "human":
                    messages.append(ChatMessage.from_user(m.text if hasattr(m, 'text') else m.content))
                else:
                    messages.append(ChatMessage.from_assistant(m.text if hasattr(m, 'text') else m.content))
            
        messages.append(ChatMessage.from_user(question))
        
        try:
            response = self.llm.run(messages=messages)
            answer = response["replies"][0].text
            return {
                "question": question,
                "answer": answer,
                "total_results": 0,
                "vector_search_used": False
            }
        except Exception as e:
            logger.error(f"GeneralAgent error: {e}")
            return {
                "question": question,
                "answer": "I'm sorry, I encountered an error while trying to respond to you.",
                "total_results": 0,
                "vector_search_used": False
            }
