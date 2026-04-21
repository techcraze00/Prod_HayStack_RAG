"""
Haystack LLM helper utilities.

Provides a synchronous chat helper for Haystack generators,
replacing LangChain's `await llm.ainvoke([SystemMessage, HumanMessage])` pattern.
"""

from haystack.dataclasses import ChatMessage


def chat_sync(generator, system: str, user: str) -> str:
    """Send a system + user message pair through a Haystack chat generator.

    Args:
        generator: A Haystack ChatGenerator (Ollama or OpenAI-compatible).
        system: System prompt text.
        user: User prompt text.

    Returns:
        The assistant's reply text.
    """
    messages = []
    if system:
        messages.append(ChatMessage.from_system(system))
    messages.append(ChatMessage.from_user(user))
    result = generator.run(messages=messages)
    return result["replies"][0].text
