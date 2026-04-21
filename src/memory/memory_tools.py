"""
MEMORY TOOLS
Management of Core Memory and sticky facts.

Functionality:
- Tools to update, retrieve, and clear Core Memory (sticky facts).
- Logic to inject Core Memory into agent system prompts.
- Management of Negative Constraints (failure rules).

Usage:
- Orchestrator uses these to enrich the context before calling workers.
"""

from typing import List, Dict, Any

def get_context_injection(core_memory: str = "", negative_constraints: List[str] = None, episodic_context: str = "", archival_context: str = "", graph_episodic_context: str = "") -> str:
    """Formats Core Memory, Negative Constraints, and RAG contexts for prompt insertion."""
    neg = negative_constraints or []

    

    injection = ""

    if core_memory:

        injection += f"\n### Core Memory (User Facts):\n{core_memory}\n"

    

    if neg:

        injection += "\n### Negative Constraints (To Avoid):\n"

        for c in neg:

            injection += f"- {c}\n"



    if episodic_context:

        injection += f"\n### Relevant Past Conversations:\n{episodic_context}\n"



    if archival_context:

        injection += f"\n### Successful Patterns/Code Found:\n{archival_context}\n"

    if graph_episodic_context:

        injection += f"\n### Graph Memory (Knowledge Graph Recall):\n{graph_episodic_context}\n"

            

    return injection
