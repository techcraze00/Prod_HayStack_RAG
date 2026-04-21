from typing import List
import json

from haystack.dataclasses import ChatMessage

from ...utils.llm import chat_sync


class QueryExpansionStrategy:
    """
    Expands queries to improve recall.

    Techniques:
    - Synonym generation
    - Question reformulation
    - Multi-perspective expansion
    """

    SYSTEM_PROMPT = """You are a query expansion expert.

Generate 3 alternative formulations of the user's question that:
1. Use different terminology/synonyms
2. Approach from different angles
3. Include related concepts

Return as JSON array: ["query1", "query2", "query3"]
"""

    def __init__(self, generator):
        self.generator = generator

    def expand(self, query: str) -> List[str]:
        """Generate query variations.

        Args:
            query: Original query.

        Returns:
            List of expanded queries (including original).
        """
        response = chat_sync(
            self.generator,
            system=self.SYSTEM_PROMPT,
            user=f"Original question: {query}",
        )

        try:
            expanded = json.loads(response)
            if isinstance(expanded, list):
                return [query] + expanded
            else:
                return [query]
        except Exception:
            return [query]
