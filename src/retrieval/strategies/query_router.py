"""
Query Router — Intelligent Query Classification and Routing

Classifies incoming queries and determines optimal retrieval strategy.
Uses LLM-based classification to route queries to appropriate search
methods with tuned parameters.
"""

from typing import Dict, Any, Optional
import json
import re

from ...utils.llm import chat_sync


class QueryRouter:
    """
    Routes queries to the optimal retrieval strategy.

    Classification types:
    - factual: Simple fact lookup → vector search
    - analytical: Deep analysis → reranking
    - comparison: Comparing entities → more results
    - aggregation: Summarising multiple sources → summary index
    """

    CLASSIFICATION_PROMPT = """Classify the following user query into exactly one category.
Respond with a JSON object containing "type" and "confidence".

Categories:
- "factual": Simple fact lookup (e.g. "What is X?", "Define Y")
- "analytical": Deep analysis requiring multi-hop reasoning (e.g. "How does X relate to Y?")
- "comparison": Comparing two or more entities (e.g. "Compare X and Y")
- "aggregation": Summarising across multiple sections or docs (e.g. "Summarise the key points")

Query: {query}

Respond ONLY with valid JSON, no markdown formatting:
{{"type": "<category>", "confidence": <0.0-1.0>}}"""

    STRATEGY_MAP = {
        "factual": {
            "use_vector": True,
            "use_summary": False,
            "use_reranking": False,
            "top_k": 3,
        },
        "analytical": {
            "use_vector": True,
            "use_summary": False,
            "use_reranking": True,
            "top_k": 8,
        },
        "comparison": {
            "use_vector": True,
            "use_summary": False,
            "use_reranking": True,
            "top_k": 10,
        },
        "aggregation": {
            "use_vector": True,
            "use_summary": True,
            "use_reranking": False,
            "top_k": 15,
        },
    }

    def __init__(self, generator):
        self.generator = generator

    def classify(self, query: str) -> Dict[str, Any]:
        """Classify a query type using LLM."""
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)

        try:
            response = chat_sync(self.generator, system="", user=prompt)

            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("type") in self.STRATEGY_MAP:
                    return result

            return {"type": "factual", "confidence": 0.5}

        except Exception:
            return {"type": "factual", "confidence": 0.3}

    def determine_strategy(
        self,
        classification: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Determine retrieval strategy from classification."""
        query_type = classification.get("type", "factual")
        strategy = {**self.STRATEGY_MAP.get(query_type, self.STRATEGY_MAP["factual"])}
        strategy["query_type"] = query_type
        strategy["confidence"] = classification.get("confidence", 0.5)

        if overrides:
            strategy.update(overrides)

        return strategy
