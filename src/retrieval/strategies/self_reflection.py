from typing import List, Dict, Any
import json

from ...utils.llm import chat_sync


class SelfReflectionStrategy:
    """
    Grades retrieval quality and decides if re-query is needed.

    Process:
    1. Grade retrieved results
    2. If quality < threshold, reformulate query
    3. Re-execute retrieval
    4. Return best results
    """

    GRADING_PROMPT = """Grade the relevance of these search results to the question:

Question: {query}

Results:
{results}

Return JSON:
{{
    "grade": "high" | "medium" | "low",
    "reason": "explanation",
    "needs_refinement": true | false
}}
"""

    def __init__(self, generator, quality_threshold: float = 0.7):
        self.generator = generator
        self.quality_threshold = quality_threshold

    def should_refine(
        self, query: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Grade retrieval quality."""
        if not results:
            return {"grade": "low", "needs_refinement": True, "reason": "No results found"}

        results_text = "\n".join(
            [f"{i+1}. {r.get('text', '')[:200]}..." for i, r in enumerate(results[:5])]
        )

        response = chat_sync(
            self.generator,
            system="You are a retrieval quality expert.",
            user=self.GRADING_PROMPT.format(query=query, results=results_text),
        )

        try:
            grade = json.loads(response)
            return grade
        except Exception:
            return {"grade": "medium", "needs_refinement": False}

    def refine_query(self, query: str, reason: str) -> str:
        """Generate refined query."""
        prompt = f"""The original query didn't retrieve good results.

Original query: {query}
Issue: {reason}

Generate a better formulated query that addresses the issue.
Return only the new query text.
"""

        response = chat_sync(
            self.generator,
            system="You are a query refinement expert.",
            user=prompt,
        )

        return response.strip()
