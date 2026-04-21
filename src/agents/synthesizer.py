import logging
from typing import Dict, Any

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return args[0] if args and callable(args[0]) else decorator

logger = logging.getLogger(__name__)


class Synthesizer:
    @traceable(name="Synthesizer.format", run_type="prompt")
    def synthesize(self, worker_result: Dict[str, Any], intent: str) -> str:
        """
        Formats and synthesizes the final response for the user based on worker output.
        """
        logger.info(f"Synthesizer formatting response for intent: {intent}")

        answer = worker_result.get("answer", "I could not generate an answer.")

        # Append graph retrieval metadata if available
        if intent == "GraphRetrieval":
            facts_count = worker_result.get("graph_facts_count", 0)
            entities = worker_result.get("graph_entities", [])
            if facts_count > 0 or entities:
                meta_parts = []
                if facts_count > 0:
                    meta_parts.append(f"{facts_count} graph facts")
                if entities:
                    meta_parts.append(f"entities: {', '.join(entities[:5])}")
                answer += f"\n\n_[Graph: {'; '.join(meta_parts)}]_"

        # Append hybrid retrieval metadata
        elif intent == "HybridRetrieval":
            vector_count = worker_result.get("vector_count", 0)
            graph_count = worker_result.get("graph_facts_count", 0)
            meta_parts = []
            if vector_count > 0:
                meta_parts.append(f"{vector_count} vector hits")
            if graph_count > 0:
                meta_parts.append(f"{graph_count} graph facts")
            if meta_parts:
                answer += f"\n\n_[Hybrid: {' + '.join(meta_parts)}]_"

        return answer
