import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Synthesizer:
    def synthesize(self, worker_result: Dict[str, Any], intent: str) -> str:
        """
        Formats and synthesizes the final response for the user based on worker output.
        """
        logger.info(f"Synthesizer formatting response for intent: {intent}")
        
        answer = worker_result.get("answer", "I could not generate an answer.")
        
        # Currently, both GeneralAgent and VectorAgent return fully formed string answers.
        # This Synthesizer layer allows future extensibility to combine multiple outputs
        # or append metadata (like 'Found in 3 documents.')
            
        return answer
