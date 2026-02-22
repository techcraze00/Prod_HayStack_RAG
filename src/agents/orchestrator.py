import logging
from .router import classify_intent
from .workers.general_agent import GeneralAgent
from .workers.vector_agent import VectorAgent
from .synthesizer import Synthesizer
from src.retrieval.session import RAGSession
from src.retrieval.agent import AdvancedRAGAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Supervisor layer for RAG3.
    Manages state context mapping, routes intents, dispatches workers, and synthesizes output.
    """
    def __init__(self, rag_agent: AdvancedRAGAgent):
        self.general_agent = GeneralAgent()
        self.vector_agent = VectorAgent(rag_agent=rag_agent)
        self.synthesizer = Synthesizer()
        
    def run(self, query: str, session: RAGSession) -> str:
        """
        Executes the agentic workflow: Route -> Dispatch -> Synthesize
        """
        logger.info(f"Orchestrator received query: {query}")
        
        # 1. Gather Context
        chat_history = session.chat_history
        memory_context = session.get_memory_context(query)
        
        # 2. Route Intent (Advanced Semantic Engine)
        # Extract a global syllabus if we have one defined in settings/memory, else use a default.
        global_syllabus = getattr(session, "syllabus", None) 
        
        intent = classify_intent(
            query, 
            history=chat_history, 
            vector_tool=self.vector_agent.rag_agent.vector_tool,
            syllabus=global_syllabus
        )
        logger.info(f"Orchestrator decided intent: {intent}")
        
        # 3. Dispatch to Worker
        worker_result = {}
        if intent == "GeneralChat":
            worker_result = self.general_agent.query(
                question=query, 
                chat_history=chat_history, 
                memory_context=memory_context
            )
        else: # Default to VectorRetrieval
            worker_result = self.vector_agent.query(
                question=query, 
                chat_history=chat_history, 
                memory_context=memory_context
            )
            
        # 4. Synthesize Final Response
        final_answer = self.synthesizer.synthesize(worker_result, intent)
        
        return final_answer
