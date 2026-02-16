import json
from typing import List, Dict, Any

from haystack.dataclasses import ChatMessage

from ...storage.base import SummaryStoreInterface
from ...utils.llm import chat_sync


class SummaryIndexStrategy:
    """
    Universal Summary Strategy.
    Generates and retrieves:
    1. Full Document Summary
    2. Topic-wise Summary
    3. Section-wise Summary
    """

    FULL_PROMPT = """Summarize this entire document comprehensively.
Include main topics, key entities, dates, and conclusions."""

    TOPIC_EXTRACT_PROMPT = """Identify the top 5 distinct topics discussed in this text.
Return ONLY a JSON array of strings: ["topic1", "topic2", ...]"""

    TOPIC_SUMMARIZE_PROMPT = """Summarize the document specifically regarding the topic: '{topic}'.
Focus only on details relevant to this topic."""

    SECTION_PROMPT = """Summarize this specific section of the text.
Capture the specific details and arguments presented here."""

    def __init__(self, generator, summary_store: SummaryStoreInterface):
        self.generator = generator
        self.summary_store = summary_store

    def build_universal_index(
        self, doc_id: str, full_text: str, metadata: Dict[str, Any]
    ):
        """Generate all summary types for a document."""

        # 1. Full Summary
        self.create_summary(doc_id, full_text, metadata, "full", self.FULL_PROMPT)

        # 2. Topic-wise Summary
        topics = self._extract_topics(full_text[:10000])
        for topic in topics:
            prompt = self.TOPIC_SUMMARIZE_PROMPT.format(topic=topic)
            meta = {**metadata, "topic": topic}
            self.create_summary(doc_id, full_text[:15000], meta, "topic", prompt)

        # 3. Section-wise Summary
        chunks = [full_text[i : i + 4000] for i in range(0, len(full_text), 4000)]
        for i, chunk in enumerate(chunks):
            if len(chunk) < 500:
                continue
            meta = {**metadata, "section_index": i}
            self.create_summary(doc_id, chunk, meta, "section", self.SECTION_PROMPT)

    def create_summary(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        summary_type: str,
        prompt_template: str,
    ) -> str:
        """Generate and store a specific type of summary."""
        try:
            summary = chat_sync(
                self.generator,
                system="You are a precise summarizer.",
                user=f"{prompt_template}\n\nText:\n{text[:15000]}",
            )

            storage_meta = {**metadata, "summary_type": summary_type}
            self.summary_store.add_summary(doc_id, summary, storage_meta)
            return summary
        except Exception as e:
            print(f"Error generating {summary_type} summary: {e}")
            return ""

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics."""
        try:
            response = chat_sync(
                self.generator,
                system="",
                user=f"{self.TOPIC_EXTRACT_PROMPT}\n\nText:\n{text}",
            )
            content = response.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception:
            return []

    def search_summaries(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search all summary types."""
        return self.summary_store.search_summaries(query, top_k)
