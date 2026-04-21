import re
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from bs4 import BeautifulSoup

from ..utils.llm import chat_sync


class TableReformatter:
    """
    Reformats HTML tables for optimal embedding and retrieval.

    Strategy:
    1. Parse table structure (headers, rows, cells)
    2. Generate natural language narrative via LLM
    3. Preserve all numeric values and relationships
    4. Fallback to row-as-sentence linearization
    """

    SYSTEM_PROMPT = """You are a table analysis expert. Transform HTML tables into a format optimized for semantic search.

Guidelines:
1. Create a natural language narrative that preserves ALL data
2. Use hierarchical structure: Headers → Row descriptions → Cell values
3. Preserve exact numeric values and units
4. Include row/column relationships explicitly
5. For multi-level headers, maintain hierarchy
6. Output format should be highly searchable

Example Output:
**Table: Quarterly Revenue by Region**
- Q1 2024:
  * North America: $2.5M (↑15% YoY)
  * Europe: $1.8M (↑8% YoY)
  * Asia: $3.2M (↑22% YoY)
- Q2 2024:
  * North America: $2.7M (↑12% YoY)
  ...
"""

    REFORMAT_PROMPT = """Reformat this HTML table into a highly searchable format:

{html_table}

Generate a structured narrative that preserves all data and relationships."""

    def __init__(self, model: str, base_url: str):
        self.generator = OllamaChatGenerator(
            model=model,
            url=base_url,
            generation_kwargs={"temperature": 0.1},
        )
        # self.generator.warm_up()

    def reformat(self, html_table: str) -> str:
        """Transform HTML table into embedding-friendly format."""
        structure = self._extract_structure(html_table)

        try:
            reformatted = chat_sync(
                self.generator,
                system=self.SYSTEM_PROMPT,
                user=self.REFORMAT_PROMPT.format(html_table=html_table),
            )

            if not self._validate_preservation(html_table, reformatted):
                return self._fallback_format(structure)

            return reformatted
        except Exception:
            return self._fallback_format(structure)

    def _extract_structure(self, html: str) -> dict:
        """Extract table structure."""
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table") or soup

        headers = []
        rows = []

        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]

        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells and cells != headers:
                rows.append(cells)

        return {"headers": headers, "rows": rows}

    def _validate_preservation(self, original: str, reformatted: str) -> bool:
        """Validate that numeric values are preserved."""
        original_numbers = set(re.findall(r"\d+\.?\d*", original))
        if not original_numbers:
            return True
        reformatted_numbers = set(re.findall(r"\d+\.?\d*", reformatted))
        preserved_ratio = len(original_numbers & reformatted_numbers) / len(original_numbers)
        return preserved_ratio >= 0.9

    def _fallback_format(self, structure: dict) -> str:
        """Fallback: row-as-sentence linearization."""
        lines = []
        headers = structure["headers"]
        rows = structure["rows"]

        if headers:
            lines.append(f"Table with columns: {', '.join(headers)}")
            lines.append("")

        for i, row in enumerate(rows):
            pairs = []
            for j, cell in enumerate(row):
                header = headers[j] if j < len(headers) else f"Column {j + 1}"
                if cell.strip():
                    pairs.append(f"{header}: {cell}")

            if pairs:
                row_label = row[0] if row else f"Row {i + 1}"
                lines.append(f"For {row_label}, {'; '.join(pairs[1:] if len(pairs) > 1 else pairs)}.")

        return "\n".join(lines)
