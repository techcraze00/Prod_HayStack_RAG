from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import re
from unstructured.partition.auto import partition


# Noise patterns to filter during consolidation
_NOISE_PATTERNS = [
    re.compile(r'^\d{1,4}$'),                          # Bare page numbers
    re.compile(r'^[\s\-—_]+$'),                         # Separators/whitespace
    re.compile(r'.*Hkkx.*', re.IGNORECASE),             # Hindi transliterations
    re.compile(r'^\[?PART\s+II.*SEC.*\]?$', re.IGNORECASE),  # Gazette section refs
]

# Section boundary patterns (start of a new logical section)
_SECTION_BOUNDARY = re.compile(
    r'^(?:'
    r'Rule\s+\d+'
    r'|Chapter[\s\-]+[IVX\d]+'
    r'|SCHEDULE[\s—\-]+[IVX\d]+'
    r'|FORM\s+[A-Z]+'
    r'|PART\s+[IVX\d]+'
    r'|Division\s+\d+'
    r'|Table\s+\d+'
    r')',
    re.IGNORECASE,
)


@dataclass
class ParsedElement:
    """Represents a parsed document element"""

    element_type: str  # "text", "table", "image"
    content: str
    metadata: Dict[str, Any]
    page_number: int


class DocumentParser:
    """
    Parses documents using unstructured library (framework-agnostic).
    """

    def __init__(self, vision_processor=None, table_reformatter=None):
        self.vision_processor = vision_processor
        self.table_reformatter = table_reformatter

    def parse(self, file_path: str) -> List[ParsedElement]:
        """Parse document with layout detection."""
        file_path = Path(file_path)

        images_dir = file_path.parent / ".images"
        images_dir.mkdir(exist_ok=True)

        elements = partition(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=str(images_dir),
        )

        parsed_elements = []
        current_page = 1

        for element in elements:
            element_type = type(element).__name__

            if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
                current_page = element.metadata.page_number or current_page

            if element_type == "Table":
                parsed_elements.append(self._process_table(element, current_page))
            elif element_type == "Image":
                parsed_elements.append(self._process_image(element, current_page))
            else:
                parsed_elements.append(
                    ParsedElement(
                        element_type="text",
                        content=str(element),
                        metadata={"element_type": element_type},
                        page_number=current_page,
                    )
                )

        return parsed_elements

    # ── Element Consolidation ─────────────────────────────────────────

    def consolidate_elements(
        self, elements: List[ParsedElement]
    ) -> List[ParsedElement]:
        """
        Consolidate fine-grained parsed elements into coherent sections.

        1. Filter noise (page numbers, gazette headers, empty strings)
        2. Merge consecutive text elements into section-level blocks
        3. Detect section boundaries (Rule/Chapter/Schedule)
        4. Attach section context as metadata
        5. Keep tables/images as separate elements with section context
        """
        consolidated = []
        current_texts: List[str] = []
        current_section = "Document"
        current_pages: List[int] = []

        def _flush():
            if not current_texts:
                return
            merged = "\n".join(current_texts).strip()
            if len(merged) < 20:
                return
            consolidated.append(
                ParsedElement(
                    element_type="text",
                    content=merged,
                    metadata={
                        "section": current_section,
                        "consolidated": True,
                    },
                    page_number=current_pages[0] if current_pages else 1,
                )
            )

        for elem in elements:
            content = elem.content.strip()

            if not content or self._is_noise(content):
                continue

            if elem.element_type in ("table", "image"):
                _flush()
                current_texts = []
                current_pages = []
                elem.metadata["section"] = current_section
                consolidated.append(elem)
                continue

            if _SECTION_BOUNDARY.match(content):
                _flush()
                current_texts = []
                current_pages = []
                current_section = content.split(".")[0].split("—")[0].strip()[:80]

            current_texts.append(content)
            current_pages.append(elem.page_number)

        _flush()
        return consolidated

    @staticmethod
    def _is_noise(text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        if text.upper() in (
            "THE GAZETTE OF INDIA : EXTRAORDINARY",
            "THE GAZETTE OF INDIA: EXTRAORDINARY",
            "NOTIFICATION",
        ):
            return True
        return any(p.match(text) for p in _NOISE_PATTERNS)

    # ── Element Processing ────────────────────────────────────────────

    def _process_table(self, element, page_number) -> ParsedElement:
        html_content = getattr(element.metadata, "text_as_html", None)

        if self.table_reformatter and html_content:
            reformatted = self.table_reformatter.reformat(html_content)
        else:
            reformatted = str(element)

        return ParsedElement(
            element_type="table",
            content=reformatted,
            metadata={"has_html": html_content is not None},
            page_number=page_number,
        )

    def _process_image(self, element, page_number) -> ParsedElement:
        image_path = getattr(element.metadata, "image_path", None)

        if self.vision_processor and image_path:
            description = self.vision_processor.describe(image_path)
        else:
            description = str(element)

        return ParsedElement(
            element_type="image",
            content=description,
            metadata={"image_path": image_path},
            page_number=page_number,
        )

    def save_parsed_json(
        self, parsed_elements: List[ParsedElement], source_file: str
    ) -> Path:
        """Save parsed document as JSON"""
        from ..config import settings

        output_dir = Path(settings.parsed_docs_dir)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{Path(source_file).stem}_parsed.json"

        data = [
            {
                "element_type": elem.element_type,
                "content": elem.content,
                "metadata": elem.metadata,
                "page_number": elem.page_number,
            }
            for elem in parsed_elements
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_file
