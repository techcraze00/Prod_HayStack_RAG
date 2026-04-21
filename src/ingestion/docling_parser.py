from typing import List
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel

from src.ingestion.unstructured_parser import DocumentParser, ParsedElement


class DoclingParser(DocumentParser):
    """
    Document parser using Docling's standard PDF pipeline (layout detection
    + OCR + table structure recognition).  Models are auto-downloaded on
    first run.

    Inherits consolidate_elements() and save_parsed_json() from DocumentParser.
    Only overrides __init__() and parse() to use Docling instead of unstructured.
    """

    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def parse(self, file_path: str) -> List[ParsedElement]:
        """Parse document using Docling standard pipeline."""
        result = self.converter.convert(file_path)
        doc = result.document

        parsed_elements = []

        for item, _level in doc.iterate_items():
            label = item.label if hasattr(item, "label") else None
            prov = getattr(item, "prov", [])
            page_no = prov[0].page_no if prov else 1

            if label == DocItemLabel.TABLE:
                md_table = item.export_to_markdown(doc=doc)
                parsed_elements.append(
                    ParsedElement(
                        element_type="table",
                        content=md_table,
                        metadata={"has_html": False, "element_type": "Table"},
                        page_number=page_no,
                    )
                )

            elif label == DocItemLabel.PICTURE:
                parsed_elements.append(
                    ParsedElement(
                        element_type="image",
                        content=getattr(item, "text", str(item)),
                        metadata={"image_path": None, "element_type": "Picture"},
                        page_number=page_no,
                    )
                )

            else:
                text = getattr(item, "text", str(item))
                if text and text.strip():
                    parsed_elements.append(
                        ParsedElement(
                            element_type="text",
                            content=text,
                            metadata={
                                "element_type": str(label) if label else "Text",
                            },
                            page_number=page_no,
                        )
                    )

        return parsed_elements
