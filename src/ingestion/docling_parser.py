from typing import List
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import (
    VlmPipelineOptions,
    InlineVlmOptions,
    InferenceFramework,
    ResponseFormat,
)
from docling_core.types.doc import DocItemLabel

from src.ingestion.unstructured_parser import DocumentParser, ParsedElement


class DoclingParser(DocumentParser):
    """
    Document parser using IBM Docling VLM pipeline with a locally
    downloaded HuggingFace model (MLX format for Apple Silicon).

    Inherits consolidate_elements() and save_parsed_json() from DocumentParser.
    Only overrides __init__() and parse() to use Docling instead of unstructured.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Docling model path does not exist: {self.model_path}"
            )

        vlm_options = InlineVlmOptions(
            repo_id=self.model_path.name,
            inference_framework=InferenceFramework.MLX,
            response_format=ResponseFormat.DOCTAGS,
            prompt="Convert this page to docling.",
            max_new_tokens=8192,
            stop_strings=["</doctag>", "<|end_of_text|>"],
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            artifacts_path=self.model_path.parent,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def parse(self, file_path: str) -> List[ParsedElement]:
        """Parse document using Docling VLM model."""
        result = self.converter.convert(file_path)
        doc = result.document

        parsed_elements = []

        for item, _level in doc.iterate_items():
            label = item.label if hasattr(item, "label") else None
            prov = getattr(item, "prov", [])
            page_no = prov[0].page_no if prov else 1

            if label == DocItemLabel.TABLE:
                md_table = item.export_to_markdown()
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
