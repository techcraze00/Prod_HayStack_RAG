from pathlib import Path
import base64

from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage


class VisionProcessor:
    """
    Processes images using LLaVA vision model via Ollama.

    Generates natural language descriptions of images
    for embedding and retrieval.
    """

    DESCRIPTION_PROMPT = """Analyze this image and provide a detailed description that would be useful for search and retrieval.

Include:
1. Main subject/content
2. Key visual elements (text, charts, diagrams)
3. Any data or numbers visible
4. Context and purpose of the image

Be specific and include all searchable details."""

    def __init__(self, model: str, base_url: str):
        self.generator = OllamaChatGenerator(
            model=model,
            url=base_url,
            generation_kwargs={"temperature": 0.1},
        )
        # self.generator.warm_up()

    def describe(self, image_path: str) -> str:
        """Generate a description of the image."""
        image_path = Path(image_path)

        if not image_path.exists():
            return f"[Image not found: {image_path}]"

        image_data = self._encode_image(image_path)
        if not image_data:
            return f"[Could not read image: {image_path}]"

        # Haystack Ollama multimodal: pass image bytes in ChatMessage
        message = ChatMessage.from_user(
            text=self.DESCRIPTION_PROMPT,
        )
        # Attach image as base64 via meta for Ollama vision
        # OllamaChatGenerator supports images through the content parts
        message.meta["images"] = [image_data]

        try:
            result = self.generator.run(messages=[message])
            return result["replies"][0].text
        except Exception as e:
            return f"[Error processing image: {e}]"

    def _encode_image(self, image_path: Path) -> str | None:
        """Encode image as base64"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None
