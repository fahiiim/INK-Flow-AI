"""Unified AI Brain processor used by backend services."""

from __future__ import annotations

from .errors import AnalysisPipelineError
from .extraction import TattooTextExtractor
from .routing import TattooRouter
from .schemas import AIExtractionOutput, TattooInquiryInput
from .vision import TattooVisionAnalyzer


class StudioAIBrain:
    """Single entry-point for tattoo inquiry analysis."""

    def __init__(
        self,
        vision_analyzer: TattooVisionAnalyzer | None = None,
        text_extractor: TattooTextExtractor | None = None,
        router: TattooRouter | None = None,
    ) -> None:
        self.vision_analyzer = vision_analyzer or TattooVisionAnalyzer()
        self.text_extractor = text_extractor or TattooTextExtractor()
        self.router = router or TattooRouter()

    def process_inquiry(
        self,
        text: str,
        image_urls: list[str],
    ) -> AIExtractionOutput:
        """Process text and images into strict AIExtractionOutput."""
        try:
            inquiry = TattooInquiryInput(
                client_text=text,
                image_urls=image_urls,
            )
        except Exception as exc:
            raise AnalysisPipelineError("Invalid inquiry payload.") from exc

        style_tags = self.vision_analyzer.analyze_styles(inquiry.image_urls)
        extracted = self.text_extractor.extract(
            client_text=inquiry.client_text,
            style_tags=style_tags,
            image_urls=inquiry.image_urls,
        )
        return self.router.route(extracted)


if __name__ == "__main__":
    brain = StudioAIBrain()
    mock_text = (
        "Hey, I want a tiny fine-line lotus on my inner wrist. "
        "Maybe around 4 cm. Black ink only. Can I come next Tuesday?"
    )
    mock_images = [
        "https://example.com/reference-1.jpg",
        "https://example.com/reference-2.jpg",
    ]

    try:
        output = brain.process_inquiry(text=mock_text, image_urls=mock_images)
        print(output.model_dump_json(indent=2))
    except Exception as exc:
        print(f"Processor test run failed: {exc}")
