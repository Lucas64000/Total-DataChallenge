"""OCR engine implementations for timestamp extraction."""

from __future__ import annotations

from typing import cast

from PIL import Image

from pipeline.etl.timestamp_ocr.data_models import OCREngineProtocol


class TrOCREngine(OCREngineProtocol):
    """
    OCR engine based on Microsoft TrOCR (``trocr-base-printed``).

    Wraps HuggingFace's ``VisionEncoderDecoderModel`` for single-line printed text recognition.
    """

    MODEL_NAME = "microsoft/trocr-base-printed"

    def __init__(self, gpu: bool = True) -> None:
        """
        Load the TrOCR model and processor.

        Args:
            gpu: Use CUDA when available. Falls back to CPU silently.
        """
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._processor = TrOCRProcessor.from_pretrained(self.MODEL_NAME)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)

        self._device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    def read(self, image: Image.Image) -> str:
        """
        Read text from a single PIL image.

        Args:
            image: Input image (any mode — converted to RGB internally).

        Returns:
            Decoded text string from the image.
        """
        return self.read_batch([image])[0]

    def read_batch(self, images: list[Image.Image]) -> list[str]:
        """
        Read text from multiple images in one batched forward pass.

        Args:
            images: PIL images to process. Empty list returns ``[]``.

        Returns:
            Decoded text strings aligned with the input order.
        """
        import torch

        if not images:
            return []

        rgb_images = [img.convert("RGB") for img in images]
        # Processor applies TrOCR feature extraction + tensor padding for the whole batch.
        pixel_values = self._processor(rgb_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values, max_length=64)

        decoded = cast(
            list[str],
            self._processor.batch_decode(generated_ids, skip_special_tokens=True),
        )
        return decoded
