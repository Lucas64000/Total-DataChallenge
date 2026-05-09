"""BioCLIP species classification engine."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from PIL import Image

from pipeline.etl.config import LABELED_SPECIES

# ------------------------------------------------------------------
# BioCLIP Classifier
# ------------------------------------------------------------------

# Species returned when no text prompt exceeds the confidence threshold.
_UNKNOWN_LABEL = "unknown"

# Prompt template used to turn a species name into a text query for BioCLIP.
_PROMPT_TEMPLATE = "a wildlife camera trap photo of {species}"


def _format_species_name(name: str) -> str:
    """Convert hyphenated ETL species name to a readable string (e.g. ``Vulpes-vulpes`` -> ``Vulpes vulpes``)."""
    return name.replace("-", " ")


class BioCLIPClassifier:
    """
    Zero-shot species classifier using BioCLIP (``hf-hub:imageomics/bioclip``).

    Encodes species name text prompts at initialization and compares them
    against image crop embeddings via cosine similarity at inference time.
    No fine-tuning is required; adding new species is as simple as extending
    the ``species`` list.
    """

    MODEL_NAME = "hf-hub:imageomics/bioclip"

    def __init__(
        self,
        gpu: bool = True,
        species: list[str] | None = None,
        unknown_threshold: float = 0.0,
    ) -> None:
        """
        Load BioCLIP and pre-compute text embeddings for each species.

        Args:
            gpu: Use CUDA when available. Falls back to CPU silently.
            species: Species labels to classify against. Defaults to the
                8 labeled species from the ETL pipeline (``LABELED_SPECIES``).
                Labels use the hyphenated ETL format (e.g. ``"Vulpes-vulpes"``).
            unknown_threshold: Minimum cosine similarity required to assign a
                species label. Predictions below this value are returned as
                ``"unknown"``. Set to ``0.0`` to always return the top match.
        """
        import open_clip

        self._unknown_threshold = unknown_threshold
        self._species: list[str] = list(species) if species is not None else list(LABELED_SPECIES)

        self._device = "cuda" if gpu and torch.cuda.is_available() else "cpu"

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)

        # Pre-compute and cache L2-normalized text embeddings for all species.
        # Re-encoding on every call would be redundant since species labels are fixed at init.
        prompts = [_PROMPT_TEMPLATE.format(species=_format_species_name(s)) for s in self._species]
        tokens = tokenizer(prompts).to(self._device)
        with torch.inference_mode():
            text_features = self._model.encode_text(tokens)
            self._text_embeddings = F.normalize(text_features, dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, crop: Image.Image) -> tuple[str, float]:
        """
        Classify one pre-cropped PIL image.

        Args:
            crop: RGB crop of a single detected subject.

        Returns:
            ``(species, confidence)`` tuple. ``species`` is ``"unknown"``
            when the top similarity is below ``unknown_threshold``.
        """
        return self.classify_batch([crop])[0]

    def classify_batch(self, crops: list[Image.Image]) -> list[tuple[str, float]]:
        """
        Classify multiple pre-cropped PIL images in one forward pass.

        Args:
            crops: RGB crops to classify. Empty list returns ``[]``.

        Returns:
            ``(species, confidence)`` tuples aligned with ``crops`` order.
        """
        if not crops:
            return []

        # Stack all crops into a single batch tensor.
        image_tensors = torch.stack(
            [self._preprocess(crop.convert("RGB")) for crop in crops]
        ).to(self._device)

        with torch.inference_mode():
            image_features = self._model.encode_image(image_tensors)
            image_features = F.normalize(image_features, dim=-1)

        # Cosine similarity matrix: (N_crops, N_species)
        similarities = image_features @ self._text_embeddings.T

        results: list[tuple[str, float]] = []
        for row in similarities:
            top_idx = int(row.argmax())
            top_conf = float(row[top_idx])
            if top_conf < self._unknown_threshold:
                results.append((_UNKNOWN_LABEL, top_conf))
            else:
                results.append((self._species[top_idx], top_conf))

        return results
