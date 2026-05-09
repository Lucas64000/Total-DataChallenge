"""Stateful wildlife analysis orchestration."""

from __future__ import annotations

from collections import Counter
from logging import Logger
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from models.classifier import BioCLIPClassifier
from models.data_models import AnalysisResult, Classification, DetectorProtocol, ClassifierProtocol, Detection
from models.detector import MegaDetector
from utils.logging_system import LogCategory, get_phototrap_logger

# ------------------------------------------------------------------
# Result Helpers
# ------------------------------------------------------------------


def _build_error_result(
    error: str,
    detections: list[Detection] | None = None,
) -> AnalysisResult:
    """
    Build a standardized failure result.

    Args:
        error: Human-readable error message.
        detections: Optional detections when available before failure.

    Returns:
        Failed analysis result.
    """
    return AnalysisResult(
        detections=detections or [],
        classifications=[],
        animal_count=0,
        success=False,
        error=error,
    )


# ------------------------------------------------------------------
# Animal Analyzer
# ------------------------------------------------------------------


class AnimalAnalyzer:
    """
    Detect and classify animals and persons in camera-trap images.

    Orchestrates the full inference pipeline: image loading, detection via
    MegaDetector, bbox cropping, species classification via BioCLIP, and
    per-species counting. Supports both single-image and batched analysis.
    """

    def __init__(
        self,
        gpu: bool = True,
        detector: DetectorProtocol | None = None,
        classifier: ClassifierProtocol | None = None,
    ) -> None:
        """
        Initialize the analyzer with a detector and classifier.

        Args:
            gpu: Whether to use GPU acceleration for the default engines.
                Ignored when custom engines are provided.
            detector: Optional pre-built detector satisfying ``DetectorProtocol``.
                When ``None``, a ``MegaDetector`` is created automatically.
            classifier: Optional pre-built classifier satisfying ``ClassifierProtocol``.
                When ``None``, a ``BioCLIPClassifier`` is created automatically.
        """
        self._logger: Logger = get_phototrap_logger().get_logger(
            LogCategory.INFERENCE, "animal_analyzer"
        )

        if detector is not None:
            self._detector = detector
            self._logger.info("Using provided detector")
        else:
            self._logger.info("Initializing MegaDetector (GPU=%s)", gpu)
            self._detector = MegaDetector(gpu=gpu)

        if classifier is not None:
            self._classifier = classifier
            self._logger.info("Using provided classifier")
        else:
            self._logger.info("Initializing BioCLIPClassifier (GPU=%s)", gpu)
            self._classifier = BioCLIPClassifier(gpu=gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image_path: Path | str) -> AnalysisResult:
        """
        Detect and classify subjects in one image.

        Args:
            image_path: Path to the source image.

        Returns:
            Analysis result with detections, classifications, and species counts.
        """
        try:
            image = self._load_image(Path(image_path))
            return self._analyze_impl(image)
        except (OSError, ValueError, RuntimeError) as exc:
            self._logger.debug("Analysis failed for %s: %s", image_path, exc, exc_info=True)
            return _build_error_result(error=str(exc))

    def analyze_batch(
        self,
        image_paths: list[Path | str],
        show_progress: bool = True,
    ) -> list[AnalysisResult]:
        """
        Detect and classify subjects across multiple images, preserving input order.

        Args:
            image_paths: Image paths to process.
            show_progress: Whether to render a progress bar.

        Returns:
            Results aligned with ``image_paths`` order.
        """
        if not image_paths:
            return []

        paths = [Path(p) for p in image_paths]
        iterator = tqdm(paths, desc="Analyzing images") if show_progress else paths

        results = [self.analyze(p) for p in iterator]

        success_count = sum(1 for r in results if r.success)
        self._logger.info(
            "Analyzed %d/%d images successfully",
            success_count,
            len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_path: Path) -> Image.Image:
        """
        Load one image as a PIL RGB image.

        Args:
            image_path: Path to the source image.

        Returns:
            RGB PIL image.
        """
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _crop_detection(self, image: Image.Image, detection: Detection) -> Image.Image:
        """
        Crop the bounding box region from an image.

        Args:
            image: Source PIL image.
            detection: Detection whose bbox defines the crop region.

        Returns:
            Cropped RGB image ready for classification.
        """
        return image.crop(detection.bbox)

    def _analyze_impl(self, image: Image.Image) -> AnalysisResult:
        """
        Run detection + classification on a single loaded image.

        Args:
            image: RGB PIL image to analyze.

        Returns:
            Analysis result with detections, classifications, and counts.
        """
        detections = self._detector.detect(image)

        if not detections:
            return AnalysisResult(
                detections=[],
                classifications=[],
                animal_count=0,
                success=True,
            )

        # Person detections bypass the species classifier — "human" is the final class.
        animal_detections = [d for d in detections if d.label == "animal"]
        person_detections = [d for d in detections if d.label == "person"]

        classifications: list[Classification] = [
            Classification(species="human", confidence=d.confidence, bbox=d.bbox)
            for d in person_detections
        ]

        if animal_detections:
            crops = [self._crop_detection(image, d) for d in animal_detections]
            species_confs = self._classifier.classify_batch(crops)
            classifications += [
                Classification(species=species, confidence=conf, bbox=det.bbox)
                for det, (species, conf) in zip(animal_detections, species_confs)
            ]

        species_counts = dict(Counter(c.species for c in classifications))

        return AnalysisResult(
            detections=detections,
            classifications=classifications,
            animal_count=len(classifications),
            species_counts=species_counts,
            success=True,
        )
