"""MegaDetector animal detection engine."""

from __future__ import annotations

from PIL import Image

from models.data_models import Detection
from utils.types import BBox

# ------------------------------------------------------------------
# MegaDetector
# ------------------------------------------------------------------

# MegaDetector V6 class ids (defined in PytorchWildlife/models/base_detector.py).
_CLASS_LABELS: dict[int, str] = {0: "animal", 1: "person"}


class MegaDetector:
    """
    Animal and person detector based on MegaDetector V6 (PytorchWildlife).

    Wraps ``pw_detection.MegaDetectorV6`` for single-image and batched
    detection. Returns both ``animal`` (class id 0) and ``person`` (class id 1)
    detections; vehicle detections (class id 2) are discarded.
    """

    # MDV6-yolov10-e offers the best accuracy/speed tradeoff among V6 variants.
    DEFAULT_VERSION = "MDV6-yolov10-e"

    def __init__(
        self,
        gpu: bool = True,
        confidence_threshold: float = 0.2,
        version: str = DEFAULT_VERSION,
    ) -> None:
        """
        Load the MegaDetector V6 model.

        Args:
            gpu: Use CUDA when available. Falls back to CPU silently.
            confidence_threshold: Minimum detector confidence to keep a detection.
            version: MegaDetector V6 variant. Options: ``MDV6-yolov9-c``,
                ``MDV6-yolov9-e``, ``MDV6-yolov10-c``, ``MDV6-yolov10-e``,
                ``MDV6-rtdetr-c``.
        """
        import torch
        from PytorchWildlife.models import detection as pw_detection

        self._confidence_threshold = confidence_threshold

        device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self._model = pw_detection.MegaDetectorV6(
            device=device,
            pretrained=True,
            version=version,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image) -> list[Detection]:
        """
        Detect animals and persons in one image.

        Args:
            image: Input PIL image (any mode — converted to RGB internally).

        Returns:
            Detections sorted by confidence (highest first).
        """
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[Image.Image]) -> list[list[Detection]]:
        """
        Detect animals and persons in multiple images.

        Args:
            images: Input PIL images. Empty list returns ``[]``.

        Returns:
            Per-image detection lists aligned with ``images`` order.
        """
        if not images:
            return []

        rgb_images = [img.convert("RGB") for img in images]
        return [
            self._parse_result(self._model.single_image_detection(img))
            for img in rgb_images
        ]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _parse_result(self, result: dict) -> list[Detection]:
        """
        Convert a PytorchWildlife detection result to Detection objects.

        Args:
            result: Dict returned by ``single_image_detection``, containing
                a ``supervision.Detections`` object under the ``"detections"`` key.

        Returns:
            Detections above the confidence threshold, sorted by confidence.
            Vehicle detections (class id 2) are excluded.
        """
        sv_detections = result.get("detections")
        if sv_detections is None or len(sv_detections) == 0:
            return []

        detections: list[Detection] = []
        for xyxy, class_id, conf in zip(
            sv_detections.xyxy,
            sv_detections.class_id,
            sv_detections.confidence,
        ):
            label = _CLASS_LABELS.get(int(class_id))
            if label is None:
                # Vehicles and any future unknown class ids are skipped.
                continue
            if float(conf) < self._confidence_threshold:
                continue
            bbox: BBox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            detections.append(Detection(bbox=bbox, confidence=float(conf), label=label))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
