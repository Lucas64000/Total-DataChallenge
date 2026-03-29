"""Stateful timestamp extraction orchestration."""

from __future__ import annotations

from logging import Logger
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pipeline.etl.config import PreprocessingConfig
from pipeline.etl.timestamp_ocr.camera_profiles import CameraProfile, get_profile
from pipeline.etl.timestamp_ocr.data_models import OCREngineProtocol, TimestampResult
from pipeline.etl.timestamp_ocr.parser import parse_timestamp
from pipeline.etl.timestamp_ocr.trocr_engine import TrOCREngine
from pipeline.etl.transform.filename_parser import detect_camera_type
from utils.logging_system import LogCategory, get_phototrap_logger
from utils.types import CameraType

# ------------------------------------------------------------------
# Result Helpers
# ------------------------------------------------------------------

def _build_error_result(
    camera_type: CameraType,
    error: str,
    raw_text: str | None = None,
) -> TimestampResult:
    """
    Build a standardized failure result.

    Args:
        camera_type: Camera type associated with the image.
        error: Human-readable error message.
        raw_text: Optional OCR text when available.

    Returns:
        Failed timestamp extraction result.
    """
    return TimestampResult(
        timestamp=None,
        camera_type=camera_type,
        raw_text=raw_text,
        success=False,
        error=error,
    )


# ------------------------------------------------------------------
# Timestamp Extractor
# ------------------------------------------------------------------

class TimestampExtractor:
    """
    Extract timestamps from camera-trap images using OCR + profile parsing.

    Orchestrates the full extraction pipeline: camera type detection, image
    cropping via camera profiles, OCR inference, and timestamp parsing.
    Supports both single-image and batched extraction.
    """

    def __init__(
        self,
        gpu: bool = True,
        ocr_engine: OCREngineProtocol | None = None,
        config: PreprocessingConfig | None = None,
    ) -> None:
        """
        Initialize the extractor with an OCR engine.

        Args:
            gpu: Whether to use GPU acceleration for the default TrOCR engine.
                Ignored when a custom ``ocr_engine`` is provided.
            ocr_engine: Optional pre-built OCR engine satisfying
                ``OCREngineProtocol``. When ``None``, a ``TrOCREngine``
                is created automatically.
            config: ETL preprocessing configuration. When ``None``, defaults are used.
        """
        self._logger: Logger = get_phototrap_logger().get_logger(
            LogCategory.PREPROCESSING, "timestamp_extractor"
        )

        self._config = config or PreprocessingConfig()
        self._ts_config = self._config.timestamp

        if ocr_engine is not None:
            self._ocr_engine = ocr_engine
            self._logger.info("Using provided OCR engine")
        else:
            self._logger.info("Initializing TrOCR engine (GPU=%s)", gpu)
            self._ocr_engine = TrOCREngine(gpu=gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image_path: Path | str,
        camera_type: CameraType | None = None,
    ) -> TimestampResult:
        """
        Extract a timestamp from one image.

        Args:
            image_path: Path to the source image.
            camera_type: Optional pre-detected camera type. If omitted, it is
                inferred from the filename.

        Returns:
            Timestamp extraction result.
        """
        image_path = Path(image_path)
        # Preserve user-provided camera type when available; otherwise infer from filename.
        resolved_type: CameraType = (
            camera_type if camera_type is not None else detect_camera_type(image_path.name)
        )
        try:
            profile = get_profile(resolved_type)
        except KeyError:
            return _build_error_result(
                camera_type="unknown",
                error=f"Unsupported camera type: {resolved_type}",
            )
        return self._extract_impl(image_path, resolved_type, profile)

    def _crop_for_profile(self, image_path: Path, profile: CameraProfile) -> Image.Image:
        """
        Load one image and return the profile-specific RGB crop.

        Args:
            image_path: Path to the source image.
            profile: Camera profile containing crop ratios.

        Returns:
            Cropped RGB image ready for OCR.
        """
        with Image.open(image_path) as img:
            width, height = img.size
            crop_region = profile.crop_region.to_pixels(width, height)
            return img.crop(crop_region).convert("RGB")

    def _result_from_raw_text(
        self,
        raw_text: str,
        camera_type: CameraType,
        profile: CameraProfile,
    ) -> TimestampResult:
        """
        Parse OCR text and build a success/failure extraction result.

        Args:
            raw_text: OCR output.
            camera_type: Camera type associated with the source image.
            profile: Parsing profile for this camera type.

        Returns:
            Timestamp extraction result.
        """
        timestamp = parse_timestamp(
            raw_text,
            profile,
            self._ts_config.min_year,
            self._ts_config.max_year,
        )

        if timestamp:
            return TimestampResult(
                timestamp=timestamp,
                camera_type=camera_type,
                raw_text=raw_text,
                success=True,
            )

        return _build_error_result(
            camera_type=camera_type,
            error=f"Could not parse timestamp from: {raw_text}",
            raw_text=raw_text,
        )

    def _extract_impl(
        self,
        image_path: Path,
        camera_type: CameraType,
        profile: CameraProfile,
    ) -> TimestampResult:
        """
        Run crop + OCR + parsing for one image/profile pair.

        Args:
            image_path: Path to the source image.
            camera_type: Camera type associated with the image.
            profile: OCR crop/parsing profile for this camera type.

        Returns:
            Timestamp extraction result.
        """
        try:
            cropped = self._crop_for_profile(image_path, profile)
            raw_text = self._ocr_engine.read(cropped)
            return self._result_from_raw_text(raw_text, camera_type, profile)

        except (OSError, ValueError, RuntimeError) as exc:
            self._logger.debug("Extraction failed for %s: %s", image_path, exc, exc_info=True)
            return _build_error_result(camera_type=camera_type, error=str(exc))

    def extract_batch(
        self,
        image_paths: list[Path | str],
        show_progress: bool = True,
        batch_size: int = 16,
    ) -> list[TimestampResult]:
        """
        Extract timestamps from multiple images while preserving input order.

        Args:
            image_paths: Image paths to process.
            show_progress: Whether to render a progress bar.
            batch_size: Number of images per OCR batch.

        Returns:
            Results aligned with ``image_paths`` order.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if not image_paths:
            return []

        # Preallocate by original index so batch processing can fill results in-order.
        results: list[TimestampResult | None] = [None] * len(image_paths)
        paths = [Path(p) for p in image_paths]

        # Keep index + metadata per image; index is used to write back deterministically.
        to_process: list[tuple[int, Path, CameraType, CameraProfile]] = []

        for idx, image_path in enumerate(paths):
            camera_type = detect_camera_type(image_path.name)
            # Profile lookup is done once per image and reused through batch processing.
            profile = get_profile(camera_type)
            to_process.append((idx, image_path, camera_type, profile))

        if to_process:
            iterator = (
                tqdm(
                    range(0, len(to_process), batch_size),
                    desc=f"OCR batching ({len(to_process)} images)",
                    total=(len(to_process) + batch_size - 1) // batch_size,
                )
                if show_progress
                else range(0, len(to_process), batch_size)
            )
            for batch_start in iterator:
                batch = to_process[batch_start : batch_start + batch_size]
                self._process_batch(batch, results)

        final_results: list[TimestampResult] = results  # type: ignore[assignment]

        success_count = sum(1 for r in final_results if r.success)
        success_rate = 100 * success_count / len(final_results)
        self._logger.info(
            "Extracted %d/%d timestamps (%.1f%% success rate)",
            success_count,
            len(final_results),
            success_rate,
        )

        return final_results

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        batch: list[tuple[int, Path, CameraType, CameraProfile]],
        results: list[TimestampResult | None],
    ) -> None:
        """
        Process one OCR batch and write results back in-place.

        Args:
            batch: List of ``(index, path, camera_type, profile)`` tuples.
            results: Preallocated result buffer indexed by original position.
        """
        crops: list[Image.Image] = []
        # Keep a 1:1 item list aligned with `crops` for zip(...) after OCR.
        valid_items: list[tuple[int, Path, CameraType, CameraProfile]] = []

        for idx, image_path, camera_type, profile in batch:
            try:
                cropped = self._crop_for_profile(image_path, profile)
                crops.append(cropped)
                # Save the original index and metadata in the same order as `crops`.
                valid_items.append((idx, image_path, camera_type, profile))
            except (OSError, ValueError, RuntimeError) as exc:
                self._logger.debug("Failed to load %s: %s", image_path, exc)
                results[idx] = _build_error_result(camera_type=camera_type, error=str(exc))

        if not crops:
            return

        try:
            ocr_texts = self._ocr_engine.read_batch(crops)
        except (OSError, ValueError, RuntimeError) as exc:
            self._logger.warning("Batch OCR failed: %s", exc)
            for idx, _, camera_type, _ in valid_items:
                results[idx] = _build_error_result(
                    camera_type=camera_type,
                    error=f"Batch OCR failed: {exc}",
                )
            return

        for (idx, _image_path, camera_type, profile), raw_text in zip(valid_items, ocr_texts):
            results[idx] = self._result_from_raw_text(raw_text, camera_type, profile)
