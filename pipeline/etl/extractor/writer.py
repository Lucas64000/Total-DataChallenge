"""Write/copy helpers for ETL extraction outputs."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from threading import Lock

from pipeline.etl.config import PreprocessingConfig
from pipeline.etl.extractor.data_models import ExtractionStats, FileData, FilePair
from pipeline.etl.extractor.validators import ImageValidator, ValidationResult, YOLOValidator


class ExtractionWriter:
    """Persist extracted files and update shared stats safely."""

    def __init__(
        self,
        config: PreprocessingConfig,
        stats: ExtractionStats,
        stats_lock: Lock,
        skip_existing: bool,
        logger: logging.Logger,
        image_validator: ImageValidator,
        yolo_validator: YOLOValidator | None = None,
        move_invalid: bool = True,
    ) -> None:
        """
        Initialize writer with shared state for thread-safe extraction.

        Args:
            config: Preprocessing paths and dry-run flag.
            stats: Shared mutable counters.
            stats_lock: Lock protecting concurrent stat increments.
            skip_existing: When ``True``, skip files already on disk.
            logger: Module logger for warnings and errors.
            image_validator: Validates image bytes before writing.
            yolo_validator: Validates YOLO annotation content. ``None`` when
                no ``classes.txt`` was found (labeled validation is skipped).
            move_invalid: When ``True``, invalid files are written to
                ``backup/invalid`` instead of being silently dropped.
        """
        self._config = config
        self._stats = stats
        self._stats_lock = stats_lock
        self._skip_existing = skip_existing
        self._logger = logger
        self._image_validator = image_validator
        self._yolo_validator = yolo_validator
        self._move_invalid = move_invalid

    def write_unlabeled_image(self, image: FileData) -> None:
        """
        Validate and write one unlabeled image to the output directory.

        Args:
            image: Lazy reference to the source image.
        """
        dest = self._resolve_unlabeled_destination(image)

        if self._skip_existing and dest.exists():
            # Incremental mode: existing output is treated as already processed.
            self._inc("skipped_existing")
            return

        # Read bytes once; reuse for validation and disk write.
        image_bytes = image.read_content()

        # Gate: image integrity check before writing to output.
        result = self._image_validator.validate(image_bytes, image.name)
        if not result.is_valid:
            self._logger.warning(
                "Invalid unlabeled image %s: %s", image.name, result.error
            )
            self._quarantine_invalid_unlabeled(image_bytes, image, result.error or "")
            self._inc("invalid_unlabeled")
            return

        # A name change means the resolver found a collision and produced a hashed fallback.
        is_collision = dest.name != image.name
        if is_collision:
            self._logger.warning(
                "Duplicate unlabeled filename detected: %s (source: %s) -> writing as %s",
                image.name,
                image.source_hint,
                dest.name,
            )

        if not self._config.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(image_bytes)

        if is_collision:
            self._inc("duplicate_unlabeled_images")
        self._inc("unlabeled_images")

    def extract_complete_pair(self, pair: FilePair) -> None:
        """
        Validate and extract one complete image/annotation pair.

        Image bytes and annotation content are validated in memory before
        any disk write occurs. Invalid pairs are quarantined and counted.

        Args:
            pair: Stem pair with both image and annotation present.

        Raises:
            ValueError: If image or annotation is missing.
        """
        if not pair.image or not pair.annotation:
            raise ValueError(f"Incomplete pair for stem '{pair.stem}'")

        dest_image = self._config.paths.labelized_images / pair.image.name
        dest_annotation = self._config.paths.labelized_annotations / pair.annotation.name

        if self._skip_existing and dest_image.exists() and dest_annotation.exists():
            # Skip only when both files already exist to preserve pair consistency.
            self._inc("skipped_existing")
            return

        # Read bytes once; reuse for validation and disk write.
        image_bytes = pair.image.read_content()
        annotation_bytes = pair.annotation.read_content()

        # Gate 1: image integrity (corrupt JPEG, wrong format, bad dimensions).
        image_result = self._image_validator.validate(image_bytes, pair.image.name)
        if not image_result.is_valid:
            self._logger.warning(
                "Invalid image %s: %s", pair.image.name, image_result.error
            )
            self._quarantine_invalid_labeled(
                image_bytes, annotation_bytes, pair, image_result.error or ""
            )
            self._inc("invalid_labeled")
            return

        # Gate 2: YOLO annotation syntax and geometry.
        if self._yolo_validator is not None:
            annotation_text = annotation_bytes.decode("utf-8", errors="replace")
            annotation_result = self._yolo_validator.validate(
                annotation_text, pair.annotation.name
            )
            if not annotation_result.is_valid:
                self._logger.warning(
                    "Invalid annotation %s: %s",
                    pair.annotation.name,
                    annotation_result.error,
                )
                self._quarantine_invalid_labeled(
                    image_bytes, annotation_bytes, pair, annotation_result.error or ""
                )
                self._inc("invalid_labeled")
                return

        if not self._config.dry_run:
            dest_image.write_bytes(image_bytes)
            dest_annotation.write_bytes(annotation_bytes)

        self._inc("pairs_extracted")
        self._inc("images_extracted")
        self._inc("annotations_extracted")

    def extract_image_only(self, pair: FilePair) -> None:
        """
        Quarantine an image orphan missing its annotation.

        Args:
            pair: Stem pair with an image but no annotation.

        Raises:
            ValueError: If image is missing.
        """
        if not pair.image:
            raise ValueError(f"Missing image for stem '{pair.stem}'")
        self._logger.warning("Orphan image detected: %s", pair.image.name)

        dest = self._config.orphans_dir / "images" / pair.image.name
        self._write_orphan_file(pair.image, dest)
        self._inc("orphan_images")

    def extract_annotation_only(self, pair: FilePair) -> None:
        """
        Quarantine an annotation orphan missing its image.

        Args:
            pair: Stem pair with an annotation but no image.

        Raises:
            ValueError: If annotation is missing.
        """
        if not pair.annotation:
            raise ValueError(f"Missing annotation for stem '{pair.stem}'")
        self._logger.warning("Orphan annotation detected: %s", pair.annotation.name)

        dest = self._config.orphans_dir / "annotations" / pair.annotation.name
        self._write_orphan_file(pair.annotation, dest)
        self._inc("orphan_annotations")

    def quarantine_labelized_duplicate_image(self, file_data: FileData) -> None:
        """
        Quarantine one duplicate labeled image file.

        Args:
            file_data: Duplicate image reference.
        """
        self._logger.warning(
            "Duplicate labeled image skipped: %s (source: %s)",
            file_data.name,
            file_data.source_hint,
        )
        self._inc("duplicate_labelized_images")
        self._write_duplicate_file(file_data, self._config.duplicates_dir / "images")

    def quarantine_labelized_duplicate_annotation(self, file_data: FileData) -> None:
        """
        Quarantine one duplicate labeled annotation file.

        Args:
            file_data: Duplicate annotation reference.
        """
        self._logger.warning(
            "Duplicate labeled annotation skipped: %s (source: %s)",
            file_data.name,
            file_data.source_hint,
        )
        self._inc("duplicate_labelized_annotations")
        self._write_duplicate_file(file_data, self._config.duplicates_dir / "annotations")

    def write_classes_file(self, content: bytes) -> None:
        """
        Write ``classes.txt`` content to the annotation directory.

        Args:
            content: Raw bytes to persist.
        """
        if self._config.dry_run:
            return
        dest = self._config.paths.classes_file
        # Parent folders are created by config.ensure_dirs() before extraction starts.
        dest.write_bytes(content)

    def _write_orphan_file(self, file_data: FileData, destination: Path) -> None:
        """
        Persist an orphan file when backup storage is enabled.

        Args:
            file_data: File to quarantine.
            destination: Target orphan quarantine path.
        """
        if not self._config.backup_enabled:
            self._logger.warning(
                "Backup disabled; orphan not quarantined: %s (source: %s)",
                file_data.name,
                file_data.source_hint,
            )
            return

        if self._skip_existing and destination.exists():
            self._inc("skipped_existing")
            return

        if not self._config.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file_data.read_content())

    def _write_duplicate_file(self, file_data: FileData, destination_dir: Path) -> None:
        """
        Persist a duplicate file into quarantine with collision-safe naming.

        Args:
            file_data: File to quarantine.
            destination_dir: Duplicate quarantine folder.
        """
        if not self._config.backup_enabled:
            self._logger.warning(
                "Backup disabled; duplicate not quarantined: %s (source: %s)",
                file_data.name,
                file_data.source_hint,
            )
            return

        destination = destination_dir / self._hashed_duplicate_name(file_data)
        if self._skip_existing and destination.exists():
            self._inc("skipped_existing")
            return

        if not self._config.dry_run:
            # Hash-based names avoid collisions when multiple duplicates share basename.
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file_data.read_content())

    def _resolve_unlabeled_destination(self, file_data: FileData) -> Path:
        """
        Resolve output destination for an unlabeled image without side-effects.

        Returns the original filename path when available. When a file with the
        same name already exists and overwrite is allowed, returns a hashed
        fallback path to avoid clobbering. Callers are responsible for logging
        and incrementing counters based on the returned path.

        Args:
            file_data: Unlabeled image descriptor.

        Returns:
            Destination path for writing (may differ from original name on collision).
        """
        destination_dir = self._config.paths.unlabeled
        primary = destination_dir / file_data.name
        if not primary.exists():
            return primary

        if self._skip_existing:
            # Incremental runs treat the existing file as already processed.
            return primary

        # Collision: return a stable hashed name to avoid overwriting existing content.
        return destination_dir / self._hashed_duplicate_name(file_data)

    @staticmethod
    def _hashed_duplicate_name(file_data: FileData) -> str:
        """
        Build a deterministic collision-safe filename for duplicate quarantine.

        Args:
            file_data: Duplicate file descriptor.

        Returns:
            Filename suffixed with short BLAKE2b hash of the source hint.
        """
        name_path = Path(file_data.name)
        digest = hashlib.blake2b(
            file_data.source_hint.encode("utf-8"), digest_size=5
        ).hexdigest()
        return f"{name_path.stem}__{digest}{name_path.suffix}"

    def _quarantine_invalid_labeled(
        self,
        image_bytes: bytes,
        annotation_bytes: bytes,
        pair: FilePair,
        reason: str,
    ) -> None:
        """
        Persist an invalid labeled pair to ``backup/invalid``.

        Keeps image and annotation together for manual inspection.

        Args:
            image_bytes: Already-read image content.
            annotation_bytes: Already-read annotation content.
            pair: The source file pair (must have both image and annotation).
            reason: Human-readable rejection reason for logs.
        """
        if not self._move_invalid or not self._config.backup_enabled:
            return
        if self._config.dry_run:
            return

        assert pair.image is not None and pair.annotation is not None
        img_dest = self._config.invalid_dir / "images" / pair.image.name
        ann_dest = self._config.invalid_dir / "annotations" / pair.annotation.name
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        ann_dest.parent.mkdir(parents=True, exist_ok=True)
        img_dest.write_bytes(image_bytes)
        ann_dest.write_bytes(annotation_bytes)

    def _quarantine_invalid_unlabeled(
        self,
        image_bytes: bytes,
        image: FileData,
        reason: str,
    ) -> None:
        """
        Persist an invalid unlabeled image to ``backup/invalid/unlabeled``.

        Args:
            image_bytes: Already-read image content.
            image: Source file descriptor.
            reason: Human-readable rejection reason for logs.
        """
        if not self._move_invalid or not self._config.backup_enabled:
            return
        if self._config.dry_run:
            return

        dest = self._config.invalid_dir / "unlabeled" / image.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_bytes)

    def _inc(self, field_name: str) -> None:
        """
        Atomically increment one counter on the shared stats object.

        Args:
            field_name: Name of the ``ExtractionStats`` field to increment.

        Raises:
            AttributeError: If ``field_name`` is not a valid stats field.
        """
        if field_name not in ExtractionStats.__dataclass_fields__:
            raise AttributeError(f"ExtractionStats has no field '{field_name}'")
        with self._stats_lock:
            # Read-modify-write is guarded to keep counters correct under concurrency.
            current = getattr(self._stats, field_name)
            setattr(self._stats, field_name, current + 1)
