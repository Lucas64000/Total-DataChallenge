"""Write/copy helpers for ETL extraction outputs."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from threading import Lock

from pipeline.etl.config import PreprocessingConfig
from pipeline.etl.extractor.data_models import ExtractionStats, FileData, FilePair


class ExtractionWriter:
    """Persist extracted files and update shared stats safely."""

    def __init__(
        self,
        config: PreprocessingConfig,
        stats: ExtractionStats,
        stats_lock: Lock,
        skip_existing: bool,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize writer with shared state for thread-safe extraction.

        Args:
            config: Preprocessing paths and dry-run flag.
            stats: Shared mutable counters.
            stats_lock: Lock protecting concurrent stat increments
            skip_existing: When ``True``, skip files already on disk.
            logger: Module logger for warnings and errors.
        """
        self._config = config
        self._stats = stats
        self._stats_lock = stats_lock
        self._skip_existing = skip_existing
        self._logger = logger

    def write_unlabeled_image(self, image: FileData) -> None:
        """
        Write one unlabeled image to the output directory.

        Args:
            image: Lazy reference to the source image.
        """
        # Resolve destination once so skip/write decisions use the same path
        dest = self._resolve_unlabeled_destination(image)

        if self._skip_existing and dest.exists():
            self._inc("skipped_existing")
            return

        if not self._config.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(image.read_content())

        self._inc("unlabeled_images")

    def extract_complete_pair(self, pair: FilePair) -> None:
        """
        Extract one complete image/annotation pair.

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
            self._inc("skipped_existing")
            return

        if not self._config.dry_run:
            dest_image.write_bytes(pair.image.read_content())
            dest_annotation.write_bytes(pair.annotation.read_content())

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

    def quarantine_unlabeled_duplicate(self, file_data: FileData) -> None:
        """
        Quarantine one duplicate unlabeled image file.

        Args:
            file_data: Duplicate unlabeled image reference.
        """
        self._logger.warning(
            "Duplicate unlabeled filename skipped: %s (source: %s)",
            file_data.name,
            file_data.source_hint,
        )
        self._inc("duplicate_unlabeled_images")
        self._write_duplicate_file(file_data, self._config.duplicates_dir / "unlabeled")

    def write_classes_file(self, content: bytes) -> None:
        """
        Write ``classes.txt`` content to the annotation directory.

        Args:
            content: Raw bytes to persist.
        """
        if self._config.dry_run:
            return
        dest = self._config.paths.classes_file
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
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file_data.read_content())

    def _resolve_unlabeled_destination(self, file_data: FileData) -> Path:
        """
        Resolve output destination for an unlabeled image.

        Keeps original filename when available. If a file with the same name
        already exists, uses a deterministic hashed suffix to avoid overwrite.

        Args:
            file_data: Unlabeled image descriptor.

        Returns:
            Destination path for writing.
        """
        destination_dir = self._config.paths.unlabeled
        primary = destination_dir / file_data.name
        if not primary.exists():
            return primary

        if self._skip_existing:
            # In incremental runs, existing file means skip
            return primary

        # When overwrite is allowed, avoid clobbering by using a stable hashed suffix
        hashed = destination_dir / self._hashed_duplicate_name(file_data)
        self._logger.warning(
            "Duplicate unlabeled filename detected: %s (source: %s) -> writing as %s",
            file_data.name,
            file_data.source_hint,
            hashed.name,
        )
        self._inc("duplicate_unlabeled_images")
        return hashed

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
            current = getattr(self._stats, field_name)
            setattr(self._stats, field_name, current + 1)
