"""High-level ETL extractor orchestration."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from pipeline.etl.class_catalog import load_class_catalog
from pipeline.etl.config import LABELED_SPECIES, PreprocessingConfig
from pipeline.etl.extractor.data_models import ExtractionStats, FilePair
from pipeline.etl.extractor.sources import SourceScanner
from pipeline.etl.extractor.validators import ImageValidator, YOLOValidator
from pipeline.etl.extractor.writer import ExtractionWriter
from utils.logging_system import LogCategory, get_phototrap_logger

# ------------------------------------------------------------------
# Extractor Orchestration
# ------------------------------------------------------------------


class Extractor:
    """
    Extract and organize camera trap data.

    Handles both regular directories and ZIP archives.
    File contents are loaded lazily on demand to minimize memory usage.
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        num_workers: int | None = None,
        skip_existing: bool = True,
        move_invalid: bool = True,
    ) -> None:
        """
        Initialize extractor runtime settings.

        Args:
            config: ETL preprocessing configuration (paths, dry-run, backup behavior).
                When ``None``, default ``PreprocessingConfig`` is used.
            num_workers: Maximum worker threads for labeled pair extraction.
                When ``None``, uses ``os.cpu_count()`` and is capped to 16.
            skip_existing: Whether to skip writing files that already exist in output.
            move_invalid: When ``True``, files that fail validation are moved to
                ``backup/invalid`` instead of being silently dropped.

        Raises:
            ValueError: If ``num_workers`` is provided and is <= 0.
        """
        self._config = config or PreprocessingConfig()
        if num_workers is not None and num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        resolved_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        # Upper bound keeps IO-bound extraction from overscheduling too many threads.
        self._num_workers = min(resolved_workers, 16)
        self._skip_existing = skip_existing
        self._move_invalid = move_invalid
        self._logger = get_phototrap_logger().get_logger(LogCategory.PREPROCESSING, "extractor")
        self._stats = ExtractionStats()
        self._stats_lock = Lock()
        self._scanner = SourceScanner(self._logger)
        self._image_validator = ImageValidator()
        self._yolo_validator: YOLOValidator | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> ExtractionStats:
        """
        Run full extraction for labeled and unlabeled assets.

        Returns:
            Aggregated extraction statistics.
        """
        self._stats = ExtractionStats()
        self._config.ensure_dirs()

        writer = self._make_writer()
        self._extract_labelized(writer)
        self._extract_unlabelized(writer)

        self._log_summary()
        return self._stats

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _make_writer(self) -> ExtractionWriter:
        """
        Build the shared writer used by extraction workers.

        Returns:
            Writer configured with shared stats, lock, and validators.
        """
        return ExtractionWriter(
            config=self._config,
            stats=self._stats,
            stats_lock=self._stats_lock,
            skip_existing=self._skip_existing,
            logger=self._logger,
            image_validator=self._image_validator,
            yolo_validator=self._yolo_validator,
            move_invalid=self._move_invalid,
        )

    def _extract_labelized(self, writer: ExtractionWriter) -> None:
        """
        Extract labeled assets and dispatch each stem pair to workers.

        Args:
            writer: Writer responsible for persisting extracted files.
        """
        source_labelized = self._config.paths.source_dir / "labelized"
        if not source_labelized.exists():
            self._logger.warning("Source directory not found: %s", source_labelized)
            return

        self._logger.info("Extracting labelized from: %s", source_labelized)

        self._copy_classes_file(source_labelized, writer)
        self._load_yolo_validator()
        scan_result = self._scanner.scan_labelized_sources(source_labelized)
        file_pairs = scan_result.pairs
        # Duplicates are quarantined immediately and excluded from main extraction flow.
        for duplicate_image in scan_result.duplicate_images:
            writer.quarantine_labelized_duplicate_image(duplicate_image)
        for duplicate_annotation in scan_result.duplicate_annotations:
            writer.quarantine_labelized_duplicate_annotation(duplicate_annotation)

        self._logger.info("Found %d unique file stems", len(file_pairs))

        pairs_list = sorted(file_pairs.values(), key=lambda pair: pair.stem)
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            # Keep future -> pair map to report precise failures from worker threads.
            futures = {
                executor.submit(self._process_pair, writer, pair): pair for pair in pairs_list
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting labelized",
                unit="pair",
            ):
                try:
                    future.result()
                except Exception:
                    pair = futures[future]
                    self._logger.error("Failed to process pair %s", pair.stem, exc_info=True)
                    with self._stats_lock:
                        self._stats.extraction_errors += 1

    def _extract_unlabelized(self, writer: ExtractionWriter) -> None:
        """
        Extract unlabeled images from directory and ZIP sources.

        Args:
            writer: Writer responsible for persisting extracted files.
        """
        source_unlabelized = self._config.paths.source_dir / "unlabelized"
        if not source_unlabelized.exists():
            self._logger.warning("Source directory not found: %s", source_unlabelized)
            return

        self._logger.info("Extracting unlabelized from: %s", source_unlabelized)

        images = self._scanner.scan_unlabeled_sources(source_unlabelized)
        self._logger.info("Found %d unlabelized images", len(images))

        # Unlabeled extraction sequential:
        # deterministic handling when multiple files share the same filename.
        for image in tqdm(images, desc="Extracting unlabelized", unit="image"):
            try:
                writer.write_unlabeled_image(image)
            except Exception:
                self._logger.error("Failed to extract %s", image.name, exc_info=True)
                with self._stats_lock:
                    self._stats.extraction_errors += 1

    def _copy_classes_file(self, root: Path, writer: ExtractionWriter) -> None:
        """
        Copy ``classes.txt`` discovered by the source scanner.

        Args:
            root: Labeled source root.
            writer: Writer used to persist ``classes.txt``.
        """
        classes_payload = self._scanner.find_classes_content(root)
        if classes_payload is not None:
            content, source_hint = classes_payload
            writer.write_classes_file(content)
            self._logger.info("Copied classes.txt from %s", source_hint)
            return

        self._logger.warning("No classes.txt found in source")

    def _load_yolo_validator(self) -> None:
        """
        Build a ``YOLOValidator`` from the extracted ``classes.txt``.

        Called after ``_copy_classes_file`` so that the output file exists.
        If ``classes.txt`` is missing or unreadable, labeled annotation
        validation is skipped (``_yolo_validator`` stays ``None``).
        """
        classes_file = self._config.paths.classes_file
        if not classes_file.exists():
            self._logger.warning(
                "No classes.txt at %s — YOLO annotation validation disabled",
                classes_file,
            )
            return

        try:
            catalog = load_class_catalog(classes_file, LABELED_SPECIES)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self._logger.warning(
                "Could not load class catalog: %s — YOLO annotation validation disabled",
                exc,
            )
            return

        self._yolo_validator = YOLOValidator(classes=list(catalog.source_classes))
        self._logger.info("YOLO validator ready: %d classes", self._yolo_validator.num_classes)

    @staticmethod
    def _process_pair(writer: ExtractionWriter, pair: FilePair) -> None:
        """
        Route one file stem to the correct extraction action.

        Args:
            writer: Writer that handles output paths and stats.
            pair: Stem-level pair that may contain image and/or annotation.
        """
        # The scanner already guarantees at most one image and one annotation per stem.
        if pair.image and pair.annotation:
            writer.extract_complete_pair(pair)
        elif pair.image:
            writer.extract_image_only(pair)
        elif pair.annotation:
            writer.extract_annotation_only(pair)

    def _log_summary(self) -> None:
        """Log aggregated extraction counters."""
        self._logger.info(
            "Extraction complete: %d pairs, %d orphan images, %d orphan annotations, %d unlabeled images, "
            "%d duplicate labeled images, %d duplicate labeled annotations, %d duplicate unlabeled images, "
            "%d invalid labeled, %d invalid unlabeled "
            "(%d skipped, %d errors)",
            self._stats.pairs_extracted,
            self._stats.orphan_images,
            self._stats.orphan_annotations,
            self._stats.unlabeled_images,
            self._stats.duplicate_labelized_images,
            self._stats.duplicate_labelized_annotations,
            self._stats.duplicate_unlabeled_images,
            self._stats.invalid_labeled,
            self._stats.invalid_unlabeled,
            self._stats.skipped_existing,
            self._stats.extraction_errors,
        )
