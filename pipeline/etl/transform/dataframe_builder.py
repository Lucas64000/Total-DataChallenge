"""
Build tabular metadata from extracted camera-trap images.

Workflow:
1. Scan labeled and unlabeled image folders.
2. Parse filename metadata and optional annotation stats.
3. Optionally attach OCR timestamp results.
4. Return a clean metadata DataFrame ready for deduplication and export.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.etl.config import IMAGE_EXTENSIONS, LABELED_SPECIES, PathConfig
from pipeline.etl.timestamp_ocr import TimestampExtractor
from pipeline.etl.transform.filename_parser import FilenameParser
from utils.logging_system import LogCategory, get_phototrap_logger


@dataclass(frozen=True, slots=True)
class _AnnotationSummary:
    """Parsed summary for one YOLO annotation file."""

    label_bbox_count: int = 0
    label_bbox_area_sum: float = 0.0
    label_bbox_area_sum_capped: float = 0.0
    dominant_species: str | None = None

    def to_stats(self) -> dict[str, float | int]:
        """Convert bbox metrics to record fields."""
        return {
            "label_bbox_count": self.label_bbox_count,
            "label_bbox_area_sum": self.label_bbox_area_sum,
            "label_bbox_area_sum_capped": self.label_bbox_area_sum_capped,
        }


class DataFrameBuilder:
    """
    Build metadata DataFrames from extracted camera-trap data.

    Each row represents one image with its parsed metadata, annotation stats,
    and optional OCR timestamp.
    """

    # The single output schema: every column the DataFrame will contain.
    COLUMNS: tuple[str, ...] = (
        "image_id",
        "path",
        "filename",
        "dataset",
        "labeled",
        "species",
        "camera_type",
        "location_id",
        "ocr_timestamp",
        "label_bbox_count",
        "label_bbox_area_sum",
        "label_bbox_area_sum_capped",
    )

    def __init__(
        self,
        paths: PathConfig | None = None,
        extract_timestamps: bool = True,
        gpu: bool = True,
    ) -> None:
        """
        Args:
            paths: ETL path configuration.
            extract_timestamps: Enable OCR timestamp extraction.
            gpu: Use GPU OCR backend when available.
        """
        self._paths = paths or PathConfig()
        self._extract_timestamps = extract_timestamps
        self._timestamp_extractor: TimestampExtractor | None = None
        self._logger = get_phototrap_logger().get_logger(
            LogCategory.PREPROCESSING, "dataframe_builder"
        )

        if extract_timestamps:
            self._timestamp_extractor = TimestampExtractor(gpu=gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Build a metadata DataFrame from all images in the ETL output folders.

        Args:
            show_progress: Forwarded to OCR batch extraction.

        Returns:
            DataFrame with ``COLUMNS`` schema, one row per image.
        """
        image_files = self._collect_image_files()
        if not image_files:
            self._logger.warning("No files found, returning empty DataFrame")
            return self._empty_dataframe()

        self._logger.info("Found %d images to process", len(image_files))

        # Build one metadata record per image.
        records = [
            self._build_record(file_path=file_path, dataset=dataset, labeled=labeled)
            for file_path, dataset, labeled in image_files
        ]

        # Enrich records with OCR timestamps (modifies records in place).
        ocr_success, ocr_fail = self._attach_ocr_results(
            records=records,
            image_files=image_files,
            show_progress=show_progress,
        )

        df = pd.DataFrame.from_records(records)
        df = self._finalize(df)
        self._log_build_summary(ocr_success=ocr_success, ocr_fail=ocr_fail)
        return df

    def to_csv(self, df: pd.DataFrame, output_path: Path | str) -> None:
        """
        Export DataFrame to CSV.

        Args:
            df: DataFrame to export.
            output_path: Destination CSV path (parent dirs created automatically).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self._logger.info("Exported %d rows to %s", len(df), output_path)

    # ------------------------------------------------------------------
    # Image collection
    # ------------------------------------------------------------------

    def _collect_image_files(self) -> list[tuple[Path, str, bool]]:
        """
        Collect image files from ETL output folders.

        Returns:
            List of ``(image_path, dataset_name, labeled_flag)`` tuples,
            sorted by filename for deterministic ordering.
        """
        image_files: list[tuple[Path, str, bool]] = []

        labeled_dir = self._paths.labelized_images
        if labeled_dir.exists():
            for file_path in sorted(labeled_dir.iterdir(), key=lambda path: path.name):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append((file_path, "labelized", True))

        unlabeled_dir = self._paths.unlabeled
        if unlabeled_dir.exists():
            for file_path in sorted(unlabeled_dir.iterdir(), key=lambda path: path.name):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append((file_path, "unlabeled", False))

        return image_files

    # ------------------------------------------------------------------
    # Record building (one record = one image)
    # ------------------------------------------------------------------

    def _build_record(self, file_path: Path, dataset: str, labeled: bool) -> dict[str, Any]:
        """
        Build one metadata record from one image file.

        For labeled images, the annotation species overrides the filename species
        when available (ground truth principle). Mismatches are logged as warnings.

        Args:
            file_path: Image path.
            dataset: Dataset name (``"labelized"`` or ``"unlabeled"``).
            labeled: Whether the image belongs to the labeled split.

        Returns:
            Dict with keys matching ``COLUMNS``.
        """
        metadata = FilenameParser.parse(file_path.name, labeled=labeled)
        image_id = self._build_image_id(dataset, file_path.name)

        if not metadata.parse_success:
            self._logger.debug("Parse failed for %s: %s", file_path.name, metadata.parse_error)

        # Use parsed location_id when available, otherwise generate a unique
        # fallback so that downstream grouping (e.g. dedup) treats this image
        # as its own isolated location instead of crashing on None.
        location_id = metadata.location_id if metadata.location_id else f"__fallback_{image_id}"

        record: dict[str, Any] = {
            "image_id": image_id,
            "path": str(file_path),
            "filename": metadata.filename,
            "dataset": dataset,
            "labeled": labeled,
            "species": metadata.species,
            "camera_type": metadata.camera_type,
            "location_id": location_id,
            "ocr_timestamp": None,  # filled later by _attach_ocr_results
        }

        if not labeled:
            # Unlabeled images have no annotation file — fill bbox stats with defaults.
            record.update(_AnnotationSummary().to_stats())
            return record

        # Parse the YOLO annotation file to get bbox stats and authoritative species.
        annotation_path = self._paths.labelized_annotations / f"{file_path.stem}.txt"
        summary = self._parse_annotation_summary(annotation_path)

        if summary.dominant_species:
            filename_species = metadata.species
            if filename_species and filename_species != summary.dominant_species:
                self._logger.warning(
                    "Species mismatch for %s: filename=%s, annotation=%s",
                    file_path.name,
                    filename_species,
                    summary.dominant_species,
                )
            # Annotation species is authoritative for labeled images.
            record["species"] = summary.dominant_species

        record.update(summary.to_stats())
        return record

    # ------------------------------------------------------------------
    # OCR timestamp enrichment
    # ------------------------------------------------------------------

    def _attach_ocr_results(
        self,
        records: list[dict[str, Any]],
        image_files: list[tuple[Path, str, bool]],
        show_progress: bool,
    ) -> tuple[int, int]:
        """
        Attach OCR timestamps to records in place.

        Only the ``ocr_timestamp`` field is written to each record.
        Detailed OCR diagnostics (raw text, error messages) are not stored —
        the success/failure counts returned here are used for summary logging.

        Args:
            records: Metadata records to enrich (modified in place).
            image_files: Input list aligned with records.
            show_progress: Forwarded OCR progress flag.

        Returns:
            ``(ocr_success_count, ocr_fail_count)``.
        """
        if not (self._extract_timestamps and self._timestamp_extractor):
            # OCR disabled — timestamps stay None (set in _build_record).
            return 0, 0

        ocr_paths: list[Path | str] = [path for path, _, _ in image_files]
        ts_results = self._timestamp_extractor.extract_batch(
            ocr_paths,
            show_progress=show_progress,
        )

        if len(ts_results) != len(records):
            msg = (
                f"Timestamp extractor returned {len(ts_results)} results for {len(records)} images"
            )
            self._logger.error(msg)
            raise RuntimeError(msg)

        ocr_success = 0
        ocr_fail = 0
        for record, ts_result in zip(records, ts_results):
            record["ocr_timestamp"] = ts_result.timestamp
            if ts_result.success:
                ocr_success += 1
            else:
                ocr_fail += 1

        return ocr_success, ocr_fail

    # ------------------------------------------------------------------
    # DataFrame finalization and type coercion
    # ------------------------------------------------------------------

    @classmethod
    def _empty_dataframe(cls) -> pd.DataFrame:
        """Return an empty DataFrame with the output schema and correct dtypes."""
        empty = pd.DataFrame(columns=list(cls.COLUMNS))
        return cls._coerce_types(empty)

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to output columns and normalize dtypes."""
        out = df.reindex(columns=list(self.COLUMNS))
        return self._coerce_types(out)

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dtypes for the output columns.

        Ensures consistent types regardless of how pandas inferred them
        from the raw records (e.g. nullable Int64 for counts, datetime for
        timestamps, bool for flags).
        """
        if "ocr_timestamp" in df.columns:
            df["ocr_timestamp"] = pd.to_datetime(df["ocr_timestamp"], errors="coerce")

        if "label_bbox_count" in df.columns:
            df["label_bbox_count"] = (
                pd.to_numeric(df["label_bbox_count"], errors="coerce").fillna(0).astype("Int64")
            )

        for col in ("label_bbox_area_sum", "label_bbox_area_sum_capped"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")

        if "labeled" in df.columns:
            df["labeled"] = df["labeled"].astype("boolean").fillna(False).astype("bool")

        return df

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_build_summary(self, ocr_success: int, ocr_fail: int) -> None:
        """Log OCR extraction success rate."""
        if not self._extract_timestamps:
            return
        total = ocr_success + ocr_fail
        success_rate = (100.0 * ocr_success / total) if total > 0 else 0.0
        self._logger.info(
            "OCR timestamps: %d/%d extracted (%.1f%% success)",
            ocr_success,
            total,
            success_rate,
        )

    # ------------------------------------------------------------------
    # Image ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_image_id(dataset: str, filename: str) -> str:
        """
        Build a deterministic image identifier from dataset + filename.

        The ID is a truncated SHA-256 hash: same input always produces the
        same ID, and different datasets with the same filename get different IDs.

        Returns:
            16-character hex string.
        """
        source = f"{dataset}/{filename}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # YOLO annotation parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_annotation_summary(
        annotation_path: Path,
        area_cap: float = 0.25,
    ) -> _AnnotationSummary:
        """
        Parse one YOLO annotation file and aggregate bbox stats.

        Each valid line contributes to the count and area sums.  An invalid
        class token does not discard the geometry — the box is still counted,
        only the class-to-species mapping is skipped.

        Args:
            annotation_path: Path to the YOLO ``.txt`` file.
            area_cap: Per-box area cap for ``label_bbox_area_sum_capped``.

        Returns:
            Summary with bbox count, area sums, and dominant species.
        """
        if not annotation_path.exists():
            return _AnnotationSummary()

        try:
            lines = annotation_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return _AnnotationSummary()

        class_counts: dict[int, int] = {}
        count = 0
        area_sum = 0.0
        area_sum_capped = 0.0

        for line in lines:
            parsed = DataFrameBuilder._parse_annotation_line(line)
            if parsed is None:
                continue

            class_idx, area = parsed
            count += 1
            area_sum += area
            area_sum_capped += min(area, area_cap)
            if class_idx is not None and 0 <= class_idx < len(LABELED_SPECIES):
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        dominant_species: str | None = None
        if class_counts:
            # Use negative count to pick the highest frequency, then smallest class index on ties.
            dominant_class = min(class_counts, key=lambda idx: (-class_counts[idx], idx))
            # Dominant species = most frequent class.
            dominant_species = LABELED_SPECIES[dominant_class]

        return _AnnotationSummary(
            label_bbox_count=count,
            label_bbox_area_sum=area_sum,
            label_bbox_area_sum_capped=area_sum_capped,
            dominant_species=dominant_species,
        )

    @staticmethod
    def _parse_annotation_line(line: str) -> tuple[int | None, float] | None:
        """
        Parse one YOLO annotation line: ``class_id x_center y_center width height``.

        Returns ``(class_idx_or_none, area)`` when the geometry is valid,
        or ``None`` when the line is malformed or the box falls outside [0, 1].
        """
        parts = line.strip().split()
        if len(parts) != 5:
            return None

        try:
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            return None

        # Geometry validation: center in [0,1], positive dimensions <= 1,
        # and the entire box must fit within the normalized image bounds.
        if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
            return None
        if width <= 0.0 or width > 1.0:
            return None
        if height <= 0.0 or height > 1.0:
            return None
        if x_center - width / 2 < 0.0 or x_center + width / 2 > 1.0:
            return None
        if y_center - height / 2 < 0.0 or y_center + height / 2 > 1.0:
            return None

        try:
            class_idx: int | None = int(parts[0])
        except ValueError:
            class_idx = None

        area = max(0.0, min(width * height, 1.0))
        return class_idx, area


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------


def build_dataframe(
    paths: PathConfig | None = None,
    extract_timestamps: bool = True,
    gpu: bool = True,
    show_progress: bool = True,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a metadata DataFrame (convenience wrapper).

    Args:
        paths: Path configuration.
        extract_timestamps: Enable OCR timestamp extraction.
        gpu: Use GPU OCR backend when available.
        show_progress: Show OCR progress when extraction is enabled.
        save_path: Optional CSV destination.

    Returns:
        Built metadata DataFrame.
    """
    builder = DataFrameBuilder(
        paths=paths,
        extract_timestamps=extract_timestamps,
        gpu=gpu,
    )
    df = builder.build(show_progress=show_progress)
    if save_path is not None:
        builder.to_csv(df, save_path)
    return df
