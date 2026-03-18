"""High-level ETL dataset validator orchestration."""

from __future__ import annotations

import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from pipeline.etl.config import IMAGE_EXTENSIONS, LABELED_SPECIES, PreprocessingConfig
from pipeline.etl.validation.data_models import ValidationStats
from pipeline.etl.validation.rules import FilenameValidator, ImageValidator, YOLOValidator
from utils.logging_system import LogCategory, get_phototrap_logger


class Validator:
    """Validate extracted files and enforce ETL data quality constraints."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        num_workers: int | None = None,
        move_invalid: bool = False,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Preprocessing paths and settings. Defaults to standard layout.
            num_workers: Thread pool size for parallel validation.
                Falls back to ``os.cpu_count()``.
            move_invalid: When ``True``, invalid files are moved to
                ``backup/invalid`` instead of being left in place.
        """
        if num_workers is not None and num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self._config = config or PreprocessingConfig()
        self._num_workers = min(num_workers or os.cpu_count() or 1, 16)
        self._logger = get_phototrap_logger().get_logger(
            LogCategory.PREPROCESSING, "validator"
        )
        self._stats = ValidationStats()
        self._stats_lock = Lock()
        self._move_invalid_files = move_invalid
        self._invalid_files: list[dict[str, str]] = []
        self._invalid_lock = Lock()
        self._backup_disabled_warning_emitted = False
        self._filename_validator = FilenameValidator()
        self._image_validator = ImageValidator()
        self._yolo_validator = YOLOValidator(classes=list(LABELED_SPECIES))

    def validate(self) -> ValidationStats:
        """Validate labeled and unlabeled extracted data."""
        self._stats = ValidationStats()
        self._invalid_files = []
        self._backup_disabled_warning_emitted = False
        self._config.ensure_dirs()

        self._load_classes()
        self._validate_labelized()
        self._validate_unlabeled()

        self._write_report()
        self._log_summary()
        return self._stats

    def _load_classes(self) -> None:
        """
        Load canonical labeled classes.

        Class IDs in annotations are expected to map to ``LABELED_SPECIES`` (8 labeled classes).
        If ``classes.txt`` is missing or contains a broader taxonomy, we rewrite it to
        the canonical labeled list so downstream steps stay consistent.
        """
        classes_file = self._config.paths.classes_file
        canonical_classes = list(LABELED_SPECIES)

        file_classes: list[str] | None = None
        if classes_file.exists():
            try:
                raw = classes_file.read_text(encoding="utf-8")
                file_classes = [line.strip() for line in raw.splitlines() if line.strip()]
            except (OSError, UnicodeDecodeError) as exc:
                self._logger.warning(
                    "Failed to read classes.txt at %s (%s): rewriting with canonical labeled classes",
                    classes_file,
                    exc,
                )
        else:
            self._logger.warning(
                "No classes.txt found at %s: creating canonical labeled classes file",
                classes_file,
            )

        if file_classes != canonical_classes:
            if self._config.dry_run:
                self._logger.info(
                    "Dry-run: would write canonical classes.txt with %d labeled classes",
                    len(canonical_classes),
                )
            else:
                try:
                    classes_file.parent.mkdir(parents=True, exist_ok=True)
                    content = "\n".join(canonical_classes) + "\n"
                    classes_file.write_text(content, encoding="utf-8")
                except OSError as exc:
                    self._logger.warning(
                        "Failed to write canonical classes.txt at %s (%s)",
                        classes_file,
                        exc,
                    )
                else:
                    self._logger.info(
                        "Wrote canonical classes.txt with %d labeled classes at %s",
                        len(canonical_classes),
                        classes_file,
                    )

        self._logger.info(
            "Loaded %d canonical labeled classes for class_id validation",
            self._yolo_validator.num_classes,
        )

    def _validate_labelized(self) -> None:
        """Validate all complete labeled image/annotation pairs."""
        images_dir = self._config.paths.labelized_images
        annotations_dir = self._config.paths.labelized_annotations

        if not images_dir.exists() or not annotations_dir.exists():
            self._logger.warning("Labelized directories not found")
            return

        image_stems = {
            f.stem
            for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        }
        annotation_stems = {
            f.stem
            for f in annotations_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        }
        complete_stems = sorted(image_stems & annotation_stems)
        self._logger.info("Found %d pairs", len(complete_stems))

        pairs = [
            (self._find_image(images_dir, stem), annotations_dir / f"{stem}.txt")
            for stem in complete_stems
        ]

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = {
                executor.submit(self._validate_pair, img, ann): (img, ann)
                for img, ann in pairs
                if img is not None
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validating pairs",
                unit="pair",
            ):
                try:
                    future.result()
                except Exception:
                    img, _ = futures[future]
                    self._logger.error(
                        "Failed to validate pair %s",
                        img.name if img else "unknown",
                        exc_info=True,
                    )

    def _find_image(self, directory: Path, stem: str) -> Path | None:
        """
        Find image path for a given stem across supported extensions.

        Args:
            directory: Directory containing labeled images.
            stem: Filename stem without extension.

        Returns:
            Matching image path, or ``None`` if no file exists.
        """
        for ext in sorted(IMAGE_EXTENSIONS):
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path
        return None

    def _validate_unlabeled(self) -> None:
        """Validate all unlabeled images."""
        unlabeled_dir = self._config.paths.unlabeled
        if not unlabeled_dir.exists():
            self._logger.warning("Unlabeled directory not found: %s", unlabeled_dir)
            return

        image_files = sorted(
            [f for f in unlabeled_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name,
        )
        self._logger.info("Validating %d unlabeled images", len(image_files))

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = {executor.submit(self._validate_unlabeled_image, img): img for img in image_files}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validating unlabeled",
                unit="image",
            ):
                try:
                    future.result()
                except Exception:
                    img = futures[future]
                    self._logger.error("Failed to validate %s", img.name, exc_info=True)

    def _validate_pair(self, image_path: Path, annotation_path: Path) -> None:
        """
        Validate one labeled image/annotation pair.

        Args:
            image_path: Image file path.
            annotation_path: Annotation file path.
        """
        filename_result = self._filename_validator.validate(image_path.name, labeled=True)
        if not filename_result.is_valid:
            self._logger.warning(
                "Unparseable filename %s: %s (keeping file, will use fallback location_id)",
                image_path.name,
                filename_result.error,
            )
            with self._stats_lock:
                self._stats.parse_warnings += 1

        image_result = self._image_validator.validate(image_path.read_bytes(), image_path.name)
        if not image_result.is_valid:
            self._logger.warning("Invalid image %s: %s", image_path.name, image_result.error)
            self._record_invalid(image_path.name, "image", image_result.error or "")
            if self._move_invalid_files:
                self._move_to_invalid(image_path, annotation_path)
            with self._stats_lock:
                self._stats.invalid_images += 1
            return

        try:
            content = annotation_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            error = f"Unreadable annotation: {exc}"
            self._logger.warning("Invalid annotation %s: %s", annotation_path.name, error)
            self._record_invalid(annotation_path.name, "annotation", error)
            if self._move_invalid_files:
                self._move_to_invalid(image_path, annotation_path)
            with self._stats_lock:
                self._stats.invalid_annotations += 1
            return

        annotation_result = self._yolo_validator.validate(content, annotation_path.name)
        if not annotation_result.is_valid:
            self._logger.warning(
                "Invalid annotation %s: %s", annotation_path.name, annotation_result.error
            )
            self._record_invalid(annotation_path.name, "annotation", annotation_result.error or "")
            if self._move_invalid_files:
                self._move_to_invalid(image_path, annotation_path)
            with self._stats_lock:
                self._stats.invalid_annotations += 1
            return

        with self._stats_lock:
            self._stats.valid_pairs += 1
            if not content.strip():
                self._stats.empty_annotations += 1

    def _validate_unlabeled_image(self, image_path: Path) -> None:
        """
        Validate one unlabeled image file.

        Args:
            image_path: Image file path.
        """
        filename_result = self._filename_validator.validate(image_path.name, labeled=False)
        if not filename_result.is_valid:
            self._logger.warning(
                "Unparseable filename %s: %s (keeping file, will use fallback location_id)",
                image_path.name,
                filename_result.error,
            )
            with self._stats_lock:
                self._stats.parse_warnings += 1

        result = self._image_validator.validate(image_path.read_bytes(), image_path.name)
        if not result.is_valid:
            self._logger.warning("Invalid unlabeled image %s: %s", image_path.name, result.error)
            self._record_invalid(image_path.name, "unlabeled", result.error or "")
            if self._move_invalid_files:
                self._move_unlabeled_to_invalid(image_path)
            with self._stats_lock:
                self._stats.invalid_unlabeled += 1
            return

        with self._stats_lock:
            self._stats.valid_unlabeled += 1

    def _record_invalid(self, filename: str, file_type: str, reason: str) -> None:
        """
        Record one invalid item for the validation report.

        Args:
            filename: Invalid filename.
            file_type: Invalid file category.
            reason: Human-readable reason.
        """
        with self._invalid_lock:
            self._invalid_files.append({"filename": filename, "type": file_type, "reason": reason})

    def _write_report(self) -> None:
        """Write CSV report with all invalid files detected during validation."""
        if self._config.dry_run:
            self._logger.info(
                "Dry-run: skipping validation_report.csv write (%d invalid)",
                len(self._invalid_files),
            )
            return

        report_path = self._config.paths.output_dir / "validation_report.csv"
        if not self._invalid_files:
            if report_path.exists():
                try:
                    report_path.unlink()
                    self._logger.info("Stale validation report removed: %s", report_path)
                except OSError as exc:
                    self._logger.warning(
                        "Failed to remove stale validation report %s: %s",
                        report_path,
                        exc,
                    )
            return

        ordered_rows = sorted(
            self._invalid_files,
            key=lambda row: (row["type"], row["filename"], row["reason"]),
        )
        with open(report_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=["filename", "type", "reason"])
            writer.writeheader()
            writer.writerows(ordered_rows)
        self._logger.info("Validation report: %s (%d invalid)", report_path, len(ordered_rows))

    def _move_to_invalid(self, image_path: Path, annotation_path: Path) -> None:
        """
        Move an invalid labeled pair to ``backup/invalid``.

        Args:
            image_path: Invalid image path.
            annotation_path: Invalid annotation path.
        """
        if self._config.dry_run:
            return
        if not self._config.backup_enabled:
            self._warn_backup_disabled()
            return

        image_dest = self._config.invalid_dir / "images" / image_path.name
        ann_dest = self._config.invalid_dir / "annotations" / annotation_path.name
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        ann_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(image_dest))
        shutil.move(str(annotation_path), str(ann_dest))

    def _move_unlabeled_to_invalid(self, image_path: Path) -> None:
        """
        Move one invalid unlabeled image to ``backup/invalid/unlabeled``.

        Args:
            image_path: Invalid unlabeled image path.
        """
        if self._config.dry_run:
            return
        if not self._config.backup_enabled:
            self._warn_backup_disabled()
            return

        dest = self._config.invalid_dir / "unlabeled" / image_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(dest))

    def _warn_backup_disabled(self) -> None:
        """Log once when invalid-file moves are requested but backups are disabled."""
        if self._backup_disabled_warning_emitted:
            return
        self._logger.warning(
            "move_invalid=True but backup_enabled=False: invalid files are counted but not moved"
        )
        self._backup_disabled_warning_emitted = True

    def _log_summary(self) -> None:
        """Log final validation counters."""
        self._logger.info(
            "Validation complete: %d valid pairs (%d empty), %d valid unlabeled | Warnings: %d unparseable filenames | Invalid: %d image, %d annotation, %d unlabeled",
            self._stats.valid_pairs,
            self._stats.empty_annotations,
            self._stats.valid_unlabeled,
            self._stats.parse_warnings,
            self._stats.invalid_images,
            self._stats.invalid_annotations,
            self._stats.invalid_unlabeled,
        )


if __name__ == "__main__":
    Validator().validate()
