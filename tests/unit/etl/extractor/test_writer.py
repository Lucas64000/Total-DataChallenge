"""Unit tests for ETL extraction writer."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock

import pytest

from pipeline.etl.config import PathConfig, PreprocessingConfig
from pipeline.etl.extractor.data_models import ExtractionStats, FileData, FilePair
from pipeline.etl.extractor.validators import ImageValidator, ValidationResult, YOLOValidator
from pipeline.etl.extractor.writer import ExtractionWriter


def _build_writer(
    root: Path,
    *,
    dry_run: bool = False,
    skip_existing: bool = False,
    backup_enabled: bool = True,
    move_invalid: bool = True,
    yolo_validator: YOLOValidator | None = None,
    image_valid: bool = True,
) -> tuple[ExtractionWriter, ExtractionStats, PreprocessingConfig]:
    """Build a writer wired to real sandbox dirs with a mocked image validator.

    The image validator is always mocked because decoding real JPEG headers
    in unit tests is slow and brittle.  Use image_valid=False to simulate a
    corrupt image and trigger the quarantine code path.

    Returns (writer, stats, config) so tests can inspect both side-effects
    (files written to disk) and counters (stats.*).
    """
    config = PreprocessingConfig(
        paths=PathConfig(
            source_dir=root / "source",
            output_dir=root / "output",
            backup_dir=root / "backup",
        ),
    )
    # Create output dirs upfront so tests can assert on file existence right away.
    config.ensure_dirs()
    config.dry_run = dry_run
    config.backup_enabled = backup_enabled

    stats = ExtractionStats()
    img_validator = MagicMock(spec=ImageValidator)
    img_validator.validate.return_value = ValidationResult(
        is_valid=image_valid, error=None if image_valid else "bad image"
    )

    writer = ExtractionWriter(
        config=config,
        stats=stats,
        stats_lock=Lock(),
        skip_existing=skip_existing,
        logger=logging.getLogger("test_writer"),
        image_validator=img_validator,
        yolo_validator=yolo_validator,
        move_invalid=move_invalid,
    )
    return writer, stats, config


def _minimal_writer() -> ExtractionWriter:
    """Build a writer with no filesystem side-effects, for error-path tests."""
    return ExtractionWriter(
        config=PreprocessingConfig(),
        stats=ExtractionStats(),
        stats_lock=Lock(),
        skip_existing=False,
        logger=logging.getLogger("test_writer"),
        image_validator=MagicMock(spec=ImageValidator),
    )


def _src_file(root: Path, name: str, content: bytes) -> FileData:
    """Create a FileData backed by a real file in a temp source directory."""
    src = root / "src_files" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(content)
    return FileData.from_path(src)


# ---------------------------------------------------------------------------
# extract_complete_pair
# ---------------------------------------------------------------------------


class TestExtractCompletePair:
    def test_writes_both_files_and_increments_stats(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox)
        pair = FilePair(
            stem="s",
            image=_src_file(sandbox, "s.jpg", b"img"),
            annotation=_src_file(sandbox, "s.txt", b"0 0.5 0.5 0.1 0.1\n"),
        )

        writer.extract_complete_pair(pair)

        assert (config.paths.labelized_images / "s.jpg").read_bytes() == b"img"
        assert (config.paths.labelized_annotations / "s.txt").read_bytes() == b"0 0.5 0.5 0.1 0.1\n"
        assert stats.pairs_extracted == 1
        assert stats.images_extracted == 1
        assert stats.annotations_extracted == 1

    def test_raises_on_incomplete_pair(self) -> None:
        writer = _minimal_writer()
        pair = FilePair(stem="x", image=FileData(stem="x", name="x.jpg"))

        with pytest.raises(ValueError, match="Incomplete pair"):
            writer.extract_complete_pair(pair)

    def test_skips_when_both_destinations_exist(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox, skip_existing=True)
        (config.paths.labelized_images / "s.jpg").write_bytes(b"old")
        (config.paths.labelized_annotations / "s.txt").write_bytes(b"old")
        pair = FilePair(
            stem="s",
            image=_src_file(sandbox, "s.jpg", b"new"),
            annotation=_src_file(sandbox, "s.txt", b"new"),
        )

        writer.extract_complete_pair(pair)

        assert stats.skipped_existing == 1
        assert stats.pairs_extracted == 0
        assert (config.paths.labelized_images / "s.jpg").read_bytes() == b"old"

    def test_quarantines_when_image_invalid(self, sandbox: Path) -> None:
        # Corrupt images must never land in the main output folder.
        # Both the image and its annotation are moved together to invalid_dir
        # so they can be reviewed/retried without polluting the training set.
        writer, stats, config = _build_writer(sandbox, image_valid=False)
        pair = FilePair(
            stem="bad",
            image=_src_file(sandbox, "bad.jpg", b"corrupt"),
            annotation=_src_file(sandbox, "bad.txt", b"ann"),
        )

        writer.extract_complete_pair(pair)

        assert stats.invalid_labeled == 1
        assert stats.pairs_extracted == 0
        assert (config.invalid_dir / "images" / "bad.jpg").exists()
        assert (config.invalid_dir / "annotations" / "bad.txt").exists()

    def test_quarantines_when_yolo_annotation_invalid(self, sandbox: Path) -> None:
        yolo = MagicMock(spec=YOLOValidator)
        yolo.validate.return_value = ValidationResult(
            is_valid=False, error="bad annotation"
        )
        writer, stats, config = _build_writer(sandbox, yolo_validator=yolo)
        pair = FilePair(
            stem="bad",
            image=_src_file(sandbox, "bad.jpg", b"ok-img"),
            annotation=_src_file(sandbox, "bad.txt", b"garbage"),
        )

        writer.extract_complete_pair(pair)

        assert stats.invalid_labeled == 1
        assert stats.pairs_extracted == 0
        assert (config.invalid_dir / "images" / "bad.jpg").exists()

    def test_skips_annotation_validation_when_no_yolo_validator(
        self, sandbox: Path
    ) -> None:
        writer, stats, _ = _build_writer(sandbox, yolo_validator=None)
        pair = FilePair(
            stem="s",
            image=_src_file(sandbox, "s.jpg", b"img"),
            annotation=_src_file(sandbox, "s.txt", b"anything-here"),
        )

        writer.extract_complete_pair(pair)

        # Without YOLO validator, pair is accepted purely on image validity.
        assert stats.pairs_extracted == 1

    def test_dry_run_increments_stats_without_writes(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox, dry_run=True)
        pair = FilePair(
            stem="s",
            image=_src_file(sandbox, "s.jpg", b"img"),
            annotation=_src_file(sandbox, "s.txt", b"ann"),
        )

        writer.extract_complete_pair(pair)

        assert stats.pairs_extracted == 1
        assert not (config.paths.labelized_images / "s.jpg").exists()


# ---------------------------------------------------------------------------
# extract_image_only / extract_annotation_only
# ---------------------------------------------------------------------------


class TestExtractOrphans:
    def test_image_only_quarantines_and_counts(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox)
        pair = FilePair(stem="o", image=_src_file(sandbox, "o.jpg", b"img"))

        writer.extract_image_only(pair)

        assert stats.orphan_images == 1
        assert (config.orphans_dir / "images" / "o.jpg").read_bytes() == b"img"

    def test_image_only_raises_when_no_image(self) -> None:
        with pytest.raises(ValueError, match="Missing image"):
            _minimal_writer().extract_image_only(FilePair(stem="x"))

    def test_image_only_skips_write_when_backup_disabled(
        self, sandbox: Path
    ) -> None:
        # backup_enabled=False means orphans are counted but not saved to disk.
        # This is used to run the pipeline without writing diagnostic files.
        writer, stats, config = _build_writer(sandbox, backup_enabled=False)
        pair = FilePair(stem="o", image=_src_file(sandbox, "o.jpg", b"img"))

        writer.extract_image_only(pair)

        assert stats.orphan_images == 1
        assert not (config.orphans_dir / "images" / "o.jpg").exists()

    def test_annotation_only_quarantines_and_counts(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox)
        pair = FilePair(stem="o", annotation=_src_file(sandbox, "o.txt", b"ann"))

        writer.extract_annotation_only(pair)

        assert stats.orphan_annotations == 1
        assert (config.orphans_dir / "annotations" / "o.txt").read_bytes() == b"ann"

    def test_annotation_only_raises_when_no_annotation(self) -> None:
        with pytest.raises(ValueError, match="Missing annotation"):
            _minimal_writer().extract_annotation_only(FilePair(stem="x"))


# ---------------------------------------------------------------------------
# write_unlabeled_image
# ---------------------------------------------------------------------------


class TestWriteUnlabeledImage:
    def test_writes_valid_image(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox)
        img = _src_file(sandbox, "u.jpg", b"uimg")

        writer.write_unlabeled_image(img)

        assert stats.unlabeled_images == 1
        assert (config.paths.unlabeled / "u.jpg").read_bytes() == b"uimg"

    def test_skips_when_destination_exists(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox, skip_existing=True)
        (config.paths.unlabeled / "u.jpg").write_bytes(b"old")
        img = _src_file(sandbox, "u.jpg", b"new")

        writer.write_unlabeled_image(img)

        assert stats.skipped_existing == 1
        assert stats.unlabeled_images == 0
        assert (config.paths.unlabeled / "u.jpg").read_bytes() == b"old"

    def test_quarantines_invalid_image(self, sandbox: Path) -> None:
        writer, stats, config = _build_writer(sandbox, image_valid=False)
        img = _src_file(sandbox, "bad.jpg", b"corrupt")

        writer.write_unlabeled_image(img)

        assert stats.invalid_unlabeled == 1
        assert stats.unlabeled_images == 0
        assert not (config.paths.unlabeled / "bad.jpg").exists()

    def test_hashes_name_on_collision(self, sandbox: Path) -> None:
        # When skip_existing=False and a file with the same name already exists,
        # the incoming file must still be written (it may come from a different camera
        # location). A hash suffix is appended to avoid overwriting: u__<hash>.jpg.
        writer, stats, config = _build_writer(sandbox, skip_existing=False)
        # Pre-create a file to force the name collision.
        (config.paths.unlabeled / "u.jpg").write_bytes(b"existing")
        img = _src_file(sandbox, "u.jpg", b"new-content")

        writer.write_unlabeled_image(img)

        assert stats.duplicate_unlabeled_images == 1
        assert stats.unlabeled_images == 1
        # Hashed file should exist alongside the original.
        hashed_files = list(config.paths.unlabeled.glob("u__*.jpg"))
        assert len(hashed_files) == 1
        assert hashed_files[0].read_bytes() == b"new-content"
        # Original untouched.
        assert (config.paths.unlabeled / "u.jpg").read_bytes() == b"existing"


# ---------------------------------------------------------------------------
# write_classes_file
# ---------------------------------------------------------------------------


class TestWriteClassesFile:
    def test_writes_content(self, sandbox: Path) -> None:
        writer, _, config = _build_writer(sandbox)
        writer.write_classes_file(b"cat\ndog\n")
        assert config.paths.classes_file.read_bytes() == b"cat\ndog\n"

    def test_dry_run_skips_write(self, sandbox: Path) -> None:
        writer, _, config = _build_writer(sandbox, dry_run=True)
        writer.write_classes_file(b"cat\ndog\n")
        assert not config.paths.classes_file.exists()


# ---------------------------------------------------------------------------
# quarantine duplicates
# ---------------------------------------------------------------------------


class TestQuarantineDuplicates:
    def test_duplicate_image_increments_stat(self, sandbox: Path) -> None:
        writer, stats, _ = _build_writer(sandbox)
        fd = _src_file(sandbox, "dup.jpg", b"img")

        writer.quarantine_labelized_duplicate_image(fd)

        assert stats.duplicate_labelized_images == 1

    def test_duplicate_annotation_increments_stat(self, sandbox: Path) -> None:
        writer, stats, _ = _build_writer(sandbox)
        fd = _src_file(sandbox, "dup.txt", b"ann")

        writer.quarantine_labelized_duplicate_annotation(fd)

        assert stats.duplicate_labelized_annotations == 1

    def test_duplicate_not_written_when_backup_disabled(
        self, sandbox: Path
    ) -> None:
        writer, stats, config = _build_writer(sandbox, backup_enabled=False)
        fd = _src_file(sandbox, "dup.jpg", b"img")

        writer.quarantine_labelized_duplicate_image(fd)

        assert stats.duplicate_labelized_images == 1
        dup_files = [f for f in config.duplicates_dir.rglob("*") if f.is_file()]
        assert not dup_files


# ---------------------------------------------------------------------------
# _hashed_duplicate_name
# ---------------------------------------------------------------------------


def test_hashed_duplicate_name_is_deterministic_and_unique() -> None:
    fd_a = FileData(stem="x", name="x.jpg", _source_path=Path("a/x.jpg"))
    fd_b = FileData(stem="x", name="x.jpg", _source_path=Path("b/x.jpg"))

    name_a = ExtractionWriter._hashed_duplicate_name(fd_a)
    name_a_again = ExtractionWriter._hashed_duplicate_name(fd_a)
    name_b = ExtractionWriter._hashed_duplicate_name(fd_b)

    assert name_a == name_a_again
    assert name_a != name_b
    assert name_a.endswith(".jpg")
    assert name_b.endswith(".jpg")


# ---------------------------------------------------------------------------
# _inc
# ---------------------------------------------------------------------------


def test_inc_rejects_unknown_field() -> None:
    with pytest.raises(AttributeError, match="no field"):
        _minimal_writer()._inc("nonexistent_counter")
