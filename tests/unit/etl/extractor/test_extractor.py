"""Business-focused tests for pipeline.etl.extractor internals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.etl.config import PathConfig, PreprocessingConfig
from pipeline.etl.extractor.core import Extractor
from pipeline.etl.extractor.data_models import FileData, FilePair, LabelizedScanResult
from pipeline.etl.extractor.sources import SourceScanner


@dataclass
class _FakePathEntry:
    """Minimal path-like object to keep scanner tests deterministic."""

    name: str
    suffix: str
    stem: str
    file: bool = True

    def is_file(self) -> bool:
        return self.file

    def as_posix(self) -> str:
        return self.name


def _build_config(root: Path) -> PreprocessingConfig:
    """Build a test config rooted inside the per-test sandbox."""
    return PreprocessingConfig(
        paths=PathConfig(
            source_dir=root / "source",
            output_dir=root / "output",
            backup_dir=root / "output" / "backup",
        )
    )


# ---------------------------------------------------------------------------
# _process_pair
# ---------------------------------------------------------------------------


class TestProcessPair:
    def test_routes_complete_pair_to_writer(self) -> None:
        writer = MagicMock()
        pair = FilePair(
            stem="sample",
            image=FileData(stem="sample", name="sample.jpg"),
            annotation=FileData(stem="sample", name="sample.txt"),
        )
        # A complete pair must go through the "normal" labeled extraction path.
        Extractor._process_pair(writer, pair)
        writer.extract_complete_pair.assert_called_once_with(pair)
        writer.extract_image_only.assert_not_called()
        writer.extract_annotation_only.assert_not_called()

    def test_routes_image_orphan_to_writer(self) -> None:
        writer = MagicMock()
        pair = FilePair(stem="sample", image=FileData(stem="sample", name="sample.jpg"))
        # Missing annotation means this file is an orphan image.
        Extractor._process_pair(writer, pair)
        writer.extract_image_only.assert_called_once_with(pair)
        writer.extract_complete_pair.assert_not_called()
        writer.extract_annotation_only.assert_not_called()

    def test_routes_annotation_orphan_to_writer(self) -> None:
        writer = MagicMock()
        pair = FilePair(
            stem="sample",
            annotation=FileData(stem="sample", name="sample.txt"),
        )
        # Missing image means this file is an orphan annotation.
        Extractor._process_pair(writer, pair)
        writer.extract_annotation_only.assert_called_once_with(pair)
        writer.extract_complete_pair.assert_not_called()
        writer.extract_image_only.assert_not_called()

    def test_does_nothing_for_empty_pair(self) -> None:
        writer = MagicMock()
        Extractor._process_pair(writer, FilePair(stem="empty"))
        writer.extract_complete_pair.assert_not_called()
        writer.extract_image_only.assert_not_called()
        writer.extract_annotation_only.assert_not_called()


# ---------------------------------------------------------------------------
# SourceScanner internals used by Extractor
# ---------------------------------------------------------------------------


class TestSourceScannerIntegration:
    def test_scan_directory_processes_nested_zip_files(self) -> None:
        scanner = SourceScanner(MagicMock())
        root = MagicMock()
        nested_zip = _FakePathEntry(name="a/nested.zip", suffix=".zip", stem="nested")
        image = _FakePathEntry(name="a/image.jpg", suffix=".jpg", stem="image")
        # Simulate a folder containing both a nested ZIP and loose files.
        root.rglob.return_value = [nested_zip, image]
        pairs: dict[str, FilePair] = {}
        with (
            patch.object(scanner, "_scan_zip") as scan_zip,
            patch.object(scanner, "_register_file") as register_file,
        ):
            scanner._scan_directory(root, pairs)
        scan_zip.assert_called_once()
        assert scan_zip.call_args.args[0] is nested_zip
        assert scan_zip.call_args.args[1] is pairs
        register_file.assert_called_once()
        assert register_file.call_args.args[0] is pairs

    def test_scan_unlabeled_directory_processes_nested_zip_files(self) -> None:
        scanner = SourceScanner(MagicMock())
        root = MagicMock()
        nested_zip = _FakePathEntry(name="u/nested.zip", suffix=".zip", stem="nested")
        image = _FakePathEntry(name="u/img.jpg", suffix=".jpg", stem="img")
        root.rglob.return_value = [nested_zip, image]
        zip_images = [FileData(stem="zip_img", name="zip_img.jpg")]

        with patch.object(scanner, "_scan_unlabeled_zip", return_value=zip_images) as scan_zip:
            out = scanner._scan_unlabeled_directory(root)
        scan_zip.assert_called_once_with(nested_zip)

        assert len(out) == 2
        assert {item.name for item in out} == {"zip_img.jpg", "u/img.jpg"}

    def test_register_file_keeps_first_duplicate_image(self) -> None:
        scanner = SourceScanner(MagicMock())
        pairs: dict[str, FilePair] = {}
        first = FileData(stem="dup", name="dup.jpg")
        second = FileData(stem="dup", name="dup.jpeg")
        scanner._register_file(pairs, first)
        scanner._register_file(pairs, second)
        # Duplicate policy is "first one wins" for stable extraction output.
        assert pairs["dup"].image is first


# ---------------------------------------------------------------------------
# Extractor lifecycle
# ---------------------------------------------------------------------------


class TestExtractorLifecycle:
    def test_extract_orchestrates_labelized_and_unlabelized_steps(self) -> None:
        # num_workers=1 keeps the test deterministic (no thread pool) while still
        # exercising the full orchestration path.
        extractor = Extractor(config=PreprocessingConfig(), num_workers=1, skip_existing=True)
        # This test verifies call flow, not real file writes.
        with (
            patch.object(extractor._config, "ensure_dirs") as ensure_dirs,
            patch.object(extractor, "_make_writer") as make_writer,
            patch.object(extractor, "_extract_labelized") as extract_labelized,
            patch.object(extractor, "_extract_unlabelized") as extract_unlabelized,
            patch.object(extractor, "_log_summary") as log_summary,
        ):
            writer = MagicMock()
            make_writer.return_value = writer
            stats = extractor.extract()
        ensure_dirs.assert_called_once()
        make_writer.assert_called_once()
        extract_labelized.assert_called_once_with(writer)
        extract_unlabelized.assert_called_once_with(writer)
        log_summary.assert_called_once()
        assert stats is extractor._stats

    def test_rejects_non_positive_workers(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be > 0"):
            Extractor(config=PreprocessingConfig(), num_workers=0)

        with pytest.raises(ValueError, match="num_workers must be > 0"):
            Extractor(config=PreprocessingConfig(), num_workers=-1)

    def test_num_workers_capped_at_16(self) -> None:
        extractor = Extractor(config=PreprocessingConfig(), num_workers=32)
        assert extractor._num_workers == 16


# ---------------------------------------------------------------------------
# _extract_unlabelized
# ---------------------------------------------------------------------------


class TestExtractUnlabelized:
    def test_writes_all_images_even_with_same_name(self, sandbox: Path) -> None:
        config = _build_config(sandbox)
        (config.paths.source_dir / "unlabelized").mkdir(parents=True, exist_ok=True)
        extractor = Extractor(config=config, num_workers=1, skip_existing=True)
        writer = MagicMock()

        # Same filename can appear multiple times when source paths are different.
        first = FileData(stem="a", name="same.jpg", _source_path=Path("a/same.jpg"))
        duplicate = FileData(stem="b", name="same.jpg", _source_path=Path("b/same.jpg"))
        other = FileData(stem="c", name="other.jpg", _source_path=Path("c/other.jpg"))
        with patch.object(
            extractor._scanner,
            "scan_unlabeled_sources",
            return_value=[duplicate, other, first],
        ):
            extractor._extract_unlabelized(writer)
        writer.quarantine_unlabeled_duplicate.assert_not_called()
        written_hints = [
            call.args[0].source_hint for call in writer.write_unlabeled_image.call_args_list
        ]
        # Keep scanner order for deterministic naming in downstream writer logic.
        assert written_hints == [duplicate.source_hint, other.source_hint, first.source_hint]

    def test_returns_early_when_source_missing(self, sandbox: Path) -> None:
        config = PreprocessingConfig(paths=PathConfig(source_dir=sandbox / "missing_source_dir"))
        extractor = Extractor(config=config, num_workers=1)
        writer = MagicMock()
        extractor._extract_unlabelized(writer)
        writer.write_unlabeled_image.assert_not_called()

    def test_increments_errors_on_exception(self, sandbox: Path) -> None:
        (sandbox / "source" / "unlabelized").mkdir(parents=True)
        extractor = Extractor(config=_build_config(sandbox), num_workers=1)
        writer = MagicMock()
        writer.write_unlabeled_image.side_effect = RuntimeError("boom")
        with patch.object(
            extractor._scanner,
            "scan_unlabeled_sources",
            return_value=[FileData(stem="a", name="a.jpg")],
        ):
            extractor._extract_unlabelized(writer)
        assert extractor._stats.extraction_errors == 1


# ---------------------------------------------------------------------------
# _extract_labelized
# ---------------------------------------------------------------------------


class TestExtractLabelized:
    def test_quarantines_duplicate_image_and_annotation(self, sandbox: Path) -> None:
        config = _build_config(sandbox)
        (config.paths.source_dir / "labelized").mkdir(parents=True, exist_ok=True)
        extractor = Extractor(config=config, num_workers=1, skip_existing=True)
        writer = MagicMock()
        pair = FilePair(
            stem="sample",
            image=FileData(stem="sample", name="sample.jpg"),
            annotation=FileData(stem="sample", name="sample.txt"),
        )
        scan_result = LabelizedScanResult(
            pairs={"sample": pair},
            duplicate_images=[FileData(stem="sample", name="sample.jpg")],
            duplicate_annotations=[FileData(stem="sample", name="sample.txt")],
        )
        # Duplicates are not extracted twice: they are quarantined for audit/debug.
        with (
            patch.object(extractor, "_copy_classes_file") as copy_classes,
            patch.object(extractor._scanner, "scan_labelized_sources", return_value=scan_result),
        ):
            extractor._extract_labelized(writer)
        copy_classes.assert_called_once()
        writer.extract_complete_pair.assert_called_once_with(pair)
        writer.quarantine_labelized_duplicate_image.assert_called_once_with(
            scan_result.duplicate_images[0]
        )
        writer.quarantine_labelized_duplicate_annotation.assert_called_once_with(
            scan_result.duplicate_annotations[0]
        )

    def test_returns_early_when_source_missing(self, sandbox: Path) -> None:
        config = PreprocessingConfig(paths=PathConfig(source_dir=sandbox / "missing_source_dir"))
        extractor = Extractor(config=config, num_workers=1)
        writer = MagicMock()
        extractor._extract_labelized(writer)
        writer.extract_complete_pair.assert_not_called()

    def test_increments_errors_on_worker_exception(self, sandbox: Path) -> None:
        (sandbox / "source" / "labelized").mkdir(parents=True)
        extractor = Extractor(config=_build_config(sandbox), num_workers=1)
        writer = MagicMock()
        writer.extract_complete_pair.side_effect = RuntimeError("boom")

        pair = FilePair(
            stem="s",
            image=FileData(stem="s", name="s.jpg"),
            annotation=FileData(stem="s", name="s.txt"),
        )
        scan_result = LabelizedScanResult(pairs={"s": pair})
        with (
            patch.object(extractor, "_copy_classes_file"),
            patch.object(extractor, "_load_yolo_validator"),
            patch.object(extractor._scanner, "scan_labelized_sources", return_value=scan_result),
        ):
            extractor._extract_labelized(writer)
        assert extractor._stats.extraction_errors == 1


# ---------------------------------------------------------------------------
# classes.txt discovery
# ---------------------------------------------------------------------------


class TestCopyClassesFile:
    def test_scans_nested_zip_locations(self, sandbox: Path) -> None:
        config = _build_config(sandbox)
        labelized_root = config.paths.source_dir / "labelized"
        nested_zip = labelized_root / "subdir" / "batch.zip"
        nested_zip.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder archive file to exercise recursive zip discovery.
        nested_zip.write_bytes(b"zip-placeholder")

        extractor = Extractor(config=config, num_workers=1, skip_existing=True)
        writer = MagicMock()
        # classes.txt may live inside nested archives, not only as a loose file.
        with patch.object(
            extractor._scanner,
            "find_classes_in_zip",
            side_effect=lambda path: b"class_a\nclass_b\n" if path == nested_zip else None,
        ) as find_in_zip:
            extractor._copy_classes_file(labelized_root, writer)

        # If classes are found in nested zip, writer must persist them once.
        find_in_zip.assert_called_once_with(nested_zip)
        writer.write_classes_file.assert_called_once_with(b"class_a\nclass_b\n")


# ---------------------------------------------------------------------------
# _load_yolo_validator
# ---------------------------------------------------------------------------


class TestLoadYoloValidator:
    def test_stays_none_when_classes_file_missing(self, sandbox: Path) -> None:
        extractor = Extractor(config=_build_config(sandbox), num_workers=1)
        # Without classes.txt we cannot validate YOLO class IDs.
        extractor._load_yolo_validator()
        assert extractor._yolo_validator is None

    def test_stays_none_when_catalog_load_fails(self, sandbox: Path) -> None:
        # A corrupt classes.txt is not fatal: YOLO validation is skipped rather
        # than blocking the entire extraction.  The error is logged upstream.
        config = _build_config(sandbox)
        config.paths.classes_file.parent.mkdir(parents=True, exist_ok=True)
        config.paths.classes_file.write_text("bad\ncontent\n", encoding="utf-8")

        extractor = Extractor(config=config, num_workers=1)
        with patch(
            "pipeline.etl.extractor.core.load_class_catalog",
            side_effect=RuntimeError("bad catalog"),
        ):
            extractor._load_yolo_validator()
        assert extractor._yolo_validator is None
