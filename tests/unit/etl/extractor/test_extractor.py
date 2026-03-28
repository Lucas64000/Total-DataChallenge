"""Business-focused tests for pipeline.etl.extractor internals."""

from __future__ import annotations

import shutil
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
    # Minimal Path-like object used to keep scanner tests fast and deterministic.
    name: str
    suffix: str
    stem: str
    file: bool = True

    def is_file(self) -> bool:
        return self.file

    def as_posix(self) -> str:
        return self.name


def test_process_pair_routes_complete_pair_to_writer() -> None:
    writer = MagicMock()
    # Complete pair = image + annotation for the same stem.
    pair = FilePair(
        stem="sample",
        image=FileData(stem="sample", name="sample.jpg"),
        annotation=FileData(stem="sample", name="sample.txt"),
    )

    # Act: route this pair to the correct writer branch.
    Extractor._process_pair(writer, pair)

    # Assert: complete pairs must go through the complete extraction path.
    writer.extract_complete_pair.assert_called_once_with(pair)
    writer.extract_image_only.assert_not_called()
    writer.extract_annotation_only.assert_not_called()


def test_process_pair_routes_image_orphan_to_writer() -> None:
    writer = MagicMock()
    # Orphan image = image exists, annotation is missing.
    pair = FilePair(stem="sample", image=FileData(stem="sample", name="sample.jpg"))

    Extractor._process_pair(writer, pair)

    # This case must be routed to the image-only workflow.
    writer.extract_image_only.assert_called_once_with(pair)
    writer.extract_complete_pair.assert_not_called()
    writer.extract_annotation_only.assert_not_called()


def test_process_pair_routes_annotation_orphan_to_writer() -> None:
    writer = MagicMock()
    # Orphan annotation = annotation exists, image is missing.
    pair = FilePair(stem="sample", annotation=FileData(stem="sample", name="sample.txt"))

    Extractor._process_pair(writer, pair)

    # This case must be routed to the annotation-only workflow.
    writer.extract_annotation_only.assert_called_once_with(pair)
    writer.extract_complete_pair.assert_not_called()
    writer.extract_image_only.assert_not_called()


def test_scan_directory_processes_nested_zip_files() -> None:
    logger = MagicMock()
    scanner = SourceScanner(logger)
    # Fake directory object: `rglob` returns one zip and one regular image.
    root = MagicMock()
    nested_zip = _FakePathEntry(name="a/nested.zip", suffix=".zip", stem="nested")
    image = _FakePathEntry(name="a/image.jpg", suffix=".jpg", stem="image")
    root.rglob.return_value = [nested_zip, image]
    pairs: dict[str, FilePair] = {}

    # Patch internal methods to observe control flow.
    with (
        patch.object(scanner, "_scan_zip") as scan_zip,
        patch.object(scanner, "_register_file") as register_file,
    ):
        scanner._scan_directory(root, pairs)

    # The scanner must recurse into zip files and still register regular files.
    scan_zip.assert_called_once()
    assert scan_zip.call_args.args[0] is nested_zip
    assert scan_zip.call_args.args[1] is pairs
    register_file.assert_called_once()
    assert register_file.call_args.args[0] is pairs


def test_scan_unlabeled_directory_processes_nested_zip_files() -> None:
    logger = MagicMock()
    scanner = SourceScanner(logger)
    root = MagicMock()
    nested_zip = _FakePathEntry(name="u/nested.zip", suffix=".zip", stem="nested")
    image = _FakePathEntry(name="u/img.jpg", suffix=".jpg", stem="img")
    root.rglob.return_value = [nested_zip, image]
    zip_images = [FileData(stem="zip_img", name="zip_img.jpg")]

    with patch.object(scanner, "_scan_unlabeled_zip", return_value=zip_images) as scan_zip:
        out = scanner._scan_unlabeled_directory(root)

    # Zip images and regular files should be merged in a single output list.
    scan_zip.assert_called_once_with(nested_zip)
    assert len(out) == 2
    assert {item.name for item in out} == {"zip_img.jpg", "u/img.jpg"}


def test_register_file_keeps_first_duplicate_image() -> None:
    logger = MagicMock()
    scanner = SourceScanner(logger)
    pairs: dict[str, FilePair] = {}
    first = FileData(stem="dup", name="dup.jpg")
    second = FileData(stem="dup", name="dup.jpeg")

    # Register two image files sharing the same stem.
    scanner._register_file(pairs, first)
    scanner._register_file(pairs, second)

    # Policy: keep the first one.
    assert pairs["dup"].image is first


def test_extractor_extract_orchestrates_labelized_and_unlabelized_steps() -> None:
    extractor = Extractor(config=PreprocessingConfig(), num_workers=1, skip_existing=True)

    # Patch each major step: this test checks orchestration order/calls only.
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

    # `extract()` must run all top-level steps and return internal stats object.
    ensure_dirs.assert_called_once()
    make_writer.assert_called_once()
    extract_labelized.assert_called_once_with(writer)
    extract_unlabelized.assert_called_once_with(writer)
    log_summary.assert_called_once()
    assert stats is extractor._stats


def test_extractor_rejects_non_positive_workers() -> None:
    # Guardrail: worker pool size must always be strictly positive.
    with pytest.raises(ValueError, match="num_workers must be > 0"):
        Extractor(config=PreprocessingConfig(), num_workers=0)

    with pytest.raises(ValueError, match="num_workers must be > 0"):
        Extractor(config=PreprocessingConfig(), num_workers=-1)


def test_extract_unlabelized_writes_all_images_even_with_same_name() -> None:
    # Use a dedicated temporary folder under repo for this test.
    root = Path("sandbox_extractor_unlabelized")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    config = PreprocessingConfig(
        paths=PathConfig(
            source_dir=root / "source",
            output_dir=root / "output",
            backup_dir=root / "output" / "backup",
        )
    )
    (config.paths.source_dir / "unlabelized").mkdir(parents=True, exist_ok=True)
    extractor = Extractor(config=config, num_workers=1, skip_existing=True)
    writer = MagicMock()

    # Same file name, different source paths -> distinct source hints.
    first = FileData(stem="a", name="same.jpg", _source_path=Path("a/same.jpg"))
    duplicate = FileData(stem="b", name="same.jpg", _source_path=Path("b/same.jpg"))
    other = FileData(stem="c", name="other.jpg", _source_path=Path("c/other.jpg"))

    # Force scanner output to a known order for deterministic assertions.
    with patch.object(
        extractor._scanner,
        "scan_unlabeled_sources",
        return_value=[duplicate, other, first],
    ):
        extractor._extract_unlabelized(writer)

    # Unlabeled duplicates are allowed when source hints are distinct.
    writer.quarantine_unlabeled_duplicate.assert_not_called()
    written_hints = [
        call.args[0].source_hint for call in writer.write_unlabeled_image.call_args_list
    ]
    # Keep scanner order to guarantee deterministic output naming downstream.
    assert written_hints == [duplicate.source_hint, other.source_hint, first.source_hint]
    # Cleanup local sandbox created by this test.
    shutil.rmtree(root, ignore_errors=True)


def test_extract_labelized_quarantines_duplicate_image_and_annotation() -> None:
    root = Path("sandbox_extractor_labelized")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    config = PreprocessingConfig(
        paths=PathConfig(
            source_dir=root / "source",
            output_dir=root / "output",
            backup_dir=root / "output" / "backup",
        )
    )
    (config.paths.source_dir / "labelized").mkdir(parents=True, exist_ok=True)
    extractor = Extractor(config=config, num_workers=1, skip_existing=True)
    writer = MagicMock()
    # One valid pair that should still be extracted.
    pair = FilePair(
        stem="sample",
        image=FileData(stem="sample", name="sample.jpg"),
        annotation=FileData(stem="sample", name="sample.txt"),
    )
    # Scanner also reports duplicates that must go to quarantine.
    scan_result = LabelizedScanResult(
        pairs={"sample": pair},
        duplicate_images=[FileData(stem="sample", name="sample.jpg")],
        duplicate_annotations=[FileData(stem="sample", name="sample.txt")],
    )

    with (
        patch.object(extractor, "_copy_classes_file") as copy_classes,
        patch.object(extractor._scanner, "scan_labelized_sources", return_value=scan_result),
    ):
        extractor._extract_labelized(writer)

    # Labelized flow must both process valid pairs and quarantine duplicates.
    copy_classes.assert_called_once()
    writer.extract_complete_pair.assert_called_once_with(pair)
    writer.quarantine_labelized_duplicate_image.assert_called_once_with(
        scan_result.duplicate_images[0]
    )
    writer.quarantine_labelized_duplicate_annotation.assert_called_once_with(
        scan_result.duplicate_annotations[0]
    )
    # Cleanup local sandbox created by this test.
    shutil.rmtree(root, ignore_errors=True)


def test_copy_classes_file_scans_nested_zip_locations() -> None:
    root = Path("sandbox_extractor_classes_nested_zip")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        config = PreprocessingConfig(
            paths=PathConfig(
                source_dir=root / "source",
                output_dir=root / "output",
                backup_dir=root / "output" / "backup",
            )
        )
        labelized_root = config.paths.source_dir / "labelized"
        nested_zip = labelized_root / "subdir" / "batch.zip"
        nested_zip.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder archive file to exercise recursive zip discovery.
        nested_zip.write_bytes(b"zip-placeholder")

        extractor = Extractor(config=config, num_workers=1, skip_existing=True)
        writer = MagicMock()

        # Return classes bytes only for the expected nested zip path.
        with patch.object(
            extractor._scanner,
            "find_classes_in_zip",
            side_effect=lambda path: b"class_a\nclass_b\n" if path == nested_zip else None,
        ) as find_in_zip:
            extractor._copy_classes_file(labelized_root, writer)

        # If classes are found in nested zip, writer must persist them once.
        find_in_zip.assert_called_once_with(nested_zip)
        writer.write_classes_file.assert_called_once_with(b"class_a\nclass_b\n")
    finally:
        shutil.rmtree(root, ignore_errors=True)
