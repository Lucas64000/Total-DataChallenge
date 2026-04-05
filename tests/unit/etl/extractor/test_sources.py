"""Unit tests for extractor source scanning behavior."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from pipeline.etl.extractor.data_models import FileData, FilePair
from pipeline.etl.extractor.sources import SourceScanner


def _build_scanner() -> SourceScanner:
    """Create a scanner with a mocked logger."""
    return SourceScanner(MagicMock())


# ---------------------------------------------------------------------------
# scan_labelized_sources
# ---------------------------------------------------------------------------


class TestScanLabelizedSources:
    def test_returns_pairs_and_duplicate_lists(self, sandbox: Path) -> None:
        root = sandbox / "labelized"
        first_dir = root / "a"
        second_dir = root / "b"
        first_dir.mkdir(parents=True)
        second_dir.mkdir(parents=True)

        (first_dir / "dup.jpg").write_bytes(b"img-1")
        (first_dir / "dup.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (second_dir / "dup.jpg").write_bytes(b"img-2")
        (second_dir / "dup.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        result = _build_scanner().scan_labelized_sources(root)

        assert "dup" in result.pairs
        assert result.pairs["dup"].image is not None
        assert result.pairs["dup"].annotation is not None
        # One canonical pair is kept, extras are tracked for quarantine.
        assert len(result.duplicate_images) == 1
        assert len(result.duplicate_annotations) == 1
        assert Path(result.duplicate_images[0].source_hint).name == "dup.jpg"
        assert Path(result.duplicate_annotations[0].source_hint).name == "dup.txt"

    def test_empty_directory(self, sandbox: Path) -> None:
        root = sandbox / "empty"
        root.mkdir()

        result = _build_scanner().scan_labelized_sources(root)

        assert len(result.pairs) == 0
        assert len(result.duplicate_images) == 0
        assert len(result.duplicate_annotations) == 0

    def test_handles_zip_and_loose_files(self, sandbox: Path) -> None:
        root = sandbox / "labelized"
        root.mkdir()

        (root / "loose.jpg").write_bytes(b"img")
        (root / "loose.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        zip_path = root / "batch.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cam/zipped.jpg", b"zip-img")
            zf.writestr("cam/zipped.txt", b"0 0.5 0.5 0.2 0.2\n")

        result = _build_scanner().scan_labelized_sources(root)

        assert "loose" in result.pairs
        assert result.pairs["loose"].image is not None
        assert result.pairs["loose"].annotation is not None
        assert "zipped" in result.pairs
        assert result.pairs["zipped"].image is not None
        assert result.pairs["zipped"].annotation is not None


# ---------------------------------------------------------------------------
# _register_file
# ---------------------------------------------------------------------------


class TestRegisterFile:
    def test_skips_hidden_stem(self) -> None:
        pairs: dict[str, FilePair] = {}

        _build_scanner()._register_file(pairs, FileData(stem=".hidden", name=".hidden.jpg"))

        assert len(pairs) == 0

    def test_skips_classes_stem(self) -> None:
        pairs: dict[str, FilePair] = {}

        _build_scanner()._register_file(pairs, FileData(stem="classes", name="classes.txt"))

        assert len(pairs) == 0

    def test_skips_unsupported_extension(self) -> None:
        pairs: dict[str, FilePair] = {}

        _build_scanner()._register_file(pairs, FileData(stem="readme", name="readme.md"))

        assert len(pairs) == 0

    def test_tracks_duplicate_annotations(self) -> None:
        scanner = _build_scanner()
        pairs: dict[str, FilePair] = {}
        duplicate_annotations: list[FileData] = []
        first = FileData(stem="dup", name="dup.txt")
        second = FileData(stem="dup", name="dup.txt", _source_path=Path("other/dup.txt"))

        scanner._register_file(pairs, first, duplicate_annotations=duplicate_annotations)
        scanner._register_file(pairs, second, duplicate_annotations=duplicate_annotations)

        assert pairs["dup"].annotation is first
        assert len(duplicate_annotations) == 1
        assert duplicate_annotations[0] is second


# ---------------------------------------------------------------------------
# scan_unlabeled_sources
# ---------------------------------------------------------------------------


class TestScanUnlabeledSources:
    def test_discovers_images_in_dirs_and_zips(self, sandbox: Path) -> None:
        root = sandbox / "unlabeled"
        root.mkdir()

        (root / "direct.jpg").write_bytes(b"img")
        subdir = root / "batch1"
        subdir.mkdir()
        (subdir / "nested.jpeg").write_bytes(b"img2")

        zip_path = root / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cam1/zip_img.jpg", b"img3")

        images = _build_scanner().scan_unlabeled_sources(root)

        names = {img.name for img in images}
        assert "direct.jpg" in names
        assert "nested.jpeg" in names
        assert "zip_img.jpg" in names

    def test_skips_hidden_files(self, sandbox: Path) -> None:
        root = sandbox / "unlabeled"
        root.mkdir()
        (root / ".hidden.jpg").write_bytes(b"skip")
        (root / "visible.jpg").write_bytes(b"keep")

        images = _build_scanner().scan_unlabeled_sources(root)

        names = {img.name for img in images}
        assert "visible.jpg" in names
        assert ".hidden.jpg" not in names


# ---------------------------------------------------------------------------
# ZIP and classes helpers
# ---------------------------------------------------------------------------


class TestZipScanning:
    def test_scan_zip_registers_pairs_from_real_zip(self, sandbox: Path) -> None:
        zip_path = sandbox / "batch.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("cam1/img001.jpg", b"img-data")
            zf.writestr("cam1/img001.txt", b"0 0.5 0.5 0.1 0.1\n")
            zf.writestr("cam1/classes.txt", b"animal\n")

        pairs: dict[str, FilePair] = {}
        _build_scanner()._scan_zip(zip_path, pairs)

        assert "img001" in pairs
        assert pairs["img001"].image is not None
        assert pairs["img001"].annotation is not None
        assert "classes" not in pairs

    def test_find_classes_in_zip_reads_from_real_zip(self, sandbox: Path) -> None:
        zip_path = sandbox / "batch.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/classes.txt", b"animal\nvehicle\n")
            zf.writestr("data/img.jpg", b"img")

        content = _build_scanner().find_classes_in_zip(zip_path)

        assert content == b"animal\nvehicle\n"

    def test_find_classes_in_zip_handles_bad_zip(self, sandbox: Path) -> None:
        bad_zip = sandbox / "bad.zip"
        bad_zip.write_bytes(b"not-a-zip")

        assert _build_scanner().find_classes_in_zip(bad_zip) is None


class TestFindClassesContent:
    def test_finds_regular_file(self, sandbox: Path) -> None:
        root = sandbox / "labelized"
        (root / "subdir").mkdir(parents=True)
        (root / "subdir" / "classes.txt").write_bytes(b"cat\ndog\n")

        result = _build_scanner().find_classes_content(root)

        assert result is not None
        content, _ = result
        assert content == b"cat\ndog\n"

    def test_returns_none_when_missing(self, sandbox: Path) -> None:
        root = sandbox / "no_classes"
        root.mkdir()

        assert _build_scanner().find_classes_content(root) is None
