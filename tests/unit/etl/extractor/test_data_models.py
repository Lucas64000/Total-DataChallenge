"""Unit tests for ETL extraction data models."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pipeline.etl.extractor.data_models import FileData


class TestFileDataReadContent:
    """Tests for FileData.read_content()."""

    def test_reads_from_disk(self, sandbox: Path) -> None:
        src = sandbox / "test.jpg"
        src.write_bytes(b"image-data")
        fd = FileData.from_path(src)
        assert fd.read_content() == b"image-data"

    def test_reads_from_zip(self, sandbox: Path) -> None:
        zip_path = sandbox / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("subdir/img.jpg", b"zip-image-data")
        fd = FileData.from_zip(zip_path, "subdir/img.jpg")
        # Content is loaded from the ZIP only when read_content() is called.
        assert fd.read_content() == b"zip-image-data"

    def test_raises_when_no_source_configured(self) -> None:
        fd = FileData(stem="orphan", name="orphan.jpg")
        # Defensive guard: FileData must always point to disk or ZIP.
        with pytest.raises(ValueError, match="No content source"):
            fd.read_content()


class TestFileDataSourceHint:
    """Tests for FileData.source_hint."""

    def test_shows_filesystem_path(self) -> None:
        fd = FileData.from_path(Path("/data/img.jpg"))
        assert fd.source_hint == str(Path("/data/img.jpg"))

    def test_shows_zip_bang_entry(self) -> None:
        fd = FileData.from_zip(Path("/data/archive.zip"), "dir/b.jpg")
        # The "zip!entry" format helps identify the exact source inside archives.
        assert fd.source_hint == f"{Path('/data/archive.zip')}!dir/b.jpg"

    def test_fallback_shows_name_only(self) -> None:
        fd = FileData(stem="x", name="x.jpg")
        assert fd.source_hint == "x.jpg"


class TestFileDataFactories:
    """Tests for FileData constructor helpers."""

    def test_from_path_extracts_stem_and_name(self) -> None:
        fd = FileData.from_path(Path("/data/img123.jpg"))
        assert fd.stem == "img123"
        assert fd.name == "img123.jpg"

    def test_from_zip_extracts_entry_stem_and_name(self) -> None:
        fd = FileData.from_zip(Path("/data/a.zip"), "cam1/FR_001.txt")
        assert fd.stem == "FR_001"
        assert fd.name == "FR_001.txt"
