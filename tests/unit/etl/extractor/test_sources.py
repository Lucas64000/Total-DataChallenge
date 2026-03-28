"""Unit tests for extractor source scanning behavior."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

from pipeline.etl.extractor.sources import SourceScanner


def test_scan_labelized_sources_returns_pairs_and_duplicate_lists() -> None:
    # Local sandbox to emulate two different source folders.
    root = Path("sandbox_sources_labelized")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    first_dir = root / "a"
    second_dir = root / "b"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    (first_dir / "dup.jpg").write_bytes(b"img-1")
    (first_dir / "dup.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (second_dir / "dup.jpg").write_bytes(b"img-2")
    (second_dir / "dup.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Same stem ("dup") appears twice across folders -> one canonical pair + duplicates list.
    scanner = SourceScanner(MagicMock())
    result = scanner.scan_labelized_sources(root)

    # Main output keeps one usable pair.
    assert "dup" in result.pairs
    assert result.pairs["dup"].image is not None
    assert result.pairs["dup"].annotation is not None
    # Extra files are tracked as duplicates for quarantine/reporting.
    assert len(result.duplicate_images) == 1
    assert len(result.duplicate_annotations) == 1
    assert Path(result.duplicate_images[0].source_hint).name == "dup.jpg"
    assert Path(result.duplicate_annotations[0].source_hint).name == "dup.txt"
    # Cleanup sandbox folder created by this test.
    shutil.rmtree(root, ignore_errors=True)
