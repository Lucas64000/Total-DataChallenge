"""Business-focused tests for pipeline.etl.timestamp_ocr.core module."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from pipeline.etl.timestamp_ocr import TimestampExtractor

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeImage:
    """Tiny PIL-like test double used to avoid real image decoding in unit tests."""

    size = (800, 600)

    def crop(self, _bbox):  # noqa: ANN001 - mimics PIL API
        return self

    def convert(self, _mode: str) -> _FakeImage:
        return self

    def __enter__(self) -> _FakeImage:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:  # noqa: ANN001 - context manager API
        return False


class _FakeOCREngine:
    """Minimal stand-in for a real OCR engine (e.g. EasyOCR, Tesseract, TrOCR).

    Args:
        read_text:    Text returned for every single-image read() call.
        batch_texts:  Per-image texts for read_batch(); falls back to read_text
                      if not provided (all images get the same text).
        fail_batch:   Set True to simulate an OCR backend crash on batch calls.
    """

    MODEL_NAME = "fake-ocr"

    def __init__(
        self,
        read_text: str = "2021-05-12 10:30:00",
        batch_texts: list[str] | None = None,
        fail_batch: bool = False,
    ) -> None:
        self.read_text = read_text
        self.batch_texts = batch_texts or []
        self.fail_batch = fail_batch

    def read(self, image: Image.Image) -> str:
        return self.read_text

    def read_batch(self, images: list[Image.Image]) -> list[str]:
        if self.fail_batch:
            raise RuntimeError("batch OCR exploded")
        if self.batch_texts:
            return self.batch_texts[: len(images)]
        return [self.read_text for _ in images]


def _patch_image_open():  # noqa: ANN202
    """Patch PIL.Image.open to return a _FakeImage."""
    return patch(
        "pipeline.etl.timestamp_ocr.core.Image.open",
        side_effect=lambda _p: _FakeImage(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Single-image extract()
# ---------------------------------------------------------------------------


class TestExtractSingle:
    """Tests for single-image extraction workflow."""

    def test_happy_path_returns_parsed_timestamp(self, reconyx_path: Path) -> None:
        extractor = TimestampExtractor(gpu=False, ocr_engine=_FakeOCREngine())

        with _patch_image_open():
            result = extractor.extract(reconyx_path)

        assert result.success is True
        assert result.timestamp == datetime(2021, 5, 12, 10, 30, 0)
        assert result.camera_type == "reconyx"
        assert result.raw_text == "2021-05-12 10:30:00"

    def test_explicit_camera_type_overrides_detection(self, reconyx_path: Path) -> None:
        # Filename says "RCNX" (reconyx), but caller forces "boly".
        extractor = TimestampExtractor(gpu=False, ocr_engine=_FakeOCREngine())

        with _patch_image_open():
            result = extractor.extract(reconyx_path, camera_type="boly")

        assert result.success is True
        assert result.camera_type == "boly"

    def test_unparseable_ocr_returns_failure_with_raw_text(self, reconyx_path: Path) -> None:
        # OCR produces garbage: parsing fails but raw_text is kept for debugging.
        engine = _FakeOCREngine(read_text="no timestamp here")
        extractor = TimestampExtractor(gpu=False, ocr_engine=engine)

        with _patch_image_open():
            result = extractor.extract(reconyx_path)

        assert result.success is False
        assert result.raw_text == "no timestamp here"
        assert result.error is not None

    def test_image_open_failure_returns_error(self, reconyx_path: Path) -> None:
        extractor = TimestampExtractor(gpu=False, ocr_engine=_FakeOCREngine())

        with patch(
            "pipeline.etl.timestamp_ocr.core.Image.open", side_effect=OSError("file not found")
        ):
            result = extractor.extract(reconyx_path)

        assert result.success is False
        assert "file not found" in (result.error or "")

    def test_profile_lookup_failure_returns_explicit_error(self, reconyx_path: Path) -> None:
        extractor = TimestampExtractor(gpu=False, ocr_engine=_FakeOCREngine())

        with patch("pipeline.etl.timestamp_ocr.core.get_profile", side_effect=KeyError("bad")):
            result = extractor.extract(reconyx_path)

        assert result.success is False
        assert result.camera_type == "unknown"
        assert "Unsupported camera type" in (result.error or "")


# ---------------------------------------------------------------------------
# Batch extract_batch()
# ---------------------------------------------------------------------------


class TestExtractBatch:
    """Tests for batch extraction workflow."""

    def test_preserves_input_order(self, reconyx_path_builder: Callable[[int], Path]) -> None:
        engine = _FakeOCREngine(
            batch_texts=[
                "2021-05-12 10:30:01",
                "2021-05-12 10:30:02",
                "2021-05-12 10:30:03",
            ]
        )
        extractor = TimestampExtractor(gpu=False, ocr_engine=engine)
        paths = [reconyx_path_builder(i) for i in range(1, 4)]

        with _patch_image_open():
            results = extractor.extract_batch(paths, show_progress=False, batch_size=3)

        assert [r.success for r in results] == [True, True, True]
        assert [r.timestamp for r in results] == [
            datetime(2021, 5, 12, 10, 30, 1),
            datetime(2021, 5, 12, 10, 30, 2),
            datetime(2021, 5, 12, 10, 30, 3),
        ]

    def test_empty_list_returns_empty(self) -> None:
        extractor = TimestampExtractor(gpu=False, ocr_engine=_FakeOCREngine())
        results = extractor.extract_batch([], show_progress=False)
        assert results == []

    def test_multi_batch_processes_all_images(
        self, reconyx_path_builder: Callable[[int], Path]
    ) -> None:
        engine = _FakeOCREngine(
            batch_texts=[
                "2021-05-12 10:00:01",
                "2021-05-12 10:00:02",
                "2021-05-12 10:00:03",
                "2021-05-12 10:00:04",
                "2021-05-12 10:00:05",
            ]
        )
        extractor = TimestampExtractor(gpu=False, ocr_engine=engine)
        paths = [reconyx_path_builder(i) for i in range(1, 6)]

        with _patch_image_open():
            results = extractor.extract_batch(paths, show_progress=False, batch_size=2)

        assert len(results) == 5
        assert all(r.success for r in results)

    def test_splits_work_into_expected_batch_sizes(
        self, reconyx_path_builder: Callable[[int], Path]
    ) -> None:
        engine = _FakeOCREngine(
            batch_texts=[
                "2021-05-12 10:00:01",
                "2021-05-12 10:00:02",
                "2021-05-12 10:00:03",
                "2021-05-12 10:00:04",
                "2021-05-12 10:00:05",
            ]
        )
        extractor = TimestampExtractor(gpu=False, ocr_engine=engine)
        paths = [reconyx_path_builder(i) for i in range(1, 6)]

        with (
            patch.object(engine, "read_batch", wraps=engine.read_batch) as read_batch_spy,
            _patch_image_open(),
        ):
            results = extractor.extract_batch(paths, show_progress=False, batch_size=2)

        assert len(results) == 5
        assert read_batch_spy.call_count == 3
        # 5 paths with batch_size=2 must produce batches [2, 2, 1].
        assert [len(call.args[0]) for call in read_batch_spy.call_args_list] == [2, 2, 1]

    def test_marks_all_items_failed_when_batch_ocr_crashes(
        self, reconyx_path_builder: Callable[[int], Path]
    ) -> None:
        engine = _FakeOCREngine(fail_batch=True)
        extractor = TimestampExtractor(gpu=False, ocr_engine=engine)
        paths = [reconyx_path_builder(1), reconyx_path_builder(2)]

        with _patch_image_open():
            results = extractor.extract_batch(paths, show_progress=False, batch_size=2)

        assert [r.success for r in results] == [False, False]
        assert all("Batch OCR failed" in (r.error or "") for r in results)

    def test_marks_only_unreadable_images_as_failed(
        self, reconyx_path_builder: Callable[[int], Path]
    ) -> None:
        engine = _FakeOCREngine(batch_texts=["2021-05-12 10:30:01"])
        extractor = TimestampExtractor(gpu=False, ocr_engine=engine)
        paths = [reconyx_path_builder(1), reconyx_path_builder(2)]

        def _open(path: Path) -> _FakeImage:
            if path.name.endswith("RCNX0002.jpg"):
                raise OSError("corrupted image")
            return _FakeImage()

        with patch("pipeline.etl.timestamp_ocr.core.Image.open", side_effect=_open):
            results = extractor.extract_batch(paths, show_progress=False, batch_size=2)

        assert results[0].success is True
        assert results[0].timestamp == datetime(2021, 5, 12, 10, 30, 1)
        assert results[1].success is False
        assert "corrupted image" in (results[1].error or "")

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_rejects_non_positive_batch_size(self, batch_size: int, reconyx_path: Path) -> None:
        extractor = TimestampExtractor(gpu=False, ocr_engine=_FakeOCREngine())

        with pytest.raises(ValueError, match="batch_size must be > 0"):
            extractor.extract_batch([reconyx_path], show_progress=False, batch_size=batch_size)
