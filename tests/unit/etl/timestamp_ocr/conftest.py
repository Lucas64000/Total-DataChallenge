"""Shared fixtures for timestamp OCR unit tests."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.etl.timestamp_ocr.camera_profiles import BolyProfile, ReconyxProfile, UnknownProfile


@pytest.fixture
def reconyx_profile() -> ReconyxProfile:
    """Concrete Reconyx profile used across parser/profile tests."""
    return ReconyxProfile()


@pytest.fixture
def boly_profile() -> BolyProfile:
    """Concrete Boly profile used across parser/profile tests."""
    return BolyProfile()


@pytest.fixture
def unknown_profile() -> UnknownProfile:
    """Fallback Unknown profile used across parser/profile tests."""
    return UnknownProfile()


@pytest.fixture
def reconyx_path(reconyx_path_builder: Callable[[int], Path]) -> Path:
    """Shared single-image path used across timestamp extractor tests."""
    return reconyx_path_builder(1)


@pytest.fixture
def reconyx_path_builder(etl_filenames: SimpleNamespace) -> Callable[[int], Path]:
    """Factory fixture to build Reconyx-style paths with a chosen index."""
    template = etl_filenames.reconyx_ocr_template
    return lambda index=1: Path(template.format(index=index))
