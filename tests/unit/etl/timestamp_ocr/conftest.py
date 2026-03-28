"""Shared fixtures for timestamp OCR unit tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline.etl.timestamp_ocr.camera_profiles import BolyProfile, ReconyxProfile, UnknownProfile


def _build_reconyx_path(index: int = 1) -> Path:
    """Build a deterministic Reconyx-style filename used in OCR tests."""
    return Path(f"FR_N0431652-111_W0000251-205_20220725_Fox_RCNX{index:04d}.jpg")


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
def reconyx_path() -> Path:
    """Shared single-image path used across timestamp extractor tests."""
    return _build_reconyx_path(1)


@pytest.fixture
def reconyx_path_builder() -> Callable[[int], Path]:
    """Factory fixture to build Reconyx-style paths with a chosen index."""
    return _build_reconyx_path
