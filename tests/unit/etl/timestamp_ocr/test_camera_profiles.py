"""Contract tests for camera profile static configuration."""

from __future__ import annotations

import pytest

from pipeline.etl.timestamp_ocr.camera_profiles import (
    BolyProfile,
    CropRegion,
    ReconyxProfile,
    UnknownProfile,
    get_profile,
)

# ---------------------------------------------------------------------------
# CropRegion
# ---------------------------------------------------------------------------


class TestCropRegion:
    @pytest.mark.parametrize(
        ("region", "width", "height", "expected_bbox"),
        [
            (
                CropRegion(x_start=0.0, y_start=0.0, x_end=0.5, y_end=0.1),
                1000,
                1000,
                (0, 0, 500, 100),
            ),
            (
                CropRegion(x_start=0.45, y_start=0.96, x_end=1.0, y_end=1.0),
                2048,
                1536,
                (921, 1474, 2048, 1536),
            ),
        ],
        ids=["square-image", "widescreen-image"],
    )
    def test_to_pixels_converts_ratios_to_expected_bbox(
        self,
        region: CropRegion,
        width: int,
        height: int,
        expected_bbox: tuple[int, int, int, int],
    ) -> None:
        # CropRegion is pure math. This test protects against accidental changes
        # in conversion rules that would shift OCR crop windows.
        assert region.to_pixels(width=width, height=height) == expected_bbox


# ---------------------------------------------------------------------------
# Profile registry / global profile contract
# ---------------------------------------------------------------------------


class TestProfileRegistry:
    @pytest.mark.parametrize(
        ("camera_type", "expected_cls", "expected_pattern_count", "expected_bbox"),
        [
            ("reconyx", ReconyxProfile, 2, (0, 0, 716, 46)),
            ("boly", BolyProfile, 4, (921, 1474, 2048, 1536)),
            ("unknown", UnknownProfile, 4, (0, 0, 2048, 153)),
        ],
        ids=["reconyx", "boly", "unknown"],
    )
    def test_registry_returns_expected_profile_contract(
        self,
        camera_type: str,
        expected_cls: type,
        expected_pattern_count: int,
        expected_bbox: tuple[int, int, int, int],
    ) -> None:
        profile = get_profile(camera_type)  # type: ignore[arg-type]

        # These checks:
        # - correct profile class
        # - expected number of parsing patterns
        # - calibrated crop area for 2048x1536 reference images
        assert isinstance(profile, expected_cls)
        assert len(profile.patterns) == expected_pattern_count
        assert profile.crop_region.to_pixels(width=2048, height=1536) == expected_bbox

    @pytest.mark.parametrize("camera_type", ["reconyx", "boly", "unknown"])
    def test_registry_returns_singleton_instances(self, camera_type: str) -> None:
        # get_profile() should be stable: same object each call.
        assert get_profile(camera_type) is get_profile(camera_type)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reconyx profile specifics
# ---------------------------------------------------------------------------


class TestReconyxProfile:
    def test_rejects_eu_format_patterns(self, reconyx_profile: ReconyxProfile) -> None:
        # Reconyx contract: only ISO variants are supported.
        text = "28-05-2016 09:39:54"
        assert all(pattern.pattern.search(text) is None for pattern in reconyx_profile.patterns)

    def test_all_patterns_use_iso_group_order(self, reconyx_profile: ReconyxProfile) -> None:
        # group_order defines how regex groups map to datetime components.
        iso_order = (0, 1, 2, 3, 4, 5)
        assert all(pattern.group_order == iso_order for pattern in reconyx_profile.patterns)


# ---------------------------------------------------------------------------
# Boly profile specifics
# ---------------------------------------------------------------------------


class TestBolyProfile:
    def test_iso_patterns_are_prioritized_before_eu(self, boly_profile: BolyProfile) -> None:
        # The parser stops at first match, so order matters.
        # ISO is most common in the dataset and must be tried first.
        iso_order = (0, 1, 2, 3, 4, 5)
        eu_order = (2, 1, 0, 3, 4, 5)
        orders = [pattern.group_order for pattern in boly_profile.patterns]
        assert orders == [iso_order, iso_order, eu_order, eu_order]

    @pytest.mark.parametrize(
        "pattern_index", [2, 3], ids=["eu-with-separators", "eu-no-separators"]
    )
    def test_eu_patterns_use_day_month_year_remapping(
        self, boly_profile: BolyProfile, pattern_index: int
    ) -> None:
        # EU regex captures (day, month, year), then remaps to (year, month, day).
        assert boly_profile.patterns[pattern_index].group_order == (2, 1, 0, 3, 4, 5)


# ---------------------------------------------------------------------------
# Unknown profile specifics
# ---------------------------------------------------------------------------


class TestUnknownProfile:
    def test_keeps_broad_fallback_pattern_set(self, unknown_profile: UnknownProfile) -> None:
        # Unknown profile is recall-first: ISO + EU, with and without time separators.
        assert len(unknown_profile.patterns) == 4
        assert [pattern.group_order for pattern in unknown_profile.patterns] == [
            (0, 1, 2, 3, 4, 5),
            (0, 1, 2, 3, 4, 5),
            (2, 1, 0, 3, 4, 5),
            (2, 1, 0, 3, 4, 5),
        ]
