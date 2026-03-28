"""Business-focused tests for pipeline.etl.timestamp_ocr.parser module."""

from __future__ import annotations

from datetime import datetime

import pytest

from pipeline.etl.timestamp_ocr.camera_profiles import BolyProfile, ReconyxProfile, UnknownProfile
from pipeline.etl.timestamp_ocr.parser import normalize_ocr_text, parse_timestamp

# ---------------------------------------------------------------------------
# normalize_ocr_text
# ---------------------------------------------------------------------------


class TestNormalizeOcrText:
    def test_fixes_uppercase_o_and_lowercase_l(self) -> None:
        assert normalize_ocr_text("2O2l-O5-l2 lO:3O:OO") == "2021-05-12 10:30:00"

    def test_fixes_lowercase_o(self) -> None:
        assert normalize_ocr_text("2o21") == "2021"

    def test_fixes_uppercase_i(self) -> None:
        assert normalize_ocr_text("I0:30:00") == "10:30:00"

    def test_fixes_pipe_character(self) -> None:
        # "|" is often misread by OCR when the "1" digit is thin.
        assert normalize_ocr_text("|0:30:00") == "10:30:00"

    def test_leaves_clean_text_unchanged(self) -> None:
        text = "2021-05-12 10:30:00"
        assert normalize_ocr_text(text) == text


# ---------------------------------------------------------------------------
# parse_timestamp — ReconyxProfile (ISO only)
# ---------------------------------------------------------------------------


class TestParseTimestampReconyx:
    @pytest.fixture
    def profile(self) -> ReconyxProfile:
        return ReconyxProfile()

    def test_parses_iso_timestamp(self, profile: ReconyxProfile) -> None:
        result = parse_timestamp("2021-05-12 10:30:00", profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)

    def test_parses_nosep_time_fallback(self, profile: ReconyxProfile) -> None:
        result = parse_timestamp("2021-05-12 103000", profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)

    def test_parses_dot_separated_time(self, profile: ReconyxProfile) -> None:
        # OCR misreads ":" as "." on some Reconyx overlays.
        result = parse_timestamp("2021-05-12 10.30.00", profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)

    def test_applies_ocr_normalization_before_matching(self, profile: ReconyxProfile) -> None:
        result = parse_timestamp("2O2l-O5-l2 lO:3O:OO", profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)

    def test_rejects_year_below_minimum(self, profile: ReconyxProfile) -> None:
        assert parse_timestamp("2014-05-12 10:30:00", profile, min_year=2015, max_year=2027) is None

    def test_rejects_year_above_maximum(self, profile: ReconyxProfile) -> None:
        assert parse_timestamp("2028-05-12 10:30:00", profile, min_year=2015, max_year=2027) is None

    def test_rejects_invalid_calendar_date(self, profile: ReconyxProfile) -> None:
        # February 31 does not exist.
        assert parse_timestamp("2021-02-31 10:30:00", profile, min_year=2015, max_year=2027) is None

    def test_returns_none_on_garbage_text(self, profile: ReconyxProfile) -> None:
        assert parse_timestamp("no timestamp here", profile, min_year=2015, max_year=2027) is None

    def test_extracts_timestamp_from_noisy_ocr_output(self, profile: ReconyxProfile) -> None:
        # Real OCR output often has junk around the timestamp.
        text = "  ~~~ 2021-05-12 10:30:00 abc"
        result = parse_timestamp(text, profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)


# ---------------------------------------------------------------------------
# parse_timestamp — BolyProfile (ISO + EU)
# ---------------------------------------------------------------------------


class TestParseTimestampBoly:
    @pytest.fixture
    def profile(self) -> BolyProfile:
        return BolyProfile()

    def test_parses_iso_format(self, profile: BolyProfile) -> None:
        result = parse_timestamp("2021-05-12 10:30:00", profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)

    def test_parses_eu_format_with_group_remapping(self, profile: BolyProfile) -> None:
        # EU format: DD-MM-YYYY — group_order remaps to (year, month, day).
        result = parse_timestamp("22-02-2016 19:20:39", profile, min_year=2015, max_year=2027)
        assert result == datetime(2016, 2, 22, 19, 20, 39)

    def test_parses_eu_nosep_format(self, profile: BolyProfile) -> None:
        result = parse_timestamp("22-02-2016 192039", profile, min_year=2015, max_year=2027)
        assert result == datetime(2016, 2, 22, 19, 20, 39)

    def test_iso_takes_priority_over_eu(self, profile: BolyProfile) -> None:
        # An ISO-formatted string should match via the ISO pattern, not the EU one.
        # This would fail if EU matched first and swapped day/year incorrectly.
        result = parse_timestamp("2016-05-28 09:39:54", profile, min_year=2015, max_year=2027)
        assert result == datetime(2016, 5, 28, 9, 39, 54)


# ---------------------------------------------------------------------------
# parse_timestamp — UnknownProfile (broadest fallback)
# ---------------------------------------------------------------------------


class TestParseTimestampUnknown:
    @pytest.fixture
    def profile(self) -> UnknownProfile:
        return UnknownProfile()

    def test_parses_iso_format(self, profile: UnknownProfile) -> None:
        result = parse_timestamp("2021-05-12 10:30:00", profile, min_year=2015, max_year=2027)
        assert result == datetime(2021, 5, 12, 10, 30, 0)

    def test_parses_eu_format(self, profile: UnknownProfile) -> None:
        result = parse_timestamp("22-02-2016 19:20:39", profile, min_year=2015, max_year=2027)
        assert result == datetime(2016, 2, 22, 19, 20, 39)
