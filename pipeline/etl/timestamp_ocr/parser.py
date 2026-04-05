"""Pure OCR text parsing utilities for timestamps."""

from __future__ import annotations

from datetime import datetime

from pipeline.etl.timestamp_ocr.camera_profiles import CameraProfile, TimestampPattern

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Apply only low-risk OCR substitutions seen on timestamp overlays.
OCR_CHAR_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "l": "1",
        "I": "1",
        "|": "1",
    }
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def normalize_ocr_text(text: str) -> str:
    """
    Normalize ambiguous OCR characters in raw text.

    Applies ``OCR_CHAR_MAP`` to fix common misreads on camera-trap overlays
    (e.g. ``O`` -> ``0``, ``l`` -> ``1``).

    Args:
        text: Raw OCR output string.

    Returns:
        Text with character substitutions applied.
    """
    return text.translate(OCR_CHAR_MAP)


# ------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------


def _try_pattern(
    text: str,
    pattern: TimestampPattern,
    min_year: int,
    max_year: int,
) -> datetime | None:
    """
    Attempt to match and extract a datetime from text using one pattern.

    Args:
        text: Normalized OCR text to search.
        pattern: Compiled regex with a ``group_order`` mapping that tells
            which capture group corresponds to year/month/day/hour/min/sec.
        min_year: Earliest acceptable year (inclusive).
        max_year: Latest acceptable year (inclusive).

    Returns:
        Parsed ``datetime`` if the pattern matches and the date is valid,
        ``None`` otherwise.
    """
    match = pattern.pattern.search(text)
    if not match:
        return None

    # Groups are captured by regex on OCR text extracted from the image overlay (capture timestamp).
    groups = match.groups()
    order = pattern.group_order

    year = int(groups[order[0]])
    month = int(groups[order[1]])
    day = int(groups[order[2]])
    hour = int(groups[order[3]])
    minute = int(groups[order[4]])
    second = int(groups[order[5]])

    if not (min_year <= year <= max_year):
        # Reject OCR hallucinations like 1023 or 3025 early before datetime construction.
        return None

    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def parse_timestamp(
    text: str,
    profile: CameraProfile,
    min_year: int,
    max_year: int,
) -> datetime | None:
    """
    Parse a timestamp from OCR text using a camera profile's patterns.

    Patterns are tried in the order defined by the profile (highest priority
    first). The first successful match is returned.

    Args:
        text: Raw OCR text (will be normalized internally).
        profile: Camera profile providing the ordered list of timestamp
            patterns to try.
        min_year: Earliest acceptable year (inclusive).
        max_year: Latest acceptable year (inclusive).

    Returns:
        Parsed ``datetime`` on success, ``None`` if no pattern matched.
    """
    normalized = normalize_ocr_text(text)

    for pattern in profile.patterns:
        result = _try_pattern(normalized, pattern, min_year, max_year)
        if result is not None:
            return result

    return None
