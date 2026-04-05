"""
Parse camera-trap filenames into structured metadata.

The parser extracts the metadata required by the ML pipeline:
- location information used for location-based CV splits
- species label
- camera family used by OCR timestamp extraction

Date tokens from filenames are intentionally ignored because they may reflect
export/rename workflow dates instead of the real capture moment. Temporal logic
uses the timestamp extracted via OCR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.types import CameraType

# ------------------------------------------------------------------
# Camera Detection
# ------------------------------------------------------------------

def detect_camera_type(identifier: str) -> CameraType:
    """
    Detect camera type from filename or camera_id.

    Args:
        identifier: Filename or camera_id string

    Returns:
        'reconyx', 'boly', or 'unknown'
    """
    upper = identifier.upper()
    if "RCNX" in upper:
        return "reconyx"
    if "IMAG" in upper:
        return "boly"
    return "unknown"


# ------------------------------------------------------------------
# Metadata Model
# ------------------------------------------------------------------

@dataclass(slots=True)
class PhototrapMetadata:
    """Essential metadata extracted from a camera trap filename."""

    filename: str
    coord_n: str | None = None
    coord_w: str | None = None
    species: str | None = None
    camera_type: CameraType = "unknown"
    labeled: bool = False
    parse_success: bool = False
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a plain dict for DataFrame record construction.

        The returned keys match the columns expected by DataFrameBuilder._build_record.
        ``location_id`` is derived from coordinates and included explicitly because
        it is a computed property, not a raw field.
        """
        return {
            "filename": self.filename,
            "coord_n": self.coord_n,
            "coord_w": self.coord_w,
            "location_id": self.location_id,
            "species": self.species,
            "camera_type": self.camera_type,
            "labeled": self.labeled,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }

    @property
    def location_id(self) -> str | None:
        """
        Unique location identifier for CV splits.

        Combines N and W coordinates.

        Returns:
            Canonical location identifier or ``None`` when coordinates are missing.
        """
        if self.coord_n and self.coord_w:
            # Some sources append a trailing letter (e.g. W1234a); drop it for stable grouping.
            coord_w_clean = re.sub(r"[a-z]$", "", self.coord_w)
            return f"{self.coord_n}_{coord_w_clean}"
        return None


# ------------------------------------------------------------------
# Filename Parser
# ------------------------------------------------------------------

class FilenameParser:
    """
    Parser for camera trap filenames.

    Expected canonical prefix:
    {COUNTRY}_{COORD_N}_{COORD_W}_{DATE}_{SPECIES}_...

    The parser intentionally locks onto the canonical prefix and ignores the
    remaining suffix. This keeps metadata extraction resilient to non-standard
    tails such as vernacular labels, duplicate markers, extra date fragments,
    or missing camera IDs.
    """

    COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")
    COORD_N_PATTERN = re.compile(r"^N[\d-]+$")
    COORD_W_PATTERN = re.compile(r"^W[\d-]+[a-z]?$")
    DATE_PATTERN = re.compile(r"^\d{8}$")
    SPECIES_PATTERN = re.compile(r"^[A-Za-z-]+$")

    @staticmethod
    def split_species(raw: str) -> list[str]:
        """
        Split a concatenated multi-species string into individual species names.

        Uses binomial nomenclature convention: genus starts with uppercase,
        epithet with lowercase. The boundary between two species is always
        a hyphen between a lowercase letter and an uppercase letter.

        Examples:
            >>> FilenameParser.split_species("Ardea-cinerea-Martes-martes")
            ['Ardea-cinerea', 'Martes-martes']
            >>> FilenameParser.split_species("Homo-sapiens-Canis-lupus-familiaris")
            ['Homo-sapiens', 'Canis-lupus-familiaris']
            >>> FilenameParser.split_species("Ardea-cinerea")
            ['Ardea-cinerea']
        """
        return re.split(r"(?<=[a-z])-(?=[A-Z])", raw)

    @classmethod
    def parse(cls, filename: str, labeled: bool = False) -> PhototrapMetadata:
        """
        Parse a filename and extract essential metadata.

        Args:
            filename: Filename (with or without extension)
            labeled: Whether this image has annotations

        Returns:
            PhototrapMetadata with extracted fields
        """
        name = Path(filename).stem
        metadata = PhototrapMetadata(filename=filename, labeled=labeled)

        parts = name.split("_")
        if len(parts) < 5:
            metadata.parse_error = "Filename does not contain the canonical 5-token prefix"
            return metadata

        country, coord_n, coord_w, date_token, species = parts[:5]

        if not cls.COUNTRY_PATTERN.fullmatch(country):
            metadata.parse_error = "Invalid country token"
            return metadata
        if not cls.COORD_N_PATTERN.fullmatch(coord_n):
            metadata.parse_error = "Invalid coord_n token"
            return metadata
        if not cls.COORD_W_PATTERN.fullmatch(coord_w):
            metadata.parse_error = "Invalid coord_w token"
            return metadata
        if not cls.DATE_PATTERN.fullmatch(date_token):
            metadata.parse_error = "Invalid date token"
            return metadata
        if not cls.SPECIES_PATTERN.fullmatch(species):
            metadata.parse_error = "Invalid species token"
            return metadata

        metadata.coord_n = coord_n
        metadata.coord_w = coord_w
        metadata.species = species
        metadata.camera_type = detect_camera_type(name)
        metadata.parse_success = True
        return metadata
