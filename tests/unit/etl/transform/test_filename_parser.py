"""Unit tests for filename parsing and camera type detection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.etl.transform.filename_parser import FilenameParser, detect_camera_type


class TestSplitSpecies:
    def test_single_two_part_species(self) -> None:
        assert FilenameParser.split_species("Ardea-cinerea") == ["Ardea-cinerea"]

    def test_single_three_part_species(self) -> None:
        assert FilenameParser.split_species("Canis-lupus-familiaris") == [
            "Canis-lupus-familiaris"
        ]

    def test_two_two_part_species(self) -> None:
        assert FilenameParser.split_species("Ardea-cinerea-Martes-martes") == [
            "Ardea-cinerea",
            "Martes-martes",
        ]

    def test_three_species(self) -> None:
        assert FilenameParser.split_species(
            "Motacilla-alba-Capreolus-capreolus-Ardea-cinerea"
        ) == ["Motacilla-alba", "Capreolus-capreolus", "Ardea-cinerea"]


class TestFilenameParserToDict:
    def test_to_dict_keeps_coordinates_when_parse_succeeds(
        self, etl_filenames: SimpleNamespace
    ) -> None:
        result = FilenameParser.parse(etl_filenames.boly_standard).to_dict()
        assert result["coord_n"] == "N0431710-871"
        assert result["coord_w"] == "W0000230-087"

    def test_to_dict_sets_coordinates_to_none_when_parse_fails(
        self, etl_filenames: SimpleNamespace
    ) -> None:
        result = FilenameParser.parse(etl_filenames.invalid).to_dict()
        assert result["coord_n"] is None
        assert result["coord_w"] is None

    def test_parse_keeps_multi_species_token_from_canonical_prefix(
        self, etl_filenames: SimpleNamespace
    ) -> None:
        result = FilenameParser.parse(etl_filenames.multi_species_reconyx)

        assert result.parse_success is True
        assert result.species == "Ardea-cinerea-Martes-martes"
        assert result.camera_type == "reconyx"
        assert result.location_id == "N0431703-408_W0000243-7760"

    def test_parse_ignores_non_standard_suffix_after_species(
        self, etl_filenames: SimpleNamespace
    ) -> None:
        result = FilenameParser.parse(etl_filenames.alt_unknown_camera)

        assert result.parse_success is True
        assert result.species == "Canis-lupus-familiaris"
        assert result.camera_type == "unknown"
        assert result.location_id == "N0431702-522_W0000243-861"


class TestDetectCameraType:
    @pytest.mark.parametrize(
        ("identifier", "expected"),
        [
            ("detect_reconyx_id", "reconyx"),
            ("detect_reconyx_filename", "reconyx"),
            ("detect_boly_id", "boly"),
            ("detect_boly_filename", "boly"),
            ("detect_unknown_filename", "unknown"),
            ("detect_unknown_alt", "unknown"),
        ],
        ids=[
            "reconyx-id",
            "reconyx-filename",
            "boly-id",
            "boly-filename",
            "unknown-filename",
            "unknown-alt",
        ],
    )
    def test_detects_expected_type_from_identifier(
        self, etl_filenames: SimpleNamespace, identifier: str, expected: str
    ) -> None:
        assert detect_camera_type(getattr(etl_filenames, identifier)) == expected
