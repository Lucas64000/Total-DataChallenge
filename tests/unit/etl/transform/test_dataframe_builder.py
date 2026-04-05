"""Unit tests for pipeline.etl.transform.dataframe_builder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.etl.transform.dataframe_builder import DataFrameBuilder, build_dataframe

# ---------------------------------------------------------------------------
# Image ID
# ---------------------------------------------------------------------------


def test_build_image_id_is_deterministic_and_dataset_scoped(
    builder_no_timestamps: DataFrameBuilder,
) -> None:
    # image_id is a short hash used to uniquely identify images in the CSV.
    # It must be reproducible across pipeline runs (same input -> same ID)
    # and must differ across datasets so a "labelized/img.jpg" and an
    # "unlabeled/img.jpg" with the same filename get different IDs.
    builder = builder_no_timestamps

    id_a = builder._build_image_id("labelized", "img_001.jpg")
    id_b = builder._build_image_id("labelized", "img_001.jpg")
    id_c = builder._build_image_id("unlabeled", "img_001.jpg")

    assert id_a == id_b
    assert id_a != id_c
    assert len(id_a) == 16


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def test_parse_annotation_summary_handles_unreadable_text() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.side_effect = UnicodeDecodeError(
        "utf-8", b"\xff", 0, 1, "invalid start byte"
    )

    # Invalid encoding should degrade gracefully to empty summary.
    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, {})

    assert summary.label_bbox_count == 0
    assert summary.label_bbox_area_sum == 0.0


def test_parse_annotation_summary_returns_empty_when_annotation_file_is_missing() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = False

    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, {})

    assert summary.label_bbox_count == 0
    assert summary.label_bbox_area_sum == 0.0
    assert summary.all_species == ()


def test_parse_annotation_summary_uses_only_valid_boxes() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "\n".join(
        [
            "0 0.5 0.5 0.2 0.3",  # area = 0.06
            "0 0.2 0.2 0.1 0.1",  # area = 0.01
            "0 0.5 0.5 0.0 0.2",  # invalid width
            "bad line",  # invalid format
        ]
    )

    # Only geometrically valid YOLO rows contribute to counts/areas.
    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, {})

    assert summary.label_bbox_count == 2
    assert summary.label_bbox_area_sum == 0.07


def test_parse_annotation_summary_keeps_geometry_when_class_token_is_invalid() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "bad 0.5 0.5 0.2 0.2"

    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, {})

    assert summary.label_bbox_count == 1
    assert summary.label_bbox_area_sum == pytest.approx(0.04)


def test_parse_annotation_summary_ignores_out_of_range_center() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "1 1.2 0.5 0.2 0.2"

    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, {})

    assert summary.label_bbox_count == 0
    assert summary.label_bbox_area_sum == 0.0
    assert summary.all_species == ()


def test_parse_annotation_summary_ignores_bbox_exceeding_bounds() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "1 0.95 0.5 0.2 0.2"

    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, {})

    assert summary.label_bbox_count == 0
    assert summary.label_bbox_area_sum == 0.0
    assert summary.all_species == ()


# Class id 1 = Canis-lupus-familiaris, class id 3 = Genetta-genetta (LABELED_SPECIES order).
_SPECIES_MAPPING = {1: 1, 3: 3}


def test_parse_annotation_summary_all_species_ordered_by_frequency() -> None:
    # Two boxes of class 1 vs. one of class 3 -> Canis-lupus-familiaris first.
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "\n".join(
        [
            "1 0.5 0.5 0.2 0.2",
            "1 0.4 0.4 0.1 0.1",
            "3 0.3 0.3 0.1 0.1",
        ]
    )

    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, _SPECIES_MAPPING)

    assert summary.all_species == ("Canis-lupus-familiaris", "Genetta-genetta")
    assert summary.all_species[0] == "Canis-lupus-familiaris"


def test_parse_annotation_summary_ignores_rows_with_invalid_bbox_geometry() -> None:
    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "\n".join(
        [
            "1 0.5 0.5 0.0 0.2",  # invalid width, ignored for species
            "1 0.5 0.5 -0.1 0.2",  # invalid width, ignored for species
            "3 0.5 0.5 0.2 0.2",  # valid row
        ]
    )

    summary = DataFrameBuilder._parse_annotation_summary(annotation_path, _SPECIES_MAPPING)

    assert summary.all_species == ("Genetta-genetta",)


def test_parse_annotation_line_returns_none_for_non_numeric_coordinates() -> None:
    parsed = DataFrameBuilder._parse_annotation_line("0 abc 0.5 0.2 0.2")
    assert parsed is None


def test_parse_annotation_line_returns_none_for_non_positive_height() -> None:
    parsed = DataFrameBuilder._parse_annotation_line("0 0.5 0.5 0.2 0.0")
    assert parsed is None


def test_parse_annotation_line_returns_none_when_bbox_exceeds_vertical_bounds() -> None:
    parsed = DataFrameBuilder._parse_annotation_line("0 0.5 0.95 0.2 0.2")
    assert parsed is None


# ---------------------------------------------------------------------------
# build() integration
# ---------------------------------------------------------------------------


def test_build_returns_structured_empty_dataframe_when_no_images_found(
    mock_transform_paths,
) -> None:
    paths = mock_transform_paths
    paths.labelized_images.exists.return_value = False
    paths.unlabeled.exists.return_value = False
    builder = DataFrameBuilder(paths=paths, extract_timestamps=False)

    # Empty input should still return a typed DataFrame with the output schema.
    out = builder.build(show_progress=False)

    assert out.empty
    assert out.columns.tolist() == list(DataFrameBuilder.COLUMNS)
    assert str(out["ocr_timestamp"].dtype).startswith("datetime64")
    assert str(out["label_bbox_count"].dtype) == "Int64"


def test_build_reads_annotation_and_species_annotation_wins(
    mock_transform_paths, etl_filenames
) -> None:
    # The filename encodes "Ardea-cinerea" (heron) but the YOLO annotation
    # says class id 1 = "Canis-lupus-familiaris" (dog). The annotation species
    # (ground truth) must take precedence over the filename species.
    image_path = MagicMock()
    image_path.name = etl_filenames.reconyx_ardea_sample
    image_path.stem = Path(etl_filenames.reconyx_ardea_sample).stem
    image_path.suffix = ".jpg"
    image_path.is_file.return_value = True
    image_path.__str__.return_value = f"data/labelized/images/{image_path.name}"

    annotation_path = MagicMock()
    annotation_path.exists.return_value = True
    annotation_path.read_text.return_value = "1 0.5 0.5 0.2 0.2\n"

    paths = mock_transform_paths
    paths.labelized_images.exists.return_value = True
    paths.labelized_images.iterdir.return_value = [image_path]
    paths.unlabeled.exists.return_value = False
    paths.labelized_annotations.__truediv__.return_value = annotation_path

    builder = DataFrameBuilder(paths=paths, extract_timestamps=False)
    out = builder.build(show_progress=False)

    assert len(out) == 1
    annotation_path.read_text.assert_called_once_with(encoding="utf-8")
    # Annotation species wins over filename species.
    assert out.iloc[0]["species"] == "Canis-lupus-familiaris"


def test_build_raises_when_ocr_batch_length_mismatches_image_count(
    mock_transform_paths, etl_filenames
) -> None:
    image_path = MagicMock()
    image_path.name = etl_filenames.reconyx_ardea_sample
    image_path.stem = Path(etl_filenames.reconyx_ardea_sample).stem
    image_path.suffix = ".jpg"
    image_path.is_file.return_value = True
    image_path.__str__.return_value = f"data/unlabeled/{image_path.name}"

    paths = mock_transform_paths
    paths.labelized_images.exists.return_value = False
    paths.unlabeled.exists.return_value = True
    paths.unlabeled.iterdir.return_value = [image_path]

    builder = DataFrameBuilder(paths=paths, extract_timestamps=False)
    builder._extract_timestamps = True
    builder._timestamp_extractor = MagicMock()
    # Simulate OCR backend bug: returns fewer results than requested images.
    builder._timestamp_extractor.extract_batch.return_value = []

    with pytest.raises(RuntimeError, match="returned 0 results for 1 images"):
        builder.build(show_progress=False)


def test_init_with_extract_timestamps_creates_timestamp_extractor(mock_transform_paths) -> None:
    with patch("pipeline.etl.transform.dataframe_builder.TimestampExtractor") as mock_extractor_cls:
        builder = DataFrameBuilder(paths=mock_transform_paths, extract_timestamps=True, gpu=False)

    mock_extractor_cls.assert_called_once_with(gpu=False)
    assert builder._timestamp_extractor is mock_extractor_cls.return_value


def test_build_logs_debug_and_keeps_missing_location_when_filename_parse_fails(
    mock_transform_paths,
    etl_filenames,
) -> None:
    image_path = MagicMock()
    image_path.name = etl_filenames.invalid
    image_path.stem = Path(etl_filenames.invalid).stem
    image_path.suffix = ".jpg"
    image_path.is_file.return_value = True
    image_path.__str__.return_value = f"data/unlabeled/{image_path.name}"

    paths = mock_transform_paths
    paths.labelized_images.exists.return_value = False
    paths.unlabeled.exists.return_value = True
    paths.unlabeled.iterdir.return_value = [image_path]

    builder = DataFrameBuilder(paths=paths, extract_timestamps=False)
    builder._logger = MagicMock()

    out = builder.build(show_progress=False)

    assert len(out) == 1
    assert pd.isna(out.iloc[0]["location_id"])
    debug_messages = [str(call.args[0]) for call in builder._logger.debug.call_args_list]
    assert any("Parse failed for %s: %s" in msg for msg in debug_messages)


def test_build_keeps_filename_multi_species_for_unlabeled_images(
    mock_transform_paths,
    etl_filenames,
) -> None:
    image_path = MagicMock()
    image_path.name = etl_filenames.multi_species_boly
    image_path.stem = Path(etl_filenames.multi_species_boly).stem
    image_path.suffix = ".jpg"
    image_path.is_file.return_value = True
    image_path.__str__.return_value = f"data/unlabeled/{image_path.name}"

    paths = mock_transform_paths
    paths.labelized_images.exists.return_value = False
    paths.unlabeled.exists.return_value = True
    paths.unlabeled.iterdir.return_value = [image_path]

    builder = DataFrameBuilder(paths=paths, extract_timestamps=False)

    out = builder.build(show_progress=False)

    assert len(out) == 1
    assert out.iloc[0]["species"] == "Homo-sapiens-Canis-lupus-familiaris"
    assert out.iloc[0]["location_id"] == "N0431652-111_W0000251-205"


def test_build_with_ocr_attaches_timestamps_and_logs_summary(
    mock_transform_paths,
    etl_filenames,
) -> None:
    img1 = MagicMock()
    img1.name = etl_filenames.reconyx_ardea_sample
    img1.stem = Path(etl_filenames.reconyx_ardea_sample).stem
    img1.suffix = ".jpg"
    img1.is_file.return_value = True
    img1.__str__.return_value = f"data/unlabeled/{img1.name}"

    img2 = MagicMock()
    img2.name = "FR_N0431704-006_W0000242-723b_20200603_Martes-martes_RCNX0035_SmallestMaxSize.jpg"
    img2.stem = Path(img2.name).stem
    img2.suffix = ".jpg"
    img2.is_file.return_value = True
    img2.__str__.return_value = f"data/unlabeled/{img2.name}"

    paths = mock_transform_paths
    paths.labelized_images.exists.return_value = False
    paths.unlabeled.exists.return_value = True
    paths.unlabeled.iterdir.return_value = [img1, img2]

    builder = DataFrameBuilder(paths=paths, extract_timestamps=False)
    builder._extract_timestamps = True
    builder._timestamp_extractor = MagicMock()
    builder._timestamp_extractor.extract_batch.return_value = [
        SimpleNamespace(timestamp="2024-01-01 10:00:00", success=True),
        SimpleNamespace(timestamp=None, success=False),
    ]
    builder._logger = MagicMock()

    out = builder.build(show_progress=False)

    assert len(out) == 2
    non_null_timestamps = [value for value in out["ocr_timestamp"].tolist() if pd.notna(value)]
    assert len(non_null_timestamps) == 1
    info_messages = [str(call.args[0]) for call in builder._logger.info.call_args_list]
    assert any("OCR timestamps: %d/%d extracted (%.1f%% success)" in msg for msg in info_messages)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def test_to_csv_writes_file_and_creates_parent_dirs(
    sandbox: Path, builder_no_timestamps: DataFrameBuilder
) -> None:
    builder = builder_no_timestamps
    df = pd.DataFrame({"image_id": ["abc"], "species": ["fox"]})
    output = sandbox / "subdir" / "output.csv"

    builder.to_csv(df, output)

    assert output.exists()
    loaded = pd.read_csv(output)
    assert len(loaded) == 1
    assert loaded["image_id"].iloc[0] == "abc"


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def test_coerce_types_converts_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "label_bbox_count": [1.0, 2.0],
            "label_bbox_area_sum": ["0.1", "0.2"],
            "labeled": [None, True],
        }
    )

    out = DataFrameBuilder._coerce_types(df)

    assert str(out["label_bbox_count"].dtype) == "Int64"
    assert out["label_bbox_area_sum"].dtype == "float64"
    assert out["labeled"].dtype == "bool"


def test_build_dataframe_uses_builder_and_writes_csv_when_save_path_is_provided() -> None:
    fake_paths = MagicMock()
    fake_df = pd.DataFrame({"image_id": ["img_1"]})
    save_path = Path("metadata.csv")

    with patch("pipeline.etl.transform.dataframe_builder.DataFrameBuilder") as mock_builder_cls:
        mock_builder = mock_builder_cls.return_value
        mock_builder.build.return_value = fake_df

        result = build_dataframe(
            paths=fake_paths,
            extract_timestamps=False,
            gpu=False,
            show_progress=False,
            save_path=save_path,
        )

    mock_builder_cls.assert_called_once_with(
        paths=fake_paths,
        extract_timestamps=False,
        gpu=False,
    )
    mock_builder.build.assert_called_once_with(show_progress=False)
    mock_builder.to_csv.assert_called_once_with(fake_df, save_path)
    assert result is fake_df
