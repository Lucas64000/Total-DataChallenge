"""Shared fixtures for transform unit tests."""

from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.etl.class_catalog import ClassCatalog
from pipeline.etl.config import LABELED_SPECIES
from pipeline.etl.transform.dataframe_builder import DataFrameBuilder


@pytest.fixture(autouse=True)
def _mock_class_catalog() -> None:
    """Patch load_class_catalog for all transform tests.

    All tests in this module create DataFrameBuilder with mock paths that have
    no real classes.txt. This fixture provides a 1:1 LABELED_SPECIES catalog so
    tests can instantiate DataFrameBuilder without hitting the filesystem.
    """
    catalog = ClassCatalog(
        source_classes=LABELED_SPECIES,
        source_to_train_class_id=MappingProxyType(
            {i: i for i in range(len(LABELED_SPECIES))}
        ),
    )
    with patch(
        "pipeline.etl.transform.dataframe_builder.load_class_catalog", return_value=catalog
    ):
        yield


@pytest.fixture
def mock_transform_paths() -> SimpleNamespace:
    """Mocked path container for DataFrameBuilder isolated unit tests."""
    return SimpleNamespace(
        labelized_images=MagicMock(),
        unlabeled=MagicMock(),
        labelized_annotations=MagicMock(),
        classes_file=MagicMock(),
        dataframe_output="metadata.csv",
    )


@pytest.fixture
def builder_no_timestamps(mock_transform_paths: SimpleNamespace) -> DataFrameBuilder:
    """DataFrameBuilder instance with filesystem mocked and OCR disabled."""
    return DataFrameBuilder(paths=mock_transform_paths, extract_timestamps=False)
