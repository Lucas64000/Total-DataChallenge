"""Business-focused tests for pipeline.etl.class_catalog."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pipeline.etl.class_catalog import load_class_catalog

TRAIN_SPECIES: tuple[str, ...] = ("lion", "elephant")


def _mock_file(content: str) -> MagicMock:
    """Return a Path-like mock with the given text content."""
    path = MagicMock()
    path.exists.return_value = True
    path.read_text.return_value = content
    path.__str__ = lambda self: "classes.txt"
    return path


def _classes(*names: str) -> str:
    return "\n".join(names)


# ---------------------------------------------------------------------------
# load_class_catalog – happy paths
# ---------------------------------------------------------------------------


def test_source_classes_reflect_file_order() -> None:
    catalog = load_class_catalog(
        _mock_file(_classes("zebra", "lion", "elephant", "giraffe")), TRAIN_SPECIES
    )
    assert catalog.source_classes == ("zebra", "lion", "elephant", "giraffe")


def test_train_mapping_indexes_correct_source_ids() -> None:
    # lion is at source index 1, elephant at 2.
    # train index: lion=0, elephant=1 (TRAIN_SPECIES)
    catalog = load_class_catalog(_mock_file(_classes("zebra", "lion", "elephant")), TRAIN_SPECIES)
    assert catalog.source_to_train_class_id == {1: 0, 2: 1}


def test_non_train_source_classes_are_excluded_from_mapping() -> None:
    catalog = load_class_catalog(
        _mock_file(_classes("zebra", "lion", "elephant", "giraffe")), TRAIN_SPECIES
    )
    # zebra (0) and giraffe (3) have no train counterpart.
    assert 0 not in catalog.source_to_train_class_id
    assert 3 not in catalog.source_to_train_class_id


def test_blank_lines_in_file_are_ignored() -> None:
    catalog = load_class_catalog(_mock_file("lion\n\nelephant\n  \n"), TRAIN_SPECIES)
    # Blank/whitespace lines must not appear in source_classes.
    assert catalog.source_classes == ("lion", "elephant")


def test_train_species_appearing_first_get_id_zero() -> None:
    catalog = load_class_catalog(_mock_file(_classes("lion", "elephant")), TRAIN_SPECIES)
    assert catalog.source_to_train_class_id[0] == 0  # lion → train 0
    assert catalog.source_to_train_class_id[1] == 1  # elephant → train 1


# ---------------------------------------------------------------------------
# load_class_catalog – error cases
# ---------------------------------------------------------------------------


def test_missing_file_raises_file_not_found() -> None:
    path = MagicMock()
    path.exists.return_value = False
    path.__str__ = lambda self: "classes.txt"
    with pytest.raises(FileNotFoundError, match="classes.txt"):
        load_class_catalog(path, TRAIN_SPECIES)


def test_empty_file_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_class_catalog(_mock_file(""), TRAIN_SPECIES)


def test_whitespace_only_file_is_treated_as_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_class_catalog(_mock_file("  \n\n  \n"), TRAIN_SPECIES)


def test_missing_one_train_species_raises_value_error() -> None:
    # elephant missing
    with pytest.raises(ValueError, match="elephant"):
        load_class_catalog(_mock_file(_classes("lion")), TRAIN_SPECIES)

def test_missing_multiple_train_species_names_all_of_them() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_class_catalog(_mock_file(_classes("zebra")), TRAIN_SPECIES)
    msg = str(exc_info.value)
    assert "lion" in msg
    assert "elephant" in msg


def test_unreadable_file_raises_runtime_error() -> None:
    path = MagicMock()
    path.exists.return_value = True
    path.read_text.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid byte")
    path.__str__ = lambda self: "classes.txt"
    with pytest.raises(RuntimeError, match="Failed to read classes.txt"):
        load_class_catalog(path, TRAIN_SPECIES)


# ---------------------------------------------------------------------------
# ClassCatalog – immutability contract
# ---------------------------------------------------------------------------


def test_catalog_is_frozen() -> None:
    catalog = load_class_catalog(_mock_file(_classes("lion", "elephant")), TRAIN_SPECIES)
    with pytest.raises((AttributeError, TypeError)):
        catalog.source_classes = ("mutated",)  # type: ignore[misc]


def test_mapping_is_read_only() -> None:
    catalog = load_class_catalog(_mock_file(_classes("lion", "elephant")), TRAIN_SPECIES)
    with pytest.raises(TypeError):
        catalog.source_to_train_class_id[0] = 99  # type: ignore[index]
