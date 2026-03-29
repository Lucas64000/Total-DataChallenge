"""Class catalog loading and mapping utilities for ETL steps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# ------------------------------------------------------------------
# Data Model
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClassCatalog:
    """Read-only class catalog and source/train mapping."""

    # Class names as they appear in classes.txt (source order).
    source_classes: tuple[str, ...]
    # Mapping from source class id (classes.txt index) to canonical train class id.
    source_to_train_class_id: Mapping[int, int]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_class_catalog(classes_file: Path, train_species: tuple[str, ...]) -> ClassCatalog:
    """
    Load and validate the class catalog from ``classes.txt``.

    Args:
        classes_file: Path to ``classes.txt`` produced by extraction.
        train_species: Ordered list of required training species.

    Returns:
        Immutable class catalog with source/train mappings.

    Raises:
        FileNotFoundError: If ``classes.txt`` is missing.
        RuntimeError: If the file cannot be read or decoded.
        ValueError: If the file is empty or missing required train species.
    """
    if not classes_file.exists():
        raise FileNotFoundError(
            f"Missing classes.txt at {classes_file}. Run extraction first."
        )

    try:
        raw = classes_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Failed to read classes.txt at {classes_file}: {exc}") from exc

    # Keep only non-empty trimmed lines: one class name per line.
    source_classes = tuple(line.strip() for line in raw.splitlines() if line.strip())
    if not source_classes:
        raise ValueError(
            f"classes.txt at {classes_file} is empty. Extraction output is invalid."
        )

    # Reverse lookup to convert class name -> source class id quickly.
    source_index = {name: idx for idx, name in enumerate(source_classes)}
    missing_train_species = [
        species_name for species_name in train_species if species_name not in source_index
    ]
    if missing_train_species:
        raise ValueError(
            "classes.txt at "
            f"{classes_file} is missing {len(missing_train_species)} required train species: "
            f"{', '.join(missing_train_species)}"
        )

    # Build source_id -> train_id remapping using train_species canonical order.
    source_to_train_class_id = {
        source_index[species_name]: train_idx
        for train_idx, species_name in enumerate(train_species)
    }

    return ClassCatalog(
        source_classes=source_classes,
        # MappingProxyType prevents accidental mutation after catalog construction.
        source_to_train_class_id=MappingProxyType(source_to_train_class_id),
    )
