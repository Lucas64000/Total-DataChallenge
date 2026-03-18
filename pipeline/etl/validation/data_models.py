"""Validation data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ValidationStats:
    """Aggregated counters produced by the ETL validation pass."""

    valid_pairs: int = 0
    empty_annotations: int = 0
    invalid_images: int = 0
    invalid_annotations: int = 0
    valid_unlabeled: int = 0
    invalid_unlabeled: int = 0
    parse_warnings: int = 0
