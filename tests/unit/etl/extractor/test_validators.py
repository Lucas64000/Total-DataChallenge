"""Business-focused tests for ETL validators."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from pipeline.etl.extractor.validators import ImageValidator, YOLOValidator


def _image_bytes(fmt: str) -> bytes:
    # Build a tiny valid image payload in memory.
    img = Image.new("RGB", (100, 100), color="green")
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def test_yolo_empty_annotation_is_valid_no_animal_case() -> None:
    # Empty TXT is allowed: this means "image with no detected animal".
    result = YOLOValidator(classes=["animal"]).validate("")
    assert result.is_valid


def test_yolo_class_id_must_be_within_loaded_classes() -> None:
    validator = YOLOValidator(classes=["cat", "dog"])
    # Class id "2" is out of range when only 2 classes are loaded (0 and 1).
    result = validator.validate("2 0.5 0.5 0.2 0.3")    # classID, bbox coords
    assert not result.is_valid
    assert "num_classes" in (result.error or "")


def test_yolo_bbox_must_stay_within_image_bounds() -> None:
    # Center x=0.9 with width=0.4 overflows right boundary.
    result = YOLOValidator(classes=["animal"]).validate("0 0.9 0.5 0.4 0.3")
    assert not result.is_valid
    assert "horizontal bounds" in (result.error or "")


def test_yolo_bbox_edge_values_are_allowed() -> None:
    # Full-image box (width=1.0, height=1.0) is still valid by design.
    result = YOLOValidator(classes=["animal"]).validate("0 0.5 0.5 1.0 1.0")
    assert result.is_valid


def test_yolo_requires_non_empty_classes() -> None:
    # Validation cannot run without an explicit class vocabulary.
    with pytest.raises(ValueError, match="classes must be provided"):
        YOLOValidator(classes=[])


def test_image_validator_accepts_supported_format() -> None:
    # JPEG bytes + .jpg extension should pass.
    result = ImageValidator().validate(_image_bytes("JPEG"), filename="ok.jpg")
    assert result.is_valid


def test_image_validator_rejects_unsupported_format_even_if_decodable() -> None:
    # GIF can be decodable by PIL, but format is intentionally outside accepted set.
    result = ImageValidator().validate(_image_bytes("GIF"), filename="nope.gif")
    assert not result.is_valid
    assert "Unsupported format" in (result.error or "")


def test_image_validator_rejects_corrupted_payload() -> None:
    # Random bytes should fail image decoding/verification.
    result = ImageValidator().validate(b"not_an_image", filename="bad.jpg")
    assert not result.is_valid
