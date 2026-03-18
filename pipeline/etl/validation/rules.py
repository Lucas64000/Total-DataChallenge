"""Reusable validators for ETL checks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pipeline.etl.transform.filename_parser import FilenameParser
from utils.logging_system import LogCategory, get_phototrap_logger


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    error: str | None = None


class YOLOValidator:
    """Validate YOLO annotation syntax and normalized geometry bounds."""

    def __init__(self, classes: list[str]) -> None:
        """
        Initialize the YOLO annotation validator.

        Args:
            classes: Known class names from ``classes.txt``.

        Raises:
            ValueError: If ``classes`` is empty after normalization.
        """
        normalized = [name.strip() for name in classes if name.strip()]
        if not normalized:
            raise ValueError("classes must be provided and non-empty")

        self.classes = normalized
        self.num_classes = len(normalized)
        self._logger = get_phototrap_logger().get_logger(
            LogCategory.PREPROCESSING, "yolo_validator"
        )

    @classmethod
    def from_classes_file(cls, path: Path) -> YOLOValidator:
        """
        Build validator from a ``classes.txt`` file.

        Args:
            path: Path to ``classes.txt``.

        Returns:
            Configured YOLO validator.
        """
        classes = path.read_text(encoding="utf-8").strip().splitlines()
        return cls(classes)

    @classmethod
    def from_classes_content(cls, content: str) -> YOLOValidator:
        """
        Build validator from raw class content.

        Args:
            content: Newline-separated class names.

        Returns:
            Configured YOLO validator.
        """
        classes = content.strip().splitlines()
        return cls(classes)

    def validate(self, content: str, filename: str = "") -> ValidationResult:
        """
        Validate YOLO annotation content line by line.

        Args:
            content: Full annotation file content.
            filename: Optional filename used for debug logs.

        Returns:
            Validation result with error details when invalid.
        """
        lines = content.strip().splitlines()
        if not lines:
            return ValidationResult(is_valid=True)

        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                error = f"Line {i}: expected 5 values, got {len(parts)}"
                self._logger.debug("[%s] %s", filename, error)
                return ValidationResult(is_valid=False, error=error)

            try:
                class_id = int(parts[0])
                if class_id < 0:
                    error = f"Line {i}: class_id must be >= 0"
                    self._logger.debug("[%s] %s", filename, error)
                    return ValidationResult(is_valid=False, error=error)
                if class_id >= self.num_classes:
                    error = (
                        f"Line {i}: class_id {class_id} >= num_classes ({self.num_classes})"
                    )
                    self._logger.debug("[%s] %s", filename, error)
                    return ValidationResult(is_valid=False, error=error)

                x_center, y_center, width, height = [float(p) for p in parts[1:]]

                for name, val in (("x_center", x_center), ("y_center", y_center)):
                    if not 0.0 <= val <= 1.0:
                        error = f"Line {i}: {name} ({val}) not in interval [0, 1]"
                        self._logger.debug("[%s] %s", filename, error)
                        return ValidationResult(is_valid=False, error=error)

                for name, val in (("width", width), ("height", height)):
                    if not 0.0 < val <= 1.0:
                        error = f"Line {i}: {name} ({val}) not in interval ]0, 1]"
                        self._logger.debug("[%s] %s", filename, error)
                        return ValidationResult(is_valid=False, error=error)

                if x_center - width / 2 < 0 or x_center + width / 2 > 1:
                    error = f"Line {i}: bbox exceeds horizontal bounds"
                    self._logger.debug("[%s] %s", filename, error)
                    return ValidationResult(is_valid=False, error=error)
                if y_center - height / 2 < 0 or y_center + height / 2 > 1:
                    error = f"Line {i}: bbox exceeds vertical bounds"
                    self._logger.debug("[%s] %s", filename, error)
                    return ValidationResult(is_valid=False, error=error)

            except ValueError as exc:
                error = f"Line {i}: {exc}"
                self._logger.debug("[%s] %s", filename, error)
                return ValidationResult(is_valid=False, error=error)

        return ValidationResult(is_valid=True)


class ImageValidator:
    """Validate image bytes for integrity, dimensions, and supported formats."""

    SUPPORTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG"})

    def __init__(self) -> None:
        self._logger = get_phototrap_logger().get_logger(
            LogCategory.PREPROCESSING, "image_validator"
        )

    def validate(self, image_bytes: bytes, filename: str = "") -> ValidationResult:
        """
        Validate image integrity and basic properties.

        Args:
            image_bytes: Raw image bytes.
            filename: Optional filename used for debug logs.

        Returns:
            Validation result with error details when invalid.
        """
        try:
            # verify() checks structural integrity but invalidates the image object,
            # so a second open is needed to read dimensions/format.
            with Image.open(BytesIO(image_bytes)) as img:
                img.verify()

            with Image.open(BytesIO(image_bytes)) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    error = f"Invalid dimensions: {width}x{height}"
                    self._logger.debug("[%s] %s", filename, error)
                    return ValidationResult(is_valid=False, error=error)

                if img.format not in self.SUPPORTED_FORMATS:
                    error = f"Unsupported format: {img.format}"
                    self._logger.debug("[%s] %s", filename, error)
                    return ValidationResult(is_valid=False, error=error)

                return ValidationResult(is_valid=True)

        except UnidentifiedImageError:
            error = "Cannot identify image file"
            self._logger.debug("[%s] %s", filename, error)
            return ValidationResult(is_valid=False, error=error)
        except Exception as exc:  # noqa: BLE001 - capture PIL/runtime issues as validation failure
            error = f"Corrupted image: {exc}"
            self._logger.debug("[%s] %s", filename, error)
            return ValidationResult(is_valid=False, error=error)


class FilenameValidator:
    """Validate camera-trap filename parseability using FilenameParser."""

    def __init__(self) -> None:
        self._logger = get_phototrap_logger().get_logger(
            LogCategory.PREPROCESSING, "filename_validator"
        )

    def validate(self, filename: str, labeled: bool = False) -> ValidationResult:
        """
        Validate filename parseability against ETL naming rules.

        Args:
            filename: Filename to validate.
            labeled: Whether the file belongs to the labeled split.

        Returns:
            Validation result with parser error when invalid.
        """
        metadata = FilenameParser.parse(filename, labeled)
        if not metadata.parse_success:
            self._logger.debug("[%s] %s", filename, metadata.parse_error)
            return ValidationResult(is_valid=False, error=metadata.parse_error)
        return ValidationResult(is_valid=True)
