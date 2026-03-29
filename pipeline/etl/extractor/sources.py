"""Source scanning utilities for ETL extraction."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from pipeline.etl.config import IMAGE_EXTENSIONS
from pipeline.etl.extractor.data_models import FileData, FilePair, LabelizedScanResult

# ------------------------------------------------------------------
# Source Scanner
# ------------------------------------------------------------------

class SourceScanner:
    """Scan filesystem and ZIP sources into ETL extraction models."""

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize scanner with a logger for warnings and progress.

        Args:
            logger: Module logger used for scan warnings and progress.
        """
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_unlabeled_sources(self, root: Path) -> list[FileData]:
        """
        Scan unlabeled sources and collect image references.

        Args:
            root: Root directory containing unlabeled assets.

        Returns:
            List of lazy image references.
        """
        images: list[FileData] = []

        for item in sorted(root.iterdir(), key=lambda p: p.name):
            # Support mixed roots where direct files, folders, and ZIPs coexist.
            if item.suffix.lower() == ".zip" and not item.stem.startswith("."):
                images.extend(self._scan_unlabeled_zip(item))
            elif item.is_dir():
                images.extend(self._scan_unlabeled_directory(item))
            elif item.suffix.lower() in IMAGE_EXTENSIONS and not item.stem.startswith("."):
                images.append(FileData.from_path(item))

        return images

    def scan_labelized_sources(self, root: Path) -> LabelizedScanResult:
        """
        Scan labeled sources, group files by stem, and collect duplicates.

        Args:
            root: Root directory containing labeled assets.

        Returns:
            Structured scan result with canonical pairs and duplicate extras.
            A duplicate means "same stem, different source path/ZIP entry".
        """
        pairs: dict[str, FilePair] = {}
        duplicate_images: list[FileData] = []
        duplicate_annotations: list[FileData] = []

        for item in sorted(root.iterdir(), key=lambda p: p.name):
            if item.suffix.lower() == ".zip" and not item.stem.startswith("."):
                self._scan_zip(item, pairs, duplicate_images, duplicate_annotations)
            elif item.is_dir():
                self._scan_directory(item, pairs, duplicate_images, duplicate_annotations)
            elif item.suffix.lower() in IMAGE_EXTENSIONS or item.suffix.lower() == ".txt":
                self._register_file(
                    pairs,
                    FileData.from_path(item),
                    duplicate_images,
                    duplicate_annotations,
                )

        # Each FilePair in the map must have at least one file:
        # _register_file always sets image or annotation before returning.
        empty_stems = [
            stem for stem, pair in pairs.items() if pair.image is None and pair.annotation is None
        ]
        if empty_stems:
            raise RuntimeError(
                f"SourceScanner produced {len(empty_stems)} empty FilePair(s) "
                f"(stems: {empty_stems[:5]}{'...' if len(empty_stems) > 5 else ''}). "
                "This is a bug in _register_file."
            )

        return LabelizedScanResult(
            pairs=pairs,
            duplicate_images=duplicate_images,
            duplicate_annotations=duplicate_annotations,
        )

    def find_classes_content(self, root: Path) -> tuple[bytes, str] | None:
        """
        Find and read the first ``classes.txt`` content under a root tree.

        Searches both regular files and ZIP archives in deterministic order.

        Args:
            root: Root directory to scan recursively.

        Returns:
            Tuple ``(content, source_hint)`` or ``None`` when not found.
        """
        for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if not path.is_file():
                continue

            if path.name == "classes.txt":
                return path.read_bytes(), str(path)

            if path.suffix.lower() == ".zip":
                zip_content = self.find_classes_in_zip(path)
                if zip_content:
                    return zip_content, str(path)

        return None

    def find_classes_in_zip(self, zip_path: Path) -> bytes | None:
        """
        Find and read ``classes.txt`` content from a ZIP file.

        Args:
            zip_path: ZIP archive path.

        Returns:
            Raw ``classes.txt`` bytes, or ``None`` if not found.
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in sorted(zf.namelist()):
                    if Path(name).name == "classes.txt":
                        return zf.read(name)
        except zipfile.BadZipFile:
            self._logger.error("Invalid ZIP: %s", zip_path)
        return None

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _scan_unlabeled_directory(self, root: Path) -> list[FileData]:
        """
        Recursively scan one unlabeled directory.

        Args:
            root: Directory to scan.

        Returns:
            List of lazy image references.
        """
        images: list[FileData] = []

        for file_path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if not file_path.is_file():
                continue
            if file_path.stem.startswith("."):
                continue
            if file_path.suffix.lower() == ".zip":
                images.extend(self._scan_unlabeled_zip(file_path))
                continue
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(FileData.from_path(file_path))

        return images

    def _scan_unlabeled_zip(self, zip_path: Path) -> list[FileData]:
        """
        Scan one unlabeled ZIP archive.

        Args:
            zip_path: ZIP archive path.

        Returns:
            List of image references.
        """
        self._logger.info("Scanning unlabelized ZIP: %s", zip_path.name)
        images: list[FileData] = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                entries: list[str] = []
                for name in zf.namelist():
                    p = Path(name)
                    if p.stem.startswith("."):
                        continue
                    if p.suffix.lower() in IMAGE_EXTENSIONS:
                        entries.append(name)

                # Keep only references, content is loaded later
                for name in sorted(entries):
                    images.append(FileData.from_zip(zip_path, name))

        except zipfile.BadZipFile:
            self._logger.error("Invalid ZIP: %s", zip_path)

        return images

    def _scan_directory(
        self,
        root: Path,
        pairs: dict[str, FilePair],
        duplicate_images: list[FileData] | None = None,
        duplicate_annotations: list[FileData] | None = None,
    ) -> None:
        """
        Recursively scan one labeled directory and register supported files.

        Args:
            root: Directory to scan.
            pairs: Mutable stem map populated in place.
            duplicate_images: Collector for duplicate image references, or ``None`` to discard.
            duplicate_annotations: Collector for duplicate annotation references, or ``None`` to discard.
        """
        for file_path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            stem = file_path.stem

            if stem.startswith(".") or stem == "classes":
                # Hidden artifacts and classes metadata are not trainable samples.
                continue

            if suffix == ".zip":
                self._scan_zip(file_path, pairs, duplicate_images, duplicate_annotations)
                continue

            if suffix not in IMAGE_EXTENSIONS and suffix != ".txt":
                continue

            self._register_file(
                pairs,
                FileData.from_path(file_path),
                duplicate_images,
                duplicate_annotations,
            )

    def _scan_zip(
        self,
        zip_path: Path,
        pairs: dict[str, FilePair],
        duplicate_images: list[FileData] | None = None,
        duplicate_annotations: list[FileData] | None = None,
    ) -> None:
        """
        Scan one labeled ZIP archive and register files.

        Args:
            zip_path: ZIP archive path.
            pairs: Mutable stem map populated in place.
            duplicate_images: Collector for duplicate image references.
            duplicate_annotations: Collector for duplicate annotation references.
        """
        self._logger.info("Loading ZIP: %s", zip_path.name)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                entries: list[str] = []
                for name in zf.namelist():
                    p = Path(name)
                    if p.stem.startswith(".") or p.stem == "classes":
                        continue
                    suffix = p.suffix.lower()
                    if suffix in IMAGE_EXTENSIONS or suffix == ".txt":
                        entries.append(name)

                # Deterministic ordering ensures stable duplicate selection.
                for name in sorted(entries):
                    self._register_file(
                        pairs,
                        FileData.from_zip(zip_path, name),
                        duplicate_images,
                        duplicate_annotations,
                    )

        except zipfile.BadZipFile:
            self._logger.error("Invalid ZIP: %s", zip_path)

    def _register_file(
        self,
        pairs: dict[str, FilePair],
        file_data: FileData,
        duplicate_images: list[FileData] | None = None,
        duplicate_annotations: list[FileData] | None = None,
    ) -> None:
        """
        Register one image or annotation into the stem map.

        Args:
            pairs: Mutable stem map populated in place.
            file_data: Candidate file reference.
            duplicate_images: Collector for duplicate image references, or ``None`` to discard.
            duplicate_annotations: Collector for duplicate annotation references, or ``None`` to discard.
        """
        suffix = Path(file_data.name).suffix.lower()
        stem = file_data.stem
        if stem.startswith(".") or stem == "classes":
            return
        if suffix not in IMAGE_EXTENSIONS and suffix != ".txt":
            return

        if stem not in pairs:
            pairs[stem] = FilePair(stem=stem)

        pair = pairs[stem]
        if suffix in IMAGE_EXTENSIONS:
            if pair.image is None:
                pair.image = file_data
            else:
                # First occurrence wins; later duplicates are tracked for quarantine/reporting.
                self._logger.warning(
                    "Duplicate image stem '%s' detected. Keeping first source '%s', skipping '%s'.",
                    stem,
                    pair.image.source_hint,
                    file_data.source_hint,
                )
                if duplicate_images is not None:
                    duplicate_images.append(file_data)
        else:
            if pair.annotation is None:
                pair.annotation = file_data
            else:
                # Same policy for annotations: stable canonical pair + explicit duplicate capture.
                self._logger.warning(
                    "Duplicate annotation stem '%s' detected. Keeping first source '%s', skipping '%s'.",
                    stem,
                    pair.annotation.source_hint,
                    file_data.source_hint,
                )
                if duplicate_annotations is not None:
                    duplicate_annotations.append(file_data)
