"""Data models used by ETL extraction pipeline."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------
# Extraction Statistics
# ------------------------------------------------------------------

@dataclass(slots=True)
class ExtractionStats:
    """Statistics from extraction."""

    pairs_extracted: int = 0
    images_extracted: int = 0
    annotations_extracted: int = 0
    orphan_images: int = 0
    orphan_annotations: int = 0
    unlabeled_images: int = 0
    duplicate_labelized_images: int = 0
    duplicate_labelized_annotations: int = 0
    duplicate_unlabeled_images: int = 0
    skipped_existing: int = 0
    extraction_errors: int = 0
    invalid_labeled: int = 0
    invalid_unlabeled: int = 0


# ------------------------------------------------------------------
# Source File References
# ------------------------------------------------------------------

@dataclass
class FileData:
    """Reference to one source file stored on disk or inside a ZIP archive."""

    # Pair key used by extractor ("img123" for img123.jpg + img123.txt)
    stem: str
    # Full filename with extension
    name: str
    # Backing path when source is a regular filesystem file (relative path)
    _source_path: Path | None = field(repr=False, default=None)
    # Backing ZIP source as (zip file path, internal entry path)
    _zip_source: tuple[Path, str] | None = field(repr=False, default=None)

    def read_content(self) -> bytes:
        """
        Load file content from disk or ZIP source.

        Returns:
            Raw file content.

        Raises:
            ValueError: If no backing source is configured.
            OSError: If disk reading fails.
            zipfile.BadZipFile: If the ZIP source is invalid.
            KeyError: If the ZIP entry does not exist.
        """
        if self._source_path is not None:
            return self._source_path.read_bytes()
        if self._zip_source is not None:
            # Open ZIP lazily at read time to keep scan phase lightweight.
            zip_path, entry_name = self._zip_source
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.read(entry_name)
        raise ValueError("No content source")

    @property
    def source_hint(self) -> str:
        """
        Return a human-readable reference to the underlying source.

        Returns:
            Filesystem path or ``zip_path!entry`` when sourced from ZIP.

        Example:
            ``original_data/labelized/batch_2024_01.zip!cam1/FR_..._0001.jpg``
        """
        if self._source_path is not None:
            return str(self._source_path)
        if self._zip_source is not None:
            zip_path, entry_name = self._zip_source
            return f"{zip_path}!{entry_name}"
        return self.name

    @classmethod
    def from_path(cls, path: Path) -> FileData:
        """
        Create a lazy reference for a filesystem file.

        Args:
            path: Source file path.

        Returns:
            File reference pointing to ``path``.
        """
        return cls(stem=path.stem, name=path.name, _source_path=path)

    @classmethod
    def from_zip(cls, zip_path: Path, entry_name: str) -> FileData:
        """
        Create a lazy reference for a ZIP archive entry.

        Args:
            zip_path: Path to the ZIP archive.
            entry_name: Internal ZIP entry path.

        Returns:
            File reference pointing to the ZIP entry.
        """
        entry = Path(entry_name)
        return cls(stem=entry.stem, name=entry.name, _zip_source=(zip_path, entry_name))


# ------------------------------------------------------------------
# Pairing Models
# ------------------------------------------------------------------

@dataclass
class FilePair:
    """A pair of image and annotation files."""

    stem: str
    image: FileData | None = None
    annotation: FileData | None = None


@dataclass(slots=True)
class LabelizedScanResult:
    """
    Result payload for labelized source scanning.

    Keeps the canonical pair mapping and the list of duplicate files that were
    skipped while preserving the first occurrence for each stem/content type.
    """

    pairs: dict[str, FilePair] = field(default_factory=dict)
    duplicate_images: list[FileData] = field(default_factory=list)
    duplicate_annotations: list[FileData] = field(default_factory=list)
