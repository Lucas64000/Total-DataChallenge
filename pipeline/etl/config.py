"""
Define ETL path and preprocessing configuration objects.

This module contains dataclasses used by extraction and validation stages:
- source/output/backup directory layout
- preprocessing toggles (dry-run, orphan policy, backups)
- helper methods to create required directory structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset((".jpg", ".jpeg", ".png"))

LABELED_SPECIES: tuple[str, ...] = (
    "Ardea-cinerea",
    "Canis-lupus-familiaris",
    "Capreolus-capreolus",
    "Genetta-genetta",
    "Martes-martes",
    "Meles-meles",
    "Sus-scrofa",
    "Vulpes-vulpes",
)


@dataclass
class PathConfig:
    """Paths configuration."""

    source_dir: Path = field(default_factory=lambda: Path("original_data"))
    output_dir: Path = field(default_factory=lambda: Path("data"))
    backup_dir: Path = field(default_factory=lambda: Path("data/backup"))

    def __post_init__(self) -> None:
        """Normalize user-provided path-like values into ``Path`` objects."""
        self.source_dir = Path(self.source_dir)
        self.output_dir = Path(self.output_dir)
        self.backup_dir = Path(self.backup_dir)

    @property
    def labelized_images(self) -> Path:
        """
        Return the labeled images output directory.

        Returns:
            Path to ``data/labelized/images``.
        """
        return self.output_dir / "labelized" / "images"

    @property
    def labelized_annotations(self) -> Path:
        """
        Return the labeled annotations output directory.

        Returns:
            Path to ``data/labelized/annotations``.
        """
        return self.output_dir / "labelized" / "annotations"

    @property
    def unlabeled(self) -> Path:
        """
        Return the unlabeled images output directory.

        Returns:
            Path to ``data/unlabeled``.
        """
        return self.output_dir / "unlabeled"

    @property
    def dataframe_output(self) -> Path:
        """
        Return the default metadata CSV output path.

        Returns:
            Path to ``data/metadata.csv``.
        """
        return self.output_dir / "metadata.csv"

    @property
    def classes_file(self) -> Path:
        """
        Return the canonical ``classes.txt`` path.

        Returns:
            Path to ``data/labelized/classes.txt``.
        """
        return self.output_dir / "labelized" / "classes.txt"

    def ensure_output_dirs(self) -> None:
        """
        Create output directory structure.

        Returns:
            ``None``.
        """
        self.labelized_images.mkdir(parents=True, exist_ok=True)
        self.labelized_annotations.mkdir(parents=True, exist_ok=True)
        self.unlabeled.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessingConfig:
    """Main configuration for ETL pipeline."""

    paths: PathConfig = field(default_factory=PathConfig)
    dry_run: bool = False
    backup_enabled: bool = True
    fail_on_orphans: bool = True

    @property
    def backup_dir(self) -> Path:
        """
        Return the backup root directory.

        Returns:
            Backup directory path.
        """
        return self.paths.backup_dir

    @property
    def duplicates_dir(self) -> Path:
        """
        Return the duplicate quarantine directory.

        Returns:
            Path to duplicate backup folder.
        """
        return self.backup_dir / "duplicates"

    @property
    def invalid_dir(self) -> Path:
        """
        Return the invalid-file quarantine directory.

        Returns:
            Path to invalid backup folder.
        """
        return self.backup_dir / "invalid"

    @property
    def orphans_dir(self) -> Path:
        """
        Return the orphan-file quarantine directory.

        Returns:
            Path to orphan backup folder.
        """
        return self.backup_dir / "orphans"

    def ensure_dirs(self) -> None:
        """
        Create all required ETL directories.

        Returns:
            ``None``.
        """
        if not self.dry_run:
            self.paths.ensure_output_dirs()
            if self.backup_enabled:
                for subdir in ["invalid", "orphans"]:
                    (self.backup_dir / subdir / "images").mkdir(parents=True, exist_ok=True)
                    (self.backup_dir / subdir / "annotations").mkdir(parents=True, exist_ok=True)
                (self.duplicates_dir / "images").mkdir(parents=True, exist_ok=True)
                (self.duplicates_dir / "annotations").mkdir(parents=True, exist_ok=True)
                (self.duplicates_dir / "unlabeled").mkdir(parents=True, exist_ok=True)
