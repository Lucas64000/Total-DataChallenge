"""
ETL pipeline orchestration.

Chains extraction, metadata building, optional deduplication, and CSV export
into a single ``ETLPipeline.run()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.etl.config import DEFAULT_DEDUP_WINDOW_SECONDS, PreprocessingConfig
from pipeline.etl.extractor.core import Extractor
from pipeline.etl.extractor.data_models import ExtractionStats
from pipeline.etl.transform.dataframe_builder import DataFrameBuilder
from pipeline.etl.transform.deduplicator import TemporalDeduplicator


@dataclass
class PipelineResult:
    """Aggregated output from one ETL run."""

    extraction: ExtractionStats
    dataframe: pd.DataFrame
    dataframe_path: Path | None = None
    dataframe_dedup: pd.DataFrame | None = None
    dataframe_dedup_path: Path | None = None


class ETLPipeline:
    """
    Full ETL pipeline: extract → build metadata → deduplicate → export.

    Usage::

        result = ETLPipeline(config=cfg, extract_timestamps=False).run(
            save_path="data/metadata.csv"
        )
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        *,
        num_workers: int | None = None,
        skip_existing: bool = True,
        move_invalid: bool = True,
        extract_timestamps: bool = True,
        gpu: bool = True,
        deduplicate: bool = False,
        dedup_window_seconds: int = DEFAULT_DEDUP_WINDOW_SECONDS,
    ) -> None:
        """
        Args:
            config: ETL configuration (paths, dry-run, backup). Defaults to
                ``PreprocessingConfig()`` when ``None``.
            num_workers: Threads for labeled pair extraction. ``None`` auto-detects.
            skip_existing: Skip files already present in the output directory.
            move_invalid: Quarantine invalid files to backup instead of dropping them.
            extract_timestamps: Run OCR timestamp extraction during metadata build.
            gpu: Use GPU for OCR when ``extract_timestamps`` is ``True``.
            deduplicate: Enable temporal burst deduplication after metadata build.
            dedup_window_seconds: Burst window in seconds passed to
                ``TemporalDeduplicator``. Ignored when ``deduplicate`` is ``False``.
        """
        self._config = config or PreprocessingConfig()
        self._num_workers = num_workers
        self._skip_existing = skip_existing
        self._move_invalid = move_invalid
        self._extract_timestamps = extract_timestamps
        self._gpu = gpu
        self._deduplicate = deduplicate
        self._dedup_window_seconds = dedup_window_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        save_path: Path | str | None = None,
        save_dedup_path: Path | str | None = None,
        show_progress: bool = True,
    ) -> PipelineResult:
        """
        Execute the full ETL pipeline.

        Steps:
            1. Extract files from source to output directories.
            2. Build a metadata DataFrame from extracted images.
            3. Optionally deduplicate temporal bursts.
            4. Export the final DataFrame to ``save_path`` when provided.
            5. Optionally export the deduplicated DataFrame to
               ``save_dedup_path`` when deduplication is enabled.

        Args:
            save_path: Destination CSV for the full metadata. Skipped when ``None``.
            save_dedup_path: Destination CSV for the deduplicated subset.
                Only used when ``deduplicate=True`` was passed to the constructor.
                Skipped when ``None``.
            show_progress: Show progress bars (OCR extraction).

        Returns:
            :class:`PipelineResult` with extraction stats, full metadata
            DataFrame, optional deduplicated DataFrame, and their respective
            CSV paths.
        """
        extraction_stats = Extractor(
            config=self._config,
            num_workers=self._num_workers,
            skip_existing=self._skip_existing,
            move_invalid=self._move_invalid,
        ).extract()

        builder = DataFrameBuilder(
            paths=self._config.paths,
            extract_timestamps=self._extract_timestamps,
            gpu=self._gpu,
        )
        df_dedup: pd.DataFrame | None = None
        resolved_dedup_path: Path | None = None
        df = builder.build(show_progress=show_progress)
        if self._deduplicate:
            df_dedup = TemporalDeduplicator(
                window_seconds=self._dedup_window_seconds,
            ).deduplicate(df)
            if save_dedup_path is not None:
                resolved_dedup_path = Path(save_dedup_path)
                builder.to_csv(df_dedup, resolved_dedup_path)

        resolved_path: Path | None = None
        if save_path is not None:
            resolved_path = Path(save_path)
            builder.to_csv(df, resolved_path)

        return PipelineResult(
            extraction=extraction_stats,
            dataframe=df,
            dataframe_path=resolved_path,
            dataframe_dedup=df_dedup,
            dataframe_dedup_path=resolved_dedup_path,
        )
