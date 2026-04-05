"""
CLI entry point for the ETL preprocessing pipeline.

Run ``python -m pipeline.etl.cli --help`` or ``make etl-help`` for usage.
"""

from __future__ import annotations

import argparse

from pipeline.etl.config import DEFAULT_DEDUP_WINDOW_SECONDS, PathConfig, PreprocessingConfig
from pipeline.etl.etl_pipeline import ETLPipeline, PipelineResult


def _positive_int(value: str) -> int:
    """Validate that *value* is a strictly positive integer."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("workers must be > 0")
    return n


def _non_negative_int(value: str) -> int:
    """Validate that *value* is a non-negative integer."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="etl",
        description="ETL preprocessing pipeline for camera-trap images.",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--source-dir",
        default="original_data",
        metavar="DIR",
        help="Raw data root (default: original_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        metavar="DIR",
        help="Extraction output root (default: data)",
    )
    parser.add_argument(
        "--backup-dir",
        default="data/backup",
        metavar="DIR",
        help="Quarantine directory for duplicates/invalid files (default: data/backup)",
    )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Worker threads for labeled pair extraction (default: auto)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="Re-extract files that already exist in the output directory",
    )
    parser.add_argument(
        "--no-move-invalid",
        dest="move_invalid",
        action="store_false",
        default=True,
        help="Drop invalid files instead of moving them to the backup directory",
    )

    # ------------------------------------------------------------------
    # OCR timestamps
    # ------------------------------------------------------------------
    parser.add_argument(
        "--no-timestamps",
        dest="extract_timestamps",
        action="store_false",
        default=True,
        help="Skip OCR timestamp extraction",
    )
    parser.add_argument(
        "--cpu",
        dest="gpu",
        action="store_false",
        default=True,
        help="Force CPU inference for OCR (default: GPU when available)",
    )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dedup",
        dest="deduplicate",
        action="store_true",
        default=False,
        help="Enable temporal burst deduplication (requires OCR timestamps)",
    )
    parser.add_argument(
        "--dedup-window-seconds",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=(
            "Burst window in seconds "
            f"(default: {DEFAULT_DEDUP_WINDOW_SECONDS}, only used with --dedup)"
        ),
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    parser.add_argument(
        "--save-path",
        default=None,
        metavar="PATH",
        help="Full metadata CSV path (default: <output-dir>/metadata.csv)",
    )
    parser.add_argument(
        "--save-dedup-path",
        default=None,
        metavar="PATH",
        help=(
            "Deduplicated metadata CSV path "
            "(default: <output-dir>/metadata_dedup.csv, only used with --dedup)"
        ),
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        default=True,
        help="Disable progress bars",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Cross-argument validation
    # ------------------------------------------------------------------
    if args.deduplicate and not args.extract_timestamps:
        parser.error("--dedup requires OCR timestamps. Remove --no-timestamps or remove --dedup")

    if args.dedup_window_seconds is not None and not args.deduplicate:
        parser.error("--dedup-window-seconds has no effect without --dedup")

    # --cpu is only relevant when OCR is enabled
    if not args.extract_timestamps and not args.gpu:
        parser.error("--cpu has no effect when --no-timestamps is set")

    # --save-dedup-path is only relevant when dedup is enabled
    if args.save_dedup_path is not None and not args.deduplicate:
        parser.error("--save-dedup-path has no effect without --dedup")

    # ------------------------------------------------------------------
    # Configuration and pipeline instantiation
    # ------------------------------------------------------------------
    dedup_window_seconds = (
        args.dedup_window_seconds
        if args.dedup_window_seconds is not None
        else DEFAULT_DEDUP_WINDOW_SECONDS
    )

    config = PreprocessingConfig(
        paths=PathConfig(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            backup_dir=args.backup_dir,
        ),
    )

    pipeline = ETLPipeline(
        config=config,
        num_workers=args.workers,
        skip_existing=args.skip_existing,
        move_invalid=args.move_invalid,
        extract_timestamps=args.extract_timestamps,
        # Cross-argument validation above already rejects --cpu with --no-timestamps.
        gpu=args.gpu,
        deduplicate=args.deduplicate,
        dedup_window_seconds=dedup_window_seconds,
    )

    save_path = args.save_path or config.paths.dataframe_output
    save_dedup_path = None
    if args.deduplicate:
        save_dedup_path = args.save_dedup_path or config.paths.dataframe_dedup_output

    result = pipeline.run(
        save_path=save_path,
        save_dedup_path=save_dedup_path,
        show_progress=args.show_progress,
    )

    _print_summary(result)


def _print_summary(result: PipelineResult) -> None:
    """Print a one-page run summary to stdout."""
    stats = result.extraction
    metadata_df = result.dataframe
    ocr_non_null = 0
    missing_location = 0

    if "ocr_timestamp" in metadata_df.columns:
        ocr_non_null = int(metadata_df["ocr_timestamp"].notna().sum())
    if "location_id" in metadata_df.columns:
        location_series = metadata_df["location_id"].astype("string").str.strip()
        missing_location = int(location_series.isna().sum())

    def _percent(count: int, total: int) -> float:
        return (100.0 * count / total) if total > 0 else 0.0

    print("\n=== ETL pipeline complete ===")
    print(f"  Extracted pairs   : {stats.pairs_extracted}")
    print(f"  Unlabeled images  : {stats.unlabeled_images}")
    print(f"  Orphan images     : {stats.orphan_images}")
    print(f"  Orphan annotations: {stats.orphan_annotations}")
    print(f"  Invalid labeled   : {stats.invalid_labeled}")
    print(f"  Invalid unlabeled : {stats.invalid_unlabeled}")
    print(f"  Metadata rows     : {len(metadata_df)}")
    print(
        f"  OCR timestamps    : {ocr_non_null}/{len(metadata_df)} "
        f"({_percent(ocr_non_null, len(metadata_df)):.1f}%)"
    )
    print(f"  Missing location  : {missing_location}")
    if result.dataframe_path:
        print(f"  CSV saved to      : {result.dataframe_path}")
    if result.dataframe_dedup is not None:
        dropped = len(metadata_df) - len(result.dataframe_dedup)
        print(
            f"  Dedup rows kept   : {len(result.dataframe_dedup)} "
            f"(-{dropped} burst duplicates)"
        )
    if result.dataframe_dedup_path:
        print(f"  Dedup CSV saved to: {result.dataframe_dedup_path}")


if __name__ == "__main__":
    main()
