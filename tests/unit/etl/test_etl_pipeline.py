"""Unit tests for pipeline.etl.etl_pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from pipeline.etl.config import PathConfig, PreprocessingConfig
from pipeline.etl.etl_pipeline import ETLPipeline


class TestETLPipelineOrchestration:
    """Tests for the ETL pipeline orchestration logic."""

    def test_run_executes_extract_and_build_in_order(
        self, preprocessing_config: PreprocessingConfig
    ) -> None:
        config = preprocessing_config
        extraction_stats = MagicMock()
        built_df = pd.DataFrame([{"filename": "img_1.jpg"}])

        with (
            patch("pipeline.etl.etl_pipeline.Extractor") as extractor_cls,
            patch("pipeline.etl.etl_pipeline.DataFrameBuilder") as builder_cls,
        ):
            extractor_cls.return_value.extract.return_value = extraction_stats
            builder_cls.return_value.build.return_value = built_df

            result = ETLPipeline(
                config=config,
                num_workers=4,
                skip_existing=False,
                move_invalid=False,
                extract_timestamps=False,
                gpu=False,
            ).run(save_path="custom.csv", show_progress=False)

        # Pipeline should build dependencies with the forwarded options.
        extractor_cls.assert_called_once_with(
            config=config,
            num_workers=4,
            skip_existing=False,
            move_invalid=False,
        )
        builder_cls.assert_called_once_with(
            paths=config.paths,
            extract_timestamps=False,
            gpu=False,
        )
        builder_cls.return_value.build.assert_called_once_with(show_progress=False)
        builder_cls.return_value.to_csv.assert_called_once_with(built_df, Path("custom.csv"))
        assert result.extraction is extraction_stats
        assert result.dataframe.equals(built_df)
        assert result.dataframe_path == Path("custom.csv")

    def test_run_without_save_path_skips_csv_export(
        self, preprocessing_config: PreprocessingConfig
    ) -> None:
        config = preprocessing_config

        with (
            patch("pipeline.etl.etl_pipeline.Extractor") as extractor_cls,
            patch("pipeline.etl.etl_pipeline.DataFrameBuilder") as builder_cls,
        ):
            extractor_cls.return_value.extract.return_value = MagicMock()
            builder_cls.return_value.build.return_value = pd.DataFrame()

            result = ETLPipeline(config=config, extract_timestamps=False, gpu=False).run(
                show_progress=False
            )

        builder_cls.return_value.to_csv.assert_not_called()
        assert result.dataframe_path is None

    def test_run_with_dedup_deduplicates_before_export(
        self, preprocessing_config: PreprocessingConfig
    ) -> None:
        config = preprocessing_config
        source_df = pd.DataFrame([{"filename": "img_1.jpg"}, {"filename": "img_2.jpg"}])
        dedup_df = pd.DataFrame([{"filename": "img_1.jpg"}])

        with (
            patch("pipeline.etl.etl_pipeline.Extractor") as extractor_cls,
            patch("pipeline.etl.etl_pipeline.DataFrameBuilder") as builder_cls,
            patch("pipeline.etl.etl_pipeline.TemporalDeduplicator") as dedup_cls,
        ):
            extractor_cls.return_value.extract.return_value = MagicMock()
            builder_cls.return_value.build.return_value = source_df
            dedup_cls.return_value.deduplicate.return_value = dedup_df

            result = ETLPipeline(
                config=config,
                extract_timestamps=False,
                gpu=False,
                deduplicate=True,
                dedup_window_seconds=10,
            ).run(
                save_path="metadata.csv",
                save_dedup_path="metadata_dedup.csv",
                show_progress=False,
            )

        dedup_cls.assert_called_once_with(window_seconds=10)
        dedup_cls.return_value.deduplicate.assert_called_once_with(source_df)
        assert builder_cls.return_value.to_csv.call_count == 2
        first_call, second_call = builder_cls.return_value.to_csv.call_args_list
        assert first_call.args[0].equals(dedup_df)
        assert first_call.args[1] == Path("metadata_dedup.csv")
        assert second_call.args[0].equals(source_df)
        assert second_call.args[1] == Path("metadata.csv")
        assert result.dataframe.equals(source_df)
        assert result.dataframe_dedup.equals(dedup_df)

    def test_run_without_dedup_skips_deduplication(
        self, preprocessing_config: PreprocessingConfig
    ) -> None:
        config = preprocessing_config

        with (
            patch("pipeline.etl.etl_pipeline.Extractor") as extractor_cls,
            patch("pipeline.etl.etl_pipeline.DataFrameBuilder") as builder_cls,
            patch("pipeline.etl.etl_pipeline.TemporalDeduplicator") as dedup_cls,
        ):
            extractor_cls.return_value.extract.return_value = MagicMock()
            builder_cls.return_value.build.return_value = pd.DataFrame()

            ETLPipeline(
                config=config,
                extract_timestamps=False,
                gpu=False,
                deduplicate=False,
            ).run(show_progress=False)

        dedup_cls.assert_not_called()

    def test_path_config_normalizes_string_inputs(self) -> None:
        config = PathConfig(
            source_dir="raw",
            output_dir="processed",
            backup_dir="processed/backup",
        )

        assert config.labelized_images == Path("processed") / "labelized" / "images"
        assert config.labelized_annotations == Path("processed") / "labelized" / "annotations"
        assert config.unlabeled == Path("processed") / "unlabeled"


class TestETLPipelineIntegration:
    """Integration tests that use real filesystem operations."""

    def test_run_with_empty_source_returns_empty_dataframe(
        self, preprocessing_config: PreprocessingConfig
    ) -> None:
        # Regression guard: running the full pipeline on an empty source directory
        # must produce an empty CSV without crashing.
        save_path = preprocessing_config.paths.output_dir / "metadata.csv"

        result = ETLPipeline(
            config=preprocessing_config,
            extract_timestamps=False,
            gpu=False,
        ).run(save_path=save_path, show_progress=False)

        assert result.dataframe.empty
        assert result.dataframe_path == save_path
