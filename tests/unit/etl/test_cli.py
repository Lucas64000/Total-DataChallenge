"""Smoke tests for pipeline.etl.cli entrypoint."""

from __future__ import annotations

import runpy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.etl.cli import _print_summary, main
from pipeline.etl.config import DEFAULT_DEDUP_WINDOW_SECONDS


class TestCLIParsing:
    """Tests for CLI argument parsing and defaults."""

    @staticmethod
    def _mock_result() -> MagicMock:
        """Minimal PipelineResult mock compatible with _print_summary."""
        return MagicMock(
            extraction=MagicMock(
                pairs_extracted=0,
                orphan_images=0,
                orphan_annotations=0,
                unlabeled_images=0,
                invalid_labeled=0,
                invalid_unlabeled=0,
            ),
            dataframe=pd.DataFrame(columns=["ocr_timestamp", "location_id"]),
            dataframe_path="data/metadata.csv",
            dataframe_dedup=None,
            dataframe_dedup_path=None,
        )

    def test_main_parses_defaults_and_runs_pipeline(self) -> None:
        # Simulate `etl` command with no extra flags.
        with (
            patch("sys.argv", ["etl"]),
            patch("pipeline.etl.cli.ETLPipeline") as pipeline_cls,
        ):
            pipeline_cls.return_value.run.return_value = self._mock_result()
            main()

        # Verify default options are forwarded to ETLPipeline.
        call_kwargs = pipeline_cls.call_args[1]
        assert call_kwargs["skip_existing"] is True
        assert call_kwargs["move_invalid"] is True
        assert call_kwargs["extract_timestamps"] is True
        assert call_kwargs["gpu"] is True
        assert call_kwargs["deduplicate"] is False
        assert call_kwargs["dedup_window_seconds"] == DEFAULT_DEDUP_WINDOW_SECONDS

    def test_main_forwards_cli_flags_to_pipeline(self) -> None:
        # Simulate a command line with explicit non-default flags.
        with (
            patch(
                "sys.argv",
                [
                    "etl",
                    "--source-dir", "raw",
                    "--cpu",
                    "--dedup",
                    "--dedup-window-seconds", "10",
                ],
            ),
            patch("pipeline.etl.cli.ETLPipeline") as pipeline_cls,
        ):
            pipeline_cls.return_value.run.return_value = self._mock_result()
            main()

        call_kwargs = pipeline_cls.call_args[1]
        assert call_kwargs["gpu"] is False
        assert call_kwargs["extract_timestamps"] is True
        assert call_kwargs["deduplicate"] is True
        assert call_kwargs["dedup_window_seconds"] == 10

    def test_main_rejects_cpu_when_timestamps_are_disabled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("sys.argv", ["etl", "--no-timestamps", "--cpu"]),
            pytest.raises(SystemExit) as exc,
        ):
            main()

        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--cpu has no effect when --no-timestamps is set" in err

    def test_main_rejects_dedup_window_without_dedup(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("sys.argv", ["etl", "--dedup-window-seconds", "5"]),
            pytest.raises(SystemExit) as exc,
        ):
            main()

        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--dedup-window-seconds has no effect without --dedup" in err

    def test_main_rejects_negative_dedup_window(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("sys.argv", ["etl", "--dedup", "--dedup-window-seconds", "-1"]),
            pytest.raises(SystemExit) as exc,
        ):
            main()

        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "value must be >= 0" in err

    def test_module_main_guard_invokes_main(self) -> None:
        # Ensures ``python -m pipeline.etl.cli`` reaches main().
        with (
            patch("sys.argv", ["etl"]),
            patch("pipeline.etl.etl_pipeline.ETLPipeline") as pipeline_cls,
        ):
            pipeline_cls.return_value.run.return_value = self._mock_result()
            runpy.run_module("pipeline.etl.cli", run_name="__main__")

        pipeline_cls.assert_called_once()

    @pytest.mark.parametrize("workers", ["0", "-1"])
    def test_main_rejects_non_positive_workers(
        self, workers: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("sys.argv", ["etl", "--workers", workers]),
            pytest.raises(SystemExit) as exc,
        ):
            main()

        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "workers must be > 0" in err


class TestCLISummary:
    def test_print_summary_reports_quality_metrics_and_dedup_delta(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = MagicMock(
            extraction=MagicMock(
                pairs_extracted=10,
                orphan_images=1,
                orphan_annotations=2,
                unlabeled_images=3,
                invalid_labeled=4,
                invalid_unlabeled=5,
            ),
            dataframe=pd.DataFrame(
                {
                    "ocr_timestamp": [pd.Timestamp("2024-01-01 10:00:00"), pd.NaT, pd.NaT],
                    "location_id": ["loc_a", pd.NA, "loc_c"],
                }
            ),
            dataframe_path="data/metadata.csv",
            dataframe_dedup=pd.DataFrame({"location_id": ["loc_a", "loc_c"]}),
            dataframe_dedup_path="data/metadata_dedup.csv",
        )

        _print_summary(result)

        out = capsys.readouterr().out
        assert "Metadata rows     : 3" in out
        assert "OCR timestamps    : 1/3 (33.3%)" in out
        assert "Missing location  : 1" in out
        assert "Dedup rows kept   : 2 (-1 burst duplicates)" in out
