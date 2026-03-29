"""Temporal deduplication utilities for ETL metadata."""

from __future__ import annotations

import pandas as pd


class TemporalDeduplicator:
    """
    Keep one representative image per temporal burst and group.

    Grouping by ``(location_id, species)``.
    Rows with missing/invalid grouping keys or timestamp are always kept.
    """

    def __init__(self, window_seconds: int = 3) -> None:
        """
        Initialize temporal deduplication.

        Args:
            window_seconds: Maximum allowed gap (seconds) between consecutive
                images in a burst.
        """
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        self.window_seconds = window_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return deduplicated rows.

        Args:
            df: Input metadata table.

        Returns:
            Deduplicated DataFrame in original row order.
        """
        self._validate_required_columns(df, ("location_id", "species", "ocr_timestamp"))
        out = df.copy()
        if out.empty:
            return out

        # Internal helper columns are prefixed to avoid collisions.
        row_order_col = "__dedup_row_order"
        parsed_ts_col = "__dedup_parsed_timestamp"
        location_key_col = "__dedup_location_key"
        species_key_col = "__dedup_species_key"

        # Keep original order to restore a stable output after group-level processing.
        out[row_order_col] = pd.Series(range(len(out)), index=out.index, dtype="int64")

        # Invalid timestamps become NaT and are excluded from burst computation.
        out[parsed_ts_col] = pd.to_datetime(out["ocr_timestamp"], format="mixed", errors="coerce")

        out[location_key_col] = self._normalize_key(out["location_id"])
        out[species_key_col] = self._normalize_key(out["species"])

        # Dedup applies only when both grouping keys and timestamp are valid.
        eligible_mask = (
            out[location_key_col].notna()
            & out[species_key_col].notna()
            & out[parsed_ts_col].notna()
        )

        # Rows with missing keys/timestamps are explicitly kept as-is.
        kept_indices = set(out.index[~eligible_mask].tolist())

        eligible_rows = out[eligible_mask]
        # Build independent streams per (location, species): bursts are compared
        # only inside the same camera-location and species combination.
        grouped = eligible_rows.groupby(
            [location_key_col, species_key_col],
            sort=False,
            dropna=False,
            observed=True,
        )

        for _, index_values in grouped.groups.items():
            # Keep one row per temporal burst for this specific group.
            kept_indices.update(
                self._select_group_representatives(
                    out, list(index_values), row_order_col, parsed_ts_col
                )
            )

        # Filter kept rows, then restore initial ordering for deterministic output.
        result = out.loc[out.index.isin(kept_indices)]
        result = result.sort_values(row_order_col, kind="mergesort")

        # Remove internal helper columns before returning the final table.
        return result.drop(
            columns=[
                row_order_col,
                parsed_ts_col,
                location_key_col,
                species_key_col,
            ]
        ).reset_index(drop=True)

    def verify(self, df: pd.DataFrame) -> None:
        """
        Check that the deduplicated DataFrame satisfies the burst invariant.

        After correct deduplication, no two consecutive images in the same
        ``(location_id, species)`` group should have a timestamp gap ``<= window_seconds``.

        Args:
            df: Deduplicated metadata DataFrame.

        Raises:
            ValueError: If any gap violation is found, with the count of violations.
        """
        if df.empty:
            return

        required = ("location_id", "species", "ocr_timestamp")
        if any(col not in df.columns for col in required):
            return

        working = df.copy()
        # Create cleaned helper columns (location, species, timestamp) for reliable checks.
        working["__v_loc"] = working["location_id"].astype("string").str.strip().replace("", pd.NA)
        working["__v_sp"] = working["species"].astype("string").str.strip().replace("", pd.NA)
        working["__v_ts"] = pd.to_datetime(working["ocr_timestamp"], errors="coerce")

        # Only rows with valid grouping keys and timestamp can be checked.
        eligible = working[
            working["__v_loc"].notna() & working["__v_sp"].notna() & working["__v_ts"].notna()
        ]
        if eligible.empty:
            return

        violations = 0
        for _, group in eligible.groupby(
            ["__v_loc", "__v_sp"], sort=False, dropna=False, observed=True
        ):
            ordered = group.sort_values("__v_ts", kind="mergesort")
            deltas = ordered["__v_ts"].diff().dropna()
            violations += int((deltas.dt.total_seconds() <= self.window_seconds).sum())

        if violations > 0:
            raise ValueError(
                f"Dedup invariant violated: {violations} consecutive gap(s) "
                f"<= {self.window_seconds}s"
            )

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------

    def _validate_required_columns(
        self, df: pd.DataFrame, required_columns: tuple[str, ...]
    ) -> None:
        """
        Raise if any required column is absent from the DataFrame.

        Args:
            df: DataFrame to check.
            required_columns: Column names that must be present.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            missing = ", ".join(missing_cols)
            raise ValueError(f"Missing required column(s): {missing}")

    @staticmethod
    def _normalize_key(series: pd.Series) -> pd.Series:
        """Normalize string keys and convert blanks to null."""
        normalized = series.astype("string").str.strip()
        return normalized.replace("", pd.NA)

    # ------------------------------------------------------------------
    # Burst Selection Logic
    # ------------------------------------------------------------------

    def _select_group_representatives(
        self,
        df: pd.DataFrame,
        indices: list[int],
        row_order_col: str,
        parsed_ts_col: str,
    ) -> set[int]:
        """
        Select one representative per temporal burst for a single group.

        A new burst starts when the gap between the current image and the last
        **kept representative** is strictly greater than ``window_seconds``.
        Images within the same burst (gap <= window from the representative) are dropped.

        Example with window=3s and sequence [t=0, t=1, t=2, t=5]:
            t=0 -> KEPT   (first image, starts burst; representative_ts = 0)
            t=1 -> DROPPED (gap from representative: 1s <= 3s)
            t=2 -> DROPPED (gap from representative: 2s <= 3s)
            t=5 -> KEPT   (gap from representative: 5s > 3s, new burst)
        """
        if not indices:
            return set()

        ordered_for_burst = sorted(
            indices,
            # Deterministic ordering: first by parsed timestamp, then by original
            # row position when timestamps are identical.
            key=lambda idx: (
                pd.Timestamp(df.at[idx, parsed_ts_col]),
                int(df.at[idx, row_order_col]),
            ),
        )

        representatives: set[int] = set()
        previous_ts: pd.Timestamp | None = None
        for idx in ordered_for_burst:
            current_ts = pd.Timestamp(df.at[idx, parsed_ts_col])
            if previous_ts is None:
                # First row in the ordered group always starts a new burst.
                representatives.add(idx)
                previous_ts = current_ts
            elif (current_ts - previous_ts).total_seconds() > self.window_seconds:
                # Gap strictly greater than window from the last representative => new burst.
                representatives.add(idx)
                previous_ts = current_ts
        return representatives
