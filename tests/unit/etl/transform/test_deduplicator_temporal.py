"""
The TemporalDeduplicator keeps one representative image per temporal burst
inside each (location_id, species) group.  This avoids over-representing a
single animal event — multiple photos taken seconds apart — in the training set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.etl.transform.deduplicator import TemporalDeduplicator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    location: str | None,
    species: str | None,
    timestamp: str | None,
) -> dict[str, object]:
    """Build a single DataFrame row with the three required columns."""
    return {"location_id": location, "species": species, "ocr_timestamp": timestamp}


def _df(*rows: dict[str, object]) -> pd.DataFrame:
    """Construct a DataFrame from raw row dicts."""
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_rejects_negative_window(self) -> None:
        # A negative window has no physical meaning; fail fast to surface bugs early.
        with pytest.raises(ValueError, match="window_seconds must be >= 0"):
            TemporalDeduplicator(window_seconds=-1)

    def test_accepts_zero_window(self) -> None:
        # window=0 means: keep the first image of each timestamp, drop identical-timestamp dupes.
        dedup = TemporalDeduplicator(window_seconds=0)
        assert dedup.window_seconds == 0


# ---------------------------------------------------------------------------
# Basic deduplication
# ---------------------------------------------------------------------------


class TestBasicDeduplication:
    def test_returns_empty_for_empty_input(self) -> None:
        dedup = TemporalDeduplicator()
        df = pd.DataFrame(columns=["location_id", "species", "ocr_timestamp"])

        out = dedup.deduplicate(df)

        assert out.empty

    def test_keeps_single_row(self) -> None:
        dedup = TemporalDeduplicator()
        df = _df(_row("N001_W001", "fox", "2024-01-01 10:00:00"))

        out = dedup.deduplicate(df)

        # A single image is always the burst representative, never dropped.
        assert len(out) == 1

    def test_drops_image_within_burst_window(self) -> None:
        # Two fox images 1 second apart with a 3-second window: the second is
        # within the burst of the first and should be dropped.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),  # 1s gap → within burst
        )

        out = dedup.deduplicate(df)

        assert len(out) == 1
        # The deduplicator does not coerce the original column — the value stays a string.
        assert out.iloc[0]["ocr_timestamp"] == "2024-01-01 10:00:00"

    def test_keeps_image_after_burst_ends(self) -> None:
        # A 5-second gap strictly exceeds the 3-second window, so the second
        # image starts a new burst and must be kept.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:05"),  # 5s gap → new burst
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_burst_gap_exactly_equal_to_window_is_dropped(self) -> None:
        # The condition is strictly greater than (>), so a gap == window is
        # still inside the burst and must be dropped.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:03"),  # exactly 3s → NOT > 3 → dropped
        )

        out = dedup.deduplicate(df)

        assert len(out) == 1

    def test_burst_gap_one_second_above_window_starts_new_burst(self) -> None:
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:04"),  # 4s > 3 → new burst
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_representative_anchor_is_last_kept_not_last_seen(self) -> None:
        # The burst anchor stays at the last KEPT (representative) image, not
        # the most recent image.  This matters when images are dropped: a later
        # image is only a "new burst" if its gap from the representative exceeds
        # the window, regardless of intervening dropped images.
        #
        # Sequence (window=3): t=0 (KEPT, anchor=0), t=2 (dropped, 2<=3),
        # t=4 (still measures from anchor=0: 4>3 → KEPT, anchor=4), t=6 (2s from 4 → dropped)
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:02"),
            _row("N001_W001", "fox", "2024-01-01 10:00:04"),
            _row("N001_W001", "fox", "2024-01-01 10:00:06"),
        )

        out = dedup.deduplicate(df)

        # t=0 kept; t=2 dropped (2s from anchor 0); t=4 kept (4s > 3); t=6 dropped (2s from anchor 4)
        assert len(out) == 2
        kept_timestamps = out["ocr_timestamp"].tolist()
        assert "2024-01-01 10:00:00" in kept_timestamps
        assert "2024-01-01 10:00:04" in kept_timestamps

    def test_earliest_timestamp_becomes_representative(self) -> None:
        # Within a group, images are sorted by timestamp before burst logic runs.
        # The earliest image becomes the representative, even if it appears later
        # in the original DataFrame row order.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),  # row 0 — later timestamp
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),  # row 1 — earlier timestamp
        )

        out = dedup.deduplicate(df)

        # Only the earliest image is the burst representative and should be kept.
        assert len(out) == 1
        assert out.iloc[0]["ocr_timestamp"] == "2024-01-01 10:00:00"


# ---------------------------------------------------------------------------
# Group independence
# ---------------------------------------------------------------------------


class TestGroupIndependence:
    def test_different_species_at_same_location_are_independent(self) -> None:
        # The deduplicator groups by (location_id, species), so a fox burst and
        # a badger at the same trap are deduped separately.  The badger should
        # never be dropped just because a fox was photographed at the same time.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),  # 1s → fox burst, dropped
            _row("N001_W001", "badger", "2024-01-01 10:00:01"),  # same time, but different group
        )

        out = dedup.deduplicate(df)

        # Fox burst keeps 1 image; badger is in its own group and is always kept.
        assert len(out) == 2
        species_kept = set(out["species"].tolist())
        assert "fox" in species_kept
        assert "badger" in species_kept

    def test_different_locations_are_independent(self) -> None:
        # Images from different camera traps should never cancel each other out,
        # even if the timestamps are close.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N002_W002", "fox", "2024-01-01 10:00:01"),  # different trap, 1s gap
        )

        out = dedup.deduplicate(df)

        # Each location is an independent group; both images must be kept.
        assert len(out) == 2

    def test_multi_burst_across_groups(self) -> None:
        # Full scenario: two locations, two species, multiple bursts each.
        # Verifies that burst logic operates entirely within each group.
        dedup = TemporalDeduplicator(window_seconds=2)
        df = _df(
            # loc_A / fox: t=0 kept, t=1 dropped (1s), t=5 kept (5s > 2)
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),
            _row("N001_W001", "fox", "2024-01-01 10:00:05"),
            # loc_A / deer: independent group, all within 1s but different species
            _row("N001_W001", "deer", "2024-01-01 10:00:00"),
            _row("N001_W001", "deer", "2024-01-01 10:00:01"),
            # loc_B / fox: completely separate group
            _row("N002_W002", "fox", "2024-01-01 10:00:00"),
        )

        out = dedup.deduplicate(df)

        # loc_A/fox: 2 kept; loc_A/deer: 1 kept; loc_B/fox: 1 kept → 4 total
        assert len(out) == 4


# ---------------------------------------------------------------------------
# Ineligible rows (always kept)
# ---------------------------------------------------------------------------


class TestIneligibleRows:
    def test_null_timestamp_row_is_always_kept(self) -> None:
        # Without a timestamp we cannot assign a burst, so the safe policy is
        # to always keep the image (i.e. never discard what we cannot analyse).
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", None),
            _row("N001_W001", "fox", None),
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_null_location_id_row_is_always_kept(self) -> None:
        # Without a location we cannot group the image; keep it to avoid data loss.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row(None, "fox", "2024-01-01 10:00:00"),
            _row(None, "fox", "2024-01-01 10:00:01"),
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_null_species_row_is_always_kept(self) -> None:
        # Without a species, grouping is ambiguous; keep such rows unconditionally.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", None, "2024-01-01 10:00:00"),
            _row("N001_W001", None, "2024-01-01 10:00:01"),
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_blank_location_id_treated_as_null(self) -> None:
        # An empty or whitespace-only string is normalised to null, so the row
        # is treated as ineligible and kept (same as a genuine None).
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("", "fox", "2024-01-01 10:00:00"),
            _row("  ", "fox", "2024-01-01 10:00:01"),  # whitespace-only
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_unparseable_timestamp_treated_as_null(self) -> None:
        # A timestamp that cannot be parsed becomes NaT (pd.to_datetime coerce),
        # which makes the row ineligible and therefore always kept.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "not-a-date"),
            _row("N001_W001", "fox", "also-bad"),
        )

        out = dedup.deduplicate(df)

        assert len(out) == 2

    def test_mixed_eligible_and_ineligible_rows(self) -> None:
        # Eligible rows participate in burst logic; ineligible ones are always
        # passed through untouched, even when interspersed with eligible rows.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),  # eligible, burst representative
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),  # eligible, dropped (within burst)
            _row(None, "fox", "2024-01-01 10:00:01"),  # ineligible (null location), kept
        )

        out = dedup.deduplicate(df)

        # 1 fox representative + 1 ineligible row = 2 kept
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_raises_when_required_column_missing(self) -> None:
        # All three columns are mandatory; raise immediately if any is absent.
        dedup = TemporalDeduplicator()

        for missing_col in ("location_id", "species", "ocr_timestamp"):
            cols = {"location_id", "species", "ocr_timestamp"} - {missing_col}
            df = pd.DataFrame(columns=list(cols))
            with pytest.raises(ValueError, match=missing_col):
                dedup.deduplicate(df)

    def test_original_row_order_is_preserved(self) -> None:
        # Output rows must appear in the same relative order as in the input,
        # so that downstream tools (writers, CSV exports) behave deterministically.
        dedup = TemporalDeduplicator(window_seconds=3)
        # Three distinct groups; each contributes one kept row.
        df = _df(
            _row("loc_C", "fox", "2024-01-01 10:00:10"),  # row 0
            _row("loc_A", "deer", "2024-01-01 10:00:00"),  # row 1
            _row("loc_B", "badger", "2024-01-01 10:00:05"),  # row 2
        )

        out = dedup.deduplicate(df)

        # All three rows are representatives (different groups), order preserved.
        assert out["location_id"].tolist() == ["loc_C", "loc_A", "loc_B"]

    def test_internal_helper_columns_not_in_output(self) -> None:
        # The deduplicator uses private __dedup_* columns internally.  They must
        # not leak into the caller's DataFrame (would pollute downstream CSVs).
        dedup = TemporalDeduplicator()
        df = _df(_row("N001_W001", "fox", "2024-01-01 10:00:00"))

        out = dedup.deduplicate(df)

        leaked = [col for col in out.columns if col.startswith("__dedup")]
        assert leaked == [], f"Internal columns leaked into output: {leaked}"

    def test_output_has_same_columns_as_input(self) -> None:
        # The deduplicator must not add or remove columns, only filter rows.
        dedup = TemporalDeduplicator()
        df = _df(_row("N001_W001", "fox", "2024-01-01 10:00:00"))
        df["extra_col"] = "extra"

        out = dedup.deduplicate(df)

        assert set(out.columns) == set(df.columns)


# ---------------------------------------------------------------------------
# verify() — post-dedup invariant check
# ---------------------------------------------------------------------------


class TestVerify:
    def test_passes_on_correctly_deduplicated_dataframe(self) -> None:
        # After proper deduplication, verify() should not raise.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),  # within burst
            _row("N001_W001", "fox", "2024-01-01 10:00:05"),  # new burst
        )
        out = dedup.deduplicate(df)

        # Should not raise — only representatives remain.
        dedup.verify(out)

    def test_raises_on_dataframe_with_burst_duplicates(self) -> None:
        # A DataFrame that still has two images within the burst window
        # should be flagged as a violation.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", "2024-01-01 10:00:00"),
            _row("N001_W001", "fox", "2024-01-01 10:00:01"),  # 1s gap <= 3s
        )

        with pytest.raises(ValueError, match="Dedup invariant violated"):
            dedup.verify(df)

    def test_passes_on_empty_dataframe(self) -> None:
        dedup = TemporalDeduplicator()
        df = pd.DataFrame(columns=["location_id", "species", "ocr_timestamp"])

        # Empty DataFrames trivially satisfy the invariant.
        dedup.verify(df)

    def test_skips_verification_when_required_columns_missing(self) -> None:
        # If the DataFrame lacks grouping columns, verify() returns silently
        # rather than crashing — the caller might have projected columns away.
        dedup = TemporalDeduplicator()
        df = pd.DataFrame({"some_col": [1, 2]})

        # Should not raise.
        dedup.verify(df)

    def test_ineligible_rows_do_not_cause_false_violations(self) -> None:
        # Rows with missing timestamps are ineligible for burst logic and
        # should not be compared against each other.
        dedup = TemporalDeduplicator(window_seconds=3)
        df = _df(
            _row("N001_W001", "fox", None),
            _row("N001_W001", "fox", None),
        )

        # Should not raise — null-timestamp rows are not checked.
        dedup.verify(df)
