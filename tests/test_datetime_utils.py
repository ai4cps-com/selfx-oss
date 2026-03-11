import datetime as dt

import pandas as pd
import pytest
import pytz

from selfx.backend.datetime_utils import (
    ensure_utc_index,
    ensure_utc_series,
    interval_difference,
    to_aware,
    to_naive_utc,
)


UTC = pytz.utc
BERLIN = pytz.timezone("Europe/Berlin")


# -----------------------------
# interval_difference
# -----------------------------
def test_interval_difference_no_overlap():
    base = (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 12, 0))
    blocks = [(dt.datetime(2026, 1, 1, 12, 0), dt.datetime(2026, 1, 1, 13, 0))]
    assert interval_difference(base, blocks) == [base]


def test_interval_difference_full_cover():
    base = (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 12, 0))
    blocks = [(dt.datetime(2026, 1, 1, 9, 0), dt.datetime(2026, 1, 1, 13, 0))]
    assert interval_difference(base, blocks) == []


def test_interval_difference_left_overlap():
    base = (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 12, 0))
    blocks = [(dt.datetime(2026, 1, 1, 9, 0), dt.datetime(2026, 1, 1, 11, 0))]
    assert interval_difference(base, blocks) == [(dt.datetime(2026, 1, 1, 11, 0), dt.datetime(2026, 1, 1, 12, 0))]


def test_interval_difference_right_overlap():
    base = (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 12, 0))
    blocks = [(dt.datetime(2026, 1, 1, 11, 0), dt.datetime(2026, 1, 1, 13, 0))]
    assert interval_difference(base, blocks) == [(dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 11, 0))]


def test_interval_difference_split():
    base = (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 12, 0))
    blocks = [(dt.datetime(2026, 1, 1, 10, 30), dt.datetime(2026, 1, 1, 11, 0))]
    assert interval_difference(base, blocks) == [
        (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 10, 30)),
        (dt.datetime(2026, 1, 1, 11, 0), dt.datetime(2026, 1, 1, 12, 0)),
    ]


def test_interval_difference_multiple_blocks_order_independent():
    base = (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 12, 0))
    blocks_a = [
        (dt.datetime(2026, 1, 1, 10, 30), dt.datetime(2026, 1, 1, 10, 45)),
        (dt.datetime(2026, 1, 1, 11, 30), dt.datetime(2026, 1, 1, 11, 40)),
    ]
    blocks_b = list(reversed(blocks_a))

    expected = [
        (dt.datetime(2026, 1, 1, 10, 0), dt.datetime(2026, 1, 1, 10, 30)),
        (dt.datetime(2026, 1, 1, 10, 45), dt.datetime(2026, 1, 1, 11, 30)),
        (dt.datetime(2026, 1, 1, 11, 40), dt.datetime(2026, 1, 1, 12, 0)),
    ]

    assert interval_difference(base, blocks_a) == expected
    assert interval_difference(base, blocks_b) == expected


# -----------------------------
# tz helpers: to_aware / to_naive_utc
# -----------------------------
def test_to_aware_localizes_naive():
    d = dt.datetime(2026, 1, 1, 12, 0)
    a = to_aware(d, BERLIN)
    assert a.tzinfo is not None
    assert a.tzinfo.zone == "Europe/Berlin"


def test_to_aware_converts_aware():
    d = BERLIN.localize(dt.datetime(2026, 1, 1, 12, 0))
    a = to_aware(d, UTC)
    assert a.tzinfo is not None
    assert a.tzinfo.zone == "UTC"


def test_to_aware_raises_if_naive_and_not_allowed():
    d = dt.datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValueError):
        to_aware(d, BERLIN, assume_local_if_naive=False)


def test_to_naive_utc_from_aware():
    d = BERLIN.localize(dt.datetime(2026, 1, 1, 12, 0))
    n = to_naive_utc(d)
    assert n.tzinfo is None
    # 12:00 in Berlin in winter is 11:00 UTC (CET = UTC+1)
    assert n == dt.datetime(2026, 1, 1, 11, 0)


def test_to_naive_utc_raises_for_naive_without_assumption():
    d = dt.datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValueError):
        to_naive_utc(d)


def test_to_naive_utc_from_naive_with_assumption():
    d = dt.datetime(2026, 1, 1, 12, 0)
    n = to_naive_utc(d, assume_tz_if_naive=BERLIN)
    assert n == dt.datetime(2026, 1, 1, 11, 0)


# -----------------------------
# ensure_utc_series
# -----------------------------
def test_ensure_utc_series_from_naive_strings_assume_utc():
    s = pd.Series(["2026-01-01 00:00:00", "2026-01-01 01:00:00"])
    out = ensure_utc_series(s, assume_tz_if_naive="UTC")
    assert str(out.dtype).endswith(", UTC]")
    assert out.iloc[0].tzinfo is not None
    assert out.iloc[0].tzinfo.zone == "UTC"
    assert out.iloc[0].to_pydatetime() == UTC.localize(dt.datetime(2026, 1, 1, 0, 0))


def test_ensure_utc_series_from_naive_strings_assume_berlin():
    s = pd.Series(["2026-01-01 12:00:00"])
    out = ensure_utc_series(s, assume_tz_if_naive="Europe/Berlin")
    # 12:00 CET -> 11:00 UTC
    assert out.iloc[0].to_pydatetime() == UTC.localize(dt.datetime(2026, 1, 1, 11, 0))


def test_ensure_utc_series_raises_when_naive_and_no_assumption():
    s = pd.Series(["2026-01-01 12:00:00"])
    with pytest.raises(ValueError):
        ensure_utc_series(s, assume_tz_if_naive=None)


def test_ensure_utc_series_from_aware_strings():
    s = pd.Series(["2026-01-01T12:00:00+01:00"])
    out = ensure_utc_series(s, assume_tz_if_naive="UTC")
    assert out.iloc[0].to_pydatetime() == UTC.localize(dt.datetime(2026, 1, 1, 11, 0))


# -----------------------------
# ensure_utc_index (column mode)
# -----------------------------
def test_ensure_utc_index_column_sort_and_set_index():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 01:00:00", "2026-01-01 00:00:00"],
            "value": [2, 1],
        }
    )

    out = ensure_utc_index(
        df,
        col="timestamp",
        assume_tz_if_naive="UTC",
        sort=True,
        set_as_index=True,
        drop=True,
    )

    assert out.index.tz is not None
    assert str(out.index.tz) in ("UTC", "UTC+00:00")
    assert list(out["value"]) == [1, 2]  # sorted by timestamp


def test_ensure_utc_index_column_missing_raises():
    df = pd.DataFrame({"ts": ["2026-01-01 00:00:00"]})
    with pytest.raises(KeyError):
        ensure_utc_index(df, col="timestamp")


# -----------------------------
# ensure_utc_index (index mode)
# -----------------------------
def test_ensure_utc_index_from_index():
    df = pd.DataFrame({"value": [1, 2]})
    df.index = pd.Index(["2026-01-01 00:00:00", "2026-01-01 01:00:00"])

    out = ensure_utc_index(df, col=None, assume_tz_if_naive="UTC", sort=True)

    assert out.index.tz is not None
    assert str(out.index.tz) in ("UTC", "UTC+00:00")
    assert out.iloc[0]["value"] == 1