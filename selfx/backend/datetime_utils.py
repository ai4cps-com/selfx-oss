"""
selfx.utils.time_utils

Datetime/timestamp helpers used across SelfX.

Design goals
- Be explicit about timezone handling (naive vs aware).
- Keep parsing/formatting functions predictable.
- Prefer pandas vectorized operations for Series / DataFrames.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import pytz

# Types
Interval = Tuple[dt.datetime, dt.datetime]
TranslateFn = Callable[[str], str]


# -----------------------------
# Interval helpers
# -----------------------------
def interval_difference(new_interval: Interval, intervals: Sequence[Interval]) -> List[Interval]:
    """
    Subtract a list of intervals from a single interval.

    Given `new_interval = (start, end)` and a list `intervals` (the "blocked" parts),
    returns a list of remaining (non-overlapping) intervals.

    Notes
    - Intervals are treated as half-open-ish in practice: [start, end).
      The overlap checks follow that convention.
    - Assumes each interval is a tuple of (start, end) with start < end.
    - `intervals` need not be sorted; order only affects intermediate splitting, not correctness.
    """
    result: List[Interval] = [new_interval]

    for block in intervals:
        new_result: List[Interval] = []
        b0, b1 = block

        for cur0, cur1 in result:
            # No overlap
            if b1 <= cur0 or b0 >= cur1:
                new_result.append((cur0, cur1))
                continue

            # Block overlaps left part of current interval
            if b0 <= cur0 and b1 < cur1:
                new_result.append((b1, cur1))
                continue

            # Block overlaps right part of current interval
            if b0 > cur0 and b1 >= cur1:
                new_result.append((cur0, b0))
                continue

            # Block splits current into two pieces
            if b0 > cur0 and b1 < cur1:
                new_result.append((cur0, b0))
                new_result.append((b1, cur1))
                continue

            # Block fully covers current -> add nothing

        result = new_result

    return result


# -----------------------------
# Parsing / validation
# -----------------------------
def str_to_datetime(
    s: Optional[str],
    offset: Optional[dt.timedelta] = None,
    convert_to_utc: bool = False,
    tz: Optional[pytz.BaseTzInfo] = None,
    str_till_us: bool = False,
) -> Optional[dt.datetime]:
    """
    Parse a datetime string into a `datetime.datetime`.

    Supported inputs
    - Default: ISO 8601-like strings parseable by `datetime.fromisoformat`,
      e.g. "2024-01-01 12:34:56", "2024-01-01T12:34:56", with or without tz offset.
    - If `str_till_us=True`: parses european format
      "%d.%m.%y %H:%M:%S.%f" or "%d.%m.%y %H:%M:%S"

    Timezone behavior
    - If `tz` is provided and parsing produces a naive datetime, it is localized with `tz.localize(...)`.
      (If the parsed datetime is already aware, it is left as-is.)
    - If `convert_to_utc=True`, the result is converted to UTC (requires an aware datetime).

    Parameters
    ----------
    s:
        Input string or None.
    offset:
        Optional timedelta added after parsing (and after UTC conversion if enabled).
    convert_to_utc:
        Whether to convert the result to UTC.
    tz:
        pytz timezone used to localize naive datetimes.
    str_till_us:
        Whether to parse using the european "%d.%m.%y %H:%M:%S(.%f)" format.

    Returns
    -------
    datetime or None
    """
    if s is None:
        return None

    if str_till_us:
        try:
            d = dt.datetime.strptime(s, "%d.%m.%y %H:%M:%S.%f")
        except ValueError:
            d = dt.datetime.strptime(s, "%d.%m.%y %H:%M:%S")
    else:
        d = dt.datetime.fromisoformat(s)

    # Localize if needed (only for naive)
    if tz is not None and d.tzinfo is None:
        d = tz.localize(d)

    if convert_to_utc:
        if d.tzinfo is None:
            raise ValueError("convert_to_utc=True requires an aware datetime (provide tz or parse an aware string).")
        d = d.astimezone(pytz.utc)

    if offset:
        d = d + offset

    return d


def is_valid_date(string: str, formats: Optional[Sequence[str]] = None) -> bool:
    """
    Check whether `string` matches any date format in `formats`.
    """
    if formats is None:
        formats = (
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%Y/%m/%d",
        )

    for fmt in formats:
        try:
            dt.datetime.strptime(string, fmt)
            return True
        except ValueError:
            continue
    return False


def robust_to_datetime(series: pd.Series) -> pd.Series:
    """
    Robustly parse a pandas Series to datetime.

    Strategy:
    - First try pandas' inference (`errors="coerce"`)
    - Then try a few common ISO variants with timezone info.
    """
    date1 = pd.to_datetime(series, errors="coerce")
    date2 = pd.to_datetime(series, errors="coerce", format="%Y-%m-%dT%H:%M:%S%z")
    date3 = pd.to_datetime(series, errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f%z")
    date4 = pd.to_datetime(series, errors="coerce", format="%Y-%m-%dT%H:%M:%S.%fZ")
    return date1.fillna(date2).fillna(date3).fillna(date4)


# -----------------------------
# Human-readable "time ago"
# -----------------------------
def time_ago(
    target_time: Union[dt.datetime, pd.Timestamp, None],
    time_now: Optional[dt.datetime] = None,
    translate: Optional[TranslateFn] = None,
    till_hour: bool = False,
) -> Optional[str]:
    """
    Return a human-friendly relative time string like "3 minutes ago".

    Notes
    - Assumes `time_now` and `target_time` are comparable (both naive or both aware).
    """
    if target_time is None or pd.isnull(target_time):
        return None

    if translate is None:
        translate = lambda s: s  # type: ignore[assignment]

    if time_now is None:
        time_now = now_tz_naive()

    if isinstance(target_time, pd.Timestamp):
        target_time = target_time.to_pydatetime()

    diff = time_now - target_time

    if till_hour and diff.total_seconds() > 3600:
        return dt_to_str_till_sec_europe(
            target_time,
            omit_date=diff.total_seconds() < 86400,
            omit_seconds=True,
        )

    if diff.days > 0:
        return translate("{} day ago").format(1) if diff.days == 1 else translate("{} days ago").format(diff.days)

    seconds = int(diff.total_seconds())
    if seconds >= 3600:
        hours = seconds // 3600
        return translate("{} hour ago").format(1) if hours == 1 else translate("{} hours ago").format(hours)
    if seconds >= 60:
        minutes = seconds // 60
        return translate("{} minute ago").format(1) if minutes == 1 else translate("{} minutes ago").format(minutes)

    return translate("{} second ago").format(1) if seconds == 1 else translate("{} seconds ago").format(seconds)


# -----------------------------
# Formatting helpers
# -----------------------------
def dt_to_str_till_sec(d: dt.datetime, short_year: bool = False) -> str:
    """Format as 'YYYY-MM-DD HH:MM:SS' (or 'YY-...' if short_year)."""
    return d.strftime("%y-%m-%d %H:%M:%S") if short_year else d.strftime("%Y-%m-%d %H:%M:%S")


def dt_to_str_till_sec_europe(
    d: dt.datetime,
    omit_date: bool = False,
    short_year: bool = False,
    omit_seconds: bool = False,
) -> str:
    """
    European formatting:
    - With date: 'DD.MM.YYYY HH:MM(:SS)'
    - Without date: 'HH:MM(:SS)'
    - With short_year: DD.MM.YY ...
    """
    if omit_date:
        return d.strftime("%H:%M") if omit_seconds else d.strftime("%H:%M:%S")

    if short_year:
        return d.strftime("%d.%m.%y %H:%M") if omit_seconds else d.strftime("%d.%m.%y %H:%M:%S")

    return d.strftime("%d.%m.%Y %H:%M") if omit_seconds else d.strftime("%d.%m.%Y %H:%M:%S")


# -----------------------------
# Simple date helpers (naive local time)
# -----------------------------
def istoday(ts: Optional[dt.datetime]) -> Optional[bool]:
    """Return True if `ts` is today (local), False if not, None if ts is None."""
    if ts is None:
        return None
    return dt.datetime.today().date() == ts.date()


def today_tz_naive() -> dt.datetime:
    """Local naive 'today 00:00:00'."""
    return dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


def tomorrow_tz_naive() -> dt.datetime:
    """
    Local naive 'tomorrow 00:00:00'.

    Uses timedelta, avoids invalid dates at month boundaries.
    """
    return today_tz_naive() + dt.timedelta(days=1)


def now_tz_naive() -> dt.datetime:
    """Local naive current time."""
    return dt.datetime.now()


# -----------------------------
# Pandas helpers
# -----------------------------
def sort_timestamps(data: Dict) -> Dict:
    """
    Sort `data[k]` DataFrames by ['timestamp', 'name'] if timestamps are not monotonic increasing.

    Expects each DataFrame to have columns: 'timestamp' and 'name'.
    """
    for k in list(data.keys()):
        df = data[k]
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values(by=["timestamp", "name"]).reset_index(drop=True)
            data[k] = df

    return data


def convert_pandas_dt_to_str(series: pd.Series, nano_sec: bool = False) -> pd.Series:
    """
    Convert a datetime-like pandas Series to strings in format '%d.%m.%y %H:%M:%S.%f'.

    If `nano_sec=True`, append 3 extra digits for nanoseconds (so you effectively get 9 digits).
    """
    fmt = "%d.%m.%y %H:%M:%S.%f"

    if not pd.api.types.is_datetime64_any_dtype(series):
        return series

    base = series.dt.strftime(fmt)

    if not nano_sec:
        return base

    nanos = series.dt.nanosecond.map(
        lambda x: f"{int(x):03d}" if not (isinstance(x, float) and math.isnan(x)) else "000"
    )
    return base + nanos


def convert_pandas_str_to_dt(series: pd.Series) -> pd.Series:
    """Parse strings formatted as '%d.%m.%y %H:%M:%S.%f' into pandas datetimes."""
    return pd.to_datetime(series, format="%d.%m.%y %H:%M:%S.%f")


# -----------------------------
# Timezone conversions
# -----------------------------
def datetime_to_utc(t: dt.datetime) -> dt.datetime:
    """
    Convert an aware datetime to UTC.

    Raises
    ------
    ValueError if `t` is naive.
    """
    if t.tzinfo is None:
        raise ValueError("datetime_to_utc() requires an aware datetime.")
    return t.astimezone(pytz.utc)


def to_aware(d: dt.datetime, tz: pytz.BaseTzInfo, *, assume_local_if_naive: bool = True) -> dt.datetime:
    """
    Ensure `d` is timezone-aware.

    If `d` is naive and `assume_local_if_naive=True`, localize it with the provided `tz`.
    If `d` is already aware, it is converted to `tz`.

    Raises
    ------
    ValueError if `d` is naive and assume_local_if_naive=False.
    """
    if d.tzinfo is None:
        if not assume_local_if_naive:
            raise ValueError("to_aware(): datetime is naive and assume_local_if_naive=False.")
        return tz.localize(d)
    return d.astimezone(tz)


def to_naive_utc(d: dt.datetime, *, assume_tz_if_naive: Optional[pytz.BaseTzInfo] = None) -> dt.datetime:
    """
    Convert datetime to *naive UTC* (tzinfo removed).

    If `d` is naive, you must supply `assume_tz_if_naive` so we know how to interpret it.
    """
    if d.tzinfo is None:
        if assume_tz_if_naive is None:
            raise ValueError("to_naive_utc(): naive datetime; provide assume_tz_if_naive to interpret it.")
        d = assume_tz_if_naive.localize(d)

    return d.astimezone(pytz.utc).replace(tzinfo=None)


# -----------------------------
# InfluxDB/pipeline-friendly helpers
# -----------------------------
def ensure_utc_series(
    s: pd.Series,
    *,
    assume_tz_if_naive: Union[str, pytz.BaseTzInfo, None] = "UTC",
) -> pd.Series:
    """
    Ensure a timestamp Series is timezone-aware in UTC.

    Behavior
    - Parses to datetime (`pd.to_datetime`).
    - If naive:
        - if assume_tz_if_naive is None -> raise
        - else localize to that tz, then convert to UTC
    - If already tz-aware -> convert to UTC

    Parameters
    ----------
    s:
        Input Series (strings/datetime/ints are accepted by pandas).
    assume_tz_if_naive:
        Timezone to assume for naive timestamps. Common choices:
        - "UTC" (often safest for machine timestamps)
        - "Europe/Berlin" (if your sources are local time)
        - None (force callers to be explicit; raises on naive)

    Returns
    -------
    Series with dtype datetime64[ns, UTC]
    """
    out = pd.to_datetime(s, errors="raise")

    if out.dt.tz is None:
        if assume_tz_if_naive is None:
            raise ValueError("ensure_utc_series(): naive timestamps; set assume_tz_if_naive.")
        tzinfo = pytz.timezone(assume_tz_if_naive) if isinstance(assume_tz_if_naive, str) else assume_tz_if_naive
        out = out.dt.tz_localize(tzinfo)

    return out.dt.tz_convert(pytz.utc)


def ensure_utc_index(
    df: pd.DataFrame,
    *,
    col: Optional[str] = "timestamp",
    assume_tz_if_naive: Union[str, pytz.BaseTzInfo, None] = "UTC",
    sort: bool = True,
    set_as_index: bool = False,
    drop: bool = False,
) -> pd.DataFrame:
    """
    Ensure a DataFrame has UTC-aware timestamps, optionally set as index, optionally sorted.

    Common use cases
    - normalize data before writing to InfluxDB
    - normalize data right after reading from InfluxDB
    - enforce consistent comparisons across sources

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Timestamp column name. If None, uses the current index as the timestamp source.
    assume_tz_if_naive:
        Timezone assumed for naive timestamps (see `ensure_utc_series`).
    sort:
        If True, sort by timestamp (and preserve stable ordering for ties).
    set_as_index:
        If True, set the normalized timestamp as the DataFrame index.
    drop:
        If set_as_index=True, drop the timestamp column.

    Returns
    -------
    DataFrame with normalized UTC timestamps (column or index).
    """
    out = df.copy()

    if col is None:
        # Normalize index
        idx = pd.to_datetime(out.index, errors="raise")
        if idx.tz is None:
            if assume_tz_if_naive is None:
                raise ValueError("ensure_utc_index(): naive index; set assume_tz_if_naive.")
            tzinfo = pytz.timezone(assume_tz_if_naive) if isinstance(assume_tz_if_naive, str) else assume_tz_if_naive
            idx = idx.tz_localize(tzinfo)
        out.index = idx.tz_convert(pytz.utc)
        if sort:
            out = out.sort_index()
        return out

    # Normalize column
    if col not in out.columns:
        raise KeyError(f"ensure_utc_index(): column '{col}' not found in DataFrame.")

    out[col] = ensure_utc_series(out[col], assume_tz_if_naive=assume_tz_if_naive)

    if sort:
        # stable sort to keep deterministic ordering with ties
        out = out.sort_values(by=[col], kind="mergesort").reset_index(drop=True)

    if set_as_index:
        out = out.set_index(col, drop=drop)
        if sort:
            out = out.sort_index()

    return out