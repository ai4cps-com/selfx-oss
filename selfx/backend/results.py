"""
results.py

Utilities for storing and retrieving computation results on the local
filesystem under the fallback directory ``Analysis``.

This module currently uses only local storage. Results are serialized with
``joblib`` and stored in a directory structure grouped by interval.

Storage layout
--------------
Results are written to::

    Analysis/{interval_prefix}/{feature}.joblib

Where:
- ``interval_prefix`` is either ``"Online"`` or a sanitized timestamp-like string
- ``feature`` is the logical result name

Main functions
--------------
- ``store_result(...)`` stores one serialized result
- ``get_result(...)`` retrieves one stored result by relative path
- ``is_stored(...)`` checks whether a result exists
- ``get_results(...)`` loads all results for a given interval
- ``delete_files(...)`` deletes all stored files for one or more intervals

Notes
-----
- Missing files in ``get_result(...)`` return ``{}`` for compatibility with the
  previous behavior.
- Failed deserialization returns ``None`` after printing the traceback.
- This module assumes that ``feature`` is already safe to use as a filename.
"""

from __future__ import annotations

from pathlib import Path
from traceback import print_exc
from typing import Any, Dict, Iterable, Optional

import joblib

from selfx.backend.utils import make_valid_filename
from selfx.backend.datetime_utils import dt_to_str_till_sec


DEFAULT_RESULTS_DIR = Path("Analysis")


def _parse_int(value: Any, default: int) -> int:
    """
    Parse an integer value, returning a default on failure.

    Parameters
    ----------
    value : Any
        Value to parse, commonly a string.
    default : int
        Fallback value if parsing fails.

    Returns
    -------
    int
        Parsed integer if possible, otherwise ``default``.

    Examples
    --------
    >>> _parse_int("5", 2)
    5
    >>> _parse_int("abc", 2)
    2
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _interval_to_prefix(interval: Optional[Any]) -> str:
    """
    Convert an interval object into a safe directory prefix.

    Parameters
    ----------
    interval : Any | None
        Interval identifier. If ``None``, the special prefix ``"Online"`` is used.
        Otherwise the value is converted to a timestamp string using
        ``dt_to_str_till_sec(...)`` and sanitized for filesystem use.

    Returns
    -------
    str
        Safe interval prefix suitable for use as a directory name.

    Examples
    --------
    - ``None`` -> ``"Online"``
    - datetime-like value -> sanitized timestamp string
    """
    if interval is None:
        return "Online"
    return make_valid_filename(dt_to_str_till_sec(interval))


def store_result(interval: Optional[Any], feature: str, result: Any) -> None:
    """
    Store a serialized result for a given interval and feature.

    The result is written as a ``.joblib`` file under the analysis directory.

    Parameters
    ----------
    interval : Any | None
        Interval identifier. If ``None``, results are stored under ``"Online"``.
    feature : str
        Logical name of the result. Used as the filename stem.
    result : Any
        Python object serializable by ``joblib``.

    Returns
    -------
    None

    Storage path
    ------------
    ``Analysis/{interval_prefix}/{feature}.joblib``
    """
    prefix = _interval_to_prefix(interval)
    out_dir = DEFAULT_RESULTS_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, out_dir / f"{feature}.joblib", compress=False)


def get_result(identifier: str) -> Any:
    """
    Load a stored result by relative path.

    Parameters
    ----------
    identifier : str
        Relative path under ``DEFAULT_RESULTS_DIR``, for example:
        ``"Online/temperature.joblib"``.

    Returns
    -------
    Any
        Deserialized Python object.

        Special cases:
        - returns ``{}`` if the file does not exist
        - returns ``None`` if deserialization fails

    Notes
    -----
    Returning ``{}`` for missing files preserves compatibility with earlier code.
    """
    file_path = DEFAULT_RESULTS_DIR / identifier

    if not file_path.exists():
        return {}

    try:
        return joblib.load(filename=file_path, mmap_mode=None)
    except Exception:
        print_exc()
        return None


def is_stored(interval: str, feature: str) -> bool:
    """
    Check whether a stored result exists.

    Parameters
    ----------
    interval : str
        Interval prefix, such as ``"Online"`` or a sanitized timestamp string.
    feature : str
        Feature name without the ``.joblib`` extension.

    Returns
    -------
    bool
        ``True`` if the corresponding file exists, otherwise ``False``.
    """
    return (DEFAULT_RESULTS_DIR / interval / f"{feature}.joblib").exists()


def get_results(interval: str) -> Dict[str, Any]:
    """
    Load all stored results for a given interval.

    Parameters
    ----------
    interval : str
        Interval prefix to load from, for example ``"Online"``.

    Returns
    -------
    dict[str, Any]
        Mapping from feature name to deserialized result object.

    Notes
    -----
    - Only files matching ``*.joblib`` are loaded.
    - Files that fail to load are skipped after printing a traceback.
    - If the interval directory does not exist, an empty dictionary is returned.
    """
    dir_path = DEFAULT_RESULTS_DIR / interval
    results: Dict[str, Any] = {}

    if not dir_path.exists():
        return results

    for path in dir_path.glob("*.joblib"):
        try:
            results[path.stem] = joblib.load(path)
        except Exception:
            print_exc()

    return results


def delete_files(intervals: Iterable[str]) -> None:
    """
    Delete all stored files for one or more intervals.

    Parameters
    ----------
    intervals : Iterable[str]
        Iterable of interval prefixes whose stored results should be removed.

    Returns
    -------
    None

    Notes
    -----
    This function deletes files recursively inside each interval directory,
    but does not currently remove now-empty directories.
    """
    for interval in intervals:
        dir_path = DEFAULT_RESULTS_DIR / interval
        if not dir_path.exists():
            continue

        for path in dir_path.rglob("*"):
            if path.is_file():
                path.unlink()