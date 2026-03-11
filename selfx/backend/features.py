"""
analysis.py

Core abstractions for running feature computations and retrieving stored
analysis results.

This module provides:

- ``Feature``:
  Base Celery task for a single analysis feature. A feature computes a result
  for a given time interval, optionally generates an LLM summary, and stores the
  result via ``selfx.backend.results``.

- ``AnalysisManager``:
  Helper for splitting time ranges into analysis intervals, discovering missing
  intervals, and loading stored results.

- ``get_analysis_intervals(...)``:
  Utility for converting a requested time range into a mapping of stable,
  storage-safe interval identifiers.

Design notes
------------
- Results are persisted through ``store_result(...)``, ``get_result(...)``, and
  ``get_results(...)`` from ``selfx.backend.results``.
- The special interval ``None`` represents online/live analysis and is stored
  under the ``"Online"`` prefix.
- ``Feature.run(...)`` is the Celery entrypoint and should not usually be
  overridden. Subclasses should implement ``perform(...)`` and optionally
  ``llm_prompt(...)``.

Expected subclass contract
--------------------------
A typical feature subclass implements:

- ``perform(start, end) -> Any``:
    Compute the feature result for the given interval.
- ``llm_prompt(result_dict) -> str | None``:
    Return a prompt for LLM summarization, or a falsey value to skip it.
- ``layout(...)``, ``register_callbacks(...)``, etc. for UI integration.

Notes
-----
- This refactor removes several broken references from the original code such as
  ``self.s3`` and unfinished in-memory frame management.
- ``AnalysisManager`` now acts purely as a storage/time-range helper.
"""

from __future__ import annotations

import datetime as dt
import logging
from traceback import print_exc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import requests
from celery import Task

from selfx.backend import datetime_utils
from selfx.backend.results import get_result, get_results, store_result
from selfx.backend.utils import make_valid_filename


DEFAULT_LLM_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_LLM_MODEL = "llama3"


def _coerce_datetime(value: Any) -> Optional[pd.Timestamp]:
    """
    Convert a supported input value to ``pandas.Timestamp``.

    Parameters
    ----------
    value : Any
        Supported values include:
        - ``None``
        - ``str``
        - ``datetime.datetime``
        - ``datetime.date``
        - ``pandas.Timestamp``

    Returns
    -------
    pandas.Timestamp | None
        Normalized timestamp, or ``None`` if the input is ``None``.

    Raises
    ------
    TypeError
        If the value type is unsupported.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, dt.datetime):
        return pd.Timestamp(value)
    if isinstance(value, dt.date):
        return pd.Timestamp(dt.datetime.combine(value, dt.datetime.min.time()))
    if isinstance(value, str):
        return pd.Timestamp(datetime_utils.str_to_datetime(value))
    raise TypeError(f"Unsupported datetime value: {type(value)!r}")


def _interval_key(start: Any) -> str:
    """
    Convert an interval start value into a storage-safe identifier.

    Parameters
    ----------
    start : Any
        Interval start timestamp or ``None``.

    Returns
    -------
    str
        ``"Online"`` when ``start`` is ``None``, otherwise a sanitized timestamp
        string.
    """
    if start is None:
        return "Online"
    ts = _coerce_datetime(start)
    return make_valid_filename(datetime_utils.dt_to_str_till_sec(ts))


class Feature(Task):
    """
    Base class for analysis features executed as Celery tasks.

    Subclasses are expected to implement ``perform(...)`` and may optionally
    implement UI-specific hooks such as ``layout(...)`` and
    ``register_callbacks(...)``.

    Attributes
    ----------
    required_features : list[str]
        Names of features that should be available before this feature runs.
    plant_name : str | None
        Optional plant/system identifier for logging or UI display.
    data_access : Any
        Optional backend/data access object injected externally.
    config : dict
        Feature-specific configuration mapping.
    tr : Any
        Optional translation/localization object.
    color_mapping : Any
        Optional UI color mapping metadata.
    periodic : bool
        Whether this feature is intended for periodic execution.
    fetching : bool
        Whether this feature represents a data-fetching task.
    """

    required_features: List[str] = []

    @classmethod
    def feature_name(cls) -> str:
        """
        Return the canonical feature name.

        Returns
        -------
        str
            The class name.
        """
        return cls.__name__

    def __init__(self, tr: Any = None, periodic: bool = False, fetching: bool = False) -> None:
        """
        Initialize the feature task.

        Parameters
        ----------
        tr : Any, optional
            Translation/localization helper.
        periodic : bool, default=False
            Whether the feature is intended to run periodically.
        fetching : bool, default=False
            Whether the task is a data-fetching task.
        """
        self.plant_name: Optional[str] = None
        self.data_access: Any = None
        self.config: Dict[str, Any] = {}
        self.tr = tr
        self.color_mapping: Any = None
        self.periodic = periodic
        self.fetching = fetching

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> Any:
        """
        Return the UI layout representation for this feature.

        Parameters
        ----------
        role : Any
            User/application role.
        analysis : Any
            Analysis payload to visualize.
        start : Any
            Interval start.
        end : Any
            Interval end.

        Returns
        -------
        Any
            UI-specific layout object.

        Notes
        -----
        Subclasses should override this method.
        """
        raise NotImplementedError

    def get_result(self, sel_date: Any, feature: Optional[str] = None) -> Any:
        """
        Load a previously stored result for this feature.

        Parameters
        ----------
        sel_date : Any
            Selected date/interval start. Supported values include ``None``,
            string, datetime-like values, and ``"Online"``.
        feature : str | None, optional
            Feature name to load. Defaults to this feature's own name.

        Returns
        -------
        Any
            Stored result object, ``{}`` if missing, or ``None`` if loading
            fails.
        """
        feature_name = feature or self.feature_name()

        if sel_date is None or sel_date == "Online":
            interval_key = "Online"
        else:
            ts = _coerce_datetime(sel_date)
            logging.info("Selected date is not online: %s", ts)
            interval_key = make_valid_filename(datetime_utils.dt_to_str_till_sec(ts))

        identifier = f"{interval_key}/{feature_name}.joblib"

        try:
            return get_result(identifier)
        except Exception:
            logging.warning("Could not load stored result for %s", identifier, exc_info=True)
            return {}

    def __repr__(self) -> str:
        """
        Return the feature's display representation.

        Returns
        -------
        str
            The class name.
        """
        return type(self).__name__

    def perform(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Any:
        """
        Execute the actual feature computation.

        Parameters
        ----------
        start : pandas.Timestamp | None
            Interval start.
        end : pandas.Timestamp | None
            Interval end.

        Returns
        -------
        Any
            Arbitrary result payload. Non-dict results are wrapped into
            ``{"result": ...}`` by ``run(...)``.

        Notes
        -----
        Subclasses should override this method.
        """
        raise NotImplementedError

    def llm_prompt(self, result: Mapping[str, Any]) -> Optional[str]:
        """
        Build the LLM prompt for a computed result.

        Parameters
        ----------
        result : Mapping[str, Any]
            Computed result payload.

        Returns
        -------
        str | None
            Prompt text for the LLM, or ``None`` / empty string to disable LLM
            summarization.
        """
        return None

    def icon(self) -> str:
        """
        Return the Material icon name for this feature.

        Returns
        -------
        str
            Material icon identifier.
        """
        return "extension"

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """
        Register UI callbacks for this feature.

        Parameters
        ----------
        dash_app : Any
            Dash application instance.
        analysis : Any
            Analysis manager or analysis context.

        Notes
        -----
        Subclasses may override this method.
        """
        return None

    def time_range_selection(self, role: Any) -> bool:
        """
        Indicate whether this feature supports manual time-range selection.

        Parameters
        ----------
        role : Any
            User/application role.

        Returns
        -------
        bool
            ``True`` if time-range selection is supported.
        """
        return True

    def is_online(self, role: Any) -> bool:
        """
        Indicate whether this feature is an online/live feature.

        Parameters
        ----------
        role : Any
            User/application role.

        Returns
        -------
        bool
            ``True`` for live/online-only features.
        """
        return False

    def _generate_llm_summary(self, result: Mapping[str, Any]) -> Optional[str]:
        """
        Generate an LLM summary for a computed result.

        Parameters
        ----------
        result : Mapping[str, Any]
            Computed result payload.

        Returns
        -------
        str | None
            Generated LLM response text, or ``None`` if generation was skipped
            or failed.
        """
        prompt = self.llm_prompt(result)
        if not prompt:
            return None

        payload = {
            "model": DEFAULT_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(DEFAULT_LLM_URL, json=payload, timeout=30)
            response.raise_for_status()
            body = response.json()
            return body.get("response")
        except Exception:
            logging.warning(
                "LLM request failed for feature %s",
                self.feature_name(),
                exc_info=True,
            )
            return None

    def run(self, start_iso: Optional[str], finish_iso: Optional[str] = None) -> bool:
        """
        Celery task entrypoint.

        This method converts the incoming ISO-like timestamps, executes the
        feature computation, optionally enriches the result with an LLM summary,
        and stores the final payload.

        Parameters
        ----------
        start_iso : str | None
            Interval start as a string, or ``None`` for online mode.
        finish_iso : str | None, optional
            Interval end as a string.

        Returns
        -------
        bool
            ``True`` if execution reached completion.

        Raises
        ------
        Exception
            Re-raises any exception from ``perform(...)`` after logging.
        """
        logging.info("Running feature %s for %s - %s", self.feature_name(), start_iso, finish_iso)

        start = pd.Timestamp(start_iso) if start_iso is not None else None
        finish = pd.Timestamp(finish_iso) if finish_iso is not None else None

        try:
            result = self.perform(start, finish)
            if not isinstance(result, dict):
                result = {"result": result}

            result["llm"] = self._generate_llm_summary(result)

            store_result(start, self.feature_name(), result)
            return True

        except Exception:
            logging.exception(
                "Task %s of %s crashed",
                self.feature_name(),
                self.plant_name,
            )
            raise


class AnalysisManager:
    """
    Helper for interval discovery and loading persisted analysis results.

    Parameters
    ----------
    freq : str
        Pandas-compatible frequency string used to split ranges into smaller
        intervals, e.g. ``"1h"`` or ``"15min"``.

    Notes
    -----
    This class does not maintain an in-memory frame cache in this refactored
    version. It uses persisted results as the source of truth.
    """

    def __init__(self, freq: str) -> None:
        """
        Initialize the analysis manager.

        Parameters
        ----------
        freq : str
            Pandas frequency string.
        """
        self.freq = freq

    def __str__(self) -> str:
        """
        Return a string representation of the manager.

        Returns
        -------
        str
            Human-readable manager description.
        """
        return f"AnalysisManager(freq={self.freq!r})"

    def get_analysis(
        self,
        start: Any,
        finish: Any,
        feature: Optional[str] = None,
    ) -> List[Any]:
        """
        Load analysis results for all intervals between ``start`` and ``finish``.

        Parameters
        ----------
        start : Any
            Requested analysis start.
        finish : Any
            Requested analysis finish.
        feature : str | None, optional
            If provided, load only that feature for each interval.
            Otherwise load all stored feature results for each interval.

        Returns
        -------
        list[Any]
            List of loaded results in interval order.
        """
        intervals = get_analysis_intervals(start, finish)
        return [self.get_frame(interval_key, feature=feature) for interval_key in intervals.keys()]

    def get_frame(self, interval_key: str, feature: Optional[str] = None) -> Any:
        """
        Load stored results for a single interval.

        Parameters
        ----------
        interval_key : str
            Storage-safe interval identifier such as ``"Online"`` or a sanitized
            timestamp string.
        feature : str | None, optional
            If provided, load only this feature from the interval.

        Returns
        -------
        Any
            Either:
            - ``dict[str, Any]`` when ``feature`` is ``None``
            - a single stored result object when ``feature`` is given
            - ``None`` if loading fails unexpectedly
        """
        try:
            if feature is None:
                return get_results(interval_key)
            return get_result(f"{interval_key}/{feature}.joblib")
        except Exception:
            print_exc()
            return None

    def get_today_non_analyzed_frames(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Return missing analysis intervals from local midnight until now.

        Returns
        -------
        list[tuple[pandas.Timestamp, pandas.Timestamp]]
            Missing sub-intervals between today's midnight and current Berlin
            time.
        """
        now = dt.datetime.now(tz=datetime_utils.tz_berlin)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_non_analyzed_intervals(midnight, now)

    def get_non_analyzed_intervals(
        self,
        start: Any,
        finish: Any,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Determine which sub-intervals do not yet have stored results.

        Parameters
        ----------
        start : Any
            Start of the requested range.
        finish : Any
            End of the requested range.

        Returns
        -------
        list[tuple[pandas.Timestamp, pandas.Timestamp]]
            Intervals of length ``self.freq`` whose corresponding storage folder
            currently does not exist or has no stored results.

        Notes
        -----
        This refactored implementation infers "analyzed" from persisted storage.
        An interval is considered analyzed if ``get_results(interval_key)``
        returns a non-empty dictionary.
        """
        start_ts = _coerce_datetime(start)
        finish_ts = _coerce_datetime(finish)

        if start_ts is None or finish_ts is None:
            return []

        boundaries = pd.date_range(start=start_ts, end=finish_ts, freq=self.freq).to_list()

        if len(boundaries) < 2:
            return []

        missing: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for interval_start, interval_end in zip(boundaries[:-1], boundaries[1:]):
            interval_key = _interval_key(interval_start)
            stored = get_results(interval_key)
            if not stored:
                missing.append((interval_start, interval_end))

        return missing

    def get_previous_frame(self, finish: Any) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Return the interval immediately preceding ``finish``.

        Parameters
        ----------
        finish : Any
            End timestamp of the desired interval.

        Returns
        -------
        tuple[pandas.Timestamp, pandas.Timestamp] | None
            Previous interval boundaries, or ``None`` if ``finish`` is invalid.
        """
        finish_ts = _coerce_datetime(finish)
        if finish_ts is None:
            return None

        offset = pd.tseries.frequencies.to_offset(self.freq)
        start_ts = finish_ts - offset
        return start_ts, finish_ts


def get_analysis_intervals(start: Any, finish: Any) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    """
    Split a requested range into daily analysis intervals.

    Parameters
    ----------
    start : Any
        Analysis start. Supported values include string and datetime-like values.
    finish : Any
        Analysis end. Supported values include string, datetime/date-like values.

    Returns
    -------
    dict[str, tuple[pandas.Timestamp | None, pandas.Timestamp | None]]
        Mapping from storage-safe interval key to ``(interval_start, interval_end)``.

        Special case:
        - If either ``start`` or ``finish`` is ``None``, returns
          ``{"Online": (None, None)}``.

    Notes
    -----
    - Intervals are generated at daily granularity.
    - Each interval maps a day-start to ``day-start + 1 day``.
    - Keys are sanitized timestamp strings suitable for filesystem storage.
    """
    if start is None or finish is None:
        return {"Online": (None, None)}

    start_ts = _coerce_datetime(start)
    finish_ts = _coerce_datetime(finish)

    all_days = pd.date_range(start=start_ts, end=finish_ts, freq="D")

    return {
        make_valid_filename(datetime_utils.dt_to_str_till_sec(day)): (
            day,
            day + dt.timedelta(days=1),
        )
        for day in all_days
    }


if __name__ == "__main__":
    manager = AnalysisManager("1h")
    missing = manager.get_non_analyzed_intervals("2023-08-14", "2023-08-15")
    for interval in missing:
        print(interval)
    print("Finished")