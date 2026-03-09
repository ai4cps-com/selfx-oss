# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nemanja Hranisavljevic
# Contact: nemanja@ai4cps.com


from __future__ import annotations

import datetime
import logging
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path
from traceback import print_exc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from celery import chain
from celery.schedules import schedule
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import redirect, request

from selfx.backend.features import AnalysisManager, get_analysis_intervals
from selfx.backend.perform import get_sorted_features, perform_requested_features
from selfx.backend.results import delete_files
from selfx.dash import colors
from selfx.dash.routing_utils import construct_id, parse_url, construct_url, get_today

from selfx.dash.layouts import get_sidebar, get_topbar


logger = logging.getLogger(__name__)

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/icon?family=Material+Icons",
    "https://fonts.googleapis.com/css2?family=Poppins&display=swap",
    dbc.icons.FONT_AWESOME,
    "selfx_style.css",  # assumed to be served from assets/
]


class SelfXDash:
    """
    Main application wrapper for the SelfX Dash dashboard.

    Responsibilities
    ---------------
    - Create and configure the Dash app instance.
    - Register plants/systems and their feature classes.
    - Instantiate feature objects and register their callbacks.
    - Provide routing (URL -> page render) and date-range/reevaluate behavior.
    """

    ROUTE_PREFIX = "/dashboard/"
    TITLE = "SelfX"

    def __init__(
        self,
        roles: Optional[Sequence[Any]] = None,
        users: Optional[Sequence[Any]] = None,
        celery_app: Any = None,
        config: Any = None,
        work_day_shift: int = 0,
        initial_date: Any = None,
        logo: Optional[Union[str, Sequence[str]]] = None,
        analysis_period: int = 60,
        content_not_ready_refresh_interval: int = 3,
        LOCK_KEY: str = "selfx_celery_task_lock",
        LOCK_TTL: int = 60,
    ) -> None:
        # ---- Inputs / config ----
        self.roles = tuple(roles) if roles is not None else ()
        self.users = tuple(users) if users is not None else ()
        self.celery_app = celery_app
        self.config = config
        self.logo = logo
        self.initial_date = initial_date

        self.work_day_shift = work_day_shift
        self.analysis_period = analysis_period
        self.content_not_ready_refresh_interval = content_not_ready_refresh_interval

        self.grafana_url = grafana_url
        self.grafana_token = grafana_token

        # ---- Internal registries / state ----
        self.unified: Dict[str, Any] = {}
        self.features: Dict[str, Any] = {}
        self._feature_obj: Dict[str, Dict[str, Any]] = {}

        self.plant_names: List[str] = []
        self.data_access: Dict[str, Any] = {}

        self.settings: Dict[str, Any] = {}
        self.preferences: Dict[str, Any] = {}

        self.LOCK_KEY = LOCK_KEY
        self.TTL = TTL

        # ---- Dash app ----
        self.app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets,
            external_scripts=[],
            suppress_callback_exceptions=True,
            routes_pathname_prefix=self.ROUTE_PREFIX,
            requests_pathname_prefix=self.ROUTE_PREFIX,
            title=self.TITLE,
        )

        # ---- Component registries / stores ----
        self.file_load: Dict[str, Any] = {}
        self.dropdown_items: Dict[str, Any] = {}
        self.date_picker: Dict[str, Any] = {}
        self.refresh: Dict[str, Any] = {}
        self.session_store: Dict[str, Any] = {}
        self.analysis: Dict[str, AnalysisManager] = {}

    # -------------------------
    # Public API
    # -------------------------
    def add_system(
        self,
        name: str,
        features: Union[Iterable[Any], Mapping[str, Iterable[Any]]] = (),
        unified: Union[bool, str] = True,
        settings: Union[bool, str] = False,
        preferences: Union[bool, str] = False,
        file_dropdown: Any = None,
        refresh: bool = False,
        freq: str = "1h",
        file_load: bool = False,
    ) -> None:
        """
        Register a system/plant in the dashboard and wire up feature callbacks.

        `features` can be:
        - iterable of Feature classes (treated as role "Default")
        - mapping role -> iterable of Feature classes

        Each feature class is expected to provide:
        - feature_name() -> str
        - register_callbacks(app, analysis_manager)
        - config : dict (parameter -> dict containing at least "value")
        """

        unified_label = "Unified" if unified is True else unified
        settings_label = "Settings" if settings is True else settings
        preferences_label = "Preferences" if preferences is True else preferences

        self.unified[name] = unified_label
        self.settings[name] = settings_label
        self.preferences[name] = preferences_label
        self.plant_names.append(name)

        # Normalize features to {role: iterable[FeatureClass]}
        if hasattr(features, "items"):
            features_by_role: Mapping[str, Iterable[Any]] = features  # type: ignore[assignment]
        else:
            features_by_role = {"Default": features}

        # role -> feature_name -> FeatureClass
        features_nested: "OrderedDict[str, OrderedDict[str, Any]]" = OrderedDict()
        for role, feature_list in features_by_role.items():
            role_dict: "OrderedDict[str, Any]" = OrderedDict()
            for feature_cls in feature_list:
                role_dict[feature_cls.feature_name()] = feature_cls
            features_nested[role] = role_dict
        self.features[name] = features_nested

        # registries
        self.file_load[name] = file_load
        self.dropdown_items[name] = file_dropdown
        self.refresh[name] = refresh
        self.analysis[name] = AnalysisManager(freq=freq)
        self._feature_obj[name] = {}

        # instantiate and register callbacks
        for _, feature_list in features_by_role.items():
            for feature_cls in feature_list:
                tool_name = feature_cls.feature_name()
                if tool_name in self._feature_obj[name]:
                    continue
                try:
                    obj = feature_cls()
                    obj.plant_name = name
                    obj.register_callbacks(self.app, self.analysis[name])
                    self._feature_obj[name][tool_name] = obj
                    logger.info("Feature initialized: %s (%s)", tool_name, feature_cls)
                except Exception:
                    logger.exception("Feature not available: %s (%s)", tool_name, feature_cls)

        # per-feature modal callbacks
        for tool, feature_object in self._feature_obj[name].items():
            logger.info("%s: registering modal callbacks for %s", name, tool)
            self._register_feature_modals(system_name=name, tool=tool, feature_object=feature_object)

    def _initialize_server(self) -> None:
        """Finalize layout and register routing callbacks."""

        @self.app.server.before_request
        def redirect_dashboard_root():
            if request.path in ["/dashboard", "/dashboard/"]:
                system = self.plant_names[0]
                user = self.users[0]
                feature = next(iter(self.features[system][user].keys()))
                return redirect(construct_url(system=system, user=self.users[0], feature=feature, start=get_today(),
                                              end=get_today()), code=302)

        self._cleanup_online_folder()
        self._ensure_default_users_roles()
        self._validate_systems_registered()
        self._set_base_layout()
        self._register_routing_callbacks()

    def run(self, port: int = 8050, debug: bool = False, host: str = "0.0.0.0", **kwargs) -> None:
        self.register_celery_tasks(LOCK_KEY=self.LOCK_KEY, LOCK_TTL=self.LOCK_TTL)
        self._initialize_server()
        self.app.run(host=host, port=port, debug=debug, **kwargs)

    # -------------------------
    # Callback registration
    # -------------------------
    def _register_routing_callbacks(self) -> None:
        self._register_update_pathname_callback()
        self._register_render_page_callback()

    def _register_update_pathname_callback(self) -> None:
        @self.app.callback(
            Output(construct_id("url"), "pathname"),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
            Input(construct_id("reevaluate"), "n_clicks"),
            State(construct_id("url"), "pathname"),
            State(construct_id("url"), "href"),
            prevent_initial_call=True,
        )
        def _update_pathname(start_date, end_date, reevaluate_clicks, pathname, href):
            logger.debug("Requested path: %s", pathname)
            system, role, feature, start, end = parse_url(pathname)

            if not reevaluate_clicks and (start == start_date and end == end_date):
                raise PreventUpdate

            if reevaluate_clicks:
                logger.info("Reevaluate triggered -> deleting cached files.")
                intervals = get_analysis_intervals(start_date, end_date)
                delete_files(intervals)
                logger.info("Cached files deleted.")

            return construct_url(system, role, feature, start_date, end_date)

    def _register_render_page_callback(self) -> None:
        @self.app.callback(
            Output(construct_id("topbar"), "children"),
            Output(construct_id("sidebar"), "children"),
            Output(construct_id("content"), "children"),
            Input(construct_id("url"), "pathname"),
            prevent_initial_call=True,
        )
        def _render_page(pathname):
            href = pathname
            system, role, feature, start, end = parse_url(pathname)

            if any(v is None for v in (system, role, feature, start, end)):
                return None, None, error_content("Bad path.", href)

            logger.info("Rendering: %s - %s - %s - %s - %s", system, role, feature, start, end)

            sidebar_layout = get_sidebar(
                self,
                system=system,
                user=role,
                feature=feature,
                start=start,
                end=end,
            )

            topbar_layout = get_topbar(
                self,
                systems=self.plant_names,
                roles=self.roles,
                logo=self._normalize_logos(),
                date_picker=self._date_picker_enabled(system, feature, role),
                system=system,
                role=role,
                feature=feature,
                start=start,
                end=end,
            )

            feature = self._resolve_feature(system, role, feature)
            if feature is None:
                return topbar_layout, sidebar_layout, error_content("No features available.", href)

            if not self._feature_allowed(system, role, feature):
                return topbar_layout, sidebar_layout, error_content(f"Unknown feature: {feature}", href)

            feature_object = self._get_feature_object(system, feature)
            if feature_object is None:
                return topbar_layout, sidebar_layout, error_content(f"Unknown feature: {feature}", href)

            perform_requested_features(self._feature_obj[system], self.celery_app, feature, system, start, end)
            return topbar_layout, sidebar_layout, self._content_container()

    def _register_feature_modals(self, system_name: str, tool: str, feature_object: Any) -> None:
        """Register config + LLM modal toggles for a single feature."""
        config_keys = list(getattr(feature_object, "config", {}).keys())

        @self.app.callback(
            [Output(construct_id(system_name, tool, "modal"), "is_open")]
            + [Output(construct_id(system_name, tool, k), "value") for k in config_keys],
            [
                Input(construct_id(system_name, tool, "configure"), "n_clicks"),
                Input(construct_id(system_name, tool, "close"), "n_clicks"),
                Input(construct_id(system_name, tool, "configure-apply"), "n_clicks"),
            ],
            [State(construct_id(system_name, tool, "modal"), "is_open")]
            + [State(construct_id(system_name, tool, k), "value") for k in config_keys],
            prevent_initial_call=True,
        )
        def _toggle_config_modal(n_open, n_close, n_apply, is_open, *values):
            if n_open or n_close or n_apply:
                is_open = not is_open

            if n_apply:
                for k, v in zip(config_keys, values):
                    self._feature_obj[system_name][tool].config[k]["value"] = v

            return (is_open,) + values

        @self.app.callback(
            Output(construct_id(system_name, tool, "modal-llm"), "is_open"),
            [
                Input(construct_id(system_name, tool, "llm"), "n_clicks"),
                Input(construct_id(system_name, tool, "close-llm"), "n_clicks"),
            ],
            State(construct_id(system_name, tool, "modal-llm"), "is_open"),
            prevent_initial_call=True,
        )
        def _toggle_llm_modal(n_open, n_close, is_open):
            return (not is_open) if (n_open or n_close) else is_open

    # -------------------------
    # Layout / lifecycle helpers
    # -------------------------
    def _cleanup_online_folder(self) -> None:
        online_dir = Path("Analysis") / "Online"
        try:
            if online_dir.exists():
                shutil.rmtree(online_dir)
                logger.info("Deleted folder: %s", online_dir)
        except Exception as exc:
            warnings.warn(f"Could not delete Online folder ({online_dir}): {exc}")

    def _ensure_default_users_roles(self) -> None:
        if not self.users:
            self.users = ("Default",)
        if not self.roles:
            self.roles = [{self.users[0]: {"password": "Default", "roles": ["Default"]}}]

    def _validate_systems_registered(self) -> None:
        if not self.plant_names:
            raise RuntimeError("No system added to the dashboard.")

    def _set_base_layout(self) -> None:
        location = dcc.Location(id=construct_id("url"))
        self.app.layout = html.Div(
            [
                location,
                html.Div(id=construct_id("topbar")),
                html.Div(id=construct_id("sidebar")),
                html.Div(id=construct_id("content"), className="content_style"),
            ]
        )

    def _content_container(self) -> html.Div:
        return html.Div(
            children=[
                dcc.Interval(
                    id=construct_id("content", "interval"),
                    disabled=True,
                    interval=self.content_not_ready_refresh_interval * 1000,
                    n_intervals=0,
                ),
                html.Div(id=construct_id("contentcontent")),
            ]
        )

    # -------------------------
    # Feature helpers
    # -------------------------
    def _normalize_logos(self) -> Optional[List[str]]:
        if self.logo is None:
            return None
        if isinstance(self.logo, str):
            return [self.logo]
        return list(self.logo)

    def _date_picker_enabled(self, system: str, feature: str, role: str) -> bool:
        try:
            obj = self._feature_obj[system][feature]
            return bool(obj.time_range_selection(role))
        except Exception:
            return True

    def _resolve_feature(self, system: str, role: str, feature: str) -> Optional[str]:
        if feature:
            return feature
        role_features: Dict[str, Any] = self.features.get(system, {}).get(role, {})
        return next(iter(role_features.keys()), None)

    def _feature_allowed(self, system: str, role: str, feature: str) -> bool:
        role_features = self.features.get(system, {}).get(role, {})
        return feature in role_features

    def _get_feature_object(self, system: str, feature: str) -> Any:
        obj = self._feature_obj.get(system, {}).get(feature)
        if obj is None:
            logger.error("Missing feature object for %s/%s", system, feature)
        return obj

    # -------------------------
    # Celery integration
    # -------------------------
    def register_celery_tasks(self, LOCK_KEY, LOCK_TTL):
        """
        Register feature tasks and (optionally) create a locked periodic chain executor.

        `plants_roles_features` is expected like:
            {plant_name: {role: [(cls, mdl), ...], ...}, ...}
        """

        for plant_name, features in self._feature_obj.items():
            registered: set[Tuple[str, str]] = set()
            periodic_objs: List[Any] = []
            all_objs: Dict[str, Any] = {}

            for user, feat in self._feature_obj.items():
                for cls, mdl in feat:
                    key = (mdl, cls)
                    if key in registered:
                        continue

                    try:
                        obj.name = plant_name + obj.feature_name()
                        self.app.register_task(obj)
                        registered.add(key)

                        if getattr(obj, "periodic", False):
                            periodic_objs.append(obj)

                        all_objs[obj.feature_name()] = obj
                    except Exception:
                        logger.warning("Feature not available: %s.%s", mdl, cls)
                        print_exc()

            if periodic_objs and ANALYSIS_IDLE_PERIOD is not None:
                sorted_features = [(x, all_objs[x]) for x in get_sorted_features(periodic_objs)]

                if "celery_app.run_feature_chain" not in app.tasks:

                    @app.task(name="celery_app.run_feature_chain")
                    def run_feature_chain():
                        lock_acquired = redis_client.set(
                            LOCK_KEY,
                            "1",
                            nx=True,  # only set if not exists
                            ex=LOCK_TTL,  # auto-expire safety
                        )
                        if not lock_acquired:
                            logger.info("Previous chain still running -> skipping")
                            return

                        try:
                            chain_tasks = [app.tasks[v.name].si(None, None) for (_, v) in sorted_features]
                            chain_tasks.append(app.tasks["tasks.release_lock"].s())
                            chain(*chain_tasks).apply_async()
                        except Exception:
                            redis_client.delete(LOCK_KEY)
                            raise

                app.conf.beat_schedule = {
                    "run-feature-chain": {
                        "task": "celery_app.run_feature_chain",
                        "schedule": schedule(datetime.timedelta(seconds=ANALYSIS_IDLE_PERIOD)),
                    },
                }
            else:
                logger.info("No periodic analysis of %s...", plant_name)


# -------------------------
# UI helpers
# -------------------------
def table(data: Any, use_columns: Optional[Sequence[str]] = None, **kwargs) -> dash_table.DataTable:
    """Simple read-only table helper that normalizes object columns to strings."""
    if "style_cell" not in kwargs:
        kwargs["style_cell"] = {"maxWidth": "500px"}

    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)

    cols = list(use_columns) if use_columns is not None else list(df.columns)

    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=df.to_dict("records"),
        style_data={"whiteSpace": "normal", "height": "auto"},
        **kwargs,
    )


def editable_table(
    df: Any,
    use_columns: Optional[Sequence[str]] = None,
    conditional_rows: Optional[Mapping[str, Sequence[int]]] = None,
    hidden_columns: Optional[Sequence[str]] = None,
    limit_numeric_precision: Optional[int] = None,
    numeric_cols: Optional[Sequence[str]] = None,
    editable_columns: Optional[Sequence[str]] = None,
    style_as_list_view: bool = False,
    **kwargs,
) -> dash_table.DataTable:
    """
    Editable Dash DataTable helper.

    - `numeric_cols` marks columns as numeric (optionally with `limit_numeric_precision`)
    - `editable_columns` marks specific columns editable
    - `conditional_rows` highlights specific (column, row_index) pairs
    """
    if df is None:
        records: List[Dict[str, Any]] = []
        columns: List[Dict[str, Any]] = []
    else:
        _df = pd.DataFrame(df)
        records = _df.to_dict("records")
        columns = [{"name": str(c), "id": str(c)} for c in _df.columns]

    if use_columns is not None:
        columns = [{"name": c, "id": c} for c in use_columns]

    # allow caller override
    if "columns" in kwargs:
        columns = kwargs.pop("columns")

    numeric_cols_set = set(numeric_cols or [])
    for c in columns:
        if c["id"] in numeric_cols_set or c["name"] in numeric_cols_set:
            c["type"] = "numeric"
            if limit_numeric_precision is not None:
                c["format"] = dash_table.Format.Format(precision=limit_numeric_precision)

    # conditional highlighting
    conditional: List[Dict[str, Any]] = []
    if conditional_rows:
        for col, rows in conditional_rows.items():
            for r in rows:
                conditional.append(
                    {
                        "if": {"row_index": r, "column_id": col},
                        "backgroundColor": colors.NEGATIVE,
                        "color": "white",
                    }
                )

    # editable columns
    editable = False
    if editable_columns is not None:
        editable_set = set(editable_columns)
        for c in columns:
            c["editable"] = c["id"] in editable_set
        editable = True

    if "style_data_conditional" in kwargs:
        conditional += kwargs.pop("style_data_conditional")

    return dash_table.DataTable(
        columns=columns,
        data=records,
        editable=editable,
        hidden_columns=list(hidden_columns) if hidden_columns else None,
        style_as_list_view=style_as_list_view,
        style_data_conditional=conditional,
        style_data={
            "whiteSpace": "normal",
            "wordBreak": "break-all",
            "overflowWrap": "break-word",
            "height": "40px",
            "lineHeight": "40px",
        },
        **kwargs,
    )


def error_content(message: Optional[str], path: str = "") -> dbc.Container:
    """Default error page content."""
    message = message or "404: Not found"
    return dbc.Container(
        [
            dcc.Interval(interval=10000),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H1(message, className="text-danger"),
            html.Hr(),
            html.P(f'The path "{path}" was not recognised...'),
        ]
    )


def get_modal(modal_id: str, title: str = "Notification", button: bool = True, button_text: str = "Acknowledge"):
    """Small modal helper with optional acknowledge button."""
    buttons = []
    if button:
        buttons.append(dbc.Button(button_text, id=construct_id(modal_id, "button"), className="configure-apply"))

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(id=construct_id(modal_id, "content")),
            dbc.ModalFooter(buttons),
        ],
        id=modal_id,
        className="modal-content",
        is_open=False,
    )


