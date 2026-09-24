"""Compatibility alias for :mod:`ai4cps.dash.routing_utils`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("ai4cps.dash.routing_utils")
