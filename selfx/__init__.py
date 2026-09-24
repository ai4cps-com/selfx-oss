"""Deprecated compatibility imports for the AI4CPS framework."""

import warnings as _warnings

from .version import __version__

_warnings.warn(
    "The selfx package is deprecated. Install ai4cps and replace selfx imports "
    "with ai4cps imports. Existing selfx imports remain available for compatibility.",
    FutureWarning,
    stacklevel=2,
)
