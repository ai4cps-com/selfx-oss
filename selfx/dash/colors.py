# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nemanja Hranisavljevic
# Contact: nemanja@ai4cps.com

from __future__ import annotations
from typing import Sequence


# Semantic color groups used across charts and UI states.
POSITIVE: list[str] = ["#04F1D9", "#B1C800", "#45AF97", "#2D00F5"]
NEGATIVE: list[str] = ["#FC871C", "#FF312E", "#FDE74C"]
NEUTRAL: list[str] = ["#191835", "#A08BFF", "#A3A3AE", "#693372", "#E8C990"]
OTHER: list[str] = ["#1F82C0", "#B2A198", "#512D38", "#CBFF4D", "#FFA3AF"]

# Base colors.
BLACK: str = "#000000"
WHITE: str = "#FFFFFF"
TRANSPARENT: str = "rgba(255,255,255,0)"

# Chart-specific colors.
BAR_CHART: list[str] = ["#274C48", "#417F79", "#67CAC1", "#81FDF1"]
TICK_LABEL_COLOR: str = "#7F7F7F"
GRID_COLOR: str = "#E3E3E3"
PLOT_BG_COLOR: str = "#FFFFFF"


def _validate_rgb(rgb: Sequence[int]) -> None:
    if len(rgb) != 3 or any(not isinstance(v, int) or v < 0 or v > 255 for v in rgb):
        raise ValueError("rgb must contain exactly 3 integers between 0 and 255")


def _validate_hex_color(hex_color: str) -> None:
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        raise ValueError("hex_color must be a string in the format '#RRGGBB'")

    try:
        int(hex_color[1:], 16)
    except ValueError as exc:
        raise ValueError("hex_color must contain valid hexadecimal digits") from exc


def rgb_to_hex(rgb: Sequence[int], *, with_hash: bool = False) -> str:
    """
    Convert an RGB triplet into a 6-digit hex string.

    Args:
        rgb: Sequence of three integers in the range 0..255.
        with_hash: Whether to prepend '#'.

    Returns:
        Hex color string like 'ff00aa' or '#ff00aa'.
    """
    _validate_rgb(rgb)
    value = "%02x%02x%02x" % tuple(rgb)
    return f"#{value}" if with_hash else value


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert '#RRGGBB' into an (R, G, B) tuple.
    """
    _validate_hex_color(hex_color)
    value = hex_color[1:]
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color: str, opacity: float | None = None) -> str:
    """
    Convert a hex color string to CSS rgb(...) or rgba(...).

    Args:
        hex_color: Color in '#RRGGBB' format.
        opacity: Optional opacity value in the range 0..1.

    Returns:
        CSS color string, either 'rgb(r,g,b)' or 'rgba(r,g,b,a)'.
    """
    _validate_hex_color(hex_color)

    if opacity is not None and not (0 <= opacity <= 1):
        raise ValueError("opacity must be between 0 and 1")

    red, green, blue = hex_to_rgb(hex_color)

    if opacity is None:
        return f"rgb({red},{green},{blue})"

    return f"rgba({red},{green},{blue},{opacity})"


def opacity(color: str, alpha: float) -> str:
    """
    Apply alpha transparency to a hex color and return '#RRGGBBAA'.

    Args:
        color: Color in '#RRGGBB' format.
        alpha: Opacity in the range 0..1.

    Returns:
        8-digit hex color string like '#FF000080'.
    """
    if color.startswith("rgb"):
        raise NotImplementedError("Applying opacity to rgb/rgba strings is not supported")

    _validate_hex_color(color)

    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")

    alpha_hex = f"{round(alpha * 255):02X}"
    return f"{color.upper()}{alpha_hex}"


def iterate(index: int) -> str:
    """
    Return a color from the combined NEUTRAL and OTHER palettes, cycling by index.
    """
    palette = NEUTRAL + OTHER
    return palette[index % len(palette)]


def contrast(hex_color: str, *, threshold: float = 186) -> str:
    """
    Return black or white depending on the perceived brightness of a background color.

    Args:
        hex_color: Color in '#RRGGBB' format.
        threshold: Brightness threshold for switching to black text.

    Returns:
        '#000000' for light backgrounds, '#FFFFFF' for dark backgrounds.
    """
    _validate_hex_color(hex_color)
    red, green, blue = hex_to_rgb(hex_color)

    brightness = red * 0.299 + green * 0.587 + blue * 0.114
    return BLACK if brightness > threshold else WHITE