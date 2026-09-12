# test_colors.py
import pytest

from selfx.dash.colors import (
    BLACK,
    NEGATIVE,
    POSITIVE,
    TRANSPARENT,
    WHITE,
    contrast,
    color_list,
    get_color_mapping,
    hex_to_rgb,
    hex_to_rgba,
    iterate,
    opacity,
    rgb_to_hex,
)


def test_rgb_to_hex_without_hash():
    assert rgb_to_hex((255, 0, 170)) == "ff00aa"


def test_rgb_to_hex_with_hash():
    assert rgb_to_hex((255, 0, 170), with_hash=True) == "#ff00aa"


def test_rgb_to_hex_invalid_length():
    with pytest.raises(ValueError):
        rgb_to_hex((255, 0))


def test_rgb_to_hex_invalid_value():
    with pytest.raises(ValueError):
        rgb_to_hex((255, -1, 0))


def test_hex_to_rgb():
    assert hex_to_rgb("#FF00AA") == (255, 0, 170)


def test_hex_to_rgb_invalid():
    with pytest.raises(ValueError):
        hex_to_rgb("FF00AA")


def test_hex_to_rgba_without_opacity():
    assert hex_to_rgba("#FF00AA") == "rgb(255,0,170)"


def test_hex_to_rgba_with_opacity():
    assert hex_to_rgba("#FF00AA", 0.5) == "rgba(255,0,170,0.5)"


def test_hex_to_rgba_invalid_opacity():
    with pytest.raises(ValueError):
        hex_to_rgba("#FF00AA", 1.5)


def test_opacity_zero():
    assert opacity("#FF00AA", 0.0) == "#FF00AA00"


def test_opacity_half():
    assert opacity("#FF00AA", 0.5) == "#FF00AA80"


def test_opacity_full():
    assert opacity("#FF00AA", 1.0) == "#FF00AAFF"


def test_opacity_rejects_rgb():
    with pytest.raises(NotImplementedError):
        opacity("rgb(255,0,0)", 0.5)


def test_iterate_wraps():
    first = iterate(0)
    wrapped = iterate(10)
    assert first == wrapped


def test_contrast_white_background():
    assert contrast("#FFFFFF") == BLACK


def test_contrast_black_background():
    assert contrast("#000000") == WHITE


def test_contrast_mid_background():
    assert contrast("#808080") in {BLACK, WHITE}


def test_transparent_constant():
    assert TRANSPARENT == "rgba(255,255,255,0)"


def test_get_color_mapping_returns_empty_mapping_by_default():
    assert get_color_mapping() == {}


def test_get_color_mapping_maps_status_colors():
    color_map = get_color_mapping(
        positive_color=["ok"],
        negative_color=["bad"],
        error_color=["error"],
        transparent_color=["hidden"],
    )

    assert color_map == {
        "ok": POSITIVE[0],
        "bad": NEGATIVE[0],
        "error": NEGATIVE[1],
        "hidden": TRANSPARENT,
    }


def test_get_color_mapping_maps_items_to_palette():
    assert get_color_mapping(items=["first", "second"]) == {
        "first": color_list[0],
        "second": color_list[1],
    }


def test_get_color_mapping_maps_grouped_items_to_same_color():
    assert get_color_mapping(items=[["a", "b"], "c"]) == {
        "a": color_list[0],
        "b": color_list[0],
        "c": color_list[1],
    }
