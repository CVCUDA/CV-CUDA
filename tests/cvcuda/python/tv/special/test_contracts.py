# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input validation not covered by successful parity calls."""

import re
from types import SimpleNamespace

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv
from cvcuda.tv._layout import (
    InterpolationMode,
    RESIZE_INTERP_MODES,
    WARP_INTERP_MODES,
    interp_from,
)


@pytest.mark.parametrize("scale", [0, -1])
def test_affine_nonpositive_scale(tvf, scale):
    import torch

    arguments = dict(angle=0, translate=[0, 0], scale=scale, shear=0)
    with pytest.raises(ValueError, match="scale should be positive"):
        tvf.affine(torch.zeros((3, 4, 5)), **arguments)
    with pytest.raises(ValueError, match="scale should be positive"):
        cvcuda_tv.affine(SimpleNamespace(layout="CHW", shape=(3, 4, 5)), **arguments)


@pytest.mark.parametrize(
    "mode,message",
    [
        pytest.param(999, "interpolation is not supported", id="999"),
        pytest.param(-1, "interpolation is not supported", id="negative"),
        pytest.param("bogus", "interpolation is not supported", id="bogus"),
        pytest.param(True, "'true' interpolation is not supported", id="true"),
        pytest.param(False, "'false' interpolation is not supported", id="false"),
        pytest.param(None, "interpolation is not supported", id="none"),
        pytest.param([], "interpolation is not supported", id="list"),
        pytest.param({}, "interpolation is not supported", id="dict"),
        pytest.param(object(), "interpolation is not supported", id="object"),
    ],
)
@pytest.mark.parametrize(
    "supported,unsupported",
    [
        (None, NotImplementedError),
        (RESIZE_INTERP_MODES, NotImplementedError),
        (WARP_INTERP_MODES, ValueError),
    ],
)
def test_invalid_interpolation_has_operator_exception(
    mode, message, supported, unsupported
):
    with pytest.raises(unsupported, match=message):
        interp_from(mode, supported, unsupported)


@pytest.mark.parametrize("mode", [0, "nearest", InterpolationMode.NEAREST])
def test_interpolation_accepts_collection(mode):
    assert interp_from(mode, ["nearest"]) == cvcuda.Interp.NEAREST


@pytest.mark.parametrize("mode", [False, True])
def test_boolean_interpolation_cannot_reuse_integer_cache(mode):
    interp_from(int(mode))
    with pytest.raises(ValueError, match="interpolation is not supported"):
        interp_from(mode, unsupported=ValueError)
    with pytest.raises(NotImplementedError, match="interpolation is not supported"):
        interp_from(mode)


@pytest.mark.parametrize(
    "call,channels",
    [
        pytest.param(call, channels, id=f"{name}-{channels}-channels")
        for name, call, invalid_channels in [
            ("rgb_to_grayscale", lambda f, x: f.rgb_to_grayscale(x), (2, 4, 5)),
            (
                "rgb_to_grayscale_3",
                lambda f, x: f.rgb_to_grayscale(x, num_output_channels=3),
                (2, 4, 5),
            ),
            ("grayscale_to_rgb", lambda f, x: f.grayscale_to_rgb(x), (2,)),
            ("adjust_brightness", lambda f, x: f.adjust_brightness(x, 1.2), (2, 4, 5)),
            ("adjust_contrast", lambda f, x: f.adjust_contrast(x, 1.2), (2, 4, 5)),
            ("adjust_saturation", lambda f, x: f.adjust_saturation(x, 1.2), (2, 4, 5)),
            ("adjust_hue", lambda f, x: f.adjust_hue(x, 0.1), (2, 4, 5)),
            ("adjust_sharpness", lambda f, x: f.adjust_sharpness(x, 1.2), (2, 4, 5)),
        ]
        for channels in invalid_channels
    ],
)
def test_channel_count_error_matches_torchvision(torch, tvf, call, channels):
    image = torch.zeros((1, channels, 8, 8), device="cuda", dtype=torch.uint8)
    with pytest.raises((TypeError, ValueError)) as expected:
        call(tvf, image)
    with pytest.raises(expected.type, match=re.escape(str(expected.value))):
        call(cvcuda_tv, image)


@pytest.mark.parametrize("channels", [0, 1, 2, 4, 5])
def test_rgb_channel_error_matches_python_unpacking(channels):
    from cvcuda.tv._color import _require_rgb

    with pytest.raises(ValueError) as expected:
        r, g, b = tuple(range(channels))
    with pytest.raises(ValueError) as actual:
        _require_rgb(channels)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("num_output_channels", [0, 2, 4])
def test_grayscale_output_channels_error_matches_torchvision(
    torch, tvf, num_output_channels
):
    image = torch.zeros((3, 8, 8), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError) as expected:
        tvf.rgb_to_grayscale(image, num_output_channels=num_output_channels)
    with pytest.raises(ValueError, match=re.escape(str(expected.value))):
        cvcuda_tv.rgb_to_grayscale(image, num_output_channels=num_output_channels)


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("adjust_brightness", -0.5, "not non-negative"),
        ("adjust_contrast", -0.5, "not non-negative"),
        ("adjust_saturation", -0.5, "not non-negative"),
        ("adjust_sharpness", -0.5, "not non-negative"),
        ("adjust_hue", -0.6, "not in"),
        ("adjust_hue", 0.6, "not in"),
        ("adjust_gamma", -0.5, "non-negative"),
    ],
)
def test_color_factor_errors(name, value, message):
    with pytest.raises(ValueError, match=message):
        getattr(cvcuda_tv, name)(None, value)
