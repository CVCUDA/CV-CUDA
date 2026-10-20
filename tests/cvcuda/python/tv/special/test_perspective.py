# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv perspective contracts beyond native tensor parity."""

import numpy as np
import pytest

from cvcuda import tv as cvcuda_tv


_CORNERS = [[0, 0], [4, 0], [4, 3], [0, 3]]


_IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0]


@pytest.mark.parametrize(
    "startpoints,endpoints,coefficients,match",
    [
        (None, None, None, "startpoints/endpoints"),
        (None, _CORNERS, None, "startpoints/endpoints"),
        (_CORNERS, None, None, "startpoints/endpoints"),
        (_CORNERS[:3], _CORNERS, None, "four corners"),
        (_CORNERS, _CORNERS + [[0, 0]], None, "four corners"),
        (None, None, _IDENTITY[:7], "8 float values"),
        (None, None, _IDENTITY + [0], "8 float values"),
    ],
)
def test_perspective_validation_matches_oracle(
    tvf, monkeypatch, startpoints, endpoints, coefficients, match
):
    import torch

    arguments = dict(
        startpoints=startpoints, endpoints=endpoints, coefficients=coefficients
    )
    with pytest.raises(ValueError, match=match):
        tvf.perspective(torch.zeros((3, 4, 5)), **arguments)

    def unexpected_solve(*args, **kwargs):
        pytest.fail("invalid perspective arguments reached the solver")

    monkeypatch.setattr(np.linalg, "solve", unexpected_solve)
    with pytest.raises(ValueError, match=match):
        cvcuda_tv.perspective(None, **arguments)


@pytest.mark.parametrize(
    "points,error,match",
    [
        (0, TypeError, None),
        ([[0]] * 4, ValueError, "four corners.*two"),
        ([[0, 0, 0]] * 4, ValueError, "four corners.*two"),
        ([[0, 0]] * 3 + [None], ValueError, None),
    ],
)
@pytest.mark.parametrize("argument", ["startpoints", "endpoints"])
def test_perspective_malformed_points(monkeypatch, points, error, match, argument):
    arguments = dict(startpoints=_CORNERS, endpoints=_CORNERS)
    arguments[argument] = points

    def unexpected_solve(*args, **kwargs):
        pytest.fail("malformed corners reached the solver")

    monkeypatch.setattr(np.linalg, "solve", unexpected_solve)
    with pytest.raises(error, match=match):
        cvcuda_tv.perspective(None, **arguments)


@pytest.mark.parametrize("coefficients", [0, [[1]] * 8, [None] * 8, ["bad"] * 8])
def test_perspective_malformed_coefficients(coefficients):
    with pytest.raises(ValueError, match="8 float values"):
        cvcuda_tv.perspective(None, None, None, coefficients=coefficients)


@pytest.mark.parametrize("use_coefficients", [False, True])
def test_perspective_identity(torch, use_coefficients):
    image = torch.arange(60, device="cuda", dtype=torch.float32).reshape(3, 4, 5)
    arguments = (
        dict(startpoints=None, endpoints=None, coefficients=_IDENTITY)
        if use_coefficients
        else dict(startpoints=_CORNERS, endpoints=_CORNERS)
    )
    actual = cvcuda_tv.perspective(image, interpolation="nearest", **arguments)
    torch.testing.assert_close(actual, image, atol=0, rtol=0)
