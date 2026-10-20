# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometric image operations for :mod:`cvcuda.tv`."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

import cvcuda

from ._layout import (
    BORDER_MAP,
    InterpolationMode,
    RESIZE_INTERP_MODES,
    WARP_INTERP_MODES,
    _Interpolation,
    _interp_key,
    dims,
    interp_from,
    sized_shape,
)

_Size: TypeAlias = int | Sequence[int]
_Fill: TypeAlias = int | float | Sequence[float] | None
_Vector: TypeAlias = Sequence[int | float]
_Points: TypeAlias = Sequence[_Vector]
_FiveCrop: TypeAlias = tuple[
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
]
_TenCrop: TypeAlias = tuple[
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
    cvcuda.Tensor,
]


def horizontal_flip(
    inpt: cvcuda.Tensor, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Flip image(s) horizontally (left-right).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Image(s) flipped left-right.

    """
    return cvcuda.flip(inpt, flipCode=1, stream=stream)


def vertical_flip(
    inpt: cvcuda.Tensor, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Flip image(s) vertically (up-down).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Image(s) flipped up-down.

    """
    return cvcuda.flip(inpt, flipCode=0, stream=stream)


def _fill_border(fill: _Fill) -> cvcuda.Border:
    """Pick the border mode that reproduces torchvision's handling of ``fill``.

    ``_apply_grid_transform`` only builds its coverage mask when a fill was
    asked for; with ``fill=None`` it returns the plain zero-padded sample. So
    the coverage border applies exactly when ``fill`` is not None -- using it
    unconditionally would weigh an unfilled warp by its own coverage and darken
    every edge.
    """
    return cvcuda.Border.CONSTANT if fill is None else cvcuda.Border.CONSTANT_COVERAGE


def _fill_value(fill: _Fill) -> list[float]:
    """Normalize torchvision's ``fill`` into a border-value list (default black).

    A scalar or one-element-sequence ``fill`` is broadcast to all four border
    components because the CV-CUDA border-value bindings zero any component the
    caller omits, which would otherwise color only the first channel of a
    multi-channel image.  This matches torchvision's behaviour of treating a
    one-element fill sequence as a scalar broadcast.
    """
    if fill is None:
        return [0.0, 0.0, 0.0, 0.0]
    if isinstance(fill, (int, float)):
        return [float(fill)] * 4
    fill = list(fill)
    if len(fill) == 1:
        return [float(fill[0])] * 4
    return [float(v) for v in fill]


def _resize_output_hw(
    size: _Size | None, h: int, w: int, max_size: int | None
) -> tuple[int, int]:
    """Reproduce torchvision's ``size`` semantics -> concrete ``(out_h, out_w)``.

    A two-element ``size`` is an exact ``[height, width]``. A scalar / one-element
    ``size`` sets the shorter edge and scales the longer one to preserve aspect
    ratio, optionally capped by ``max_size``. ``size=None`` scales the longer edge
    to ``max_size``.

    Mirrors torchvision's ``_compute_resized_output_size``, including where it
    rejects a ``max_size`` it cannot honour, so callers see the same errors.
    """
    if isinstance(size, int):
        size = [size]
    elif max_size is not None and size is not None and len(size) != 1:
        raise ValueError(
            "max_size should only be passed if size is None or specifies the "
            "length of the smaller edge, i.e. size should be an int or a "
            "sequence of length 1 in torchscript mode."
        )

    short, long_edge = (w, h) if w <= h else (h, w)

    if size is None:
        if not isinstance(max_size, int):
            raise ValueError(
                f"max_size must be an integer when size is None, but got {max_size} instead."
            )
        new_short, new_long = int(max_size * short / long_edge), max_size
    else:
        size = list(size)
        if len(size) != 1:
            return int(size[0]), int(size[1])

        requested_short = size[0]
        new_short, new_long = requested_short, int(requested_short * long_edge / short)
        new_short, new_long = _limit_resize_size(size, new_short, new_long, max_size)

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return new_h, new_w


def _limit_resize_size(size, new_short, new_long, max_size):
    if max_size is not None:
        if max_size <= new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the "
                f"requested size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            return int(max_size * new_short / new_long), max_size
    return new_short, new_long


def _resize(
    inpt: cvcuda.Tensor,
    size: _Size | None,
    interpolation: _Interpolation,
    max_size: int | None,
    antialias: bool | None,
    stream: cvcuda.Stream | None,
) -> cvcuda.Tensor:
    """Shared core for ``resize`` / ``resized_crop``."""
    _, h, w, _ = dims(inpt)
    out_h, out_w = _resize_output_hw(size, h, w, max_size)
    interp = interp_from(interpolation, RESIZE_INTERP_MODES, NotImplementedError)
    if (out_h, out_w) == (h, w):
        # torchvision returns the input itself for a no-op resize, and its tests
        # assert that identity, whatever the interpolation mode.
        return inpt
    if _interp_key(interpolation) == "nearest":
        interp = cvcuda.Interp.NEAREST_FLOOR
    return cvcuda.tvresize(
        inpt,
        size,
        interp=interp,
        max_size=max_size,
        antialias=antialias,
        stream=stream,
    )


def resize(
    inpt: cvcuda.Tensor,
    size: _Size | None,
    interpolation: _Interpolation = InterpolationMode.BILINEAR,
    max_size: int | None = None,
    antialias: bool | None = True,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Resize image(s) to a target size using :func:`cvcuda.tvresize`.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        size (int | list[int]): Target size; a two-element ``[h, w]`` is exact,
            a scalar sets the shorter edge preserving aspect ratio.
        interpolation (InterpolationMode | str | int, optional): Interpolation
            mode. Supports nearest, nearest-exact, bilinear, and bicubic.
            Defaults to bilinear.
        max_size (int, optional): Cap for the longer edge when ``size`` is scalar.
        antialias (bool, optional): Apply an antialiasing filter when downscaling.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Resized image(s).

    Raises:
        NotImplementedError: If ``interpolation`` is unsupported.

    """
    return _resize(inpt, size, interpolation, max_size, antialias, stream)


def resized_crop(
    inpt: cvcuda.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: _Size,
    interpolation: _Interpolation = InterpolationMode.BILINEAR,
    antialias: bool | None = True,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Crop a region from image(s) then resize it to a target size.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        top, left, height, width (int): Crop rectangle (rows from ``top``,
            columns from ``left``).
        size (int | list[int]): Target size for the resize.
        interpolation (InterpolationMode | str | int, optional): Interpolation
            mode. Supports nearest, nearest-exact, bilinear, and bicubic.
            Defaults to bilinear.
        antialias (bool, optional): Apply an antialiasing filter when downscaling.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Cropped-and-resized image(s).

    Raises:
        NotImplementedError: If ``interpolation`` is unsupported.

    """
    cropped = crop(inpt, top, left, height, width, stream=stream)
    return _resize(cropped, size, interpolation, None, antialias, stream)


def crop(
    inpt: cvcuda.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Crop a rectangular region from image(s).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        top, left (int): Top-left corner of the crop (row, column).
        height, width (int): Crop extent.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: The cropped region.

    Note:
        A rectangle extending past the input is cropped to the overlap and then
        zero-padded back out to the requested extent, matching torchvision.

    """
    _, h, w, _ = dims(inpt)
    right = left + width
    bottom = top + height

    if left >= 0 and top >= 0 and right <= w and bottom <= h:
        rect = cvcuda.RectI(x=left, y=top, width=width, height=height)
        return cvcuda.customcrop(inpt, rect, stream=stream)

    # Same decomposition torchvision uses: slice the in-bounds overlap, then pad
    # the shortfall on each side with zeros.
    pad_left = max(min(right, 0) - left, 0)
    pad_right = max(right - max(w, left), 0)
    pad_top = max(min(bottom, 0) - top, 0)
    pad_bottom = max(bottom - max(h, top), 0)

    src_left, src_top = max(left, 0), max(top, 0)
    overlap_w = max(min(right, w) - src_left, 0)
    overlap_h = max(min(bottom, h) - src_top, 0)

    if overlap_w == 0 or overlap_h == 0:
        # The rectangle misses the image entirely, so every output pixel is fill.
        # CV-CUDA has no zero-extent tensor, so build the result from a single
        # zeroed pixel instead: convertto with scale 0 and offset 0 clears it.
        seed = cvcuda.customcrop(
            inpt, cvcuda.RectI(x=0, y=0, width=1, height=1), stream=stream
        )
        zero = cvcuda.convertto(seed, seed.dtype, 0.0, 0.0, stream=stream)
        return cvcuda.copymakeborder(
            zero,
            cvcuda.Border.CONSTANT,
            [0.0, 0.0, 0.0, 0.0],
            top=0,
            bottom=height - 1,
            left=0,
            right=width - 1,
            stream=stream,
        )

    overlap = cvcuda.customcrop(
        inpt,
        cvcuda.RectI(x=src_left, y=src_top, width=overlap_w, height=overlap_h),
        stream=stream,
    )
    return cvcuda.copymakeborder(
        overlap,
        cvcuda.Border.CONSTANT,
        [0.0, 0.0, 0.0, 0.0],
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        stream=stream,
    )


def center_crop(
    inpt: cvcuda.Tensor,
    output_size: _Size,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Crop the center region from image(s).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        output_size (int | list[int]): Center-crop ``[height, width]`` (a scalar
            crops a square).
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: The center crop.

    """
    crop_h, crop_w = _crop_size(output_size)
    _, h, w, _ = dims(inpt)
    if crop_h > h or crop_w > w:
        pad_left = max((crop_w - w) // 2, 0)
        pad_top = max((crop_h - h) // 2, 0)
        pad_right = max((crop_w - w + 1) // 2, 0)
        pad_bottom = max((crop_h - h + 1) // 2, 0)
        inpt = pad(
            inpt,
            [pad_left, pad_top, pad_right, pad_bottom],
            stream=stream,
        )
        _, h, w, _ = dims(inpt)
    top = int(round((h - crop_h) / 2.0))
    left = int(round((w - crop_w) / 2.0))
    return crop(inpt, top, left, crop_h, crop_w, stream=stream)


def pad(
    inpt: cvcuda.Tensor,
    padding: _Size,
    fill: _Fill = None,
    padding_mode: str = "constant",
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Pad image(s) on all sides.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        padding (int | list[int]): A common border width, ``[horizontal,
            vertical]``, or ``[left, top, right, bottom]``.
        fill (int | float | list[float], optional): Constant fill value(s).
        padding_mode (str, optional): One of ``constant``, ``edge``, ``reflect``,
            ``symmetric``. Defaults to ``constant``.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Padded image(s).

    """
    if isinstance(padding, int):
        left = right = top = bottom = padding
    elif not isinstance(padding, (list, tuple)):
        raise TypeError(
            f"`padding` should be an integer or tuple or list of integers, but got {padding}"
        )
    elif len(padding) not in (1, 2, 4):
        raise ValueError(
            "Padding must be an int or a 1, 2, or 4 element tuple or list. "
            f"Found {len(padding)}"
        )
    elif len(padding) == 1:
        left = right = top = bottom = padding[0]
    elif len(padding) == 2:
        left = right = padding[0]
        top = bottom = padding[1]
    else:
        left, top, right, bottom = padding[0], padding[1], padding[2], padding[3]
    if not isinstance(padding_mode, str) or padding_mode not in BORDER_MAP:
        raise ValueError(
            f"padding_mode must be one of {', '.join(BORDER_MAP)}, got {padding_mode!r}"
        )
    border = BORDER_MAP[padding_mode]
    value = _fill_value(fill)
    return cvcuda.copymakeborder(
        inpt,
        border,
        value,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        stream=stream,
    )


def _inverse_affine_matrix(
    center: _Vector,
    angle: float,
    translate: _Vector,
    scale: float,
    shear: _Vector,
) -> NDArray[np.float32]:
    """torchvision's inverse (output->input) affine matrix, in pixel coordinates.

    Reproduces ``torchvision.transforms.functional._get_inverse_affine_matrix``
    (``inverted=True``): ``M = C * RSS^-1 * C^-1 * T^-1``. Returned as a 2x3
    ``numpy.float32`` array for ``cvcuda.warp_affine`` with ``WARP_INVERSE_MAP``.
    """
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])
    cx, cy = center
    tx, ty = translate

    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    m = [d, -b, 0.0, -c, a, 0.0]
    m = [x / scale for x in m]
    m[2] += m[0] * (-cx - tx) + m[1] * (-cy - ty)
    m[5] += m[3] * (-cx - tx) + m[4] * (-cy - ty)
    m[2] += cx
    m[5] += cy
    return np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]]], dtype=np.float32)


def _expanded_output_size(
    matrix: NDArray[np.float32], w: int, h: int
) -> tuple[int, int]:
    """Canvas that fits the rotated image, as ``(out_h, out_w)``.

    Reproduces torchvision's ``_compute_affine_output_size``: push the four
    corners through the inverse matrix with the image centre at the origin and
    measure the bounding box. The 1e-4 truncation is torchvision's guard against
    ``ceil`` promoting a floating-point crumb to a whole pixel. Kept in float32
    so the rounding lands on the same integers torchvision produces.
    """
    half_w = 0.5 * w
    half_h = 0.5 * h
    corners = np.array(
        [
            [-half_w, -half_h, 1.0],
            [-half_w, half_h, 1.0],
            [half_w, half_h, 1.0],
            [half_w, -half_h, 1.0],
        ],
        dtype=np.float32,
    )
    mapped = corners @ matrix.T
    halfs = np.array([half_w, half_h], dtype=np.float32)
    minimum = mapped.min(axis=0) + halfs
    maximum = mapped.max(axis=0) + halfs

    tol = np.float32(1e-4)
    inv_tol = np.float32(1.0 / 1e-4)
    upper = np.ceil(np.trunc(maximum * inv_tol) * tol)
    lower = np.floor(np.trunc(minimum * inv_tol) * tol)
    size = upper - lower
    return int(size[1]), int(size[0])


def _pixel_center(coord: float | None, size: int) -> float:
    """Map a torchvision ``center`` component (origin top-left) into the pixel-grid
    center convention CV-CUDA's warp uses. ``None`` -> the image center."""
    base = size * 0.5 if coord is None else coord
    return base - 0.5


def rotate(
    inpt: cvcuda.Tensor,
    angle: float,
    interpolation: _Interpolation = InterpolationMode.NEAREST,
    expand: bool = False,
    center: _Vector | None = None,
    fill: _Fill = None,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Rotate image(s) by ``angle`` degrees counter-clockwise.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        angle (float): Counter-clockwise rotation in degrees.
        interpolation (InterpolationMode | str | int, optional): Interpolation
            mode. Supports nearest and bilinear. Defaults to nearest.
        expand (bool, optional): Grow the output so the whole rotated image
            fits, instead of keeping the input extent. Defaults to ``False``.
        center (list[float], optional): Center of rotation, origin top-left;
            defaults to the image center. Ignored for the expanded extent, as
            in torchvision.
        fill (int | float | list[float], optional): Constant fill for exposed
            pixels.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Rotated image(s).

    Raises:
        ValueError: If ``interpolation`` is unsupported.

    """
    _, h, w, _ = dims(inpt)
    cx = _pixel_center(None if center is None else center[0], w)
    cy = _pixel_center(None if center is None else center[1], h)
    # rotate feeds -angle to torchvision's affine matrix helper (its CCW convention).
    matrix = _inverse_affine_matrix((cx, cy), -angle, (0.0, 0.0), 1.0, (0.0, 0.0))
    flags = int(interp_from(interpolation, WARP_INTERP_MODES, ValueError)) | int(
        cvcuda.Interp.WARP_INVERSE_MAP
    )

    out_h, out_w = h, w
    if expand:
        # torchvision sizes the expanded canvas from a matrix whose centre offset
        # is measured from the image centre without the half-pixel shift the warp
        # uses.
        offset = (
            (0.0, 0.0) if center is None else (center[0] - w * 0.5, center[1] - h * 0.5)
        )
        out_h, out_w = _expanded_output_size(
            _inverse_affine_matrix(offset, -angle, (0.0, 0.0), 1.0, (0.0, 0.0)), w, h
        )
        # Re-anchor the map onto the expanded canvas. torchvision builds its
        # sampling grid from the output centre and lands it on the *image*
        # centre, not the rotation centre, so the shift ignores `center`.
        shift = np.array([(w - out_w) * 0.5, (h - out_h) * 0.5], dtype=np.float32)
        matrix = matrix.copy()
        matrix[:, 2] += matrix[:, :2] @ shift

    border = _fill_border(fill)
    border_value = _fill_value(fill)
    if not expand:
        return cvcuda.warp_affine(
            inpt,
            matrix,
            flags,
            border_mode=border,
            border_value=border_value,
            stream=stream,
        )
    dst = cvcuda.Tensor(
        sized_shape(inpt.layout, inpt.shape, out_h, out_w),
        inpt.dtype,
        inpt.layout,
    )
    return cvcuda.warp_affine_into(
        dst,
        inpt,
        matrix,
        flags,
        border_mode=border,
        border_value=border_value,
        stream=stream,
    )


def affine(
    inpt: cvcuda.Tensor,
    angle: float,
    translate: _Vector,
    scale: float,
    shear: int | float | _Vector,
    interpolation: _Interpolation = InterpolationMode.NEAREST,
    fill: _Fill = None,
    center: _Vector | None = None,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Apply an affine transformation to image(s).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        angle (float): Rotation in degrees.
        translate (list[int]): Horizontal / vertical translation in pixels.
        scale (float): Isotropic scale factor.
        shear (float | list[float]): Shear angle(s) in degrees (``x`` or
            ``[x, y]``).
        interpolation (InterpolationMode | str | int, optional): Interpolation
            mode. Supports nearest and bilinear. Defaults to nearest.
        fill (int | float | list[float], optional): Constant fill for exposed
            pixels.
        center (list[float], optional): Center of the transform, origin top-left;
            defaults to the image center.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Transformed image(s).

    Raises:
        ValueError: If ``interpolation`` is unsupported or ``scale`` is not positive.

    """
    if scale <= 0:
        raise ValueError("Argument scale should be positive")
    # torchvision distinguishes the two one-value spellings: a bare number shears
    # x only, but a one-element sequence shears both axes by that angle.
    if isinstance(shear, (int, float)):
        shear = [float(shear), 0.0]
    elif len(shear) == 1:
        shear = [float(shear[0]), float(shear[0])]
    else:
        shear = [float(shear[0]), float(shear[1])]
    _, h, w, _ = dims(inpt)
    cx = _pixel_center(None if center is None else center[0], w)
    cy = _pixel_center(None if center is None else center[1], h)
    matrix = _inverse_affine_matrix(
        (cx, cy), angle, (float(translate[0]), float(translate[1])), scale, shear
    )
    flags = int(interp_from(interpolation, WARP_INTERP_MODES, ValueError)) | int(
        cvcuda.Interp.WARP_INVERSE_MAP
    )
    value = _fill_value(fill)
    return cvcuda.warp_affine(
        inpt,
        matrix,
        flags,
        border_mode=_fill_border(fill),
        border_value=value,
        stream=stream,
    )


def _perspective_coeffs(
    startpoints: _Points, endpoints: _Points
) -> NDArray[np.float64]:
    """torchvision's perspective coefficients ``(a..h)`` solved in float64.

    Reproduces ``torchvision.transforms.functional._get_perspective_coeffs``: the
    8x8 linear system mapping each endpoint (output corner) to its startpoint
    (input corner).
    """
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError(
            f"Please provide exactly four corners, got {len(startpoints)} startpoints "
            f"and {len(endpoints)} endpoints."
        )
    startpoints = np.asarray(startpoints, dtype=np.float64)
    endpoints = np.asarray(endpoints, dtype=np.float64)
    if (
        startpoints.shape != (4, 2)
        or endpoints.shape != (4, 2)
        or not np.isfinite(startpoints).all()
        or not np.isfinite(endpoints).all()
    ):
        raise ValueError(
            "Please provide exactly four corners with two finite coordinates each"
        )
    a_matrix = np.zeros((8, 8), dtype=np.float64)
    b_vector = np.zeros(8, dtype=np.float64)
    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i] = [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]]
        a_matrix[2 * i + 1] = [
            0,
            0,
            0,
            p1[0],
            p1[1],
            1,
            -p2[1] * p1[0],
            -p2[1] * p1[1],
        ]
        b_vector[2 * i] = p2[0]
        b_vector[2 * i + 1] = p2[1]
    return np.linalg.solve(a_matrix, b_vector)


# Half-pixel change of basis between torchvision's grid and CV-CUDA's warp centers.
_HALF_POST: NDArray[np.float64] = np.array(
    [[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], dtype=np.float64
)
_HALF_PRE: NDArray[np.float64] = np.array(
    [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64
)


def perspective(
    inpt: cvcuda.Tensor,
    startpoints: _Points | None,
    endpoints: _Points | None,
    interpolation: _Interpolation = InterpolationMode.BILINEAR,
    fill: _Fill = None,
    coefficients: Sequence[float] | None = None,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Apply a perspective transformation to image(s).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        startpoints (list[list[int]]): Four source corners
            (top-left, top-right, bottom-right, bottom-left).
        endpoints (list[list[int]]): Four destination corners, same order.
        interpolation (InterpolationMode | str | int, optional): Interpolation
            mode. Supports nearest and bilinear. Defaults to bilinear.
        fill (int | float | list[float], optional): Constant fill for exposed
            pixels.
        coefficients (list[float], optional): Precomputed ``(a..h)`` coefficients;
            overrides ``startpoints``/``endpoints``.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Transformed image(s).

    Raises:
        ValueError: If ``interpolation`` is unsupported, the points do not describe
            four coordinate pairs, or ``coefficients`` does not contain eight values.

    """
    if coefficients is None:
        if startpoints is None or endpoints is None:
            raise ValueError(
                "Either the startpoints/endpoints or the coefficients must have non `None` values."
            )
        coefficients = _perspective_coeffs(startpoints, endpoints)
    else:
        try:
            coefficients = np.asarray(coefficients, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Argument coefficients should have 8 float values"
            ) from error
        if coefficients.shape != (8,) or not np.isfinite(coefficients).all():
            raise ValueError("Argument coefficients should have 8 float values")
    a, b, c, d, e, f, g, hh = coefficients
    coeff_matrix = np.array([[a, b, c], [d, e, f], [g, hh, 1.0]], dtype=np.float64)
    matrix = _HALF_POST @ coeff_matrix @ _HALF_PRE
    matrix = (matrix / matrix[2, 2]).astype(np.float32)
    flags = int(interp_from(interpolation, WARP_INTERP_MODES, ValueError)) | int(
        cvcuda.Interp.WARP_INVERSE_MAP
    )
    value = _fill_value(fill)
    return cvcuda.warp_perspective(
        inpt,
        matrix,
        flags,
        border_mode=_fill_border(fill),
        border_value=value,
        stream=stream,
    )


def _linspace(size: int) -> NDArray[np.float32]:
    """``torch.linspace((1 - size) / size, (size - 1) / size, size)`` in float32,
    bit for bit: Torch steps from the start for the first half and from the end
    for the second."""
    start = np.float32((1 - size) / size)
    end = np.float32((size - 1) / size)
    step = np.float64((end - start) / np.float32(max(size - 1, 1)))
    index = np.arange(size)
    return np.where(
        index < size // 2, start + step * index, end - step * (size - 1 - index)
    ).astype(np.float32)


def _device_tensor(array: NDArray[np.float32]) -> cvcuda.Tensor:
    """Upload a float32 ``(H, W)`` array as an ``NCHW`` ``(1, 1, H, W)`` tensor."""
    return cvcuda.as_tensor(cvcuda.Image(array[..., None], cvcuda.Format.F32))


@functools.lru_cache(maxsize=16)
def _identity_grid(
    height: int, width: int, device: int
) -> tuple[cvcuda.Tensor, cvcuda.Tensor]:
    """torchvision's identity sampling grid, negated and flattened to
    ``(1, 1, H, 2W)``, plus a unit scale, so ``cvcuda.normalize`` adds it to a
    displacement on the device. ``normalize`` takes 1, 3 or 4 channels, hence
    the ``(x, y)`` pairs fused into the row."""
    grid = np.empty((height, width, 2), np.float32)
    grid[..., 0] = -_linspace(width)
    grid[..., 1] = -_linspace(height)[:, None]
    return (
        _device_tensor(grid.reshape(height, 2 * width)),
        _device_tensor(np.ones((1, 1), np.float32)),
    )


def elastic(
    inpt: cvcuda.Tensor,
    displacement: cvcuda.Tensor,
    interpolation: _Interpolation = InterpolationMode.BILINEAR,
    fill: _Fill = None,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Apply an elastic deformation to image(s).

    As in torchvision, ``displacement`` is relative: it is added to the
    identity sampling grid, in normalized ``[-1, 1]`` coordinates with
    ``align_corners=False``. One displacement applies to every image of a batch.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        displacement (cvcuda.Tensor): Relative normalized displacement of shape
            ``(1, H, W, 2)``, last axis ``(dx, dy)``, in ``NHWC`` layout.
        interpolation (InterpolationMode | str | int, optional): Interpolation
            mode. Defaults to bilinear.
        fill (int | float | list[float], optional): Constant fill for exposed
            pixels.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Deformed image(s).

    Raises:
        ValueError: If ``displacement`` is not of shape ``(1, H, W, 2)``.

    """
    interp = interp_from(interpolation)
    value = _fill_value(fill)

    _, height, width, _ = dims(inpt)
    expected = (1, height, width, 2)
    if tuple(displacement.shape) != expected:
        raise ValueError(
            f"Argument displacement shape should be {expected}, "
            f"but given {tuple(displacement.shape)}"
        )
    # remap wants a float32 map whatever the image dtype; a caller handing us a
    # half-precision grid would otherwise be rejected deep inside the operator.
    if displacement.dtype != cvcuda.Type.F32:
        displacement = cvcuda.convertto(displacement, cvcuda.Type.F32, stream=stream)

    identity, one = _identity_grid(height, width, inpt.cuda().__dlpack_device__()[1])
    flat = cvcuda.reshape(displacement, (1, 1, height, 2 * width), "NCHW")
    grid = cvcuda.normalize(flat, identity, one, stream=stream)
    # remap requires the map's batch rank to match the input.
    if "N" in str(inpt.layout):
        grid = cvcuda.reshape(grid, expected, "NHWC")
    else:
        grid = cvcuda.reshape(grid, expected[1:], "HWC")

    return cvcuda.remap(
        inpt,
        grid,
        src_interp=interp,
        map_interp=cvcuda.Interp.NEAREST,
        map_type=cvcuda.Remap.ABSOLUTE_NORMALIZED,
        align_corners=False,
        border=_fill_border(fill),
        border_value=value,
        stream=stream,
    )


def _crop_size(size: _Size) -> tuple[int, int]:
    """Normalize a five/ten-crop ``size`` into ``(crop_height, crop_width)``."""
    if isinstance(size, int):
        return size, size
    if len(size) == 1:
        return size[0], size[0]
    return size[0], size[1]


def _five_crop_rects(inpt: cvcuda.Tensor, size: _Size) -> list[cvcuda.RectI]:
    """The five crop rectangles -- four corners then the center -- as
    ``cvcuda.RectI`` for an ``inpt`` of the given ``size``. torchvision's center
    anchor is ``round((extent - crop) / 2)``."""
    crop_h, crop_w = _crop_size(size)
    _, h, w, _ = dims(inpt)
    corners = [
        (0, 0),
        (0, w - crop_w),
        (h - crop_h, 0),
        (h - crop_h, w - crop_w),
        (int(round((h - crop_h) / 2.0)), int(round((w - crop_w) / 2.0))),
    ]
    return [
        cvcuda.RectI(x=left, y=top, width=crop_w, height=crop_h)
        for top, left in corners
    ]


def five_crop(
    inpt: cvcuda.Tensor,
    size: _Size,
    *,
    stream: cvcuda.Stream | None = None,
) -> _FiveCrop:
    """Crop image(s) into the four corners and the center (five crops).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        size (int | list[int]): Crop ``[height, width]`` (a scalar crops a
            square).
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        tuple[cvcuda.Tensor, ...]: Top-left, top-right, bottom-left,
            bottom-right, and center crops, in that order.

    """
    rects = _five_crop_rects(inpt, size)
    return tuple(cvcuda.customcrop(inpt, rect, stream=stream) for rect in rects)


def ten_crop(
    inpt: cvcuda.Tensor,
    size: _Size,
    vertical_flip: bool = False,
    *,
    stream: cvcuda.Stream | None = None,
) -> _TenCrop:
    """Crop image(s) into five crops plus their flipped versions (ten crops).

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        size (int | list[int]): Crop ``[height, width]`` (a scalar crops a
            square).
        vertical_flip (bool, optional): Flip vertically instead of horizontally
            for the second five crops.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        tuple[cvcuda.Tensor, ...]: Five crops of the input followed by five crops
            of its flipped copy.

    """
    rects = _five_crop_rects(inpt, size)
    flipped = cvcuda.flip(inpt, flipCode=0 if vertical_flip else 1, stream=stream)
    return tuple(
        cvcuda.customcrop(source, rect, stream=stream)
        for source in (inpt, flipped)
        for rect in rects
    )


__all__: list[str] = [
    "affine",
    "center_crop",
    "crop",
    "elastic",
    "five_crop",
    "horizontal_flip",
    "pad",
    "perspective",
    "resize",
    "resized_crop",
    "rotate",
    "ten_crop",
    "vertical_flip",
]
