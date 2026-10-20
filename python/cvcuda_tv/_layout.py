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

"""Shared helpers for the cvcuda.tv shims: shape helpers and enum maps.

Every CV-CUDA operator behind the shims accepts interleaved (HWC/NHWC) and
planar (CHW/NCHW) image layouts natively, so the shims call the operators
directly on the caller's layout; unsupported layouts are rejected by the
C++ operator itself.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
import enum
from typing import TypeAlias

import cvcuda


class InterpolationMode(str, enum.Enum):
    """Drop-in mirror of ``torchvision.transforms.InterpolationMode``.

    Subclassing ``str`` lets the members reduce to plain string values, so the
    shims can use them as signature defaults without importing torchvision.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


_Interpolation: TypeAlias = InterpolationMode | enum.Enum | str | int
_Axes: TypeAlias = tuple[int, int, int]

# TensorLayout hashes and compares by its labels, so both key kinds are safe.
_STR_AXES: dict[str, _Axes] = {
    "HWC": (0, 1, 2),
    "NHWC": (1, 2, 3),
    "CHW": (1, 2, 0),
    "NCHW": (2, 3, 1),
}
_HWC_AXES: dict[str | cvcuda.TensorLayout, _Axes] = {
    **_STR_AXES,
    **{getattr(cvcuda.TensorLayout, key): axes for key, axes in _STR_AXES.items()},
}


def _axes(layout: str | cvcuda.TensorLayout) -> _Axes:
    axes = _HWC_AXES.get(layout)
    if axes is None:
        layout = str(layout)
        axes = (layout.index("H"), layout.index("W"), layout.index("C"))
    return axes


def dims(inpt: cvcuda.Tensor) -> tuple[cvcuda.TensorLayout, int, int, int]:
    """Return ``(layout, height, width, channels)`` for a tagged tensor.

    ``layout`` is the tensor's ``cvcuda.TensorLayout`` and is accepted anywhere
    a layout key is taken (e.g. :func:`sized_shape`).
    """
    layout = inpt.layout
    shape = inpt.shape
    h, w, c = _axes(layout)
    return (layout, shape[h], shape[w], shape[c])


def sized_shape(
    layout: str | cvcuda.TensorLayout,
    shape: Sequence[int],
    out_h: int,
    out_w: int,
) -> tuple[int, ...]:
    """Copy ``shape`` with its H and W entries replaced by ``out_h``/``out_w``."""
    h, w, _ = _axes(layout)
    out = list(shape)
    out[h] = out_h
    out[w] = out_w
    return tuple(out)


INTERP_MAP: dict[str, cvcuda.Interp] = {
    "nearest": cvcuda.Interp.NEAREST,
    "nearest_exact": cvcuda.Interp.NEAREST,
    "bilinear": cvcuda.Interp.LINEAR,
    "bicubic": cvcuda.Interp.CUBIC,
    "box": cvcuda.Interp.BOX,
    "hamming": cvcuda.Interp.HAMMING,
    "lanczos": cvcuda.Interp.LANCZOS,
}

# Interpolation modes each operator family accepts, mirroring torchvision's v2
# functional on tensor inputs: resize/resized_crop take the resampling subset (and
# raise NotImplementedError otherwise), the warps take only nearest/bilinear (and
# raise ValueError otherwise).
RESIZE_INTERP_MODES: frozenset[str] = frozenset(
    {"nearest", "nearest_exact", "bilinear", "bicubic"}
)
WARP_INTERP_MODES: frozenset[str] = frozenset({"nearest", "bilinear"})

# torchvision pad padding_mode -> cvcuda.Border; reflect=REFLECT_101, symmetric=REFLECT.
BORDER_MAP: dict[str, cvcuda.Border] = {
    "constant": cvcuda.Border.CONSTANT,
    "edge": cvcuda.Border.REPLICATE,
    "replicate": cvcuda.Border.REPLICATE,
    "reflect": cvcuda.Border.REFLECT101,
    "symmetric": cvcuda.Border.REFLECT,
    "circular": cvcuda.Border.WRAP,
}


# Legacy PIL integer interpolation constants; an unknown int raises instead of coercing.
_PIL_INT_INTERP: dict[int, str] = {
    0: "nearest",
    1: "lanczos",
    2: "bilinear",
    3: "bicubic",
    4: "box",
    5: "hamming",
}


def _interp_key(mode: _Interpolation) -> str:
    """Normalize ``mode`` to an INTERP_MAP key without validating it."""
    key = getattr(mode, "value", mode)
    if isinstance(key, int) and not isinstance(key, bool):
        key = _PIL_INT_INTERP.get(key, key)
    return str(key).lower().replace("-", "_")


def interp_from(
    mode: _Interpolation,
    supported: Collection[str] | None = None,
    unsupported: type[Exception] = NotImplementedError,
) -> cvcuda.Interp:
    """Resolve a torchvision interpolation argument to a ``cvcuda.Interp``.

    ``mode`` may be an ``InterpolationMode`` enum, its string value (``.value`` is
    hyphenated, e.g. ``nearest-exact``), or a legacy PIL int constant. When
    ``supported`` is given, a mode outside it raises ``unsupported`` -- the same
    exception torchvision raises for that operator.
    """
    key = _interp_key(mode)
    if key not in INTERP_MAP or (supported is not None and key not in supported):
        # TorchVision's own suite matches on "got <mode>", the phrasing PyTorch core
        # uses when it rejects an interpolation, so keep that substring intact. The
        # mode is spelled as torchvision spells it, hyphenated rather than
        # underscored, since that is what the caller passed.
        spelled = key.replace("_", "-")
        raise unsupported(
            f"cvcuda.tv: {key!r} interpolation is not supported by this operator "
            f"(got {spelled})"
        )
    return INTERP_MAP[key]
