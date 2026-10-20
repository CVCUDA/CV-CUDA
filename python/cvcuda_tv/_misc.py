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

"""Miscellaneous image operations for :mod:`cvcuda.tv`."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import operator
from typing import TypeAlias

import numpy as np

import cvcuda

from ._layout import dims

_ScalarOrSequence: TypeAlias = int | float | Sequence[float]
_KernelSize: TypeAlias = int | Sequence[int]


def normalize(
    inpt: cvcuda.Tensor,
    mean: _ScalarOrSequence,
    std: _ScalarOrSequence,
    inplace: bool = False,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Normalize image(s) with per-channel mean and standard deviation.

    Computes ``(inpt - mean) / std``. A one-element ``mean`` or ``std`` is
    applied to every channel. Float16, float32, and float64 inputs are supported.

    Args:
        inpt (cvcuda.Tensor): Float16, float32, or float64 input image(s).
        mean (float | list[float]): Per-channel means.
        std (float | list[float]): Nonzero per-channel standard deviations.
        inplace (bool, optional): Write the result into ``inpt`` and return it.
            Defaults to ``False``.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Normalized image(s).

    Raises:
        TypeError: If ``inpt`` is not floating point.
        ValueError: If any value in ``std`` is zero.

    """
    dtype = inpt.dtype
    if dtype not in (cvcuda.Type.F16, cvcuda.Type.F32, cvcuda.Type.F64):
        raise TypeError(f"Input tensor should be a float tensor. Got {dtype}.")

    mean_list = (
        [float(mean)] if isinstance(mean, (int, float)) else [float(m) for m in mean]
    )
    std_list = (
        [float(std)] if isinstance(std, (int, float)) else [float(s) for s in std]
    )
    if not all(std_list):
        raise ValueError("std evaluated to zero, leading to division by zero.")

    # torchvision divides by std; cvcuda computes (in - base) * scale, so pass the reciprocal of std
    # with flags=0 (scaling, not stddev). This is a multiply-by-reciprocal vs divide, ~1 ULP apart.
    scale = [1.0 / s for s in std_list]

    if inplace:
        return cvcuda.normalize_into(inpt, inpt, mean_list, scale, stream=stream)
    return cvcuda.normalize(inpt, mean_list, scale, stream=stream)


def gaussian_blur(
    inpt: cvcuda.Tensor,
    kernel_size: _KernelSize,
    sigma: _ScalarOrSequence | None = None,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Blur image(s) with a Gaussian kernel.

    The image boundary is reflected before filtering. When ``sigma`` is omitted,
    each value is derived from its kernel size as ``kernel_size * 0.15 + 0.35``.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        kernel_size (int | list[int]): Gaussian kernel size ``[kw, kh]`` (an int is
            broadcast to both axes); must be odd and positive.
        sigma (float | list[float], optional): Gaussian standard deviation(s); when
            omitted it is derived from ``kernel_size``.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Blurred image(s).

    """
    if isinstance(kernel_size, int):
        ksize = [kernel_size, kernel_size]
    else:
        try:
            ksize = list(kernel_size)
        except TypeError:
            ksize = [kernel_size, kernel_size]
    if len(ksize) != 2 or any(isinstance(k, bool) for k in ksize):
        raise ValueError("kernel_size must be an integer or a sequence of two integers")
    try:
        ksize = [operator.index(k) for k in ksize]
    except TypeError as exc:
        raise ValueError(
            "kernel_size must be an integer or a sequence of two integers"
        ) from exc
    if any(k <= 0 or k % 2 == 0 for k in ksize):
        raise ValueError(f"kernel_size must contain odd positive integers. Got {ksize}")
    sig = _gaussian_sigma(sigma, ksize)

    return cvcuda.gaussian(
        inpt,
        (int(ksize[0]), int(ksize[1])),
        (float(sig[0]), float(sig[1])),
        border=cvcuda.Border.REFLECT101,
        stream=stream,
    )


def _gaussian_sigma(sigma, ksize):
    if sigma is None:
        sig = [k * 0.15 + 0.35 for k in ksize]
    elif isinstance(sigma, (list, tuple)):
        sig = list(sigma)
        if len(sig) == 1:
            sig = [sig[0], sig[0]]
        elif len(sig) != 2:
            raise ValueError(
                f"If sigma is a sequence, its length should be 2. Got {len(sig)}"
            )
    elif isinstance(sigma, (int, float)):
        sig = [float(sigma), float(sigma)]
    else:
        raise TypeError(
            f"sigma should be either float or sequence of floats. Got {type(sigma)}"
        )
    if any(s <= 0.0 for s in sig):
        raise ValueError(f"sigma should have positive values. Got {sig}")
    return sig


def gaussian_noise(
    inpt: cvcuda.Tensor,
    mean: float = 0.0,
    sigma: float = 0.1,
    clip: bool = True,
    *,
    seed: int | None = None,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Add independent Gaussian noise to every image element.

    ``mean`` and ``sigma`` use normalized image units for uint8, float16, and
    float32 inputs. Reusing an explicit ``seed`` reproduces the same output;
    omitting it advances the random sequence.

    Args:
        inpt (cvcuda.Tensor): Input image(s), uint8, float16, or float32 with 1 or
            3 channels.
        mean (float): Mean of the sampled normal distribution.
        sigma (float): Non-negative standard deviation of the distribution.
        clip (bool): Clamp to [0, 255] for uint8 or [0, 1] for floating point.
            With False, uint8 wraps and floating-point output remains unbounded.
        seed (int, optional): Non-negative 64-bit per-call seed. Keyword-only.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: Image(s) with added noise.

    Raises:
        TypeError: If ``seed`` is not an integer or ``None``.
        ValueError: If ``sigma`` is negative, ``seed`` is outside the uint64
            range, or the input dtype or channel count is unsupported.

    """
    mean = float(mean)
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"sigma shouldn't be negative. Got {sigma}")

    seed_value = None
    if seed is not None:
        if isinstance(seed, bool):
            raise TypeError("seed must be a non-negative 64-bit integer or None")
        try:
            seed_value = operator.index(seed)
        except TypeError as exc:
            raise TypeError(
                "seed must be a non-negative 64-bit integer or None"
            ) from exc
        if not 0 <= seed_value <= (1 << 64) - 1:
            raise ValueError("seed must be in the range [0, 2**64 - 1]")

    _, _, _, channels = dims(inpt)
    if channels not in (1, 3):
        raise ValueError(
            f"Input tensor is expected to have 1 or 3 channels, got {channels}"
        )

    if inpt.dtype == cvcuda.Type.U8:
        scale = 255.0
    elif inpt.dtype in (cvcuda.Type.F16, cvcuda.Type.F32):
        scale = 1.0
    else:
        raise ValueError(
            "Input tensor is expected to be in uint8, float16, or float32 dtype, "
            f"got dtype={inpt.dtype}"
        )

    return cvcuda.gaussiannoise(
        inpt,
        mean * scale,
        sigma * scale,
        True,
        seed=seed_value,
        clip=clip,
        stream=stream,
    )


@functools.lru_cache(maxsize=None)
def _dst_args(dtype: object) -> tuple[np.dtype, cvcuda.Round]:
    """Per-``dtype`` ``(target type, round mode)``; ``cvcuda.Type`` is ``numpy.dtype``."""
    # numpy cannot parse a torch.dtype, but its name is the numpy name.
    name = str(dtype)
    dst = np.dtype(name.removeprefix("torch.") if name.startswith("torch.") else dtype)
    # torchvision truncates toward zero casting to int; request TRUNCATE for parity.
    return dst, cvcuda.Round.NEAREST if dst.kind == "f" else cvcuda.Round.TRUNCATE


@functools.lru_cache(maxsize=None)
def _scale_factor(src_np: np.dtype, dst_np: np.dtype) -> float:
    """torchvision's dtype-range remap factor, cached per (source, target) pair."""
    dst_float = dst_np.kind == "f"
    src_float = src_np.kind == "f"

    if src_float and dst_float:
        return 1.0
    if not src_float and dst_float:
        return 1.0 / float(np.iinfo(src_np).max)
    if src_float and not dst_float:
        # float -> int: torchvision scales by (max + 1 - eps) with eps = 1e-3.
        return float(np.iinfo(dst_np).max) + 1.0 - 1e-3
    src_bits = int(np.iinfo(src_np).max).bit_length()
    dst_bits = int(np.iinfo(dst_np).max).bit_length()
    return 2.0 ** (dst_bits - src_bits)


def to_dtype(
    inpt: cvcuda.Tensor,
    dtype: object = np.float32,
    scale: bool = False,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Convert image(s) to a target dtype, optionally rescaling values.

    With ``scale=False`` this is a plain dtype cast. With ``scale=True``, integer
    and floating-point ranges are remapped, for example uint8 ``[0, 255]`` to
    float ``[0, 1]``. ``dtype`` may be a NumPy dtype, a ``torch.dtype``, or a
    dtype name.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        dtype (numpy.dtype | torch.dtype | str): Target dtype.
        scale (bool): When True, rescale values across the dtype ranges.
        stream (cvcuda.Stream, optional): CUDA stream. Defaults to the current
            stream.

    Returns:
        cvcuda.Tensor: ``inpt`` converted to the target dtype.

    Raises:
        TypeError: If ``dtype`` does not resolve to a supported CV-CUDA type.

    """
    dst_type, round_mode = _dst_args(dtype)
    src_type = inpt.dtype
    if src_type == dst_type:
        return inpt

    if not scale:
        return cvcuda.convertto(inpt, dst_type, round=round_mode, stream=stream)

    factor = _scale_factor(src_type, dst_type)
    return cvcuda.convertto(inpt, dst_type, factor, round=round_mode, stream=stream)


__all__: list[str] = ["gaussian_blur", "gaussian_noise", "normalize", "to_dtype"]
