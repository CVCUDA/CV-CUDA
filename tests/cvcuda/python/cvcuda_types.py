# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

from typing import TYPE_CHECKING

import cvcuda
import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if TYPE_CHECKING:
    from collections.abc import Callable


# CV-CUDA <-> CV-CUDA
# ===================
# maps cvcuda.Format to a cvcuda.Type
FORMAT_TO_TYPE: dict[cvcuda.Format, cvcuda.Type] = {
    # BGR
    cvcuda.Format.BGR8: cvcuda.Type.U8,
    cvcuda.Format.BGR8p: cvcuda.Type.U8,
    cvcuda.Format.BGRA8: cvcuda.Type.U8,
    cvcuda.Format.BGRA8p: cvcuda.Type.U8,
    cvcuda.Format.BGRf16: cvcuda.Type.F16,
    cvcuda.Format.BGRf16p: cvcuda.Type.F16,
    cvcuda.Format.BGRAf16: cvcuda.Type.F16,
    cvcuda.Format.BGRAf16p: cvcuda.Type.F16,
    cvcuda.Format.BGRf32: cvcuda.Type.F32,
    cvcuda.Format.BGRf32p: cvcuda.Type.F32,
    cvcuda.Format.BGRAf32: cvcuda.Type.F32,
    cvcuda.Format.BGRAf32p: cvcuda.Type.F32,
    # RGB
    cvcuda.Format.RGB8: cvcuda.Type.U8,
    cvcuda.Format.RGB8p: cvcuda.Type.U8,
    cvcuda.Format.RGBA8: cvcuda.Type.U8,
    cvcuda.Format.RGBA8p: cvcuda.Type.U8,
    cvcuda.Format.RGBf16: cvcuda.Type.F16,
    cvcuda.Format.RGBf16p: cvcuda.Type.F16,
    cvcuda.Format.RGBAf16: cvcuda.Type.F16,
    cvcuda.Format.RGBAf16p: cvcuda.Type.F16,
    cvcuda.Format.RGBf32: cvcuda.Type.F32,
    cvcuda.Format.RGBf32p: cvcuda.Type.F32,
    cvcuda.Format.RGBAf32: cvcuda.Type.F32,
    cvcuda.Format.RGBAf32p: cvcuda.Type.F32,
    cvcuda.Format.RGB8_1U_U8: cvcuda.Type.U8,
    cvcuda.Format.RGB8_3D_F32: cvcuda.Type.F32,
    cvcuda.Format.RGB8_7U_U8: cvcuda.Type.U8,
    cvcuda.Format.RGBA8_3POS3D_U32: cvcuda.Type.U32,
    cvcuda.Format.RGBA8_3U_U16: cvcuda.Type.U16,
    cvcuda.Format.RGBA8_UNASSOCIATED_ALPHA: cvcuda.Type.U8,
    # CMYK
    cvcuda.Format.CMYK8: cvcuda.Type.U8,
    # HSV
    cvcuda.Format.HSV8: cvcuda.Type.U8,
    # NV12
    cvcuda.Format.NV12: cvcuda.Type.U8,
    cvcuda.Format.NV12_ER: cvcuda.Type.U8,
    # cvcuda.Format.NV12_BL: cvcuda.Type.U8,
    # cvcuda.Format.NV12_ER_BL: cvcuda.Type.U8,
    # NV24
    cvcuda.Format.NV24: cvcuda.Type.U8,
    cvcuda.Format.NV24_ER: cvcuda.Type.U8,
    # cvcuda.Format.NV24_BL: cvcuda.Type.U8,
    # cvcuda.Format.NV24_ER_BL: cvcuda.Type.U8,
    # UYVY
    cvcuda.Format.UYVY: cvcuda.Type.U8,
    cvcuda.Format.UYVY_ER: cvcuda.Type.U8,
    # cvcuda.Format.UYVY_BL: cvcuda.Type.U8,
    # cvcuda.Format.UYVY_ER_BL: cvcuda.Type.U8,
    # VYUY
    cvcuda.Format.VYUY: cvcuda.Type.U8,
    cvcuda.Format.VYUY_ER: cvcuda.Type.U8,
    # cvcuda.Format.VYUY_BL: cvcuda.Type.U8,
    # cvcuda.Format.VYUY_ER_BL: cvcuda.Type.U8,
    # Y16
    cvcuda.Format.Y16: cvcuda.Type.U16,
    cvcuda.Format.Y16_ER: cvcuda.Type.U16,
    # cvcuda.Format.Y16_BL: cvcuda.Type.U16,
    # cvcuda.Format.Y16_ER_BL: cvcuda.Type.U16,
    # Y8
    cvcuda.Format.Y8: cvcuda.Type.U8,
    cvcuda.Format.Y8_ER: cvcuda.Type.U8,
    # cvcuda.Format.Y8_BL: cvcuda.Type.U8,
    # cvcuda.Format.Y8_ER_BL: cvcuda.Type.U8,
    # YCCK
    cvcuda.Format.YCCK8: cvcuda.Type.U8,
    # YUV
    cvcuda.Format.YUV8p: cvcuda.Type.U8,
    cvcuda.Format.YUV8p_ER: cvcuda.Type.U8,
    # YUYV
    cvcuda.Format.YUYV: cvcuda.Type.U8,
    cvcuda.Format.YUYV_ER: cvcuda.Type.U8,
    # cvcuda.Format.YUYV_BL: cvcuda.Type.U8,
    # cvcuda.Format.YUYV_ER_BL: cvcuda.Type.U8,
    # Scalar
    cvcuda.Format.U8: cvcuda.Type.U8,
    # cvcuda.Format.U8_BL: cvcuda.Type.U8,
    cvcuda.Format.U16: cvcuda.Type.U16,
    cvcuda.Format.U32: cvcuda.Type.U32,
    cvcuda.Format.S8: cvcuda.Type.S8,
    cvcuda.Format.S16: cvcuda.Type.S16,
    # cvcuda.Format.S16_BL: cvcuda.Type.S16,
    cvcuda.Format.S32: cvcuda.Type.S32,
    cvcuda.Format.F16: cvcuda.Type.F16,
    cvcuda.Format.F32: cvcuda.Type.F32,
    cvcuda.Format.F64: cvcuda.Type.F64,
    cvcuda.Format.C64: cvcuda.Type.C64,
    cvcuda.Format.C128: cvcuda.Type.C128,
    # Scalar multi-plane
    cvcuda.Format._2F16: cvcuda.Type.F16,
    cvcuda.Format._2F32: cvcuda.Type.F32,
    cvcuda.Format._2S16: cvcuda.Type.S16,
    # cvcuda.Format._2S16_BL: cvcuda.Type.S16,
    cvcuda.Format._2C64: cvcuda.Type.C64,
    cvcuda.Format._2C128: cvcuda.Type.C128,
}
# all cvcuda.Format values
FORMATS: list[cvcuda.Format] = list(FORMAT_TO_TYPE.keys())
FORMAT_SET: set[cvcuda.Format] = set(FORMATS)
SCALAR_FORMATS: list[cvcuda.Format] = [
    cvcuda.Format.U8,
    cvcuda.Format.U16,
    cvcuda.Format.U32,
    cvcuda.Format.S8,
    cvcuda.Format.S16,
    cvcuda.Format.S32,
    cvcuda.Format.F16,
    cvcuda.Format.F32,
    cvcuda.Format.F64,
    cvcuda.Format.C64,
    cvcuda.Format.C128,
]
SCALAR_FORMATS_SET: set[cvcuda.Format] = set(SCALAR_FORMATS)
SCALAR_TYPES: list[cvcuda.Type] = [FORMAT_TO_TYPE[f] for f in SCALAR_FORMATS]
TYPE_TO_FORMAT: dict[cvcuda.Type, cvcuda.Format] = {
    FORMAT_TO_TYPE[f]: f for f in SCALAR_FORMATS
}
SCALAR_TYPES_SET: set[cvcuda.Type] = set(SCALAR_TYPES)

# Packed types (multi-channel values packed into single element)
PACKED_TYPES: set[cvcuda.Type] = {
    # 2-channel packed
    cvcuda.Type._2U8,
    cvcuda.Type._2S8,
    cvcuda.Type._2U16,
    cvcuda.Type._2S16,
    cvcuda.Type._2U32,
    cvcuda.Type._2S32,
    cvcuda.Type._2U64,
    cvcuda.Type._2S64,
    cvcuda.Type._2F16,
    cvcuda.Type._2F32,
    cvcuda.Type._2F64,
    cvcuda.Type._2C64,
    cvcuda.Type._2C128,
    # 3-channel packed
    cvcuda.Type._3U8,
    cvcuda.Type._3S8,
    cvcuda.Type._3U16,
    cvcuda.Type._3S16,
    cvcuda.Type._3U32,
    cvcuda.Type._3S32,
    cvcuda.Type._3U64,
    cvcuda.Type._3S64,
    cvcuda.Type._3F16,
    cvcuda.Type._3F32,
    cvcuda.Type._3F64,
    cvcuda.Type._3C64,
    # 4-channel packed
    cvcuda.Type._4U8,
    cvcuda.Type._4S8,
    cvcuda.Type._4U16,
    cvcuda.Type._4S16,
    cvcuda.Type._4U32,
    cvcuda.Type._4S32,
    cvcuda.Type._4U64,
    cvcuda.Type._4S64,
    cvcuda.Type._4F16,
    cvcuda.Type._4F32,
    cvcuda.Type._4F64,
    cvcuda.Type._4C64,
}

# Packed formats (only a limited set exist in cvcuda.Format)
PACKED_FORMATS: set[cvcuda.Format] = {
    cvcuda.Format._2F16,
    cvcuda.Format._2F32,
    cvcuda.Format._2S16,
    cvcuda.Format._2C64,
    cvcuda.Format._2C128,
}

# Mapping from (scalar_type, channels) to packed type
SCALAR_TO_PACKED_TYPE: dict[tuple[cvcuda.Type, int], cvcuda.Type] = {
    # 1 channel (identity)
    (cvcuda.Type.U8, 1): cvcuda.Type.U8,
    (cvcuda.Type.S8, 1): cvcuda.Type.S8,
    (cvcuda.Type.U16, 1): cvcuda.Type.U16,
    (cvcuda.Type.S16, 1): cvcuda.Type.S16,
    (cvcuda.Type.U32, 1): cvcuda.Type.U32,
    (cvcuda.Type.S32, 1): cvcuda.Type.S32,
    (cvcuda.Type.U64, 1): cvcuda.Type.U64,
    (cvcuda.Type.S64, 1): cvcuda.Type.S64,
    (cvcuda.Type.F16, 1): cvcuda.Type.F16,
    (cvcuda.Type.F32, 1): cvcuda.Type.F32,
    (cvcuda.Type.F64, 1): cvcuda.Type.F64,
    (cvcuda.Type.C64, 1): cvcuda.Type.C64,
    (cvcuda.Type.C128, 1): cvcuda.Type.C128,
    # 2 channels
    (cvcuda.Type.U8, 2): cvcuda.Type._2U8,
    (cvcuda.Type.S8, 2): cvcuda.Type._2S8,
    (cvcuda.Type.U16, 2): cvcuda.Type._2U16,
    (cvcuda.Type.S16, 2): cvcuda.Type._2S16,
    (cvcuda.Type.U32, 2): cvcuda.Type._2U32,
    (cvcuda.Type.S32, 2): cvcuda.Type._2S32,
    (cvcuda.Type.U64, 2): cvcuda.Type._2U64,
    (cvcuda.Type.S64, 2): cvcuda.Type._2S64,
    (cvcuda.Type.F16, 2): cvcuda.Type._2F16,
    (cvcuda.Type.F32, 2): cvcuda.Type._2F32,
    (cvcuda.Type.F64, 2): cvcuda.Type._2F64,
    (cvcuda.Type.C64, 2): cvcuda.Type._2C64,
    (cvcuda.Type.C128, 2): cvcuda.Type._2C128,
    # 3 channels
    (cvcuda.Type.U8, 3): cvcuda.Type._3U8,
    (cvcuda.Type.S8, 3): cvcuda.Type._3S8,
    (cvcuda.Type.U16, 3): cvcuda.Type._3U16,
    (cvcuda.Type.S16, 3): cvcuda.Type._3S16,
    (cvcuda.Type.U32, 3): cvcuda.Type._3U32,
    (cvcuda.Type.S32, 3): cvcuda.Type._3S32,
    (cvcuda.Type.U64, 3): cvcuda.Type._3U64,
    (cvcuda.Type.S64, 3): cvcuda.Type._3S64,
    (cvcuda.Type.F16, 3): cvcuda.Type._3F16,
    (cvcuda.Type.F32, 3): cvcuda.Type._3F32,
    (cvcuda.Type.F64, 3): cvcuda.Type._3F64,
    (cvcuda.Type.C64, 3): cvcuda.Type._3C64,
    # 4 channels
    (cvcuda.Type.U8, 4): cvcuda.Type._4U8,
    (cvcuda.Type.S8, 4): cvcuda.Type._4S8,
    (cvcuda.Type.U16, 4): cvcuda.Type._4U16,
    (cvcuda.Type.S16, 4): cvcuda.Type._4S16,
    (cvcuda.Type.U32, 4): cvcuda.Type._4U32,
    (cvcuda.Type.S32, 4): cvcuda.Type._4S32,
    (cvcuda.Type.U64, 4): cvcuda.Type._4U64,
    (cvcuda.Type.S64, 4): cvcuda.Type._4S64,
    (cvcuda.Type.F16, 4): cvcuda.Type._4F16,
    (cvcuda.Type.F32, 4): cvcuda.Type._4F32,
    (cvcuda.Type.F64, 4): cvcuda.Type._4F64,
    (cvcuda.Type.C64, 4): cvcuda.Type._4C64,
}


def is_packed_type(dtype: cvcuda.Type) -> bool:
    return dtype in PACKED_TYPES


def get_packed_type(
    scalar_type: cvcuda.Type, channels: int, *, always_get_scalar: bool = False
) -> cvcuda.Type | None:
    if scalar_type not in SCALAR_TYPES_SET:
        raise ValueError(f"Invalid scalar type: {scalar_type}")
    if channels not in {1, 2, 3, 4} and not always_get_scalar:
        raise ValueError(f"Invalid channels: {channels}")
    packed_type = SCALAR_TO_PACKED_TYPE.get((scalar_type, channels))
    if packed_type is None and always_get_scalar:
        return SCALAR_TO_PACKED_TYPE[(scalar_type, 1)]
    return packed_type


# CV-CUDA <-> NumPy
# =================
# maps cvcuda.Type to a np.dtype
TYPE_TO_NP_DTYPE: dict[cvcuda.Type, np.dtype] = {
    cvcuda.Type.U8: np.uint8,
    cvcuda.Type._2U8: np.dtype("2u1"),
    cvcuda.Type._3U8: np.dtype("3u1"),
    cvcuda.Type._4U8: np.dtype("4u1"),
    cvcuda.Type.S8: np.int8,
    cvcuda.Type._2S8: np.dtype("2i1"),
    cvcuda.Type._3S8: np.dtype("3i1"),
    cvcuda.Type._4S8: np.dtype("4i1"),
    cvcuda.Type.U16: np.uint16,
    cvcuda.Type._2U16: np.dtype("2u2"),
    cvcuda.Type._3U16: np.dtype("3u2"),
    cvcuda.Type._4U16: np.dtype("4u2"),
    cvcuda.Type.S16: np.int16,
    cvcuda.Type._2S16: np.dtype("2i2"),
    cvcuda.Type._3S16: np.dtype("3i2"),
    cvcuda.Type._4S16: np.dtype("4i2"),
    cvcuda.Type.U32: np.uint32,
    cvcuda.Type._2U32: np.dtype("2u4"),
    cvcuda.Type._3U32: np.dtype("3u4"),
    cvcuda.Type._4U32: np.dtype("4u4"),
    cvcuda.Type.S32: np.int32,
    cvcuda.Type._2S32: np.dtype("2i4"),
    cvcuda.Type._3S32: np.dtype("3i4"),
    cvcuda.Type._4S32: np.dtype("4i4"),
    cvcuda.Type.U64: np.uint64,
    cvcuda.Type._2U64: np.dtype("2u8"),
    cvcuda.Type._3U64: np.dtype("3u8"),
    cvcuda.Type._4U64: np.dtype("4u8"),
    cvcuda.Type.S64: np.int64,
    cvcuda.Type._2S64: np.dtype("2i8"),
    cvcuda.Type._3S64: np.dtype("3i8"),
    cvcuda.Type._4S64: np.dtype("4i8"),
    cvcuda.Type.F16: np.float16,
    cvcuda.Type._2F16: np.dtype("2e"),
    cvcuda.Type._3F16: np.dtype("3e"),
    cvcuda.Type._4F16: np.dtype("4e"),
    cvcuda.Type.F32: np.float32,
    cvcuda.Type._2F32: np.dtype("2f"),
    cvcuda.Type._3F32: np.dtype("3f"),
    cvcuda.Type._4F32: np.dtype("4f"),
    cvcuda.Type.F64: np.float64,
    cvcuda.Type._2F64: np.dtype("2d"),
    cvcuda.Type._3F64: np.dtype("3d"),
    cvcuda.Type._4F64: np.dtype("4d"),
    cvcuda.Type.C64: np.csingle,
    cvcuda.Type._2C64: np.dtype("2c8"),
    cvcuda.Type._3C64: np.dtype("3c8"),
    cvcuda.Type._4C64: np.dtype("4c8"),
    cvcuda.Type.C128: np.cdouble,
    cvcuda.Type._2C128: np.dtype("2c16"),
}
# maps np.dtype to a cvcuda.Type
NP_DTYPE_TO_TYPE: dict[np.dtype, cvcuda.Type] = {
    np.dtype(v): k for k, v in TYPE_TO_NP_DTYPE.items()
}
# all cvcuda.Type values
TYPES: list[cvcuda.Type] = list(TYPE_TO_NP_DTYPE.keys())
TYPE_SET: set[cvcuda.Type] = set(TYPES)
# all np.dtype values (with overlap to cvcuda.Type)
NP_DTYPES: list[np.dtype] = list(TYPE_TO_NP_DTYPE.values())
NP_DTYPE_SET: set[np.dtype] = set(NP_DTYPES)
# maps cvcuda.Format to a np.dtype
FORMAT_TO_NP_DTYPE: dict[cvcuda.Format, np.dtype] = {
    k: TYPE_TO_NP_DTYPE[v] for k, v in FORMAT_TO_TYPE.items()
}

# CV-CUDA <-> PyTorch (optional -- all tables are empty when torch is absent)
# ===========================================================================
if _HAS_TORCH:
    TYPE_TO_TORCH_DTYPE: dict = {
        cvcuda.Type.U8: torch.uint8,
        cvcuda.Type.S8: torch.int8,
        cvcuda.Type.S16: torch.int16,
        cvcuda.Type.S32: torch.int32,
        cvcuda.Type.S64: torch.int64,
        cvcuda.Type.F16: torch.float16,
        cvcuda.Type.F32: torch.float32,
        cvcuda.Type.F64: torch.float64,
        cvcuda.Type.C64: torch.complex64,
        cvcuda.Type.C128: torch.complex128,
    }
    # Older PyTorch versions (common on Jetson / aarch64) lack unsigned integer
    # dtypes, so every downstream dict
    # comprehension and lookup must guard with ``if t in TYPE_TO_TORCH_DTYPE``.
    if hasattr(torch, "uint16"):
        TYPE_TO_TORCH_DTYPE[cvcuda.Type.U16] = torch.uint16
        TYPE_TO_TORCH_DTYPE[cvcuda.Type.U32] = torch.uint32
        TYPE_TO_TORCH_DTYPE[cvcuda.Type.U64] = torch.uint64
    TORCH_DTYPE_TO_TYPE: dict = {v: k for k, v in TYPE_TO_TORCH_DTYPE.items()}
    TORCH_DTYPES: list = list(TORCH_DTYPE_TO_TYPE.keys())
    TORCH_DTYPE_SET: set = set(TORCH_DTYPES)
    FORMAT_TO_TORCH_DTYPE: dict = {
        k: TYPE_TO_TORCH_DTYPE[v]
        for k, v in FORMAT_TO_TYPE.items()
        if v in TYPE_TO_TORCH_DTYPE
    }
    NP_DTYPE_TO_TORCH_DTYPE: dict = {
        TYPE_TO_NP_DTYPE[k]: v for k, v in TYPE_TO_TORCH_DTYPE.items()
    }
    TORCH_DTYPE_TO_NP_DTYPE: dict = {v: k for k, v in NP_DTYPE_TO_TORCH_DTYPE.items()}
    TORCH_DTYPE_TO_FORMAT: dict = {
        TYPE_TO_TORCH_DTYPE[t]: TYPE_TO_FORMAT[t]
        for t in SCALAR_TYPES
        if t in TYPE_TO_TORCH_DTYPE
    }
else:
    TYPE_TO_TORCH_DTYPE = {}
    TORCH_DTYPE_TO_TYPE = {}
    TORCH_DTYPES = []
    TORCH_DTYPE_SET = set()
    FORMAT_TO_TORCH_DTYPE = {}
    NP_DTYPE_TO_TORCH_DTYPE = {}
    TORCH_DTYPE_TO_NP_DTYPE = {}
    TORCH_DTYPE_TO_FORMAT = {}

NP_DTYPE_TO_FORMAT: dict[np.dtype, cvcuda.Format] = {
    TYPE_TO_NP_DTYPE[t]: TYPE_TO_FORMAT[t] for t in SCALAR_TYPES
}


# Auto-conversion functions
# =========================
_valid_types = (cvcuda.Format, cvcuda.Type, np.dtype)
if _HAS_TORCH:
    _valid_types = (cvcuda.Format, cvcuda.Type, np.dtype, torch.dtype)


def as_cvcuda_format(
    fmt: "cvcuda.Format | cvcuda.Type | np.dtype | torch.dtype",
) -> cvcuda.Format:
    """Convert any supported dtype representation to a ``cvcuda.Format``."""
    if fmt in FORMAT_SET:
        return fmt
    if fmt in TYPE_SET:
        return TYPE_TO_FORMAT[fmt]
    if fmt in NP_DTYPE_SET:
        return NP_DTYPE_TO_FORMAT[fmt]
    if fmt in TORCH_DTYPE_SET:
        return TORCH_DTYPE_TO_FORMAT[fmt]

    raise ValueError(f"Invalid format: {fmt}. Valid formats are: {FORMATS}")


def as_cvcuda_dtype(
    dtype: "cvcuda.Format | cvcuda.Type | np.dtype | torch.dtype",
) -> cvcuda.Type:
    """Convert any supported dtype representation to a ``cvcuda.Type``."""
    if dtype in TYPE_SET:
        return dtype

    if dtype in FORMAT_SET:
        return FORMAT_TO_TYPE[dtype]
    if dtype in NP_DTYPE_SET:
        return NP_DTYPE_TO_TYPE[dtype]
    if dtype in TORCH_DTYPE_SET:
        return TORCH_DTYPE_TO_TYPE[dtype]

    raise ValueError(f"Invalid dtype: {dtype}. Valid types are: {_valid_types}")


def as_np_dtype(
    dtype: "cvcuda.Format | cvcuda.Type | np.dtype | torch.dtype",
) -> np.dtype:
    """Convert any supported dtype representation to a ``np.dtype``."""
    if dtype in NP_DTYPE_SET:
        return dtype

    if dtype in FORMAT_SET:
        return FORMAT_TO_NP_DTYPE[dtype]
    if dtype in TYPE_SET:
        return TYPE_TO_NP_DTYPE[dtype]
    if dtype in TORCH_DTYPE_SET:
        return TORCH_DTYPE_TO_NP_DTYPE[dtype]

    raise ValueError(f"Invalid dtype: {dtype}. Valid types are: {_valid_types}")


def as_torch_dtype(dtype):
    """Convert any supported dtype representation to a ``torch.dtype``.

    Requires torch to be installed; raises ``RuntimeError`` if it is not.
    Not all cvcuda types have a torch equivalent in every PyTorch version. In
    those cases a ``ValueError`` is raised instead of a ``KeyError``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("as_torch_dtype() requires torch to be installed")

    if dtype in TORCH_DTYPE_SET:
        return dtype

    if dtype in FORMAT_SET and dtype in FORMAT_TO_TORCH_DTYPE:
        return FORMAT_TO_TORCH_DTYPE[dtype]
    if dtype in TYPE_SET and dtype in TYPE_TO_TORCH_DTYPE:
        return TYPE_TO_TORCH_DTYPE[dtype]
    if dtype in NP_DTYPE_SET and dtype in NP_DTYPE_TO_TORCH_DTYPE:
        return NP_DTYPE_TO_TORCH_DTYPE[dtype]

    raise ValueError(f"Invalid dtype: {dtype}. Valid types are: {_valid_types}")


# Mapping from (dtype, channels) to format for image creation
DTYPE_CHANNELS_TO_FORMAT: dict[tuple[cvcuda.Type, int], cvcuda.Format] = {
    # 1 channel
    (cvcuda.Type.U8, 1): cvcuda.Format.U8,
    (cvcuda.Type.U16, 1): cvcuda.Format.U16,
    (cvcuda.Type.U32, 1): cvcuda.Format.U32,
    (cvcuda.Type.S8, 1): cvcuda.Format.S8,
    (cvcuda.Type.S16, 1): cvcuda.Format.S16,
    (cvcuda.Type.S32, 1): cvcuda.Format.S32,
    (cvcuda.Type.F16, 1): cvcuda.Format.F16,
    (cvcuda.Type.F32, 1): cvcuda.Format.F32,
    (cvcuda.Type.F64, 1): cvcuda.Format.F64,
    (cvcuda.Type.C64, 1): cvcuda.Format.C64,
    (cvcuda.Type.C128, 1): cvcuda.Format.C128,
    # 3 channels (RGB variants)
    (cvcuda.Type.U8, 3): cvcuda.Format.RGB8,
    (cvcuda.Type.F16, 3): cvcuda.Format.RGBf16,
    (cvcuda.Type.F32, 3): cvcuda.Format.RGBf32,
    # 4 channels (RGBA variants)
    (cvcuda.Type.U8, 4): cvcuda.Format.RGBA8,
    (cvcuda.Type.F16, 4): cvcuda.Format.RGBAf16,
    (cvcuda.Type.F32, 4): cvcuda.Format.RGBAf32,
}


def dtype_channels_to_format(dtype: cvcuda.Type, channels: int) -> cvcuda.Format:
    key = (dtype, channels)
    if key in DTYPE_CHANNELS_TO_FORMAT:
        return DTYPE_CHANNELS_TO_FORMAT[key]

    raise ValueError(f"No format available for dtype={dtype}, channels={channels}")


def get_typestr(dtype) -> str:
    if dtype in FORMAT_SET:
        return np.dtype(FORMAT_TO_NP_DTYPE[dtype]).str
    if dtype in TYPE_SET:
        return np.dtype(TYPE_TO_NP_DTYPE[dtype]).str
    if dtype in NP_DTYPE_SET:
        return np.dtype(dtype).str
    if TORCH_DTYPE_SET and dtype in TORCH_DTYPE_SET:
        return np.dtype(TORCH_DTYPE_TO_NP_DTYPE[dtype]).str

    raise ValueError(f"Cannot get typestr for dtype: {dtype}.")


# define other constants
LAYOUTS_STR: list[str] = [
    "CDHW",
    "CFDHW",
    "CFHW",
    "CHW",
    "CW",
    "DHW",
    "DHWC",
    "FCDHW",
    "FCHW",
    "FDHW",
    "FDHWC",
    "FHW",
    "FHWC",
    "HW",
    "HWC",
    "NCDHW",
    "NCFDHW",
    "NCFHW",
    "NCHW",
    "NCW",
    "NDHW",
    "NDHWC",
    "NFCDHW",
    "NFCHW",
    "NFDHW",
    "NFDHWC",
    "NFHW",
    "NFHWC",
    "NHW",
    "NHWC",
    "NW",
    "NWC",
    "W",
    "WC",
]
LAYOUTS: dict[int, dict[str, list | set]] = {}
for layout in LAYOUTS_STR:
    rank = len(layout)
    if rank not in LAYOUTS:
        LAYOUTS[rank] = {
            "list": [],
            "set": set(),
        }
    LAYOUTS[rank]["list"].append(layout)
    LAYOUTS[rank]["set"].add(layout)

# Image-compatible layouts: only layouts containing both H and W dimensions.
# Layouts without H (e.g., NWC, NCW, WC, CW) cause crashes in image operators
# because they lack the spatial dimensions that image operations require.
IMAGE_LAYOUTS_3D: set[str] = {"CHW", "DHW", "FHW", "HWC", "NHW"}
IMAGE_LAYOUTS_4D: set[str] = {
    "CDHW",
    "CFHW",
    "DHWC",
    "FCHW",
    "FDHW",
    "FHWC",
    "NCHW",
    "NDHW",
    "NFHW",
    "NHWC",
}
IMAGE_LAYOUTS: set[str] = IMAGE_LAYOUTS_3D | IMAGE_LAYOUTS_4D

# Channel counts for testing
CHANNELS: set[int] = {1, 2, 3, 4, 5, 6}

LAYOUT_TO_SHAPE: dict[str, Callable[[int, int, int, int], tuple[int, ...]]] = {
    "W": lambda N, H, W, C: (W,),
    "WC": lambda N, H, W, C: (W, C),
    "CW": lambda N, H, W, C: (C, W),
    "HW": lambda N, H, W, C: (H, W),
    "HWC": lambda N, H, W, C: (H, W, C),
    "CHW": lambda N, H, W, C: (C, H, W),
    "FHW": lambda N, H, W, C: (1, H, W),
    "FHWC": lambda N, H, W, C: (1, H, W, C),
    "FCHW": lambda N, H, W, C: (1, C, H, W),
    "NW": lambda N, H, W, C: (N, W),
    "NWC": lambda N, H, W, C: (N, W, C),
    "NCW": lambda N, H, W, C: (N, C, W),
    "NHW": lambda N, H, W, C: (N, H, W),
    "NHWC": lambda N, H, W, C: (N, H, W, C),
    "NCHW": lambda N, H, W, C: (N, C, H, W),
    "NFHW": lambda N, H, W, C: (N, 1, H, W),
    "NFHWC": lambda N, H, W, C: (N, 1, H, W, C),
    "NFCHW": lambda N, H, W, C: (N, 1, C, H, W),
    "DHW": lambda N, H, W, C: (1, H, W),
    "DHWC": lambda N, H, W, C: (1, H, W, C),
    "CDHW": lambda N, H, W, C: (C, 1, H, W),
    "FDHW": lambda N, H, W, C: (1, 1, H, W),
    "FDHWC": lambda N, H, W, C: (1, 1, H, W, C),
    "FCDHW": lambda N, H, W, C: (1, C, 1, H, W),
    "CFHW": lambda N, H, W, C: (C, 1, H, W),
    "CFDHW": lambda N, H, W, C: (C, 1, 1, H, W),
    "NDHW": lambda N, H, W, C: (N, 1, H, W),
    "NDHWC": lambda N, H, W, C: (N, 1, H, W, C),
    "NCDHW": lambda N, H, W, C: (N, C, 1, H, W),
    "NFDHW": lambda N, H, W, C: (N, 1, 1, H, W),
    "NFDHWC": lambda N, H, W, C: (N, 1, 1, H, W, C),
    "NFCDHW": lambda N, H, W, C: (N, 1, C, 1, H, W),
    "NCFHW": lambda N, H, W, C: (N, C, 1, H, W),
    "NCFDHW": lambda N, H, W, C: (N, C, 1, 1, H, W),
}


def resolve_shape(
    layout: str,
    channels: int,
    size: tuple[int, int] = (24, 24),
    batch_size: int = 1,
) -> tuple[int, ...]:
    if layout not in LAYOUT_TO_SHAPE:
        raise ValueError(f"Invalid layout: {layout}.")
    return LAYOUT_TO_SHAPE[layout](batch_size, size[0], size[1], channels)


def get_dim_index(layout: cvcuda.TensorLayout | str, dim: str) -> int | None:
    if isinstance(layout, cvcuda.TensorLayout):
        layout = str(layout)
    idx = layout.find(dim)
    return idx if idx >= 0 else None
