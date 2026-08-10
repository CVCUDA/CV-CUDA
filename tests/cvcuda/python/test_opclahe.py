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

import cvcuda

import numpy as np
import pytest as t
import cvcuda_util as util


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input_shape, layout",
    [
        ((1, 64, 80, 1), "NHWC"),
        ((3, 127, 119, 1), "NHWC"),
        ((1, 1, 64, 80), "NCHW"),
        ((3, 1, 127, 119), "NCHW"),
    ],
)
def test_op_clahe(input_shape, layout):
    src = cvcuda.Tensor(input_shape, cvcuda.Type.U8, layout)

    out = cvcuda.clahe(src)
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype

    out2 = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.clahe_into(out2, src, clip_limit=2.0, tile_grid_size=(7, 9))
    assert tmp is out2
    assert out2.layout == src.layout
    assert out2.shape == src.shape
    assert out2.dtype == src.dtype

    stream = cvcuda.Stream()
    out3 = cvcuda.clahe(src, stream=stream)
    assert out3.layout == src.layout
    assert out3.shape == src.shape
    assert out3.dtype == src.dtype

    out4 = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp2 = cvcuda.clahe_into(
        dst=out4,
        src=src,
        clip_limit=2.0,
        tile_grid_size=(7, 9),
        stream=stream,
    )
    assert tmp2 is out4
    assert out4.layout == src.layout
    assert out4.shape == src.shape
    assert out4.dtype == src.dtype


@t.mark.parametrize(
    "input_shape, layout",
    [
        ((64, 80, 1), "HWC"),
        ((127, 119, 1), "HWC"),
        ((1, 64, 80), "CHW"),
        ((1, 127, 119), "CHW"),
    ],
)
def test_op_clahe_hwc(input_shape, layout):
    src = cvcuda.Tensor(input_shape, cvcuda.Type.U8, layout)

    out = cvcuda.clahe(src, clip_limit=2.0, tile_grid_size=(8, 8))
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype

    out2 = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.clahe_into(out2, src, clip_limit=2.0, tile_grid_size=(8, 8))
    assert tmp is out2
    assert out2.layout == src.layout
    assert out2.shape == src.shape
    assert out2.dtype == src.dtype


def test_op_clahe_varshape():
    src = util.create_image_batch(4, cvcuda.Format.Y8, max_size=(127, 121), rng=RNG)

    out = cvcuda.clahe(src)
    assert out.uniqueformat == src.uniqueformat
    assert len(out) == len(src)
    assert out.capacity == src.capacity
    assert out.maxsize == src.maxsize

    tmp = cvcuda.clahe_into(out, src, clip_limit=2.0, tile_grid_size=(8, 8))
    assert tmp is out
    assert out.uniqueformat == src.uniqueformat
    assert len(out) == len(src)
    assert out.capacity == src.capacity
    assert out.maxsize == src.maxsize

    stream = cvcuda.Stream()
    out2 = cvcuda.clahe(src=src, clip_limit=2.0, tile_grid_size=(8, 8), stream=stream)
    assert out2.uniqueformat == src.uniqueformat
    assert len(out2) == len(src)
    assert out2.capacity == src.capacity
    assert out2.maxsize == src.maxsize

    tmp2 = cvcuda.clahe_into(
        dst=out2,
        src=src,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        stream=stream,
    )
    assert tmp2 is out2
    assert out2.uniqueformat == src.uniqueformat
    assert len(out2) == len(src)
    assert out2.capacity == src.capacity
    assert out2.maxsize == src.maxsize


def test_op_clahe_negative():
    src_rgb = cvcuda.Tensor((1, 32, 48, 3), cvcuda.Type.U8, "NHWC")
    with t.raises(Exception):
        cvcuda.clahe(src_rgb)

    src_rgb_nchw = cvcuda.Tensor((1, 3, 32, 48), cvcuda.Type.U8, "NCHW")
    with t.raises(RuntimeError, match="CLAHE supports only single-channel tensors"):
        cvcuda.clahe(src_rgb_nchw)

    src_f16 = cvcuda.Tensor((1, 32, 48, 1), cvcuda.Type.F16, "NHWC")
    with t.raises(Exception):
        cvcuda.clahe(src_f16)

    src = cvcuda.Tensor((1, 32, 48, 1), cvcuda.Type.U8, "NHWC")
    with t.raises(Exception):
        cvcuda.clahe(src, clip_limit=0.0)
    with t.raises(Exception):
        cvcuda.clahe(src, tile_grid_size=(0, 8))
    with t.raises(Exception):
        cvcuda.clahe(src, tile_grid_size=(8, 0))
    with t.raises(Exception):
        cvcuda.clahe(src, tile_grid_size=(-1, 8))


def test_op_clahe_varshape_negative():
    src_rgb = util.create_image_batch(3, cvcuda.Format.RGB8, max_size=(64, 64), rng=RNG)
    with t.raises(Exception):
        cvcuda.clahe(src_rgb)

    src_y8 = util.create_image_batch(3, cvcuda.Format.Y8, max_size=(64, 64), rng=RNG)
    with t.raises(Exception):
        cvcuda.clahe(src_y8, tile_grid_size=(0, 8))
