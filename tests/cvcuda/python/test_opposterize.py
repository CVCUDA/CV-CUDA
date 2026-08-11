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

import pytest
import numpy as np
import cupy

import cvcuda_util as util

RNG = np.random.default_rng(0)


def posterize_ref(src, bits):
    info = np.iinfo(src.dtype)
    mask = (info.max << (info.bits - bits)) & info.max
    return (src & mask).astype(src.dtype)


def assert_posterized(got, src, bits):
    ref = posterize_ref(src, bits)
    got = np.asarray(got)
    if got.shape == ref.shape[:-1] and ref.shape[-1] == 1:
        got = got[..., np.newaxis]
    else:
        assert (
            got.shape == ref.shape
        ), f"shape mismatch: got={got.shape}, expected={ref.shape}"
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize(
    "shape, dtype, layout, bits",
    [
        ((5, 16, 23, 4), np.uint8, "NHWC", 4),
        ((4, 9, 3), np.uint8, "HWC", 2),
        ((3, 88, 13, 1), np.uint16, "NHWC", 5),
        ((2, 4, 16, 23), np.uint8, "NCHW", 3),
        ((3, 8, 8), np.uint8, "CHW", 1),
    ],
)
def test_op_posterize(shape, dtype, layout, bits):
    src_h = util.generate_data(shape, dtype, rng=RNG)
    src = util.to_cvcuda_tensor(src_h, layout)

    out = cvcuda.posterize(src, bits)
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert_posterized(cupy.asarray(out.cuda()).get(), src_h, bits)

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.posterize_into(src=src, dst=out, bits=bits, stream=stream)
    stream.sync()
    assert tmp is out
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert_posterized(cupy.asarray(out.cuda()).get(), src_h, bits)


@pytest.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, bits",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 256, 4),
        (1, cvcuda.Format.U16, (33, 48), 1234, 5),
        (4, cvcuda.Format.RGBA8, (26, 52), 256, 2),
    ],
)
def test_op_posterize_varshape(num_images, img_format, img_size, max_pixel, bits):
    w, h = img_size
    dtype = util.get_numpy_dtype_for_format(img_format)
    srcs_h = [
        util.generate_data((h, w, img_format.channels), dtype, max_pixel, RNG)
        for _ in range(num_images)
    ]
    src_batch = cvcuda.ImageBatchVarShape(num_images)
    for src_h in srcs_h:
        src_batch.pushback(util.to_cvcuda_image(src_h))

    out = cvcuda.posterize(src_batch, bits)
    assert len(out) == len(src_batch)
    assert out.capacity == src_batch.capacity
    assert out.uniqueformat == src_batch.uniqueformat
    assert out.maxsize == src_batch.maxsize
    for got_img, src_h in zip(out, srcs_h):
        assert_posterized(cupy.asarray(got_img.cuda()).get(), src_h, bits)

    stream = cvcuda.Stream()
    out = util.clone_image_batch(src_batch)
    tmp = cvcuda.posterize_into(src=src_batch, dst=out, bits=bits, stream=stream)
    stream.sync()
    assert tmp is out
    assert len(out) == len(src_batch)
    assert out.capacity == src_batch.capacity
    for got_img, src_h in zip(out, srcs_h):
        assert_posterized(cupy.asarray(got_img.cuda()).get(), src_h, bits)


def test_op_posterize_negative_dtype():
    # float32 is outside the supported dtype set (u8/u16) and must be rejected.
    src = cvcuda.Tensor((1, 16, 16, 3), np.float32, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.posterize(src, 4)
