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


def invert_ref(src):
    """Subtract from the integer maximum with saturation, or from one for floats."""
    if np.issubdtype(src.dtype, np.floating):
        return (np.float32(1.0) - src).astype(src.dtype)
    bound = np.iinfo(src.dtype).max
    # Python integers keep signed 64-bit boundary subtraction from overflowing.
    return np.asarray(
        [min(bound, bound - int(value)) for value in src.flat], dtype=src.dtype
    ).reshape(src.shape)


def assert_inverted(got, src):
    ref = invert_ref(src)
    # A single-channel image reads back as (H, W) while the oracle is (H, W, 1);
    # normalize only that known case and assert the exact shape otherwise, so a
    # real shape/layout regression still fails the test.
    got = np.asarray(got)
    if got.shape == ref.shape[:-1] and ref.shape[-1] == 1:
        got = got[..., np.newaxis]
    else:
        assert (
            got.shape == ref.shape
        ), f"shape mismatch: got={got.shape}, expected={ref.shape}"
    if np.issubdtype(src.dtype, np.floating):
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)
    else:
        np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize(
    "shape, dtype, layout",
    [
        ((5, 16, 23, 4), np.uint8, "NHWC"),  # interleaved RGBA u8
        ((4, 9, 3), np.uint8, "HWC"),  # interleaved RGB u8, no batch
        ((3, 88, 13, 1), np.uint16, "NHWC"),  # u16 single channel
        ((2, 4, 16, 23), np.float32, "NCHW"),  # planar float
        ((3, 8, 8), np.float32, "CHW"),  # planar float, no batch
        ((5, 16, 23, 3), np.float16, "NHWC"),  # interleaved f16
        ((5, 16, 23, 3), np.float64, "NHWC"),
        ((2, 4, 16, 23), np.float64, "NCHW"),
        (
            (3, 88, 13, 1),
            np.float16,
            "NHWC",
        ),  # f16 single channel (scalar-kernel fallback)
        ((2, 4, 16, 23), np.float16, "NCHW"),  # planar f16
    ],
)
def test_op_invert(shape, dtype, layout):
    src_h = util.generate_data(shape, dtype, rng=RNG)
    t_src = util.to_cvcuda_tensor(src_h, layout)

    # allocating variant
    out = cvcuda.invert(t_src)
    assert out.layout == t_src.layout
    assert out.shape == t_src.shape
    assert out.dtype == t_src.dtype
    assert_inverted(cupy.asarray(out.cuda()).get(), src_h)

    # into variant
    stream = cvcuda.Stream()
    out = cvcuda.Tensor(t_src.shape, t_src.dtype, t_src.layout)
    tmp = cvcuda.invert_into(src=t_src, dst=out, stream=stream)
    stream.sync()
    assert tmp is out
    assert out.layout == t_src.layout
    assert out.shape == t_src.shape
    assert out.dtype == t_src.dtype
    assert_inverted(cupy.asarray(out.cuda()).get(), src_h)


@pytest.mark.parametrize(
    "num_images, img_format, img_size, max_pixel",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 256),
        (7, cvcuda.Format.RGBf32, (62, 35), 1.0),
        (1, cvcuda.Format.U16, (33, 48), 1234),
        (4, cvcuda.Format.RGBA8, (26, 52), 256),
        (5, cvcuda.Format.RGBf16, (62, 35), 1.0),
        (
            2,
            cvcuda.Format.F16,
            (33, 48),
            1.0,
        ),  # f16 single channel (scalar-kernel fallback)
    ],
)
def test_op_invert_varshape(num_images, img_format, img_size, max_pixel):
    # Build the batch from known host data so output values can be checked.
    w, h = img_size
    dtype = util.get_numpy_dtype_for_format(img_format)
    srcs_h = [
        util.generate_data((h, w, img_format.channels), dtype, max_pixel, RNG)
        for _ in range(num_images)
    ]
    src_batch = cvcuda.ImageBatchVarShape(num_images)
    for s in srcs_h:
        src_batch.pushback(util.to_cvcuda_image(s))

    # allocating variant
    out = cvcuda.invert(src_batch)
    assert len(out) == len(src_batch)
    assert out.capacity == src_batch.capacity
    assert out.uniqueformat == src_batch.uniqueformat
    assert out.maxsize == src_batch.maxsize
    for got_img, src_h in zip(out, srcs_h):
        assert_inverted(cupy.asarray(got_img.cuda()).get(), src_h)

    # into variant
    stream = cvcuda.Stream()
    out = util.clone_image_batch(src_batch)
    tmp = cvcuda.invert_into(src=src_batch, dst=out, stream=stream)
    stream.sync()
    assert tmp is out
    assert len(out) == len(src_batch)
    assert out.capacity == src_batch.capacity
    for got_img, src_h in zip(out, srcs_h):
        assert_inverted(cupy.asarray(got_img.cuda()).get(), src_h)


def test_op_invert_negative_dtype():
    src = cvcuda.Tensor((1, 16, 16, 3), np.int8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.invert(src)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
@pytest.mark.parametrize("layout", ["NHWC", "HWC", "NCHW", "CHW"])
def test_op_invert_signed_boundaries(dtype, layout):
    """Check signed saturation and exact subtraction through both tensor APIs."""
    limits = np.iinfo(dtype)
    values = np.array(
        [limits.min, limits.min + 1, -1, 0, 1, limits.max - 1, limits.max],
        dtype=dtype,
    )
    shapes = {
        "NHWC": (1, 1, 7, 3),
        "HWC": (1, 7, 3),
        "NCHW": (1, 3, 1, 7),
        "CHW": (3, 1, 7),
    }
    host = np.resize(values, shapes[layout])
    src = util.to_cvcuda_tensor(host, layout)
    assert_inverted(util.to_cpu_numpy_buffer(cvcuda.invert(src).cuda()), host)
    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    assert cvcuda.invert_into(dst, src) is dst
    assert_inverted(util.to_cpu_numpy_buffer(dst.cuda()), host)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_op_invert_signed_varshape(dtype, channels):
    """Check signed boundary values in differently sized images and both APIs."""
    limits = np.iinfo(dtype)
    values = np.array([limits.min, -1, 0, 1, limits.max], dtype=dtype)
    hosts = [np.resize(values, (h, w, channels)) for h, w in [(2, 7), (3, 5)]]
    src = cvcuda.ImageBatchVarShape(len(hosts))
    for host in hosts:
        src.pushback(util.to_cvcuda_image(host))
    dst = util.clone_image_batch(src)
    allocated = cvcuda.invert(src)
    assert cvcuda.invert_into(dst, src) is dst
    for result in (allocated, dst):
        for image, host in zip(result, hosts, strict=True):
            assert_inverted(util.to_cpu_numpy_buffer(image.cuda()), host)
