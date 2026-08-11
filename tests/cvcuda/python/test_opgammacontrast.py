# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cvcuda_tools as cv_tools
import cvcuda_util as util

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_gamma",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            12,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (10, 15),
            256.0,
            6,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (8, 8),
            256.0,
            7,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (11, 23),
            256.0,
            9,
        ),
    ],
)
def test_op_gamma_contrastvarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_gamma,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    gamma = util.create_tensor((nimages), np.float32, "N", max_gamma, rng=RNG)

    out = cvcuda.gamma_contrast(input, gamma)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)

    tmp = cvcuda.gamma_contrast_into(
        src=input,
        dst=out,
        gamma=gamma,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


@pytest.mark.parametrize(
    "shape, layout, dtype",
    [
        ((4, 16, 23, 3), "NHWC", np.uint8),
        ((16, 23, 3), "HWC", np.float32),
        ((2, 3, 16, 23), "NCHW", np.uint8),
        ((3, 16, 23), "CHW", np.float32),
        ((4, 16, 23, 4), "NHWC", np.float32),
    ],
)
@pytest.mark.parametrize("gamma, gain", [(0.5, 1.0), (1.8, 0.9), (0.7, 1.2)])
def test_op_gamma_contrast_scalar(shape, layout, dtype, gamma, gain):
    # Host-scalar (float) gamma_contrast overload: out = gain * in**gamma with gamma/gain
    # passed by value (no gamma tensor). Validated against a numpy gold.
    is_float = np.issubdtype(dtype, np.floating)
    if is_float:
        h_src = RNG.random(shape, dtype=np.float32)
    else:
        h_src = RNG.integers(0, 256, size=shape, dtype=dtype)
    src = util.to_cvcuda_tensor(h_src, layout)

    out = cvcuda.gamma_contrast(src, gamma, gain)
    assert tuple(out.shape) == tuple(src.shape)
    assert out.dtype == src.dtype
    assert str(out.layout) == layout
    got = util.to_cpu_numpy_buffer(out.cuda())

    if is_float:
        gold = np.clip(gain * np.power(h_src, gamma), 0.0, 1.0).astype(np.float32)
        np.testing.assert_allclose(got, gold, atol=5e-7, rtol=0)
    else:
        norm = h_src.astype(np.float32) / 255.0
        gold = np.clip(np.rint(gain * np.power(norm, gamma) * 255.0), 0, 255).astype(
            dtype
        )
        assert np.max(np.abs(got.astype(np.int32) - gold.astype(np.int32))) <= 1

    # _into overload writes into the provided tensor and is deterministic.
    dst = util.to_cvcuda_tensor(np.zeros_like(h_src), layout)
    tmp = cvcuda.gamma_contrast_into(dst=dst, src=src, gamma=gamma, gain=gain)
    assert tmp is dst
    np.testing.assert_array_equal(util.to_cpu_numpy_buffer(dst.cuda()), got)


def test_op_gamma_contrast_scalar_matches_tensor():
    # The float overload (scalar gamma, default gain=1) must dispatch to the scalar path
    # and be bit-exact with the device-tensor overload fed a gamma tensor filled with the
    # same value -- proving both overload dispatch and kernel parity.
    h_src = RNG.integers(0, 256, size=(3, 16, 23, 3), dtype=np.uint8)
    src = util.to_cvcuda_tensor(h_src, "NHWC")
    gamma = 0.75

    out_scalar = cvcuda.gamma_contrast(src, gamma)
    gamma_tensor = util.to_cvcuda_tensor(np.full((3,), gamma, np.float32), "N")
    out_tensor = cvcuda.gamma_contrast(src, gamma_tensor)

    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(out_scalar.cuda()),
        util.to_cpu_numpy_buffer(out_tensor.cuda()),
    )


@pytest.mark.parametrize(
    "round_mode,expected", [(cvcuda.Round.NEAREST, 2), (cvcuda.Round.TRUNCATE, 1)]
)
def test_op_gamma_contrast_scalar_round_mode(round_mode, expected):
    h_src = np.array([[[[255]]]], dtype=np.uint8)
    src = util.to_cvcuda_tensor(h_src, "NHWC")

    out = cvcuda.gamma_contrast(src, 1.0, 1.5 / 255.0, round=round_mode)

    got = util.to_cpu_numpy_buffer(out.cuda())
    np.testing.assert_array_equal(got, np.full_like(h_src, expected))

    dst = util.to_cvcuda_tensor(np.zeros_like(h_src), "NHWC")
    returned = cvcuda.gamma_contrast_into(
        dst=dst,
        src=src,
        gamma=1.0,
        gain=1.5 / 255.0,
        round=round_mode,
    )
    assert returned is dst
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(dst.cuda()), np.full_like(h_src, expected)
    )


def _gamma_contrast_op(data):
    num_images = len(data)
    gamma = util.create_tensor((num_images,), np.float32, "N", max_random=2.0, rng=RNG)
    return cvcuda.gamma_contrast(data, gamma)


def _gamma_contrast_tensor_op(data):
    num_samples = data.shape[0] if len(data.shape) == 4 else 1
    gamma = util.create_tensor((num_samples,), np.float32, "N", max_random=2.0, rng=RNG)
    return cvcuda.gamma_contrast(data, gamma)


def _gamma_contrast_scalar_op(data):
    # Host-scalar gamma/gain overload; the values are arbitrary valid scalars since
    # these generated tests validate the input contract, not output values.
    return cvcuda.gamma_contrast(data, 0.75, 1.1)


globals().update(
    cv_tools.make_op_tests(
        name="gammacontrast",
        runner_info=[
            ("image_batch", _gamma_contrast_op, None),
        ],
        supported_formats={
            # 1 channel
            cvcuda.Format.U8,
            cvcuda.Format.U16,
            cvcuda.Format.S16,
            cvcuda.Format.S32,
            cvcuda.Format.F32,
            # 2 channels
            cvcuda.Format._2F32,
            # 3 channels
            cvcuda.Format.RGB8,
            cvcuda.Format.BGR8,
            cvcuda.Format.RGBf32,
            cvcuda.Format.BGRf32,
            # 4 channels
            cvcuda.Format.RGBA8,
            cvcuda.Format.BGRA8,
            cvcuda.Format.RGBAf32,
            cvcuda.Format.BGRAf32,
            # 3-channel planar (NCHW/CHW)
            cvcuda.Format.RGB8p,
            cvcuda.Format.BGR8p,
            cvcuda.Format.RGBf32p,
            cvcuda.Format.BGRf32p,
            # 4-channel planar (NCHW/CHW)
            cvcuda.Format.RGBA8p,
            cvcuda.Format.BGRA8p,
            cvcuda.Format.RGBAf32p,
            cvcuda.Format.BGRAf32p,
        },
    )
)

globals().update(
    cv_tools.make_op_tests(
        name="gammacontrast",
        runner_info=[
            ("tensor", _gamma_contrast_tensor_op, None),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[
            (None, "NCHW", 2),
            (None, "CHW", 2),
        ],
    )
)

# The host-scalar gamma/gain overload declares the same dense-tensor input contract
# as the gamma-tensor overload (its legacy infer mirrors that path's validation).
globals().update(
    cv_tools.make_op_tests(
        name="gammacontrast_scalar",
        runner_info=[
            ("tensor", _gamma_contrast_scalar_op, None),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[
            (None, "NCHW", 2),
            (None, "CHW", 2),
        ],
    )
)
