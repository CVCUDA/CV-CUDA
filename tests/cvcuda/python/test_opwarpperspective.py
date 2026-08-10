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

import cvcuda_util as util
import cvcuda_tools as cv_tools

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "input_args, xform, flags, border_mode, border_value",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [
                [1, 2, 0],
                [2, 1, 1],
                [0, 0, 1],
            ],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            [
                [1, 2, 0],
                [2, 1, 1],
                [0, 0, 1],
            ],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [
                [2, 2, 0],
                [3, 1, 0],
                [0, 0, 1],
            ],
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [
                [2, 2, 1],
                [3, 1, 2],
                [0, 0, 1],
            ],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [
                [1, 2, 0],
                [2, 1, 1],
                [0, 0, 1],
            ],
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
        (
            ((11, 21, 4), np.uint8, "HWC"),
            [
                [1, 2, 0],
                [2, 1, 1],
                [0, 0, 1],
            ],
            cvcuda.Interp.LINEAR | cvcuda.Interp.WARP_INVERSE_MAP,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
    ],
)
def test_op_warp_perspective(input_args, xform, flags, border_mode, border_value):
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.warp_perspective(
        input, xform, flags, border_mode=border_mode, border_value=border_value
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.warp_perspective_into(
        src=input,
        dst=out,
        xform=xform,
        flags=flags,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


def test_op_warp_perspective_extreme_projection_replicate_issue_249():
    # Regression test for CVCUDA issue #249: warp_perspective with a projective
    # matrix whose singular line falls inside the destination image produces
    # source coordinates near +/-INT32_MAX. With BORDER_REPLICATE this used to
    # trigger an illegal CUDA memory access. Syncing the stream is required to
    # surface the kernel error as a test failure.
    xform = [
        [8.08776838e-02, 2.36326631e00, -4.08795000e02],
        [-1.28514739e-02, 2.55201343e-01, -8.45896673e01],
        [-2.68404432e-04, -6.57235630e-04, 1.00000000e00],
    ]
    input = cvcuda.Tensor((1208, 1928, 3), np.uint8, "HWC")
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    stream = cvcuda.Stream()
    cvcuda.warp_perspective_into(
        src=input,
        dst=out,
        xform=xform,
        flags=cvcuda.Interp.LINEAR,
        border_mode=cvcuda.Border.REPLICATE,
        border_value=[0],
        stream=stream,
    )
    stream.sync()


@pytest.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_xval, flags, bmode, border_value",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            7,
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [],
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            7,
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [1, 2, 3, 4],
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            5,
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [0],
        ),
        (
            3,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            4,
            cvcuda.Interp.CUBIC,
            cvcuda.Border.REPLICATE,
            [2, 1, 0],
        ),
    ],
)
def test_op_warp_perspectivevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_xval,
    flags,
    bmode,
    border_value,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    xform = util.create_tensor(
        (nimages, 9), np.float32, "NC", max_random=max_xval, rng=RNG
    )

    out = cvcuda.warp_perspective(
        input, xform, flags, border_mode=bmode, border_value=border_value
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)
    tmp = cvcuda.warp_perspective_into(
        src=input,
        dst=out,
        xform=xform,
        flags=flags,
        border_mode=bmode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _warp_perspective_params(dtype, layout, channels):
    return {
        "xform": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        "flags": int(cvcuda.Interp.NEAREST),
        "border_mode": cvcuda.Border.CONSTANT,
        "border_value": np.array([], dtype=np.float32),
    }


def _warp_perspective_varshape_params(dtype, layout, channels):
    return {
        "xform": util.create_tensor((2, 9), np.float32, "NC", max_random=1, rng=RNG),
        "flags": int(cvcuda.Interp.NEAREST),
        "border_mode": cvcuda.Border.CONSTANT,
        "border_value": np.array([], dtype=np.float32),
    }


globals().update(
    cv_tools.make_op_tests(
        name="warp_perspective",
        runner_info=[
            ("tensor", cvcuda.warp_perspective, _warp_perspective_params),
            ("image_batch", cvcuda.warp_perspective, _warp_perspective_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
    )
)
