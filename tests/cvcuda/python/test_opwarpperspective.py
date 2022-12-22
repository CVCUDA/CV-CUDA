# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest as t
import numpy as np
import util


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input, xform, flags, border_mode, border_value",
    [
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [],
        ),
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array(
                [
                    [1, 2, 0],
                    [2, 1, 1],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array(
                [
                    [1, 2, 0],
                    [2, 1, 1],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
        (
            cvcuda.Tensor([11, 21, 4], np.uint8, "HWC"),
            np.array(
                [
                    [2, 2, 0],
                    [3, 1, 0],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.NEAREST,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            cvcuda.Tensor([11, 21, 4], np.uint8, "HWC"),
            np.array(
                [
                    [2, 2, 1],
                    [3, 1, 2],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.LINEAR,
            cvcuda.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            cvcuda.Tensor([11, 21, 4], np.uint8, "HWC"),
            np.array(
                [
                    [1, 2, 0],
                    [2, 1, 1],
                    [0, 0, 1],
                ]
            ),
            cvcuda.Interp.LINEAR,
            cvcuda.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
    ],
)
def test_op_warp_perspective(input, xform, flags, border_mode, border_value):
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


@t.mark.parametrize(
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
    nimages, format, max_size, max_pixel, max_xval, flags, bmode, border_value
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
