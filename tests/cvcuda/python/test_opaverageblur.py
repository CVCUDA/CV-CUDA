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
    "input, kernel_size, kernel_anchor, border",
    [
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            [3, 3],
            [1, 1],
            cvcuda.Border.CONSTANT,
        ),
        (
            cvcuda.Tensor([4, 4, 3], np.float32, "HWC"),
            [5, 5],
            [0, 0],
            cvcuda.Border.REPLICATE,
        ),
        (
            cvcuda.Tensor([3, 88, 13, 3], np.uint16, "NHWC"),
            [7, 7],
            [2, 2],
            cvcuda.Border.REFLECT,
        ),
        (
            cvcuda.Tensor([3, 4, 4], np.int32, "HWC"),
            [9, 9],
            [-1, -1],
            cvcuda.Border.WRAP,
        ),
        (
            cvcuda.Tensor([1, 2, 3, 4], np.int16, "NHWC"),
            [11, 11],
            [8, 8],
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_averageblur(input, kernel_size, kernel_anchor, border):
    out = cvcuda.averageblur(input, kernel_size, kernel_anchor, border)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.averageblur_into(
        src=input,
        dst=out,
        kernel_size=kernel_size,
        kernel_anchor=kernel_anchor,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, max_kernel_size, border",
    [
        (
            10,
            cvcuda.Format.RGB8,
            (123, 321),
            256,
            (3, 3),
            cvcuda.Border.CONSTANT,
        ),
        (
            7,
            cvcuda.Format.RGBf32,
            (62, 35),
            1.0,
            (5, 5),
            cvcuda.Border.REPLICATE,
        ),
        (
            1,
            cvcuda.Format.U8,
            (33, 48),
            123,
            (7, 7),
            cvcuda.Border.REFLECT,
        ),
        (
            13,
            cvcuda.Format.S16,
            (26, 52),
            1234,
            (9, 9),
            cvcuda.Border.WRAP,
        ),
        (
            6,
            cvcuda.Format.S32,
            (77, 42),
            123456,
            (11, 11),
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_averageblurvarshape(
    num_images, img_format, img_size, max_pixel, max_kernel_size, border
):

    input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    kernel_size = util.create_tensor(
        (num_images, 2),
        np.int32,
        "NC",
        max_random=max_kernel_size,
        rng=RNG,
        transform_dist=util.dist_odd,
    )

    kernel_anchor = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_kernel_size, rng=RNG
    )

    out = cvcuda.averageblur(
        input,
        max_kernel_size,
        kernel_size,
        kernel_anchor,
        border,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.averageblur_into(
        src=input,
        dst=out,
        max_kernel_size=max_kernel_size,
        kernel_size=kernel_size,
        kernel_anchor=kernel_anchor,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
