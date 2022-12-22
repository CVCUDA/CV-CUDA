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
    "input, diameter, sigma_color, sigma_space, border",
    [
        (
            cvcuda.Tensor([5, 9, 9, 4], np.uint8, "NHWC"),
            9,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            cvcuda.Tensor([9, 9, 3], np.uint8, "HWC"),
            7,
            3,
            10,
            cvcuda.Border.WRAP,
        ),
        (
            cvcuda.Tensor([5, 21, 21, 4], np.uint8, "NHWC"),
            6,
            15,
            9,
            cvcuda.Border.REPLICATE,
        ),
        (
            cvcuda.Tensor([21, 21, 3], np.uint8, "HWC"),
            12,
            2,
            5,
            cvcuda.Border.REFLECT,
        ),
    ],
)
def test_op_bilateral_filter(input, diameter, sigma_color, sigma_space, border):
    out = cvcuda.bilateral_filter(
        input, diameter, sigma_color, sigma_space, border=border
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.bilateral_filter_into(
        src=input,
        dst=out,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_diameter, max_sc, max_ss, border",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            12,
            2,
            5,
            cvcuda.Border.REFLECT,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            6,
            15,
            9,
            cvcuda.Border.REPLICATE,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            7,
            3,
            10,
            cvcuda.Border.WRAP,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (11, 23),
            256.0,
            9,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
    ],
)
def test_op_bilateral_filtervarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_diameter,
    max_sc,
    max_ss,
    border,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    diameter = util.create_tensor(
        (nimages), np.int32, "N", max_random=max_diameter, rng=RNG
    )

    sigma_color = util.create_tensor(
        (nimages), np.float32, "N", max_random=max_sc, rng=RNG
    )

    sigma_space = util.create_tensor(
        (nimages), np.float32, "N", max_random=max_ss, rng=RNG
    )
    out = cvcuda.bilateral_filter(
        input, diameter, sigma_color, sigma_space, border=border
    )

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)

    tmp = cvcuda.bilateral_filter_into(
        src=input,
        dst=out,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
