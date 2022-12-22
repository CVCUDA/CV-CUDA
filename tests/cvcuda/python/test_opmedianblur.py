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
    "input, ksize",
    [
        (
            cvcuda.Tensor([5, 9, 9, 4], np.uint8, "NHWC"),
            [5, 5],
        ),
        (
            cvcuda.Tensor([9, 9, 3], np.uint8, "HWC"),
            [5, 5],
        ),
        (
            cvcuda.Tensor([5, 21, 21, 4], np.uint8, "NHWC"),
            [15, 15],
        ),
        (
            cvcuda.Tensor([21, 21, 3], np.uint8, "HWC"),
            [15, 15],
        ),
    ],
)
def test_op_median_blur(input, ksize):
    out = cvcuda.median_blur(input, ksize)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.median_blur_into(
        src=input,
        dst=out,
        ksize=ksize,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_ksize",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            [11, 11],
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            [25, 25],
        ),
    ],
)
def test_op_median_blurvarshape(nimages, format, max_size, max_pixel, max_ksize):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    ksize = util.create_tensor(
        (nimages, 2),
        np.int32,
        "NC",
        max_random=max_ksize,
        rng=RNG,
        transform_dist=util.dist_odd,
    )

    out = cvcuda.median_blur(input, ksize)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)
    tmp = cvcuda.median_blur_into(
        src=input,
        dst=out,
        ksize=ksize,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
