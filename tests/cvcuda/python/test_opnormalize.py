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
    "input,base,scale,globalscale,globalshift,epsilon,flags",
    [
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            cvcuda.Tensor([1, 1], np.float32, "HW"),
            cvcuda.Tensor([1, 1], np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            cvcuda.Tensor([16, 1], np.float32, "HW"),
            cvcuda.Tensor([16, 1], np.float32, "HW"),
            1,
            2,
            3,
            cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        ),
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            cvcuda.Tensor([1, 23], np.float32, "HW"),
            cvcuda.Tensor([1, 23], np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            cvcuda.Tensor([16, 23], np.float32, "HW"),
            cvcuda.Tensor([16, 23], np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
    ],
)
def test_op_normalize(input, base, scale, globalscale, globalshift, epsilon, flags):
    out = cvcuda.normalize(input, base, scale)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.normalize_into(out, input, base, scale)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.normalize(
        src=input,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.normalize_into(
        src=input,
        dst=out,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "nimages,format,max_size,max_pixel,base,scale,globalscale,globalshift,epsilon,flags",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            cvcuda.Tensor([1, 1, 1, 5], np.float32, "NHWC"),
            cvcuda.Tensor([1, 1, 1, 5], np.float32, "NHWC"),
            1,
            2,
            3,
            None,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            cvcuda.Tensor([1, 1, 1, 5], np.float32, "NHWC"),
            cvcuda.Tensor([1, 1, 1, 5], np.float32, "NHWC"),
            1,
            2,
            3,
            cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        ),
    ],
)
def test_op_rotatevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    base,
    scale,
    globalscale,
    globalshift,
    epsilon,
    flags,
):
    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    out = cvcuda.normalize(input, base, scale)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    out = util.clone_image_batch(input)
    tmp = cvcuda.normalize_into(out, input, base, scale)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.cuda.Stream()
    out = cvcuda.normalize(
        src=input,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    tmp = cvcuda.normalize_into(
        src=input,
        dst=out,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
