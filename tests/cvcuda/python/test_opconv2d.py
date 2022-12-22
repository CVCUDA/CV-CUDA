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
    "input, kernel, kernel_anchor, border",
    [
        (
            util.create_image_batch(
                10, cvcuda.Format.RGB8, size=(123, 321), max_random=256, rng=RNG
            ),
            util.create_image_batch(
                10, cvcuda.Format.F32, size=(3, 3), max_random=1, rng=RNG
            ),
            util.create_tensor((10, 2), np.int32, "NC", max_random=(3, 3), rng=RNG),
            cvcuda.Border.CONSTANT,
        ),
        (
            util.create_image_batch(7, cvcuda.Format.RGBf32, max_random=1, rng=RNG),
            util.create_image_batch(
                7, cvcuda.Format.F32, size=(5, 5), max_random=3, rng=RNG
            ),
            util.create_tensor((7, 2), np.int32, "NC", max_random=(5, 5), rng=RNG),
            cvcuda.Border.REPLICATE,
        ),
        (
            util.create_image_batch(1, cvcuda.Format.U8, max_random=123, rng=RNG),
            util.create_image_batch(
                1, cvcuda.Format.F32, size=(7, 7), max_random=2, rng=RNG
            ),
            util.create_tensor((1, 2), np.int32, "NC", max_random=(7, 7), rng=RNG),
            cvcuda.Border.REFLECT,
        ),
        (
            util.create_image_batch(6, cvcuda.Format.S16, max_random=1234, rng=RNG),
            util.create_image_batch(
                6, cvcuda.Format.F32, max_size=(9, 9), max_random=4, rng=RNG
            ),
            util.create_tensor((6, 2), np.int32, "NC", max_random=(1, 1), rng=RNG),
            cvcuda.Border.WRAP,
        ),
        (
            util.create_image_batch(9, cvcuda.Format.S32, max_random=12345, rng=RNG),
            util.create_image_batch(
                9, cvcuda.Format.F32, max_size=(4, 4), max_random=2, rng=RNG
            ),
            util.create_tensor((9, 2), np.int32, "NC", max_random=(4, 4), rng=RNG),
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_conv2dvarshape(input, kernel, kernel_anchor, border):
    out = cvcuda.conv2d(
        input,
        kernel,
        kernel_anchor,
        border,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.conv2d_into(
        src=input,
        dst=out,
        kernel=kernel,
        kernel_anchor=kernel_anchor,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
