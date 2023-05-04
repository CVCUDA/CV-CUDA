# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cvcuda_util as util


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input_args, kernel_args, kernel_anchor_args, border",
    [
        (
            (10, cvcuda.Format.RGB8, (123, 321), (128, 128), 256, RNG),
            (10, cvcuda.Format.F32, (3, 3), (128, 128), 1, RNG),
            ((10, 2), np.int32, "NC", (3, 3), RNG, None),
            cvcuda.Border.CONSTANT,
        ),
        (
            (7, cvcuda.Format.RGBf32, (0, 0), (128, 128), 1, RNG),
            (7, cvcuda.Format.F32, (5, 5), (128, 128), 3, RNG),
            ((7, 2), np.int32, "NC", (5, 5), RNG, None),
            cvcuda.Border.REPLICATE,
        ),
        (
            (1, cvcuda.Format.U8, (0, 0), (128, 128), 123, RNG),
            (1, cvcuda.Format.F32, (7, 7), (128, 128), 2, RNG),
            ((1, 2), np.int32, "NC", (7, 7), RNG, None),
            cvcuda.Border.REFLECT,
        ),
        (
            (6, cvcuda.Format.S16, (0, 0), (128, 128), 1234, RNG),
            (6, cvcuda.Format.F32, (0, 0), (9, 9), 4, RNG),
            ((6, 2), np.int32, "NC", (1, 1), RNG, None),
            cvcuda.Border.WRAP,
        ),
        (
            (9, cvcuda.Format.S32, (0, 0), (128, 128), 12345, RNG),
            (9, cvcuda.Format.F32, (0, 0), (4, 4), 2, RNG),
            ((9, 2), np.int32, "NC", (4, 4), RNG, None),
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_conv2dvarshape(input_args, kernel_args, kernel_anchor_args, border):
    input = util.create_image_batch(*input_args)
    kernel = util.create_image_batch(*kernel_args)
    kernel_anchor = util.create_tensor(*kernel_anchor_args)

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
