# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "input_args, code, output_args",
    [
        (
            (5, [16, 23], cvcuda.Format.BGR8),
            cvcuda.ColorConversion.BGR2RGB,
            (5, [16, 23], cvcuda.Format.RGB8),
        ),
        (
            (3, [86, 22], cvcuda.Format.RGBA8),
            cvcuda.ColorConversion.RGBA2BGRA,
            (3, [86, 22], cvcuda.Format.BGRA8),
        ),
        (
            (7, [13, 21], cvcuda.Format.Y8),
            cvcuda.ColorConversion.GRAY2BGR,
            (7, [13, 21], cvcuda.Format.BGR8),
        ),
        (
            (9, [66, 99], cvcuda.Format.HSV8),
            cvcuda.ColorConversion.HSV2RGB,
            (9, [66, 99], cvcuda.Format.RGB8),
        ),
        (
            ((1, 61, 62, 3), np.uint8, "NHWC"),
            cvcuda.ColorConversion.YUV2RGB,
            ((1, 61, 62, 3), np.uint8, "NHWC"),
        ),
        (
            ((60, 62, 1), np.uint8, "HWC"),
            cvcuda.ColorConversion.YUV2RGB_NV12,
            ((40, 62, 3), np.uint8, "HWC"),
        ),
        (
            ((2, 40, 62, 3), np.uint8, "NHWC"),
            cvcuda.ColorConversion.BGR2YUV_NV21,
            ((2, 60, 62, 1), np.uint8, "NHWC"),
        ),
    ],
)
def test_op_cvtcolor(input_args, code, output_args):
    input = cvcuda.Tensor(*input_args)
    output = cvcuda.Tensor(*output_args)

    out = cvcuda.cvtcolor(input, code)
    assert out.shape == output.shape
    assert out.dtype == output.dtype

    stream = cvcuda.Stream()
    tmp = cvcuda.cvtcolor_into(
        src=input,
        dst=output,
        code=code,
        stream=stream,
    )
    assert tmp is output


@t.mark.parametrize(
    "num_images, in_format, img_size, max_pixel, code, out_format",
    [
        (
            10,
            cvcuda.Format.RGB8,
            (123, 321),
            256,
            cvcuda.ColorConversion.RGB2RGBA,
            cvcuda.Format.RGBA8,
        ),
        (
            8,
            cvcuda.Format.BGRA8,
            (23, 21),
            256,
            cvcuda.ColorConversion.BGRA2RGB,
            cvcuda.Format.RGB8,
        ),
        (
            6,
            cvcuda.Format.RGB8,
            (23, 21),
            256,
            cvcuda.ColorConversion.RGB2GRAY,
            cvcuda.Format.Y8_ER,
        ),
        (
            4,
            cvcuda.Format.HSV8,
            (23, 21),
            256,
            cvcuda.ColorConversion.HSV2RGB,
            cvcuda.Format.RGB8,
        ),
        (
            2,
            cvcuda.Format.Y8_ER,
            (23, 21),
            256,
            cvcuda.ColorConversion.GRAY2BGR,
            cvcuda.Format.BGR8,
        ),
    ],
)
def test_op_cvtcolorvarshape(
    num_images, in_format, img_size, max_pixel, code, out_format
):
    input = util.create_image_batch(
        num_images, in_format, size=img_size, max_random=max_pixel, rng=RNG
    )
    output = util.create_image_batch(
        num_images, out_format, size=img_size, max_random=max_pixel, rng=RNG
    )
    out = cvcuda.cvtcolor(input, code)
    assert len(out) == len(output)
    assert out.capacity == output.capacity
    assert out.maxsize == output.maxsize

    stream = cvcuda.Stream()
    tmp = cvcuda.cvtcolor_into(
        src=input,
        dst=output,
        code=code,
        stream=stream,
    )
    assert tmp is output
    assert len(output) == len(input)
    assert output.capacity == input.capacity
    assert output.maxsize == input.maxsize
