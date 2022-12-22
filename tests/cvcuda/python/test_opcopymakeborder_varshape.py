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
from random import randint


@t.mark.parametrize(
    "num_images, format, min_out_size, max_out_size, border_mode, border_value",
    [
        (
            4,
            cvcuda.RGBf32,
            (1, 1),
            (128, 128),
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            5,
            cvcuda.RGB8,
            (1, 1),
            (128, 128),
            cvcuda.Border.CONSTANT,
            [12, 3, 4, 55],
        ),
        (
            9,
            cvcuda.RGBA8,
            (1, 1),
            (128, 128),
            cvcuda.Border.WRAP,
            [0],
        ),
        (
            12,
            cvcuda.RGBAf32,
            (1, 1),
            (128, 128),
            cvcuda.Border.REPLICATE,
            [0],
        ),
        (
            8,
            cvcuda.RGB8,
            (1, 1),
            (128, 128),
            cvcuda.Border.REFLECT,
            [0],
        ),
        (
            10,
            cvcuda.RGBA8,
            (1, 1),
            (128, 128),
            cvcuda.Border.REFLECT101,
            [0],
        ),
    ],
)
def test_op_copymakeborder(
    num_images, format, min_out_size, max_out_size, border_mode, border_value
):
    max_out_w = randint(min_out_size[0], max_out_size[0])
    max_out_h = randint(min_out_size[1], max_out_size[1])

    input = cvcuda.ImageBatchVarShape(num_images)
    varshape_out = cvcuda.ImageBatchVarShape(num_images)
    out_heights = []
    out_widths = []
    for i in range(num_images):
        w = randint(1, max_out_w)
        h = randint(1, max_out_h)
        img_i = cvcuda.Image([w, h], format)
        input.pushback(img_i)
        w_out = randint(w, max_out_size[0])
        h_out = randint(h, max_out_size[1])
        img_o = cvcuda.Image([w_out, h_out], format)
        varshape_out.pushback(img_o)
        out_heights.append(h_out)
        out_widths.append(w_out)

    top_tensor = cvcuda.Tensor([1, 1, num_images, 1], cvcuda.Type.S32, "NHWC")
    left_tensor = cvcuda.Tensor([1, 1, num_images, 1], cvcuda.Type.S32, "NHWC")

    tensor_out = cvcuda.Tensor(num_images, [max_out_w, max_out_h], format)

    out = cvcuda.copymakeborderstack(
        input,
        top=top_tensor,
        left=left_tensor,
        out_height=max_out_h,
        out_width=max_out_w,
    )
    assert out.layout == tensor_out.layout
    assert out.shape == tensor_out.shape
    assert out.dtype == tensor_out.dtype

    stream = cvcuda.Stream()
    tmp = cvcuda.copymakeborderstack_into(
        src=input,
        dst=tensor_out,
        top=top_tensor,
        left=left_tensor,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is tensor_out

    out = cvcuda.copymakeborder(
        src=input,
        top=top_tensor,
        left=left_tensor,
        out_heights=out_heights,
        out_widths=out_widths,
        stream=stream,
    )
    assert out.uniqueformat is not None
    assert out.uniqueformat == varshape_out.uniqueformat
    for res, ref in zip(out, varshape_out):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.copymakeborder_into(
        src=input,
        dst=varshape_out,
        top=top_tensor,
        left=left_tensor,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is varshape_out
