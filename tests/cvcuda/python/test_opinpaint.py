# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "input_args, mask_args, inpaintRadius",
    [
        (
            ((1, 460, 640, 3), cvcuda.Type.U8, "NHWC"),
            ((1, 460, 640, 1), cvcuda.Type.U8, "NHWC"),
            5.0,
        ),
        (
            ((5, 640, 460, 3), cvcuda.Type.S32, "NHWC"),
            ((5, 640, 460, 1), cvcuda.Type.U8, "NHWC"),
            5.0,
        ),
        (
            ((4, 1920, 1080, 3), cvcuda.Type.F32, "NHWC"),
            ((4, 1920, 1080, 1), cvcuda.Type.U8, "NHWC"),
            5.0,
        ),
    ],
)
def test_op_inpaint(input_args, mask_args, inpaintRadius):
    input = cvcuda.Tensor(*input_args)
    mask = cvcuda.Tensor(*mask_args)

    out = cvcuda.inpaint(input, mask, inpaintRadius)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.inpaint_into(out, input, mask, inpaintRadius)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.inpaint(
        src=input,
        masks=mask,
        inpaintRadius=inpaintRadius,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.inpaint_into(
        src=input,
        dst=out,
        masks=mask,
        inpaintRadius=inpaintRadius,
        stream=stream,
    )
    assert tmp is out


@t.mark.parametrize(
    "num_images, format, min_size, max_size, inpaintRadius",
    [
        (
            1,
            cvcuda.Format.RGB8,
            (460, 640),
            (480, 720),
            5.0,
        ),
        (
            5,
            cvcuda.Format.S32,
            (640, 460),
            (720, 480),
            5.0,
        ),
        (
            4,
            cvcuda.Format.RGBf32,
            (1920, 1080),
            (1920, 1080),
            5.0,
        ),
    ],
)
def test_op_inpaint_varshape(num_images, format, min_size, max_size, inpaintRadius):

    input = cvcuda.ImageBatchVarShape(num_images)
    masks = cvcuda.ImageBatchVarShape(num_images)
    output = cvcuda.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = randint(min_size[0], max_size[0])
        h = randint(min_size[1], max_size[1])
        img_in = cvcuda.Image([w, h], format)
        input.pushback(img_in)
        mask = cvcuda.Image([w, h], cvcuda.Format.U8)
        masks.pushback(mask)
        img_out = cvcuda.Image([w, h], format)
        output.pushback(img_out)

    tmp = cvcuda.inpaint(input, masks, inpaintRadius)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.inpaint_into(output, input, masks, inpaintRadius)
    assert tmp is output

    stream = cvcuda.Stream()
    tmp = cvcuda.inpaint(
        src=input,
        masks=masks,
        inpaintRadius=inpaintRadius,
        stream=stream,
    )
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.inpaint_into(
        src=input,
        dst=output,
        masks=masks,
        inpaintRadius=inpaintRadius,
        stream=stream,
    )
    assert tmp is output
