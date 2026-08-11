# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from random import randint

import cvcuda

import pytest
import numpy as np
import cvcuda_util as util
import cvcuda_tools as cv_tools


@pytest.mark.parametrize(
    "input_args, top, bottom, left, right, border_mode, border_value",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            1,
            2,
            3,
            4,
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            1,
            2,
            3,
            4,
            cvcuda.Border.CONSTANT,
            [12, 3, 4, 55],
        ),
        (
            ((16, 23, 4), np.uint8, "HWC"),
            2,
            2,
            2,
            2,
            cvcuda.Border.WRAP,
            [0],
        ),
        (
            ((16, 23, 4), np.uint8, "HWC"),
            2,
            2,
            2,
            2,
            cvcuda.Border.WRAP,
            [0],
        ),
        (
            ((16, 23, 3), np.uint8, "HWC"),
            10,
            12,
            35,
            18,
            cvcuda.Border.REPLICATE,
            [0],
        ),
        (
            ((16, 23, 1), np.float32, "HWC"),
            11,
            1,
            20,
            3,
            cvcuda.Border.REFLECT,
            [0],
        ),
        (
            ((16, 23, 3), np.float32, "HWC"),
            11,
            1,
            20,
            3,
            cvcuda.Border.REFLECT101,
            [0],
        ),
    ],
)
def test_op_copymakeborder(
    input_args, top, bottom, left, right, border_mode, border_value
):
    input = cvcuda.Tensor(*input_args)
    out_shape = [i for i in input.shape]
    cdim = len(out_shape) - 1
    out_shape[cdim - 2] += top + bottom
    out_shape[cdim - 1] += left + right
    out_shape = tuple(out_shape)
    out = cvcuda.copymakeborder(input, top=top, bottom=bottom, left=left, right=right)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(out_shape, input.dtype, input.layout)
    tmp = cvcuda.copymakeborder_into(
        src=input,
        dst=out,
        top=top,
        left=left,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "num_images, format, min_out_size, max_out_size, border_mode, border_value",
    [
        (
            4,
            cvcuda.Format.RGBf32,
            (1, 1),
            (128, 128),
            cvcuda.Border.CONSTANT,
            [0],
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (1, 1),
            (128, 128),
            cvcuda.Border.CONSTANT,
            [12, 3, 4, 55],
        ),
        (
            9,
            cvcuda.Format.RGBA8,
            (1, 1),
            (128, 128),
            cvcuda.Border.WRAP,
            [0],
        ),
        (
            12,
            cvcuda.Format.RGBAf32,
            (1, 1),
            (128, 128),
            cvcuda.Border.REPLICATE,
            [0],
        ),
        (
            8,
            cvcuda.Format.RGB8,
            (1, 1),
            (128, 128),
            cvcuda.Border.REFLECT,
            [0],
        ),
        (
            10,
            cvcuda.Format.RGBA8,
            (1, 1),
            (128, 128),
            cvcuda.Border.REFLECT101,
            [0],
        ),
    ],
)
def test_op_copymakeborder_varshape(
    num_images, format, min_out_size, max_out_size, border_mode, border_value
):
    max_out_w = randint(min_out_size[0], max_out_size[0])
    max_out_h = randint(min_out_size[1], max_out_size[1])

    input = cvcuda.ImageBatchVarShape(num_images)
    varshape_out = cvcuda.ImageBatchVarShape(num_images)
    out_heights = []
    out_widths = []
    for _ in range(num_images):
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

    top_tensor = util.to_cvcuda_tensor(
        np.zeros((1, 1, num_images, 1), dtype=np.int32), "NHWC"
    )
    left_tensor = util.to_cvcuda_tensor(
        np.zeros((1, 1, num_images, 1), dtype=np.int32), "NHWC"
    )

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


def _copymakeborder_params(dtype, layout, channels):
    return {
        "top": 1,
        "bottom": 1,
        "left": 1,
        "right": 1,
    }


def _copymakeborder_varshape_params(dtype, layout, channels):
    return {
        "top": util.to_cvcuda_tensor(np.array([[[[1], [1]]]], dtype=np.int32), "NHWC"),
        "left": util.to_cvcuda_tensor(np.array([[[[1], [1]]]], dtype=np.int32), "NHWC"),
        "out_heights": [26, 26],
        "out_widths": [26, 26],
        "border_value": [0],
    }


globals().update(
    cv_tools.make_op_tests(
        name="copymakeborder",
        runner_info=[
            ("tensor", cvcuda.copymakeborder, _copymakeborder_params),
            ("image_batch", cvcuda.copymakeborder, _copymakeborder_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        # 2 channels only supported for U8
        exclude_dlc=[
            (cvcuda.Type.U16, None, 2),
            (cvcuda.Type.S16, None, 2),
            (cvcuda.Type.F32, None, 2),
            (None, "NCHW", 2),
            (None, "CHW", 2),
        ],
    )
)


def test_op_copymakeborder_varshape_widths_mismatch_raises():
    num_images = 3
    fmt = cvcuda.Format.RGB8
    input = cvcuda.ImageBatchVarShape(num_images)
    for _ in range(num_images):
        input.pushback(cvcuda.Image([16, 16], fmt))

    top = util.to_cvcuda_tensor(np.zeros((1, 1, num_images, 1), dtype=np.int32), "NHWC")
    left = util.to_cvcuda_tensor(
        np.zeros((1, 1, num_images, 1), dtype=np.int32), "NHWC"
    )

    out_heights = [32] * num_images
    out_widths = [32] * (num_images - 1)

    with pytest.raises(RuntimeError, match="out_widths"):
        cvcuda.copymakeborder(
            src=input,
            top=top,
            left=left,
            out_heights=out_heights,
            out_widths=out_widths,
        )
