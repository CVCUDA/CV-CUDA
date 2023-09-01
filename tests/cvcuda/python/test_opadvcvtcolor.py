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
import numpy as np


@t.mark.parametrize(
    "input,dtype,code",
    [
        (((1, 1, 1, 3), np.uint8, "NHWC"), np.uint8, cvcuda.ColorConversion.BGR2YUV),
        (((2, 2, 30, 3), np.uint8, "NHWC"), np.uint8, cvcuda.ColorConversion.RGB2YUV),
        (((1, 70, 1, 3), np.uint8, "NHWC"), np.uint8, cvcuda.ColorConversion.YUV2BGR),
        (
            ((1, 165, 230, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.YUV2RGB,
        ),
        (((162, 231, 3), np.uint8, "HWC"), np.uint8, cvcuda.ColorConversion.YUV2RGB),
    ],
)
def test_op_advcvtcolor(input, dtype, code):

    input = cvcuda.Tensor(*input)
    color_specs = [
        cvcuda.ColorSpec.BT601,
        cvcuda.ColorSpec.BT709,
        cvcuda.ColorSpec.BT2020,
    ]

    for spec in color_specs:
        out = cvcuda.advcvtcolor(input, code, spec)
        assert out.layout == input.layout
        assert out.shape == input.shape
        assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    for spec in color_specs:
        out = cvcuda.advcvtcolor_into(input, out, code, spec)
        assert out.layout == input.layout
        assert out.shape == input.shape
        assert out.dtype == input.dtype


@t.mark.parametrize(
    "input,dtype,code",
    [
        # yuv must be even and Nv12/21 yuv must at least contain 3 rows and 2 columns
        (
            ((4, 2, 2, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.YUV2BGR_NV12,
        ),
        (
            ((1, 60, 2, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.YUV2RGB_NV21,
        ),
        (
            ((1, 150, 2, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.YUV2BGR_NV21,
        ),
        (
            ((1, 46, 220, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.YUV2RGB_NV21,
        ),
        (
            ((426, 20, 3), np.uint8, "HWC"),
            np.uint8,
            cvcuda.ColorConversion.YUV2RGB_NV21,
        ),
    ],
)
def test_op_advcvtcolor_FromNV(input, dtype, code):

    # scale input size to fit NV12/21 if conversion is from NV12/21 and set c to 1
    if input[2] == "HWC":
        inputNV = (int((input[0][0] * 3) / 2), input[0][1], 1), input[1], input[2]
    else:
        inputNV = (
            (input[0][0], int((input[0][1] * 3) / 2), input[0][2], 1),
            input[1],
            input[2],
        )

    inputTensor = cvcuda.Tensor(*inputNV)
    color_specs = [
        cvcuda.ColorSpec.BT601,
        cvcuda.ColorSpec.BT709,
        cvcuda.ColorSpec.BT2020,
    ]
    for spec in color_specs:
        out = cvcuda.advcvtcolor(inputTensor, code, spec)
        assert out.layout == inputTensor.layout
        assert out.dtype == inputTensor.dtype

    outTensor = cvcuda.Tensor(out.shape, dtype, out.layout)
    for spec in color_specs:
        cvcuda.advcvtcolor_into(outTensor, inputTensor, code, spec)
        assert outTensor.layout == inputTensor.layout
        assert outTensor.dtype == inputTensor.dtype


@t.mark.parametrize(
    "input,dtype,code",
    [
        (
            ((1, 230, 230, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.BGR2YUV_NV12,
        ),
        (
            ((4, 10, 20, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.RGB2YUV_NV12,
        ),
        (
            ((2, 2, 230, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.BGR2YUV_NV21,
        ),
        (
            ((1, 2, 300, 3), np.uint8, "NHWC"),
            np.uint8,
            cvcuda.ColorConversion.RGB2YUV_NV21,
        ),
        (((2, 30, 3), np.uint8, "HWC"), np.uint8, cvcuda.ColorConversion.RGB2YUV_NV21),
    ],
)
def test_op_advcvtcolor_toNV(input, dtype, code):

    inputTensor = cvcuda.Tensor(*input)
    color_specs = [
        cvcuda.ColorSpec.BT601,
        cvcuda.ColorSpec.BT709,
        cvcuda.ColorSpec.BT2020,
    ]
    for spec in color_specs:
        out = cvcuda.advcvtcolor(inputTensor, code, spec)
        assert out.layout == inputTensor.layout
        assert out.dtype == inputTensor.dtype

    outTensor = cvcuda.Tensor(out.shape, dtype, out.layout)
    for spec in color_specs:
        cvcuda.advcvtcolor_into(outTensor, inputTensor, code, spec)
        assert outTensor.layout == inputTensor.layout
        assert outTensor.dtype == inputTensor.dtype
