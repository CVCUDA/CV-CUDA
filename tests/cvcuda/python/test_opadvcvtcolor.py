# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import numpy as np
import cvcuda_tools as cv_tools


@pytest.mark.parametrize(
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
        (((1, 3, 19, 23), np.uint8, "NCHW"), np.uint8, cvcuda.ColorConversion.RGB2YUV),
        (((3, 17, 21), np.uint8, "CHW"), np.uint8, cvcuda.ColorConversion.YUV2BGR),
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


@pytest.mark.parametrize(
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
        (
            ((1, 3, 24, 32), np.uint8, "NCHW"),
            np.uint8,
            cvcuda.ColorConversion.YUV2BGR_NV12,
        ),
        (
            ((3, 20, 28), np.uint8, "CHW"),
            np.uint8,
            cvcuda.ColorConversion.YUV2RGB_NV21,
        ),
    ],
)
def test_op_advcvtcolor_FromNV(input, dtype, code):

    # scale input size to fit NV12/21 if conversion is from NV12/21 and set c to 1
    if input[2] == "HWC":
        inputNV = (int((input[0][0] * 3) / 2), input[0][1], 1), input[1], input[2]
    elif input[2] == "NHWC":
        inputNV = (
            (input[0][0], int((input[0][1] * 3) / 2), input[0][2], 1),
            input[1],
            input[2],
        )
    elif input[2] == "CHW":
        inputNV = (1, int((input[0][1] * 3) / 2), input[0][2]), input[1], input[2]
    else:
        inputNV = (
            (input[0][0], 1, int((input[0][2] * 3) / 2), input[0][3]),
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


@pytest.mark.parametrize(
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
        (
            ((1, 3, 24, 32), np.uint8, "NCHW"),
            np.uint8,
            cvcuda.ColorConversion.BGR2YUV_NV12,
        ),
        (((3, 20, 28), np.uint8, "CHW"), np.uint8, cvcuda.ColorConversion.RGB2YUV_NV21),
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


_valid_conversions: list[tuple[cvcuda.ColorConversion, int, bool]] = [
    # Interleaved 444: RGB/BGR <-> YUV (3-ch in, 3-ch out)
    (cvcuda.ColorConversion.RGB2YUV, 3, False),
    (cvcuda.ColorConversion.BGR2YUV, 3, False),
    (cvcuda.ColorConversion.YUV2RGB, 3, False),
    (cvcuda.ColorConversion.YUV2BGR, 3, False),
    # RGB/BGR -> NV12/NV21 (3 or 4-ch in)
    (cvcuda.ColorConversion.RGB2YUV_NV12, 3, False),
    (cvcuda.ColorConversion.RGB2YUV_NV12, 4, False),
    (cvcuda.ColorConversion.RGB2YUV_NV21, 3, False),
    (cvcuda.ColorConversion.RGB2YUV_NV21, 4, False),
    (cvcuda.ColorConversion.BGR2YUV_NV12, 3, False),
    (cvcuda.ColorConversion.BGR2YUV_NV12, 4, False),
    (cvcuda.ColorConversion.BGR2YUV_NV21, 3, False),
    (cvcuda.ColorConversion.BGR2YUV_NV21, 4, False),
    # NV12/NV21 -> RGB/BGR (1-ch in with H*3/2, 3 or 4-ch out)
    (cvcuda.ColorConversion.YUV2RGB_NV12, 1, True),
    (cvcuda.ColorConversion.YUV2BGR_NV12, 1, True),
    (cvcuda.ColorConversion.YUV2RGB_NV21, 1, True),
    (cvcuda.ColorConversion.YUV2BGR_NV21, 1, True),
]
# Curated subset of unsupported (code, channels) pairs.  We cannot derive
# this automatically from _valid_conversions because some untested conversion/
# channel combos crash the operator instead of raising an error.
_invalid_conversions: list[tuple[cvcuda.ColorConversion, int, bool]] = [
    # 444 codes only support 3-ch (invalid: 1, 2, 4)
    (cvcuda.ColorConversion.RGB2YUV, 1, False),
    (cvcuda.ColorConversion.RGB2YUV, 2, False),
    (cvcuda.ColorConversion.RGB2YUV, 4, False),
    (cvcuda.ColorConversion.BGR2YUV, 1, False),
    (cvcuda.ColorConversion.BGR2YUV, 2, False),
    (cvcuda.ColorConversion.BGR2YUV, 4, False),
    (cvcuda.ColorConversion.YUV2RGB, 1, False),
    (cvcuda.ColorConversion.YUV2RGB, 2, False),
    (cvcuda.ColorConversion.YUV2RGB, 4, False),
    (cvcuda.ColorConversion.YUV2BGR, 1, False),
    (cvcuda.ColorConversion.YUV2BGR, 2, False),
    (cvcuda.ColorConversion.YUV2BGR, 4, False),
    # To-NV codes support 3 or 4-ch (invalid: 1, 2)
    (cvcuda.ColorConversion.RGB2YUV_NV12, 1, False),
    (cvcuda.ColorConversion.RGB2YUV_NV12, 2, False),
    (cvcuda.ColorConversion.RGB2YUV_NV21, 1, False),
    (cvcuda.ColorConversion.RGB2YUV_NV21, 2, False),
    (cvcuda.ColorConversion.BGR2YUV_NV12, 1, False),
    (cvcuda.ColorConversion.BGR2YUV_NV12, 2, False),
    (cvcuda.ColorConversion.BGR2YUV_NV21, 1, False),
    (cvcuda.ColorConversion.BGR2YUV_NV21, 2, False),
    # From-NV codes support 1-ch input (invalid: 2, 3, 4)
    (cvcuda.ColorConversion.YUV2RGB_NV12, 2, True),
    (cvcuda.ColorConversion.YUV2RGB_NV12, 3, True),
    (cvcuda.ColorConversion.YUV2RGB_NV12, 4, True),
    (cvcuda.ColorConversion.YUV2BGR_NV12, 2, True),
    (cvcuda.ColorConversion.YUV2BGR_NV12, 3, True),
    (cvcuda.ColorConversion.YUV2BGR_NV12, 4, True),
    (cvcuda.ColorConversion.YUV2RGB_NV21, 2, True),
    (cvcuda.ColorConversion.YUV2RGB_NV21, 3, True),
    (cvcuda.ColorConversion.YUV2RGB_NV21, 4, True),
    (cvcuda.ColorConversion.YUV2BGR_NV21, 2, True),
    (cvcuda.ColorConversion.YUV2BGR_NV21, 3, True),
    (cvcuda.ColorConversion.YUV2BGR_NV21, 4, True),
]
_supported_layouts = {"NHWC", "HWC", "NCHW", "CHW"}


def _create_input(channels: int, is_from_nv: bool, layout: str):
    # Use even dimensions for NV12/21 compatibility
    height, width = 24, 32
    if layout == "HWC":
        if is_from_nv:
            shape = ((height * 3) // 2, width, channels)
        else:
            shape = (height, width, channels)
    elif layout == "NHWC":
        if is_from_nv:
            shape = (1, (height * 3) // 2, width, channels)
        else:
            shape = (1, height, width, channels)
    elif layout == "CHW":
        if is_from_nv:
            shape = (channels, (height * 3) // 2, width)
        else:
            shape = (channels, height, width)
    else:  # NCHW
        if is_from_nv:
            shape = (1, channels, (height * 3) // 2, width)
        else:
            shape = (1, channels, height, width)
    return cvcuda.Tensor(shape, np.uint8, layout)


def _op(code: cvcuda.ColorConversion, channels: int, is_from_nv: bool, layout: str):
    cvcuda.advcvtcolor(
        _create_input(channels, is_from_nv, layout), code, cvcuda.ColorSpec.BT601
    )


@pytest.mark.parametrize(
    "code,channels,is_from_nv",
    [
        pytest.param(c, ch, nv, id=f"{c.name}-{ch}ch")
        for c, ch, nv in _valid_conversions
    ],
)
@pytest.mark.parametrize("layout", _supported_layouts)
def test_op_advcvtcolor_valid_conversions(code, channels, is_from_nv, layout):
    _op(code, channels, is_from_nv, layout)


@pytest.mark.parametrize(
    "code,channels,is_from_nv",
    [
        pytest.param(c, ch, nv, id=f"{c.name}-{ch}ch")
        for c, ch, nv in _invalid_conversions
    ],
)
def test_op_advcvtcolor_invalid_conversions(code, channels, is_from_nv):
    with pytest.raises(RuntimeError):
        _op(code, channels, is_from_nv, "NHWC")


def _advcvtcolor(src: cvcuda.Tensor, spec: cvcuda.ColorSpec):
    return cvcuda.advcvtcolor(src, code=cvcuda.ColorConversion.RGB2YUV, spec=spec)


def _advcvtcolor_params(dtype, layout, channels, spec):
    return {
        "spec": spec,
    }


globals().update(
    cv_tools.make_op_tests(
        name="advcvtcolor",
        runner_info=[("tensor", _advcvtcolor, _advcvtcolor_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts=_supported_layouts,
        supported_channels={3},
        extra_params={
            "spec": {
                cvcuda.ColorSpec.BT601,
                cvcuda.ColorSpec.BT709,
                cvcuda.ColorSpec.BT2020,
            }
        },
    )
)
