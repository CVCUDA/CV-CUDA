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

import cvcuda

import pytest
import numpy as np
import cvcuda_types as cv_types
import cvcuda_tools as cv_tools
import cvcuda_util as util

RNG = np.random.default_rng(0)


def _create_test_image_batch(num_images, img_format, size, max_pixel):
    if img_format.planes > 1:
        image_batch = cvcuda.ImageBatchVarShape(num_images)
        for _ in range(num_images):
            image_batch.pushback(cvcuda.Image(size, img_format))
        return image_batch

    return util.create_image_batch(
        num_images, img_format, size=size, max_random=max_pixel, rng=RNG
    )


@pytest.mark.parametrize(
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
        (
            ((2, 3, 16, 23), np.uint8, "NCHW"),
            cvcuda.ColorConversion.RGB2RGBA,
            ((2, 4, 16, 23), np.uint8, "NCHW"),
        ),
        (
            ((3, 17, 19), np.uint8, "CHW"),
            cvcuda.ColorConversion.RGB2GRAY,
            ((1, 17, 19), np.uint8, "CHW"),
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


@pytest.mark.parametrize(
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
        (
            3,
            cvcuda.Format.BGR8p,
            (23, 21),
            256,
            cvcuda.ColorConversion.BGR2BGRA,
            cvcuda.Format.BGRA8p,
        ),
        (
            3,
            cvcuda.Format.BGR8p,
            (23, 21),
            256,
            cvcuda.ColorConversion.BGR2RGB,
            cvcuda.Format.RGB8p,
        ),
    ],
)
def test_op_cvtcolorvarshape(
    num_images, in_format, img_size, max_pixel, code, out_format
):
    input_batch = _create_test_image_batch(num_images, in_format, img_size, max_pixel)
    output = _create_test_image_batch(num_images, out_format, img_size, max_pixel)
    out = cvcuda.cvtcolor(input_batch, code)
    if in_format.planes > 1:
        assert out.uniqueformat == output.uniqueformat
    assert len(out) == len(output)
    assert out.capacity == output.capacity
    assert out.maxsize == output.maxsize

    stream = cvcuda.Stream()
    tmp = cvcuda.cvtcolor_into(
        src=input_batch,
        dst=output,
        code=code,
        stream=stream,
    )
    assert tmp is output
    assert len(output) == len(input_batch)
    assert output.capacity == input_batch.capacity
    assert output.maxsize == input_batch.maxsize


def test_op_cvtcolorvarshape_planar_rejects_unsupported_auto_output_dtype():
    input_batch = _create_test_image_batch(2, cvcuda.Format.RGBf32p, (23, 21), 1.0)

    with pytest.raises(RuntimeError, match="Unsupported planar var-shape CvtColor"):
        cvcuda.cvtcolor(input_batch, cvcuda.ColorConversion.RGB2BGR)


_valid_conversions: list[tuple[cvcuda.ColorConversion, int]] = [
    # BGR <-> RGB (3 channels only for BGR2RGB/RGB2BGR)
    (cvcuda.ColorConversion.BGR2RGB, 3),
    (cvcuda.ColorConversion.RGB2BGR, 3),
    # BGRA <-> BGR (4 channels in, 3 out)
    (cvcuda.ColorConversion.BGRA2BGR, 4),
    (cvcuda.ColorConversion.RGBA2BGR, 4),
    # BGR <-> BGRA (3 channels in, 4 out)
    (cvcuda.ColorConversion.BGR2BGRA, 3),
    (cvcuda.ColorConversion.BGR2RGBA, 3),
    # RGBA <-> BGRA (4 channels)
    (cvcuda.ColorConversion.RGBA2BGRA, 4),
    (cvcuda.ColorConversion.BGRA2RGBA, 4),
    # GRAY conversions (1-ch in for GRAY2*, 3-ch in for *2GRAY)
    (cvcuda.ColorConversion.GRAY2BGR, 1),
    (cvcuda.ColorConversion.GRAY2RGB, 1),
    (cvcuda.ColorConversion.BGR2GRAY, 3),
    (cvcuda.ColorConversion.RGB2GRAY, 3),
    # HSV conversions (3 channels)
    (cvcuda.ColorConversion.HSV2RGB, 3),
    (cvcuda.ColorConversion.HSV2BGR, 3),
    (cvcuda.ColorConversion.RGB2HSV, 3),
    (cvcuda.ColorConversion.BGR2HSV, 3),
]

_invalid_conversions: list[tuple[cvcuda.ColorConversion, int]] = [
    # BGR2RGB requires exactly 3 channels (invalid: 1, 2, 4)
    (cvcuda.ColorConversion.BGR2RGB, 1),
    (cvcuda.ColorConversion.BGR2RGB, 2),
    (cvcuda.ColorConversion.BGR2RGB, 4),
    # RGBA2BGRA requires 4 channels (invalid: 1, 2, 3)
    (cvcuda.ColorConversion.RGBA2BGRA, 1),
    (cvcuda.ColorConversion.RGBA2BGRA, 2),
    (cvcuda.ColorConversion.RGBA2BGRA, 3),
    # GRAY2BGR requires 1 channel (invalid: 2, 3, 4)
    (cvcuda.ColorConversion.GRAY2BGR, 2),
    (cvcuda.ColorConversion.GRAY2BGR, 3),
    (cvcuda.ColorConversion.GRAY2BGR, 4),
    # BGR2GRAY requires 3 channels (invalid: 1, 2, 4)
    (cvcuda.ColorConversion.BGR2GRAY, 1),
    (cvcuda.ColorConversion.BGR2GRAY, 2),
    (cvcuda.ColorConversion.BGR2GRAY, 4),
    # HSV2RGB requires 3 channels (invalid: 1, 2, 4)
    (cvcuda.ColorConversion.HSV2RGB, 1),
    (cvcuda.ColorConversion.HSV2RGB, 2),
    (cvcuda.ColorConversion.HSV2RGB, 4),
]


def _create_input(channels: int, layout: str):
    height, width = 24, 32
    if layout == "HWC":
        shape = (height, width, channels)
    elif layout == "NHWC":
        shape = (1, height, width, channels)
    elif layout == "CHW":
        shape = (channels, height, width)
    else:  # NCHW
        shape = (1, channels, height, width)
    return cvcuda.Tensor(shape, np.uint8, layout)


def _op(code: cvcuda.ColorConversion, channels: int, layout: str):
    cvcuda.cvtcolor(_create_input(channels, layout), code)


_supported_layouts = {"NHWC", "HWC", "NCHW", "CHW"}


@pytest.mark.parametrize(
    "code,channels",
    [pytest.param(c, ch, id=f"{c.name}-{ch}ch") for c, ch in _valid_conversions],
)
@pytest.mark.parametrize("layout", _supported_layouts)
def test_op_cvtcolor_valid_conversions(code, channels, layout):
    _op(code, channels, layout)


@pytest.mark.parametrize(
    "code,channels",
    [pytest.param(c, ch, id=f"{c.name}-{ch}ch") for c, ch in _invalid_conversions],
)
def test_op_cvtcolor_invalid_conversions(code, channels):
    with pytest.raises(RuntimeError):
        _op(code, channels, "NHWC")


def _cvtcolor_op(src):
    return cvcuda.cvtcolor(src, code=cvcuda.ColorConversion.BGR2RGB)


_supported_dtypes = {cvcuda.Type.U8, cvcuda.Type.U16}
_supported_channels = {3}


@pytest.mark.parametrize("dtype", _supported_dtypes)
@pytest.mark.parametrize("layout", _supported_layouts)
@pytest.mark.parametrize("channels", _supported_channels)
def test_op_cvtcolor_input(dtype, layout, channels):
    cv_tools.assert_layouts(
        _cvtcolor_op, layout, dtype=dtype, wrapper="tensor", channels=channels
    )


@pytest.mark.parametrize("dtype", cv_types.SCALAR_TYPES_SET - _supported_dtypes)
def test_op_cvtcolor_dtype_negative(dtype):
    cv_tools.assert_dtypes(_cvtcolor_op, dtype, wrapper="tensor", negative=True)


@pytest.mark.parametrize("layout", cv_types.IMAGE_LAYOUTS - _supported_layouts)
def test_op_cvtcolor_layout_negative(layout):
    cv_tools.assert_layouts(
        _cvtcolor_op,
        layout,
        dtype=cvcuda.Type.U8,
        wrapper="tensor",
        channels=3,
        negative=True,
    )


@pytest.mark.parametrize("channels", {1, 2, 4})
def test_op_cvtcolor_channels_negative(channels):
    cv_tools.assert_layouts(
        _cvtcolor_op,
        "NHWC",
        dtype=cvcuda.Type.U8,
        wrapper="tensor",
        channels=channels,
        negative=True,
    )


def _cvtcolor_params(dtype, layout, channels):
    return {"code": cvcuda.ColorConversion.BGR2RGB}


def _cvtcolor_varshape_params(dtype, layout, channels):
    return {"code": cvcuda.ColorConversion.BGR2RGB}


globals().update(
    cv_tools.make_op_tests(
        name="cvtcolor",
        runner_info=[
            ("tensor", cvcuda.cvtcolor, _cvtcolor_params),
            ("image_batch", cvcuda.cvtcolor, _cvtcolor_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.U16},
        supported_layouts=_supported_layouts,
        supported_channels={3},
    )
)


# Regression: COLORCVT_MAX and CVT_MAX were sentinel end-of-enum markers
# leaked into Python; they are not real conversion codes and were removed.
@pytest.mark.parametrize("name", ["COLORCVT_MAX", "CVT_MAX"])
def test_color_conversion_sentinels_are_not_exposed(name):
    assert not hasattr(cvcuda.ColorConversion, name)
