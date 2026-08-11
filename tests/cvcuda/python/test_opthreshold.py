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
import cvcuda_tools as cv_tools
import pytest


@pytest.mark.parametrize(
    "input_args, thtype",
    [
        (
            ((1, 460, 640, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.BINARY,
        ),
        (
            ((5, 640, 460, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.BINARY_INV,
        ),
        (
            ((4, 1920, 1080, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TRUNC,
        ),
        (
            ((2, 1000, 1000, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TOZERO,
        ),
        (
            ((3, 100, 100, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TOZERO_INV,
        ),
        (
            ((5, 460, 640, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.OTSU | cvcuda.ThresholdType.BINARY,
        ),
        (
            ((1, 1000, 1000, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TRIANGLE | cvcuda.ThresholdType.BINARY_INV,
        ),
    ],
)
def test_op_threshold(input_args, thtype):
    input = cvcuda.Tensor(*input_args)

    parameter_shape = (input.shape[0],)
    thresh = cvcuda.Tensor(parameter_shape, cvcuda.Type.F64, "N")
    maxval = cvcuda.Tensor(parameter_shape, cvcuda.Type.F64, "N")

    out = cvcuda.threshold(input, thresh, maxval, thtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.threshold_into(out, input, thresh, maxval, thtype)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.threshold(
        src=input,
        thresh=thresh,
        maxval=maxval,
        type=thtype,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.threshold_into(
        src=input,
        dst=out,
        thresh=thresh,
        maxval=maxval,
        type=thtype,
        stream=stream,
    )
    assert tmp is out


@pytest.mark.parametrize(
    "input_args, thtype",
    [
        (
            ((2, 3, 64, 48), cvcuda.Type.U8, "NCHW"),
            cvcuda.ThresholdType.BINARY,
        ),
        (
            ((4, 32, 48), cvcuda.Type.F32, "CHW"),
            cvcuda.ThresholdType.TRUNC,
        ),
        (
            ((3, 1, 40, 32), cvcuda.Type.U8, "NCHW"),
            cvcuda.ThresholdType.OTSU | cvcuda.ThresholdType.BINARY,
        ),
    ],
)
def test_op_threshold_planar(input_args, thtype):
    input = cvcuda.Tensor(*input_args)
    batch = input.shape[0] if input_args[2].startswith("N") else 1

    thresh = cvcuda.Tensor((batch,), cvcuda.Type.F64, "N")
    maxval = cvcuda.Tensor((batch,), cvcuda.Type.F64, "N")

    out = cvcuda.threshold(input, thresh, maxval, thtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.threshold_into(out, input, thresh, maxval, thtype)
    assert tmp is out


@pytest.mark.parametrize(
    "num_images, format, min_size, max_size, thtype",
    [
        (
            1,
            cvcuda.Format.RGB8,
            (460, 640),
            (480, 720),
            cvcuda.ThresholdType.BINARY,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (640, 460),
            (720, 480),
            cvcuda.ThresholdType.BINARY_INV,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (1920, 1080),
            (1920, 1080),
            cvcuda.ThresholdType.TRUNC,
        ),
        (
            2,
            cvcuda.Format.RGB8,
            (1000, 1000),
            (1000, 1000),
            cvcuda.ThresholdType.TOZERO,
        ),
        (
            3,
            cvcuda.Format.RGB8,
            (100, 100),
            (100, 100),
            cvcuda.ThresholdType.TOZERO_INV,
        ),
        (
            5,
            cvcuda.Format.U8,
            (460, 640),
            (460, 640),
            cvcuda.ThresholdType.OTSU | cvcuda.ThresholdType.BINARY,
        ),
        (
            1,
            cvcuda.Format.U8,
            (1000, 1000),
            (1000, 1000),
            cvcuda.ThresholdType.TRIANGLE | cvcuda.ThresholdType.BINARY_INV,
        ),
    ],
)
def test_op_threshold_varshape(num_images, format, min_size, max_size, thtype):

    parameter_shape = (num_images,)
    thresh = cvcuda.Tensor(parameter_shape, cvcuda.Type.F64, "N")
    maxval = cvcuda.Tensor(parameter_shape, cvcuda.Type.F64, "N")

    input = cvcuda.ImageBatchVarShape(num_images)
    output = cvcuda.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = randint(min_size[0], max_size[0])
        h = randint(min_size[1], max_size[1])
        img_in = cvcuda.Image([w, h], format)
        input.pushback(img_in)
        img_out = cvcuda.Image([w, h], format)
        output.pushback(img_out)

    tmp = cvcuda.threshold(input, thresh, maxval, thtype)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.threshold_into(output, input, thresh, maxval, thtype)
    assert tmp is output

    stream = cvcuda.Stream()
    tmp = cvcuda.threshold(
        src=input,
        thresh=thresh,
        maxval=maxval,
        type=thtype,
        stream=stream,
    )
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.threshold_into(
        src=input,
        dst=output,
        thresh=thresh,
        maxval=maxval,
        type=thtype,
        stream=stream,
    )
    assert tmp is output


@pytest.mark.parametrize(
    "fmt, thtype",
    [
        (cvcuda.Format.RGB8p, cvcuda.ThresholdType.BINARY),
        (cvcuda.Format.RGBf32p, cvcuda.ThresholdType.TRUNC),
        (cvcuda.Format.RGBAf32p, cvcuda.ThresholdType.TOZERO_INV),
    ],
)
def test_op_threshold_varshape_planar(fmt, thtype):
    num_images = 2
    thresh = cvcuda.Tensor((num_images,), cvcuda.Type.F64, "N")
    maxval = cvcuda.Tensor((num_images,), cvcuda.Type.F64, "N")

    input = cvcuda.ImageBatchVarShape(num_images)
    output = cvcuda.ImageBatchVarShape(num_images)
    for i in range(num_images):
        size = (48 + i, 32 + i)
        input.pushback(cvcuda.Image(size, fmt))
        output.pushback(cvcuda.Image(size, fmt))

    tmp = cvcuda.threshold(input, thresh, maxval, thtype)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output, strict=True):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.threshold_into(output, input, thresh, maxval, thtype)
    assert tmp is output


def _threshold_binary_params(dtype, layout, channels):
    return {
        "thresh": cvcuda.Tensor((1,), cvcuda.Type.F64, "N"),
        "maxval": cvcuda.Tensor((1,), cvcuda.Type.F64, "N"),
        "type": cvcuda.ThresholdType.BINARY,
    }


def _threshold_otsu_params(dtype, layout, channels):
    return {
        "thresh": cvcuda.Tensor((1,), cvcuda.Type.F64, "N"),
        "maxval": cvcuda.Tensor((1,), cvcuda.Type.F64, "N"),
        "type": cvcuda.ThresholdType.OTSU | cvcuda.ThresholdType.BINARY,
    }


def _threshold_triangle_params(dtype, layout, channels):
    return {
        "thresh": cvcuda.Tensor((1,), cvcuda.Type.F64, "N"),
        "maxval": cvcuda.Tensor((1,), cvcuda.Type.F64, "N"),
        "type": cvcuda.ThresholdType.TRIANGLE | cvcuda.ThresholdType.BINARY,
    }


globals().update(
    cv_tools.make_op_tests(
        name="threshold_binary",
        runner_info=[
            ("tensor", cvcuda.threshold, _threshold_binary_params),
            ("image_batch", cvcuda.threshold, _threshold_binary_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[(None, "NCHW", 2), (None, "CHW", 2)],
    )
)

globals().update(
    cv_tools.make_op_tests(
        name="threshold_otsu",
        runner_info=[
            ("tensor", cvcuda.threshold, _threshold_otsu_params),
            ("image_batch", cvcuda.threshold, _threshold_otsu_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 1),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1},
    )
)

globals().update(
    cv_tools.make_op_tests(
        name="threshold_triangle",
        runner_info=[
            ("tensor", cvcuda.threshold, _threshold_triangle_params),
            ("image_batch", cvcuda.threshold, _threshold_triangle_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 1),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1},
    )
)
