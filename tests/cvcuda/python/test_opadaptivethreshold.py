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

import cvcuda_util as util
import cvcuda_tools as cv_tools

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "tensor_args, adaptive_method, threshold_type",
    [
        (
            ((4, 360, 640, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.AdaptiveThresholdType.MEAN_C,
            cvcuda.ThresholdType.BINARY,
        ),
        (
            ((3, 640, 360, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.AdaptiveThresholdType.GAUSSIAN_C,
            cvcuda.ThresholdType.BINARY,
        ),
        (
            ((2, 1280, 720, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.AdaptiveThresholdType.MEAN_C,
            cvcuda.ThresholdType.BINARY_INV,
        ),
        (
            ((1, 1920, 1080, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.AdaptiveThresholdType.GAUSSIAN_C,
            cvcuda.ThresholdType.BINARY_INV,
        ),
        (
            ((360, 640, 1), cvcuda.Type.U8, "HWC"),
            cvcuda.AdaptiveThresholdType.MEAN_C,
            cvcuda.ThresholdType.BINARY,
        ),
        (
            ((4, 1, 360, 640), cvcuda.Type.U8, "NCHW"),
            cvcuda.AdaptiveThresholdType.MEAN_C,
            cvcuda.ThresholdType.BINARY,
        ),
        (
            ((1, 360, 640), cvcuda.Type.U8, "CHW"),
            cvcuda.AdaptiveThresholdType.GAUSSIAN_C,
            cvcuda.ThresholdType.BINARY_INV,
        ),
    ],
)
def test_op_adaptivethreshold(tensor_args, adaptive_method, threshold_type):
    max_value = 127.0
    block_size = 3
    c = 2
    input = cvcuda.Tensor(*tensor_args)
    out = cvcuda.adaptivethreshold(
        input, max_value, adaptive_method, threshold_type, block_size, c
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.adaptivethreshold_into(
        out, input, max_value, adaptive_method, threshold_type, block_size, c
    )
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.adaptivethreshold(
        src=input,
        max_value=max_value,
        adaptive_method=adaptive_method,
        threshold_type=threshold_type,
        block_size=block_size,
        c=c,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.adaptivethreshold_into(
        src=input,
        dst=out,
        max_value=max_value,
        adaptive_method=adaptive_method,
        threshold_type=threshold_type,
        block_size=block_size,
        c=c,
        stream=stream,
    )
    assert tmp is out


@pytest.mark.parametrize(
    "num_images, img_size, adaptive_method, threshold_type, max_block_size",
    [
        (
            10,
            (123, 321),
            cvcuda.AdaptiveThresholdType.MEAN_C,
            cvcuda.ThresholdType.BINARY,
            11,
        ),
        (
            7,
            (62, 35),
            cvcuda.AdaptiveThresholdType.GAUSSIAN_C,
            cvcuda.ThresholdType.BINARY,
            8,
        ),
        (
            1,
            (33, 48),
            cvcuda.AdaptiveThresholdType.MEAN_C,
            cvcuda.ThresholdType.BINARY_INV,
            7,
        ),
        (
            8,
            (26, 52),
            cvcuda.AdaptiveThresholdType.GAUSSIAN_C,
            cvcuda.ThresholdType.BINARY_INV,
            5,
        ),
    ],
)
def test_op_adaptivethresholdvarshape(
    num_images, img_size, adaptive_method, threshold_type, max_block_size
):

    input = util.create_image_batch(
        num_images, cvcuda.Format.U8, size=img_size, max_random=256, rng=RNG
    )

    max_odd_block_size = (
        max_block_size if max_block_size % 2 == 1 else max_block_size - 1
    )
    block_size = util.to_cvcuda_tensor(
        RNG.integers(
            1,
            (max_odd_block_size + 1) // 2,
            size=(num_images),
            dtype=np.int32,
        )
        * 2
        + 1,
        "N",
    )

    max_value = util.create_tensor(
        (num_images),
        np.float64,
        "N",
        max_random=256,
        rng=RNG,
    )

    c = util.create_tensor(
        (num_images),
        np.float64,
        "N",
        max_random=100,
        rng=RNG,
    )

    out = cvcuda.adaptivethreshold(
        input, max_value, adaptive_method, threshold_type, max_block_size, block_size, c
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.adaptivethreshold_into(
        src=input,
        dst=out,
        max_value=max_value,
        adaptive_method=adaptive_method,
        threshold_type=threshold_type,
        max_block_size=max_block_size,
        block_size=block_size,
        c=c,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _adaptivethreshold_params(dtype, layout, channels):
    return {
        "max_value": 127.0,
        "adaptive_method": cvcuda.AdaptiveThresholdType.MEAN_C,
        "threshold_type": cvcuda.ThresholdType.BINARY,
        "block_size": 3,
        "c": 2,
    }


def _adaptivethreshold_varshape_params(dtype, layout, channels):
    return {
        "max_value": util.to_cvcuda_tensor(
            np.array([127.0, 127.0], dtype=np.float64), "N"
        ),
        "adaptive_method": cvcuda.AdaptiveThresholdType.MEAN_C,
        "threshold_type": cvcuda.ThresholdType.BINARY,
        "max_block_size": 3,
        "block_size": util.to_cvcuda_tensor(np.array([3, 3], dtype=np.int32), "N"),
        "c": util.to_cvcuda_tensor(np.array([2.0, 2.0], dtype=np.float64), "N"),
    }


globals().update(
    cv_tools.make_op_tests(
        name="adaptivethreshold",
        runner_info=[
            ("tensor", cvcuda.adaptivethreshold, _adaptivethreshold_params),
            (
                "image_batch",
                cvcuda.adaptivethreshold,
                _adaptivethreshold_varshape_params,
            ),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 1),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1},
    )
)
