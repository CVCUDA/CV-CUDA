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


@t.mark.parametrize(
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

    block_size = util.create_tensor(
        (num_images),
        np.int32,
        "N",
        max_random=max_block_size,
        rng=RNG,
        transform_dist=util.dist_odd,
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

    stream = cvcuda.cuda.Stream()
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
