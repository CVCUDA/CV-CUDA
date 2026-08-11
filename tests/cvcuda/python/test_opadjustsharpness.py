# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cvcuda_util as util

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "tensor_params, sharpness_factor",
    [
        (((5, 16, 23, 4), np.uint8, "NHWC"), 2.0),
        (((4, 9, 3), np.uint8, "HWC"), 0.0),
        (((3, 88, 13, 1), np.uint16, "NHWC"), 1.5),
        (((2, 4, 16, 23), np.float32, "NCHW"), 0.5),
        (((3, 8, 8), np.float32, "CHW"), 2.0),
    ],
)
def test_op_adjust_sharpness(tensor_params, sharpness_factor):
    input = cvcuda.Tensor(*tensor_params)

    out = cvcuda.adjust_sharpness(input, sharpness_factor)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.adjust_sharpness_into(
        src=input, dst=out, sharpness_factor=sharpness_factor, stream=stream
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, sharpness_factor",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 256, 2.0),
        (7, cvcuda.Format.RGBf32, (62, 35), 1.0, 0.5),
        (1, cvcuda.Format.U16, (33, 48), 1234, 1.5),
        (4, cvcuda.Format.RGBA8, (26, 52), 256, 0.0),
    ],
)
def test_op_adjust_sharpness_varshape(
    num_images, img_format, img_size, max_pixel, sharpness_factor
):
    input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    out = cvcuda.adjust_sharpness(input, sharpness_factor)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.adjust_sharpness_into(
        src=input, dst=out, sharpness_factor=sharpness_factor, stream=stream
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity


def test_op_adjust_sharpness_negative_dtype():
    # float16 is outside the supported dtype set (u8/u16/f32) and must be rejected.
    input = cvcuda.Tensor((1, 16, 16, 3), np.float16, "NHWC")
    with pytest.raises(Exception):
        cvcuda.adjust_sharpness(input, 2.0)


def test_op_adjust_sharpness_negative_factor():
    input = cvcuda.Tensor((1, 16, 16, 3), np.uint8, "NHWC")
    with pytest.raises(Exception):
        cvcuda.adjust_sharpness(input, -0.1)


def _adjust_sharpness_params(dtype, layout, channels):
    return {"sharpness_factor": 2.0}


globals().update(
    cv_tools.make_op_tests(
        name="adjustsharpness",
        runner_info=[
            ("tensor", cvcuda.adjust_sharpness, _adjust_sharpness_params),
            ("image_batch", cvcuda.adjust_sharpness, _adjust_sharpness_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.U16, cvcuda.Type.F32},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
    )
)
