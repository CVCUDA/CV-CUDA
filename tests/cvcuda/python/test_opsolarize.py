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

import cvcuda_util as util

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "tensor_params, threshold",
    [
        (((5, 16, 23, 4), np.uint8, "NHWC"), 128.0),
        (((4, 9, 3), np.uint8, "HWC"), 100.0),
        (((3, 88, 13, 1), np.uint16, "NHWC"), 32768.0),
        (((2, 4, 16, 23), np.float32, "NCHW"), 0.5),
        (((3, 8, 8), np.float32, "CHW"), 0.5),
    ],
)
def test_op_solarize(tensor_params, threshold):
    input = cvcuda.Tensor(*tensor_params)

    out = cvcuda.solarize(input, threshold)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.solarize_into(src=input, dst=out, threshold=threshold, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, threshold",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 256, 128.0),
        (7, cvcuda.Format.RGBf32, (62, 35), 1.0, 0.5),
        (1, cvcuda.Format.U16, (33, 48), 1234, 600.0),
        (4, cvcuda.Format.RGBA8, (26, 52), 256, 100.0),
    ],
)
def test_op_solarize_varshape(num_images, img_format, img_size, max_pixel, threshold):
    input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    out = cvcuda.solarize(input, threshold)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.solarize_into(src=input, dst=out, threshold=threshold, stream=stream)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity


def test_op_solarize_negative_dtype():
    # float16 is outside the supported dtype set (u8/u16/f32) and must be rejected.
    input = cvcuda.Tensor((1, 16, 16, 3), np.float16, "NHWC")
    with pytest.raises(Exception):
        cvcuda.solarize(input, 0.5)
