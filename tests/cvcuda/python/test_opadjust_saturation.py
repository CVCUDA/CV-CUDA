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
    "tensor_params, saturation",
    [
        (((5, 16, 23, 3), np.uint8, "NHWC"), 0.5),
        (((4, 9, 3), np.uint8, "HWC"), 0.0),
        (((3, 3, 88, 13), np.uint8, "NCHW"), 1.7),
        (((2, 3, 16, 23), np.float32, "NCHW"), 0.5),
        (((8, 8, 1), np.float32, "HWC"), 0.5),
        (((3, 12, 12, 1), np.uint8, "NHWC"), 0.3),
    ],
)
def test_op_adjustsaturation(tensor_params, saturation):
    src = cvcuda.Tensor(*tensor_params)

    out = cvcuda.adjust_saturation(src, saturation)
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.adjust_saturation_into(
        src=src, dst=out, saturation=saturation, stream=stream
    )
    assert tmp is out
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype


@pytest.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, saturation",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 256, 0.5),
        (7, cvcuda.Format.RGBf32, (62, 35), 1.0, 1.4),
        (4, cvcuda.Format.U8, (26, 52), 256, 0.5),
    ],
)
def test_op_adjustsaturation_varshape(
    num_images, img_format, img_size, max_pixel, saturation
):
    src = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    out = cvcuda.adjust_saturation(src, saturation)
    assert len(out) == len(src)
    assert out.capacity == src.capacity
    assert out.uniqueformat == src.uniqueformat
    assert out.maxsize == src.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(src)
    tmp = cvcuda.adjust_saturation_into(
        src=src, dst=out, saturation=saturation, stream=stream
    )
    assert tmp is out
    assert len(out) == len(src)
    assert out.capacity == src.capacity


def test_op_adjustsaturation_negative_dtype():
    # uint16 is outside the supported dtype set (u8/f32) and must be rejected.
    src = cvcuda.Tensor((1, 16, 16, 3), np.uint16, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.adjust_saturation(src, 0.5)


def test_op_adjustsaturation_negative_channels():
    # 4-channel (RGBA) is outside the supported channel set (1/3) and must be rejected.
    src = cvcuda.Tensor((1, 16, 16, 4), np.uint8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.adjust_saturation(src, 0.5)


def test_op_adjustsaturation_negative_saturation():
    src = cvcuda.Tensor((1, 16, 16, 3), np.uint8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.adjust_saturation(src, -1.0)


def _adjust_saturation_params(dtype, layout, channels):
    return {"saturation": 0.5}


globals().update(
    cv_tools.make_op_tests(
        name="adjust_saturation",
        runner_info=[
            ("tensor", cvcuda.adjust_saturation, _adjust_saturation_params),
            ("image_batch", cvcuda.adjust_saturation, _adjust_saturation_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.F32},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3},
    )
)
