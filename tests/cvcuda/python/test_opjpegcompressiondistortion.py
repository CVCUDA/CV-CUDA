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
    "tensor_params, quality",
    [
        (((5, 16, 23, 3), np.uint8, "NHWC"), 50),
        (((4, 9, 3), np.uint8, "HWC"), 1),
        (((3, 3, 88, 13), np.uint8, "NCHW"), 100),
        (((3, 16, 23), np.uint8, "CHW"), 75),
        (((8, 8, 1), np.uint8, "HWC"), 50),
        (((3, 12, 12, 1), np.uint8, "NHWC"), 10),
    ],
)
def test_op_jpegcompressiondistortion(tensor_params, quality):
    src = cvcuda.Tensor(*tensor_params)

    out = cvcuda.jpeg_compression_distortion(src, quality)
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.jpeg_compression_distortion_into(
        src=src, dst=out, quality=quality, stream=stream
    )
    assert tmp is out
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype


def test_op_jpegcompressiondistortion_quality_tensor():
    batch = 4
    src = cvcuda.Tensor((batch, 16, 23, 3), np.uint8, "NHWC")
    quality = util.create_tensor((batch,), np.int32, "N", max_random=100, rng=RNG)

    out = cvcuda.jpeg_compression_distortion(src, quality)
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == src.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.jpeg_compression_distortion_into(
        src=src, dst=out, quality=quality, stream=stream
    )
    assert tmp is out
    assert out.shape == src.shape


@pytest.mark.parametrize(
    "num_images, img_format, img_size, quality",
    [
        (10, cvcuda.Format.RGB8, (123, 321), 50),
        (7, cvcuda.Format.RGB8, (62, 35), 10),
        (4, cvcuda.Format.U8, (26, 52), 90),
    ],
)
def test_op_jpegcompressiondistortion_varshape(
    num_images, img_format, img_size, quality
):
    src = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=256, rng=RNG
    )

    out = cvcuda.jpeg_compression_distortion(src, quality)
    assert len(out) == len(src)
    assert out.capacity == src.capacity
    assert out.uniqueformat == src.uniqueformat
    assert out.maxsize == src.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(src)
    tmp = cvcuda.jpeg_compression_distortion_into(
        src=src, dst=out, quality=quality, stream=stream
    )
    assert tmp is out
    assert len(out) == len(src)
    assert out.capacity == src.capacity


def test_op_jpegcompressiondistortion_varshape_quality_tensor():
    num_images = 5
    src = util.create_image_batch(
        num_images, cvcuda.Format.RGB8, size=(40, 30), max_random=256, rng=RNG
    )
    quality = util.create_tensor((num_images,), np.int32, "N", max_random=100, rng=RNG)

    out = cvcuda.jpeg_compression_distortion(src, quality)
    assert len(out) == len(src)
    assert out.uniqueformat == src.uniqueformat


def test_op_jpegcompressiondistortion_negative_dtype():
    # float32 is outside the supported dtype set (u8 only) and must be rejected.
    src = cvcuda.Tensor((1, 16, 16, 3), np.float32, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.jpeg_compression_distortion(src, 50)


def test_op_jpegcompressiondistortion_negative_channels():
    # 4-channel (RGBA) is outside the supported channel set (1/3) and must be rejected.
    src = cvcuda.Tensor((1, 16, 16, 4), np.uint8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.jpeg_compression_distortion(src, 50)


@pytest.mark.parametrize("quality", [0, 101, -5])
def test_op_jpegcompressiondistortion_negative_scalar_quality(quality):
    # The scalar path host-validates quality; out-of-range values are rejected, not clamped.
    src = cvcuda.Tensor((1, 16, 16, 3), np.uint8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.jpeg_compression_distortion(src, quality)


@pytest.mark.parametrize(
    "quality_params",
    [
        ((2,), np.float32, "N"),  # wrong dtype
        ((1,), np.int32, "N"),  # wrong length (batch is 2)
        ((2, 1), np.int32, "NC"),  # wrong rank
    ],
)
def test_op_jpegcompressiondistortion_negative_quality_tensor(quality_params):
    src = cvcuda.Tensor((2, 16, 16, 3), np.uint8, "NHWC")
    quality = cvcuda.Tensor(*quality_params)
    with pytest.raises(RuntimeError):
        cvcuda.jpeg_compression_distortion(src, quality)


def _jpeg_compression_distortion_params(dtype, layout, channels):
    return {"quality": 50}


globals().update(
    cv_tools.make_op_tests(
        name="jpeg_compression_distortion",
        runner_info=[
            (
                "tensor",
                cvcuda.jpeg_compression_distortion,
                _jpeg_compression_distortion_params,
            ),
            (
                "image_batch",
                cvcuda.jpeg_compression_distortion,
                _jpeg_compression_distortion_params,
            ),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3},
    )
)
