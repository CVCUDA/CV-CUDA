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

import numpy as np
import pytest
import cvcuda_tools as cv_tools
import cvcuda_util as util
import cupy

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "shape,dtype,layout,order",
    [
        ((2, 9, 11, 4), np.uint8, "NHWC", [3, 1, 1, 0]),
        ((7, 5, 3), np.uint16, "HWC", [2, -1, 0]),
        ((2, 3, 8, 6), np.int16, "NCHW", [2, 0, 1]),
        ((4, 6, 5), np.int32, "CHW", [3, 2, -1, 0]),
        ((1, 5, 7, 2), np.float32, "NHWC", [1, 0]),
    ],
)
def test_op_channelreorder_tensor(shape, dtype, layout, order):
    src_h = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    src = util.to_cvcuda_tensor(src_h, layout)
    channel_axis = layout.index("C")
    expected = np.zeros_like(src_h)
    for dst_channel, src_channel in enumerate(order):
        if src_channel >= 0:
            dst_slice = [slice(None)] * len(shape)
            src_slice = [slice(None)] * len(shape)
            dst_slice[channel_axis] = dst_channel
            src_slice[channel_axis] = src_channel
            expected[tuple(dst_slice)] = src_h[tuple(src_slice)]

    out = cvcuda.channelreorder(src, order)
    assert out.shape == src.shape
    assert out.layout == src.layout
    assert out.dtype == src.dtype
    np.testing.assert_array_equal(cupy.asarray(out.cuda()).get(), expected)

    stream = cvcuda.Stream()
    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    returned = cvcuda.channelreorder_into(dst, src, order, stream=stream)
    stream.sync()
    assert returned is dst
    np.testing.assert_array_equal(cupy.asarray(dst.cuda()).get(), expected)


def test_op_channelreorder_tensor_negative():
    src = cvcuda.Tensor((1, 5, 7, 3), np.uint8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.channelreorder(src, [0, 1])
    with pytest.raises(RuntimeError):
        cvcuda.channelreorder(src, [0, 1, 3])
    with pytest.raises(RuntimeError):
        cvcuda.channelreorder_into(src, src, [2, 1, 0])


def test_op_channelreorder_varshape():

    input = util.create_image_batch(10, cvcuda.Format.RGB8, size=(123, 321), rng=RNG)
    order = util.create_tensor((10, 3), np.int32, "NC", max_random=(2, 2, 2), rng=RNG)

    out = cvcuda.channelreorder(input, order)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    order = util.create_tensor(
        (10, 4), np.int32, "NC", max_random=(3, 3, 3, 3), rng=RNG
    )
    out = cvcuda.channelreorder(input, order, format=cvcuda.Format.BGRA8)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == cvcuda.Format.BGRA8
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.channelreorder_into(src=input, dst=out, orders=order, stream=stream)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _channelreorder(data: cvcuda.ImageBatchVarShape) -> cvcuda.ImageBatchVarShape:
    num_images = len(data)
    num_channels = data.uniqueformat.channels
    order_data = np.tile(np.arange(num_channels, dtype=np.int32), (num_images, 1))
    order = cvcuda.as_tensor(cupy.asarray(order_data), "NC")
    return cvcuda.channelreorder(data, order)


def _channelreorder_tensor_params(dtype, layout, channels):
    return {"order": list(range(channels))}


globals().update(
    cv_tools.make_op_tests(
        name="channelreorder",
        runner_info=[
            ("tensor", cvcuda.channelreorder, _channelreorder_tensor_params),
            ("image_batch", _channelreorder, None),
        ],
        supported_formats={
            # 1-channel formats
            cvcuda.Format.U8,
            cvcuda.Format.U16,
            cvcuda.Format.S16,
            cvcuda.Format.S32,
            cvcuda.Format.F32,
            # 3-channel formats
            cvcuda.Format.RGB8,
            cvcuda.Format.BGR8,
            cvcuda.Format.RGB8p,
            cvcuda.Format.BGR8p,
            cvcuda.Format.RGBf32,
            cvcuda.Format.BGRf32,
            cvcuda.Format.RGBf32p,
            cvcuda.Format.BGRf32p,
            # 4-channel formats
            cvcuda.Format.RGBA8,
            cvcuda.Format.BGRA8,
            cvcuda.Format.RGBA8p,
            cvcuda.Format.BGRA8p,
            cvcuda.Format.RGBAf32,
            cvcuda.Format.BGRAf32,
            cvcuda.Format.RGBAf32p,
            cvcuda.Format.BGRAf32p,
        },
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[(None, "NCHW", 2), (None, "CHW", 2)],
    )
)
