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
import threading
import torch


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input_args,out_shape,interp,fmt",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            (5, 132, 15, 4),
            cvcuda.Interp.LINEAR,
            cvcuda.Format.RGB8,
        ),
        (
            ((5, 31, 31, 4), np.uint8, "NHWC"),
            (5, 55, 55, 4),
            cvcuda.Interp.LINEAR,
            cvcuda.Format.RGB8,
        ),
        (
            ((5, 55, 55, 4), np.uint8, "NHWC"),
            (5, 31, 31, 4),
            cvcuda.Interp.LINEAR,
            cvcuda.Format.RGB8,
        ),
        (
            ((5, 16, 23, 4), np.float32, "NHWC"),
            (5, 132, 15, 4),
            cvcuda.Interp.LINEAR,
            cvcuda.Format.RGBf32,
        ),
        (
            ((5, 31, 31, 4), np.float32, "NHWC"),
            (5, 55, 55, 4),
            cvcuda.Interp.LINEAR,
            cvcuda.Format.RGBf32,
        ),
        (
            ((5, 55, 55, 4), np.float32, "NHWC"),
            (5, 31, 31, 4),
            cvcuda.Interp.LINEAR,
            cvcuda.Format.RGBf32,
        ),
        (
            ((5, 55, 55, 4), np.float32, "NHWC"),
            (5, 31, 31, 4),
            cvcuda.Interp.CUBIC,
            cvcuda.Format.RGBf32,
        ),
        (
            ((5, 55, 55, 4), np.float32, "NHWC"),
            (5, 31, 31, 4),
            cvcuda.Interp.LANCZOS,
            cvcuda.Format.RGBf32,
        ),
        (
            ((5, 55, 55, 4), np.float32, "NHWC"),
            (5, 31, 31, 4),
            cvcuda.Interp.HAMMING,
            cvcuda.Format.RGBf32,
        ),
        (
            ((5, 55, 55, 4), np.float32, "NHWC"),
            (5, 31, 31, 4),
            cvcuda.Interp.BOX,
            cvcuda.Format.RGBf32,
        ),
    ],
)
def test_op_pillowresize(input_args, out_shape, interp, fmt):
    input = cvcuda.Tensor(*input_args)

    out = cvcuda.pillowresize(input, out_shape, fmt, interp)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(out_shape, input.dtype, input.layout)
    tmp = cvcuda.pillowresize_into(out, input, fmt, interp)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    tmp = cvcuda.pillowresize_into(
        src=input,
        dst=out,
        format=fmt,
        interp=interp,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "nimages, format, max_size, max_pixel, interp",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            cvcuda.Interp.LINEAR,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (14, 14),
            256.0,
            cvcuda.Interp.LINEAR,
        ),
        (
            7,
            cvcuda.Format.RGBf32,
            (10, 15),
            256.0,
            cvcuda.Interp.LINEAR,
        ),
    ],
)
def test_op_pillowresizevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    interp,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    base_output = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    sizes = []
    for image in base_output:
        sizes.append([image.width, image.height])

    out = cvcuda.pillowresize(input, sizes)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == base_output.maxsize

    out = cvcuda.pillowresize(
        input,
        sizes,
        interp,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == base_output.maxsize

    stream = cvcuda.Stream()

    tmp = cvcuda.pillowresize_into(
        src=input,
        dst=base_output,
        interp=interp,
        stream=stream,
    )
    assert tmp is base_output
    assert len(base_output) == len(input)
    assert base_output.capacity == input.capacity
    assert base_output.uniqueformat == input.uniqueformat
    assert base_output.maxsize == base_output.maxsize


def test_op_pillowresize_reused_from_cache():
    fmt = cvcuda.Format.RGBA8
    src = [
        cvcuda.Tensor((3, 51, 15, 4), np.uint8, "NHWC"),
        util.create_image_batch(2, fmt, max_size=(12, 35), rng=RNG),
    ]
    dst = [
        cvcuda.Tensor((3, 13, 31, 4), np.uint8, "NHWC"),
        cvcuda.Tensor((3, 46, 33, 4), np.uint8, "NHWC"),
        util.create_image_batch(2, fmt, max_size=(47, 32), rng=RNG),
    ]

    cvcuda.pillowresize_into(dst[0], src[0], fmt)

    items_in_cache = cvcuda.cache_size()

    cvcuda.pillowresize_into(dst[1], src[0], fmt)
    cvcuda.pillowresize_into(dst[2], src[1])

    assert cvcuda.cache_size() == items_in_cache


def test_op_pillowresize_gpuload():
    src_shape = (5, 1080, 1920, 4)
    dst_shape = (5, 108, 192, 4)
    fmt, dtype, layout = cvcuda.Format.RGBA8, np.uint8, "NHWC"
    src = cvcuda.Tensor(src_shape, dtype, layout)
    dst = cvcuda.Tensor(dst_shape, dtype, layout)

    torch0 = torch.zeros(src_shape, dtype=torch.int32, device="cuda")
    torch1 = torch.zeros(src_shape, dtype=torch.int32, device="cuda")

    thread = threading.Thread(
        target=lambda: (torch.abs(torch0, out=torch1), torch.square(torch1, out=torch0))
    )
    thread.start()

    tmp = cvcuda.pillowresize_into(dst, src, fmt, cvcuda.Interp.LANCZOS)
    assert tmp is dst
    assert tmp.layout == layout
    assert tmp.shape == dst_shape
    assert tmp.dtype == dtype

    thread.join()
    assert torch0.shape == src_shape
    assert torch1.shape == src_shape


def test_op_pillowresize_user_stream_with_tensor():
    stream = cvcuda.Stream()
    src_shape = (5, 1080, 1920, 4)
    dst_shape = (5, 108, 192, 4)
    fmt, dtype, layout = cvcuda.Format.RGBA8, np.uint8, "NHWC"
    src = cvcuda.Tensor(src_shape, dtype, layout)

    with stream:
        dst = cvcuda.pillowresize(src, dst_shape, fmt, cvcuda.Interp.BOX)
        assert dst.layout == layout
        assert dst.shape == dst_shape
        assert dst.dtype == dtype


@t.mark.parametrize(
    "batch_size",
    [
        3,
        4,
    ],
)
def test_op_pillowresize_user_stream_with_image_batch(batch_size):
    stream = cvcuda.Stream()
    src_shape = (batch_size, 1080, 1920, 4)
    dst_shape = (batch_size, 108, 192, 4)
    dst_sizes = [(dst_shape[2], dst_shape[1]) for _ in range(dst_shape[0])]
    src = util.create_image_batch(
        src_shape[0],
        cvcuda.Format.RGBA8,
        max_size=(dst_shape[2], dst_shape[1]),
        rng=RNG,
    )

    with stream:
        dst = cvcuda.pillowresize(src, dst_sizes, cvcuda.Interp.BOX)
        assert len(dst) == len(src)
        assert dst.uniqueformat == dst.uniqueformat
        assert dst.maxsize == dst_sizes[0]
