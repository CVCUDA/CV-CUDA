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


@t.mark.skip(reason="test currently fails CVCUDA-558 tracking")
def test_op_pillowresize_gpuload():
    stream = cvcuda.Stream()
    src_shape = (5, 1080, 1920, 4)
    dst_sizes = [(src_shape[1] // 10, src_shape[2] // 10) for _ in range(src_shape[0])]
    src = util.create_image_batch(
        src_shape[0], cvcuda.Format.RGBA8, max_size=src_shape[1:3], rng=RNG
    )
    host_data = np.ones(src_shape, np.float32)
    torch_dst = torch.tensor(0.0, device="cuda")

    thread = threading.Thread(
        target=lambda: torch.square(
            torch.as_tensor(host_data, device="cuda").abs().max(), out=torch_dst
        )
    )
    thread.start()

    with stream:
        dst = cvcuda.pillowresize(src, dst_sizes)
        assert len(dst) == len(src)
        assert dst.uniqueformat == dst.uniqueformat
        assert dst.maxsize == dst_sizes[0]

    thread.join()
    assert torch_dst.cpu() == 1
