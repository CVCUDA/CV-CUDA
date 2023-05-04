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

import pytest as t
import nvcv
import numpy as np
import torch
import nvcv_util as util


def test_image_creation_works():
    img = nvcv.Image((7, 5), nvcv.Format.NV12)
    assert img.width == 7
    assert img.height == 5
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.NV12


def test_image_creation_arg_keywords():
    img = nvcv.Image(size=(7, 5), format=nvcv.Format.NV12)
    assert img.width == 7
    assert img.height == 5
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.NV12


buffmt_common = [
    # packed formats
    ([5, 7, 1], np.uint8, nvcv.Format.U8),
    ([5, 7, 1], np.uint8, nvcv.Format.U8),
    ([5, 7, 1], np.uint8, nvcv.Format.U8),
    ([5, 7], np.uint8, nvcv.Format.U8),
    ([5, 7, 1], np.int8, nvcv.Format.S8),
    ([5, 7, 1], np.uint16, nvcv.Format.U16),
    ([5, 7, 1], np.int16, nvcv.Format.S16),
    ([5, 7, 2], np.int16, nvcv.Format._2S16),
    ([5, 7, 1], np.float32, nvcv.Format.F32),
    ([5, 7, 1], np.float64, nvcv.Format.F64),
    ([5, 7, 2], np.float32, nvcv.Format._2F32),
    ([5, 7, 3], np.uint8, nvcv.Format.RGB8),
    ([5, 7, 4], np.uint8, nvcv.Format.RGBA8),
    ([1, 5, 7], np.uint8, nvcv.Format.U8),
    ([1, 5, 7, 4], np.uint8, nvcv.Format.RGBA8),
    ([5, 7], np.csingle, nvcv.Format.C64),
    ([5, 7], np.cdouble, nvcv.Format.C128),
    ([5, 7], np.dtype("2f"), nvcv.Format._2F32),
]


@t.mark.parametrize("shape,dt,format", buffmt_common)
def test_wrap_host_buffer_infer_imgformat(shape, dt, format):
    img = nvcv.Image(np.ndarray(shape, dt))
    assert img.width == 7
    assert img.height == 5
    assert img.format == format

    img = nvcv.as_image(util.to_cuda_buffer(np.ndarray(shape, dt)))
    assert img.width == 7
    assert img.height == 5
    assert img.format == format


@t.mark.parametrize(
    "shape,dt,format",
    buffmt_common
    + [
        ([5, 7, 1], np.uint8, nvcv.Format.Y8),
        ([5, 7, 3], np.uint8, nvcv.Format.BGR8),
        ([5, 7, 4], np.uint8, nvcv.Format.BGRA8),
    ],
)
def test_wrap_host_buffer_explicit_format(shape, dt, format):
    img = nvcv.Image(np.ndarray(shape, dt), format)
    assert img.width == 7
    assert img.height == 5
    assert img.format == format

    img = nvcv.as_image(util.to_cuda_buffer(np.ndarray(shape, dt)), format)
    assert img.width == 7
    assert img.height == 5
    assert img.format == format


buffmt2_common = [
    # packed formats
    (
        [((6, 8), np.uint8, torch.uint8), ((3, 4, 2), np.uint8, torch.uint8)],
        nvcv.Format.NV12_ER,
    )
]


@t.mark.parametrize("buffers,format", buffmt2_common)
def test_wrap_host_buffer_infer_imgformat_multiple_planes(buffers, format):
    img = nvcv.Image([np.ndarray(buf[0], buf[1]) for buf in buffers])
    assert img.width == 8
    assert img.height == 6
    assert img.format == format

    img = nvcv.as_image(
        [
            torch.zeros(size=buf[0], dtype=buf[2], device="cuda").cuda()
            for buf in buffers
        ]
    )
    assert img.width == 8
    assert img.height == 6
    assert img.format == format


@t.mark.parametrize("buffers,format", buffmt2_common)
def test_wrap_host_buffer_explicit_format2(buffers, format):
    img = nvcv.Image([np.ndarray(buf[0], buf[1]) for buf in buffers], format)
    assert img.width == 8
    assert img.height == 6
    assert img.format == format

    img = nvcv.as_image(
        [
            torch.zeros(size=buf[0], dtype=buf[2], device="cuda").cuda()
            for buf in buffers
        ],
        format,
    )
    assert img.width == 8
    assert img.height == 6
    assert img.format == format


@t.mark.parametrize(
    "shape,dt,planes,height,width,channels",
    [
        ([2, 7, 6], np.uint8, 2, 7, 6, 2),
        ([1, 2, 7, 6], np.uint8, 2, 7, 6, 2),
        ([2, 7, 3], np.uint8, 1, 2, 7, 3),
        ([1, 7, 3], np.uint8, 1, 1, 7, 3),
        ([7, 3], np.uint8, 1, 7, 3, 1),
        ([7, 1], np.uint8, 1, 7, 1, 1),
        ([1, 3], np.uint8, 1, 1, 3, 1),
        ([1, 1], np.uint8, 1, 1, 1, 1),
        ([5, 7, 3], np.uint8, 1, 5, 7, 3),
    ],
)
def test_wrap_host_buffer_infer_format_geometry(
    shape, dt, planes, height, width, channels
):
    img = nvcv.Image(np.ndarray(shape, dt))
    assert img.width == width
    assert img.height == height
    assert img.format.planes == planes
    assert img.format.channels == channels

    img = nvcv.as_image(util.to_cuda_buffer(np.ndarray(shape, dt)))
    assert img.width == width
    assert img.height == height
    assert img.format.planes == planes
    assert img.format.channels == channels


def test_wrap_host_buffer_arg_keywords():
    img = nvcv.Image(buffer=np.ndarray([5, 7], np.float32), format=nvcv.Format.F32)
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32

    img = nvcv.as_image(
        buffer=util.to_cuda_buffer(np.ndarray([5, 7], np.float32)),
        format=nvcv.Format.F32,
    )
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32


def test_wrap_host_buffer_infer_format_arg_keywords():
    img = nvcv.Image(buffer=np.ndarray([5, 7], np.float32))
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32

    img = nvcv.as_image(buffer=util.to_cuda_buffer(np.ndarray([5, 7], np.float32)))
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32


def test_wrap_host_image_with_format__buffer_has_unsupported_type():
    with t.raises(ValueError):
        nvcv.Image(np.array([1 + 2j, 4 + 7j]), nvcv.Format._2F32)


def test_wrap_host_image__buffer_has_unsupported_type():
    with t.raises(ValueError):
        nvcv.Image(np.array([1 + 2j, 4 + 7j]))


def test_wrap_host_image__format_and_buffer_type_mismatch():
    with t.raises(ValueError):
        nvcv.Image(np.array([1.4, 2.85]), nvcv.Format.U8)


def test_wrap_host_image__only_pitch_linear():
    with t.raises(ValueError):
        nvcv.Image(np.ndarray([6, 4], np.uint8), nvcv.Format.Y8_BL)


def test_wrap_host_image__css_with_one_plane_failure():
    with t.raises(ValueError):
        nvcv.Image(np.ndarray([6, 4], np.uint8), nvcv.Format.NV12)


@t.mark.parametrize(
    "shape",
    [
        (5, 3, 4),  # Buffer shape HCW not supported
        (5, 7, 4),  # Buffer shape doesn't correspond to image format
    ],
)
def test_wrap_host_image_with_format__invalid_shape(shape):
    with t.raises(ValueError):
        nvcv.Image(np.ndarray(shape, np.uint8), nvcv.Format.RGB8)


@t.mark.parametrize(
    "shape",
    [
        # When buffer's number of dimensions is 4, first dimension must be 1, not 2
        (
            2,
            3,
            4,
            5,
        ),
        # Number of dimensions must be between 1 and 4, not 5
        (1, 1, 15, 7, 1),
        # Number of dimensions must be between 1 and 4, not 0
        (0,),
        # Buffer shape not supported
        (8, 7, 9),
    ],
)
def test_wrap_host_invalid_dims(shape):
    with t.raises(ValueError):
        nvcv.Image(np.ndarray(shape))


@t.mark.parametrize(
    "s",
    [
        # Fastest changing dimension must be packed, i.e.,
        # have stride equal to 1 bytes(s), not 2
        (2 * 2, 2),
        # Buffer strides must all be >= 0
        (0, 1),
    ],
)
def test_wrap_host_invalid_strides(s):
    with t.raises(ValueError):
        nvcv.Image(
            np.ndarray(
                shape=(3, 2), strides=s, buffer=bytearray(s[0] * 3), dtype=np.uint8
            )
        )


@t.mark.parametrize(
    "shapes",
    [
        # When wrapping multiple buffers, buffers with 4
        # dimensions must have first dimension == 1, not 2
        [
            (3, 4),
            (2, 2, 3, 1),
        ],
        # Number of buffer#1's dimensions must be
        # between 1 and 4, not 5
        [
            (3, 4),
            (5, 2, 2, 3, 1),
        ],
    ],
)
def test_wrap_host_multiplane_invalid_dims(shapes):
    buffers = []
    for shape in shapes:
        buffers.append(np.ndarray(shape, np.uint8))

    with t.raises(ValueError):
        nvcv.Image(buffers)


def test_image_wrap_invalid_cuda_buffer():
    class NonCudaMemory(object):
        pass

    obj = NonCudaMemory()
    obj.__cuda_array_interface__ = {
        "shape": (1, 1),
        "typestr": "i",
        "data": (419, True),
        "version": 3,
    }

    with t.raises(RuntimeError):
        nvcv.as_image(obj)


def test_image_create_packed():
    img = nvcv.Image((37, 11), nvcv.Format.U8, rowalign=1)
    assert img.cuda().strides == (37, 1)


def test_image_create_zeros_packed():
    img = nvcv.Image.zeros((37, 11), nvcv.Format.U8, rowalign=1)
    assert img.cuda().strides == (37, 1)


def test_image_create_from_host_packed():
    img = nvcv.Image(np.ndarray((11, 37), np.uint8), rowalign=1)
    assert img.cuda().strides == (37, 1)


@t.mark.parametrize(
    "size,format,layout,out_dtype, out_shape, simple_layout",
    [
        ((257, 231), nvcv.Format.U8, None, np.uint8, (231, 257), "HWC"),
        ((257, 231), nvcv.Format.U8, "HWC", np.uint8, (231, 257, 1), "HWC"),
        ((257, 231), nvcv.Format.U8, "CHW", np.uint8, (1, 231, 257), "CHW"),
        (
            (257, 231),
            nvcv.Format.U8,
            "xyCrodHlimaWab",
            np.uint8,
            (1, 1, 1, 1, 1, 1, 231, 1, 1, 1, 1, 257, 1, 1),
            "CHW",
        ),
        ((257, 231), nvcv.Format.RGBAf32, None, np.float32, (231, 257, 4), "HWC"),
        ((257, 231), nvcv.Format.RGBA8, "HWC", np.uint8, (231, 257, 4), "HWC"),
        ((257, 231), nvcv.Format.RGBA8p, None, np.uint8, (4, 231, 257), "CHW"),
        ((257, 231), nvcv.Format.RGBAf32p, "CHW", np.float32, (4, 231, 257), "CHW"),
        (
            (258, 232),
            nvcv.Format.NV12,
            None,
            [np.uint8, np.uint8],
            [(232, 258, 1), (232 // 2, 258 // 2, 2)],
            "HWC",
        ),
        (
            (258, 232),
            nvcv.Format.NV12,
            "HWC",
            [np.uint8, np.uint8],
            [(232, 258, 1), (232 // 2, 258 // 2, 2)],
            "HWC",
        ),
        # For YUYV and friends things get a bit funky
        ((258, 232), nvcv.Format.YUYV, None, np.uint8, (232, 258, 2), "HWC"),
        ((258, 232), nvcv.Format.YUYV, "HWC", np.uint8, (232, 258, 2), "HWC"),
    ],
)
def test_image_export_cuda_buffer(
    size, format, layout, out_dtype, out_shape, simple_layout
):
    img = nvcv.Image(size, format)

    mem = img.cuda(layout)
    if type(mem) is list:
        for i in range(0, len(mem)):
            assert mem[i].dtype == out_dtype[i]
            assert mem[i].shape == out_shape[i]
    else:
        assert mem.dtype == out_dtype
        assert mem.shape == out_shape

    # external buffer must not be reused
    assert img.cuda(layout) is not mem
    if layout is not None:
        newmem = img.cuda()
        assert newmem is not mem
        assert newmem is not img.cuda()
        assert newmem is not img.cuda(layout)

    cuda_buffer = img.cuda(simple_layout)
    if type(cuda_buffer) is not list:
        cuda_buffer = [cuda_buffer]

    rng = np.random.default_rng(0)

    gold_buffer = list()
    # Write values in it on CUDA side
    for buf in cuda_buffer:
        gold_buffer.append((rng.random(size=buf.shape) * 255).astype(buf.dtype))
        torch.as_tensor(buf, device="cuda").copy_(torch.as_tensor(gold_buffer[-1]))

    # Get values back on cpu
    host_buffer = img.cpu(simple_layout)
    if type(host_buffer) is not list:
        host_buffer = [host_buffer]

    if type(out_dtype) is not list:
        out_dtype = [out_dtype]

    # compare to see if they are correct
    for b in range(0, len(host_buffer)):
        np.testing.assert_array_equal(
            host_buffer[b], gold_buffer[b], "buffer #" + str(b) + " mismatch"
        )


def test_image_export_cuda_buffer_strides():
    # torch returns packed buffers
    timg = torch.zeros((11, 37), dtype=torch.uint8, device="cuda")
    img = nvcv.as_image(timg)

    data = img.cuda()

    assert data.strides == (37, 1)


def test_image_zeros():
    img = nvcv.Image.zeros((67, 34), nvcv.Format.F32)
    assert (img.cpu() == np.zeros((34, 67), np.float32)).all()


def test_image_is_kept_alive_by_cuda_array_interface():
    nvcv.clear_cache()

    img1 = nvcv.Image((640, 480), nvcv.Format.U8)

    iface1 = img1.cuda()

    data_buffer1 = iface1.__cuda_array_interface__["data"][0]

    del img1

    img2 = nvcv.Image((640, 480), nvcv.Format.U8)
    assert img2.cuda().__cuda_array_interface__["data"][0] != data_buffer1

    del img2
    # remove img2 from cache, but not img1, as it's being
    # held by iface
    nvcv.clear_cache()

    # now img1 is free for reuse
    del iface1

    img3 = nvcv.Image((640, 480), nvcv.Format.U8)
    assert img3.cuda().__cuda_array_interface__["data"][0] == data_buffer1
