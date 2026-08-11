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
    "fg_args, bg_args, fgMask_args, outChannels",
    [
        (
            ((5, 9, 9, 3), np.uint8, "NHWC"),
            ((5, 9, 9, 3), np.uint8, "NHWC"),
            ((5, 9, 9, 1), np.uint8, "NHWC"),
            3,
        ),
        (
            ((9, 9, 3), np.uint8, "HWC"),
            ((9, 9, 3), np.uint8, "HWC"),
            ((9, 9, 1), np.uint8, "HWC"),
            4,
        ),
        (
            ((5, 21, 10, 3), np.uint8, "NHWC"),
            ((5, 21, 10, 3), np.uint8, "NHWC"),
            ((5, 21, 10, 1), np.uint8, "NHWC"),
            4,
        ),
        (
            ((5, 3, 21, 10), np.uint8, "NCHW"),
            ((5, 3, 21, 10), np.uint8, "NCHW"),
            ((5, 1, 21, 10), np.uint8, "NCHW"),
            4,
        ),
        (
            ((3, 9, 9), np.uint8, "CHW"),
            ((3, 9, 9), np.uint8, "CHW"),
            ((1, 9, 9), np.uint8, "CHW"),
            3,
        ),
    ],
)
def test_op_composite(fg_args, bg_args, fgMask_args, outChannels):
    foreground = cvcuda.Tensor(*fg_args)
    background = cvcuda.Tensor(*bg_args)
    fgMask = cvcuda.Tensor(*fgMask_args)

    out = cvcuda.composite(foreground, background, fgMask, outChannels)
    assert out.layout == foreground.layout
    channel_idx = str(out.layout).find("C")
    expected_shape = list(foreground.shape)
    expected_shape[channel_idx] = outChannels
    assert out.shape == tuple(expected_shape)
    assert out.dtype == foreground.dtype

    stream = cvcuda.Stream()

    out = cvcuda.Tensor(tuple(expected_shape), foreground.dtype, foreground.layout)
    tmp = cvcuda.composite_into(
        foreground=foreground,
        dst=out,
        background=background,
        fgmask=fgMask,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == foreground.layout
    assert out.shape == tuple(expected_shape)
    assert out.dtype == foreground.dtype


@pytest.mark.parametrize(
    "nimages, max_size, outChannels",
    [
        (
            5,
            (10, 20),
            3,
        ),
        (
            8,
            (10, 20),
            4,
        ),
    ],
)
def test_op_compositevarshape(nimages, max_size, outChannels):
    foreground = util.create_image_batch(
        nimages, cvcuda.Format.RGB8, max_size=max_size, max_random=255, rng=RNG
    )

    background = util.clone_image_batch(foreground)
    fgMask = util.clone_image_batch(foreground, img_format=cvcuda.Format.U8)

    out = cvcuda.composite(foreground, background, fgMask, outChannels)
    assert len(out) == len(foreground)
    assert out.capacity == foreground.capacity
    if outChannels == 3:
        assert out.uniqueformat == cvcuda.Format.RGB8
    if outChannels == 4:
        assert out.uniqueformat == cvcuda.Format.RGBA8

    stream = cvcuda.Stream()

    if outChannels == 3:
        out = util.clone_image_batch(foreground)
    if outChannels == 4:
        out = util.clone_image_batch(foreground, img_format=cvcuda.Format.RGBA8)
    tmp = cvcuda.composite_into(
        foreground=foreground,
        dst=out,
        background=background,
        fgmask=fgMask,
        stream=stream,
    )

    assert tmp is out
    assert len(out) == len(foreground)
    assert out.capacity == foreground.capacity


@pytest.mark.parametrize(
    "outChannels, expected_format",
    [
        (3, cvcuda.Format.RGB8p),
        (4, cvcuda.Format.RGBA8p),
    ],
)
def test_op_compositevarshape_planar(outChannels, expected_format):
    foreground = util.create_image_batch(
        3, cvcuda.Format.RGB8p, max_size=(10, 20), max_random=255, rng=RNG
    )

    background = util.clone_image_batch(foreground)
    fgMask = util.clone_image_batch(foreground, img_format=cvcuda.Format.U8)

    out = cvcuda.composite(foreground, background, fgMask, outChannels)
    assert len(out) == len(foreground)
    assert out.capacity == foreground.capacity
    assert out.uniqueformat == expected_format

    stream = cvcuda.Stream()

    out = util.clone_image_batch(foreground, img_format=expected_format)
    tmp = cvcuda.composite_into(
        foreground=foreground,
        dst=out,
        background=background,
        fgmask=fgMask,
        stream=stream,
    )

    assert tmp is out
    assert len(out) == len(foreground)
    assert out.capacity == foreground.capacity


def test_op_compositevarshape_preserves_input_capacity():
    capacity = 4
    num_images = 2
    foreground = cvcuda.ImageBatchVarShape(capacity)
    background = cvcuda.ImageBatchVarShape(capacity)
    fgMask = cvcuda.ImageBatchVarShape(capacity)

    for i in range(num_images):
        size = (8 + i, 9 + i)
        foreground.pushback(cvcuda.Image(size, cvcuda.Format.RGB8))
        background.pushback(cvcuda.Image(size, cvcuda.Format.RGB8))
        fgMask.pushback(cvcuda.Image(size, cvcuda.Format.U8))

    out = cvcuda.composite(foreground, background, fgMask)

    assert len(out) == num_images
    assert out.capacity == capacity
    out.pushback(cvcuda.Image((10, 11), cvcuda.Format.RGB8))
    assert len(out) == num_images + 1


def _composite(
    src: cvcuda.Tensor,
    layout: str = "NHWC",
    out_channels: int = 3,
    mask_channels: int = 1,
):
    shape = list(src.shape)
    mask_shape = list(shape)
    channel_idx = layout.find("C")
    mask_shape[channel_idx] = mask_channels
    background = cvcuda.Tensor(tuple(shape), src.dtype, layout)
    fgMask = cvcuda.Tensor(tuple(mask_shape), src.dtype, layout)
    return cvcuda.composite(src, background, fgMask, out_channels)


def _composite_varshape(
    src: cvcuda.ImageBatchVarShape,
    layout: str = "NHWC",
    out_channels: int = 3,
    mask_channels: int = 1,
):
    num_image = len(src)
    bg_batch = cvcuda.ImageBatchVarShape(num_image)
    fg_batch = cvcuda.ImageBatchVarShape(num_image)
    mask_batch = cvcuda.ImageBatchVarShape(num_image)
    for image in src:
        bg_batch.pushback(cvcuda.Image(image.size, image.format))
        fg_batch.pushback(cvcuda.Image(image.size, image.format))
        mask_batch.pushback(cvcuda.Image(image.size, cvcuda.Format.U8))
    return cvcuda.composite(fg_batch, bg_batch, mask_batch, out_channels)


def _composite_params(dtype, layout, channels, out_channels=3, mask_channels=1):
    return {
        "layout": layout,
        "out_channels": out_channels,
        "mask_channels": mask_channels,
    }


globals().update(
    cv_tools.make_op_tests(
        name="composite",
        runner_info=[
            ("tensor", _composite, _composite_params),
            ("image_batch", _composite_varshape, _composite_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={3},
        extra_params={"out_channels": {3, 4}},
        extra_params_negative={"mask_channels": {2, 3, 4}, "out_channels": {0, 2, 5}},
        exclude_extra_params=[
            ("mask_channels", "image_batch"),
        ],
        negative_exceptions=[RuntimeError, ValueError],
    )
)
