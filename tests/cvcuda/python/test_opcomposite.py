# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import util

RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "foreground, background, fgMask, outChannels",
    [
        (
            cvcuda.Tensor([5, 9, 9, 3], np.uint8, "NHWC"),
            cvcuda.Tensor([5, 9, 9, 3], np.uint8, "NHWC"),
            cvcuda.Tensor([5, 9, 9, 1], np.uint8, "NHWC"),
            3,
        ),
        (
            cvcuda.Tensor([9, 9, 3], np.uint8, "HWC"),
            cvcuda.Tensor([9, 9, 3], np.uint8, "HWC"),
            cvcuda.Tensor([9, 9, 1], np.uint8, "HWC"),
            4,
        ),
        (
            cvcuda.Tensor([5, 21, 10, 3], np.uint8, "NHWC"),
            cvcuda.Tensor([5, 21, 10, 3], np.uint8, "NHWC"),
            cvcuda.Tensor([5, 21, 10, 1], np.uint8, "NHWC"),
            4,
        ),
    ],
)
def test_op_composite(foreground, background, fgMask, outChannels):
    out = cvcuda.composite(foreground, background, fgMask, outChannels)
    assert out.layout == foreground.layout
    assert out.shape[-1] == outChannels
    if out.layout == "NHWC":
        assert out.shape[0:3] == foreground.shape[0:3]
    if out.layout == "HWC":
        assert out.shape[0:2] == foreground.shape[0:2]
    assert out.dtype == foreground.dtype

    stream = cvcuda.Stream()

    out_shape = foreground.shape
    out_shape[-1] = outChannels
    out = cvcuda.Tensor(out_shape, foreground.dtype, foreground.layout)
    tmp = cvcuda.composite_into(
        foreground=foreground,
        dst=out,
        background=background,
        fgmask=fgMask,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == foreground.layout
    assert out.shape[-1] == outChannels
    if out.layout == "NHWC":
        assert out.shape[0:3] == foreground.shape[0:3]
    if out.layout == "HWC":
        assert out.shape[0:2] == foreground.shape[0:2]
    assert out.dtype == foreground.dtype


@t.mark.parametrize(
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

    out = cvcuda.composite(foreground, background, fgMask)
    assert len(out) == len(foreground)
    assert out.capacity == foreground.capacity

    stream = cvcuda.cuda.Stream()

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
