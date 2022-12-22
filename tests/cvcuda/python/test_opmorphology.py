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
    "input, morphologyType, maskSize, anchor, iteration, border ",
    [
        (
            cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [-1, -1],
            [-1, -1],
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            cvcuda.Tensor([4, 4, 3], np.float32, "HWC"),
            cvcuda.MorphologyType.DILATE,
            [2, 1],
            [-1, -1],
            1,
            cvcuda.Border.REPLICATE,
        ),
        (
            cvcuda.Tensor([3, 88, 13, 3], np.uint16, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [2, 2],
            [-1, -1],
            2,
            cvcuda.Border.REFLECT,
        ),
        (
            cvcuda.Tensor([3, 4, 4], np.uint16, "HWC"),
            cvcuda.MorphologyType.DILATE,
            [3, 3],
            [-1, -1],
            1,
            cvcuda.Border.WRAP,
        ),
        (
            cvcuda.Tensor([1, 2, 3, 4], np.uint8, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [-1, -1],
            [1, 1],
            1,
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_morphology(input, morphologyType, maskSize, anchor, iteration, border):
    out = cvcuda.morphology(
        input, morphologyType, maskSize, anchor, iteration=iteration, border=border
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.morphology_into(
        src=input,
        dst=out,
        morphologyType=morphologyType,
        maskSize=maskSize,
        anchor=anchor,
        iteration=iteration,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, \
     morphologyType, max_mask, max_anchor, iteration, border ",
    [
        (
            10,
            cvcuda.Format.RGB8,
            (123, 321),
            256,
            cvcuda.MorphologyType.ERODE,
            3,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            7,
            cvcuda.Format.RGBf32,
            (62, 35),
            1.0,
            cvcuda.MorphologyType.DILATE,
            4,
            2,
            2,
            cvcuda.Border.REPLICATE,
        ),
        (
            1,
            cvcuda.Format.F32,
            (33, 48),
            1234,
            cvcuda.MorphologyType.DILATE,
            5,
            1,
            3,
            cvcuda.Border.REFLECT,
        ),
        (
            3,
            cvcuda.Format.U8,
            (23, 18),
            123,
            cvcuda.MorphologyType.DILATE,
            5,
            4,
            4,
            cvcuda.Border.WRAP,
        ),
        (
            6,
            cvcuda.Format.F32,
            (77, 42),
            123456,
            cvcuda.MorphologyType.ERODE,
            6,
            3,
            1,
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_morphology_varshape(
    num_images,
    img_format,
    img_size,
    max_pixel,
    morphologyType,
    max_mask,
    max_anchor,
    iteration,
    border,
):

    input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    masks = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_mask, rng=RNG
    )

    anchors = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_anchor, rng=RNG
    )

    out = cvcuda.morphology(
        input, morphologyType, masks, anchors, iteration=iteration, border=border
    )

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)
    tmp = cvcuda.morphology_into(
        src=input,
        dst=out,
        morphologyType=morphologyType,
        masks=masks,
        anchors=anchors,
        iteration=iteration,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
