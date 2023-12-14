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
import torch


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input_args, morphologyType, maskSize, anchor, iteration, border ",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [-1, -1],
            [-1, -1],
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            ((4, 4, 3), np.float32, "HWC"),
            cvcuda.MorphologyType.DILATE,
            [2, 1],
            [-1, -1],
            1,
            cvcuda.Border.REPLICATE,
        ),
        (
            ((3, 88, 13, 3), np.uint16, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [2, 2],
            [-1, -1],
            2,
            cvcuda.Border.REFLECT,
        ),
        (
            ((3, 4, 4), np.uint16, "HWC"),
            cvcuda.MorphologyType.DILATE,
            [3, 3],
            [-1, -1],
            1,
            cvcuda.Border.WRAP,
        ),
        (
            ((1, 2, 3, 4), np.uint8, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [-1, -1],
            [1, 1],
            1,
            cvcuda.Border.REFLECT101,
        ),
        (
            ((1, 2, 3, 4), np.uint8, "NHWC"),
            cvcuda.MorphologyType.ERODE,
            [-1, -1],
            [1, 1],
            5,
            cvcuda.Border.REFLECT101,
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            cvcuda.MorphologyType.OPEN,
            [-1, -1],
            [-1, -1],
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            ((4, 4, 3), np.float32, "HWC"),
            cvcuda.MorphologyType.CLOSE,
            [2, 1],
            [-1, -1],
            1,
            cvcuda.Border.REPLICATE,
        ),
        (
            ((3, 88, 13, 3), np.uint16, "NHWC"),
            cvcuda.MorphologyType.OPEN,
            [2, 2],
            [-1, -1],
            2,
            cvcuda.Border.REFLECT,
        ),
        (
            ((3, 4, 4), np.uint16, "HWC"),
            cvcuda.MorphologyType.CLOSE,
            [3, 3],
            [-1, -1],
            1,
            cvcuda.Border.WRAP,
        ),
        (
            ((1, 2, 3, 4), np.uint8, "NHWC"),
            cvcuda.MorphologyType.OPEN,
            [-1, -1],
            [1, 1],
            1,
            cvcuda.Border.REFLECT101,
        ),
    ],
)
def test_op_morphology(input_args, morphologyType, maskSize, anchor, iteration, border):
    input = cvcuda.Tensor(*input_args)
    workspace = cvcuda.Tensor(*input_args)

    workspace_param = None if iteration in [0, 1] else workspace
    workspace_param = (
        workspace
        if morphologyType in [cvcuda.MorphologyType.OPEN, cvcuda.MorphologyType.CLOSE]
        else workspace_param
    )

    out = cvcuda.morphology(
        input,
        morphologyType,
        maskSize,
        anchor,
        iteration=iteration,
        border=border,
        workspace=workspace_param,
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
        workspace=workspace_param,
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
        (
            2,
            cvcuda.Format.RGB8,
            (100, 200),
            256,
            cvcuda.MorphologyType.OPEN,
            3,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            5,
            cvcuda.Format.RGBf32,
            (60, 30),
            1.0,
            cvcuda.MorphologyType.OPEN,
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
            cvcuda.MorphologyType.OPEN,
            5,
            1,
            3,
            cvcuda.Border.REFLECT,
        ),
        # New test cases for CLOSE
        (
            3,
            cvcuda.Format.U8,
            (20, 10),
            123,
            cvcuda.MorphologyType.CLOSE,
            5,
            4,
            4,
            cvcuda.Border.WRAP,
        ),
        (
            6,
            cvcuda.Format.F32,
            (70, 40),
            123456,
            cvcuda.MorphologyType.CLOSE,
            6,
            3,
            1,
            cvcuda.Border.REFLECT101,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (80, 40),
            256,
            cvcuda.MorphologyType.CLOSE,
            3,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            7,
            cvcuda.Format.RGBf32,
            (70, 35),
            1.0,
            cvcuda.MorphologyType.CLOSE,
            4,
            2,
            2,
            cvcuda.Border.REPLICATE,
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

    workspace = input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    masks = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_mask, rng=RNG
    )

    anchors = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_anchor, rng=RNG
    )

    workspace_param = None if iteration in [0, 1] else workspace
    workspace_param = (
        workspace
        if morphologyType in [cvcuda.MorphologyType.OPEN, cvcuda.MorphologyType.CLOSE]
        else workspace_param
    )

    out = cvcuda.morphology(
        input,
        morphologyType,
        masks,
        anchors,
        iteration=iteration,
        border=border,
        workspace=workspace_param,
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
        workspace=workspace_param,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def test_op_morphology_input_output():
    width = 3
    height = 3
    number = 3

    # Create a tensor filled with zeros
    source = torch.zeros((number, height, width, 1), dtype=torch.uint8).cuda()

    # Set the middle pixel of each image to 1
    source[:, 1, 1, :] = 1

    image = source.clone()  # Copy source tensor, so we can compare later
    image = cvcuda.as_tensor(image, "NHWC")

    workspace = cvcuda.Tensor(image.shape, image.dtype, image.layout)

    outDilate = cvcuda.morphology(image, cvcuda.MorphologyType.DILATE, [4, 4], [-1, -1])
    outErode = cvcuda.morphology(image, cvcuda.MorphologyType.ERODE, [4, 4], [-1, -1])

    outOpen = cvcuda.morphology(
        image, cvcuda.MorphologyType.OPEN, [4, 4], [-1, -1], workspace=workspace
    )
    outClose = cvcuda.morphology(
        image, cvcuda.MorphologyType.CLOSE, [4, 4], [-1, -1], workspace=workspace
    )

    outDilate = torch.as_tensor(outDilate.cuda(), device=torch.device("cuda", 0))
    outErode = torch.as_tensor(outErode.cuda(), device=torch.device("cuda", 0))
    outOpen = torch.as_tensor(outOpen.cuda(), device=torch.device("cuda", 0))
    outClose = torch.as_tensor(outClose.cuda(), device=torch.device("cuda", 0))

    expectedDilate = torch.ones((number, height, width, 1), dtype=torch.uint8).cuda()
    expectedErode = torch.zeros((number, height, width, 1), dtype=torch.uint8).cuda()
    expectedOpen = torch.zeros((number, height, width, 1), dtype=torch.uint8).cuda()
    expectedClose = torch.ones((number, height, width, 1), dtype=torch.uint8).cuda()

    assert torch.all(outDilate.eq(expectedDilate))
    assert torch.all(outErode.eq(expectedErode))
    assert torch.all(outOpen.eq(expectedOpen))
    assert torch.all(outClose.eq(expectedClose))
