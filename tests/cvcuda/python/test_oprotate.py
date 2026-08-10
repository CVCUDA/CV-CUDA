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
import cvcuda_tools as cv_tools
import cvcuda_util as util

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "input_args, angle_deg, shift, interpolation",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            30,
            [3, 4],
            cvcuda.Interp.NEAREST,
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            60,
            [3, 4],
            cvcuda.Interp.LINEAR,
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            90,
            [3, 4],
            cvcuda.Interp.CUBIC,
        ),
        (
            ((7, 12, 3), np.uint8, "HWC"),
            30,
            [2, 3],
            cvcuda.Interp.NEAREST,
        ),
        (
            ((7, 12, 3), np.uint8, "HWC"),
            60,
            [2, 3],
            cvcuda.Interp.LINEAR,
        ),
        (
            ((7, 12, 3), np.uint8, "HWC"),
            90,
            [2, 3],
            cvcuda.Interp.CUBIC,
        ),
    ],
)
def test_op_rotate(input_args, angle_deg, shift, interpolation):
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.rotate(input, angle_deg, shift, interpolation)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.rotate_into(
        src=input,
        dst=out,
        angle_deg=angle_deg,
        shift=shift,
        interpolation=interpolation,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_angle_deg, max_shift, interpolation",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            180,
            [5, 5],
            cvcuda.Interp.NEAREST,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            180,
            [5, 5],
            cvcuda.Interp.LINEAR,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            180,
            [5, 5],
            cvcuda.Interp.CUBIC,
        ),
    ],
)
def test_op_rotatevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_angle_deg,
    max_shift,
    interpolation,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    angle_deg = util.create_tensor(
        (nimages), np.float64, "N", max_random=max_angle_deg, rng=RNG
    )

    shift = util.create_tensor(
        (nimages, 2), np.float64, "NC", max_random=max_shift, rng=RNG
    )

    out = cvcuda.rotate(
        input,
        angle_deg,
        shift,
        interpolation,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)
    tmp = cvcuda.rotate_into(
        src=input,
        dst=out,
        angle_deg=angle_deg,
        shift=shift,
        interpolation=interpolation,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _rotate_params(dtype, layout, channels):
    return {
        "angle_deg": 45.0,
        "shift": [0, 0],
        "interpolation": cvcuda.Interp.LINEAR,
    }


def _rotate_varshape_params(dtype, layout, channels):
    return {
        "angle_deg": util.create_tensor((2,), np.float64, "N", max_random=180, rng=RNG),
        "shift": util.create_tensor((2, 2), np.float64, "NC", max_random=5, rng=RNG),
        "interpolation": cvcuda.Interp.LINEAR,
    }


globals().update(
    cv_tools.make_op_tests(
        name="rotate",
        runner_info=[
            ("tensor", cvcuda.rotate, _rotate_params),
            ("image_batch", cvcuda.rotate, _rotate_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F32,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
    )
)
