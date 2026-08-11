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
    "tensor_args, diameter, sigma_color, sigma_space, border",
    [
        (
            ((5, 9, 9, 4), np.uint8, "NHWC"),
            9,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
        (
            ((9, 9, 3), np.uint8, "HWC"),
            7,
            3,
            10,
            cvcuda.Border.WRAP,
        ),
        (
            ((5, 21, 21, 4), np.uint8, "NHWC"),
            6,
            15,
            9,
            cvcuda.Border.REPLICATE,
        ),
        (
            ((21, 21, 3), np.uint8, "HWC"),
            12,
            2,
            5,
            cvcuda.Border.REFLECT,
        ),
    ],
)
def test_op_bilateral_filter(tensor_args, diameter, sigma_color, sigma_space, border):
    input = cvcuda.Tensor(*tensor_args)
    out = cvcuda.bilateral_filter(
        input, diameter, sigma_color, sigma_space, border=border
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.bilateral_filter_into(
        src=input,
        dst=out,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_diameter, max_sc, max_ss, border",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            12,
            2,
            5,
            cvcuda.Border.REFLECT,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            6,
            15,
            9,
            cvcuda.Border.REPLICATE,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            256.0,
            7,
            3,
            10,
            cvcuda.Border.WRAP,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (11, 23),
            256.0,
            9,
            1,
            1,
            cvcuda.Border.CONSTANT,
        ),
    ],
)
def test_op_bilateral_filtervarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_diameter,
    max_sc,
    max_ss,
    border,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    diameter = util.create_tensor(
        (nimages), np.int32, "N", max_random=max_diameter, rng=RNG
    )

    sigma_color = util.create_tensor(
        (nimages), np.float32, "N", max_random=max_sc, rng=RNG
    )

    sigma_space = util.create_tensor(
        (nimages), np.float32, "N", max_random=max_ss, rng=RNG
    )
    out = cvcuda.bilateral_filter(
        input, diameter, sigma_color, sigma_space, border=border
    )

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)

    tmp = cvcuda.bilateral_filter_into(
        src=input,
        dst=out,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize


def _bilateralfilter_params(dtype, layout, channels):
    return {
        "diameter": 9,
        "sigma_color": 1.0,
        "sigma_space": 1.0,
    }


def _bilateralfilter_varshape_params(dtype, layout, channels):
    return {
        "diameter": util.to_cvcuda_tensor(np.array([9, 9], dtype=np.int32), "N"),
        "sigma_color": util.to_cvcuda_tensor(
            np.array([1.0, 1.0], dtype=np.float32), "N"
        ),
        "sigma_space": util.to_cvcuda_tensor(
            np.array([1.0, 1.0], dtype=np.float32), "N"
        ),
    }


globals().update(
    cv_tools.make_op_tests(
        name="bilateral_filter",
        runner_info=[
            ("tensor", cvcuda.bilateral_filter, _bilateralfilter_params),
            ("image_batch", cvcuda.bilateral_filter, _bilateralfilter_varshape_params),
        ],
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
