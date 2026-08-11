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
import cvcuda_types as cv_types
import cvcuda_tools as cv_tools

min_scale = 0.08
max_scale = 1.0
min_ratio = 0.75
max_ratio = 1.333333333
seed = 0

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "input_args,out_shape,interp",
    [
        (
            ((5, 16, 23, 1), np.uint8, "NHWC"),
            (5, 132, 15, 1),
            cvcuda.Interp.NEAREST,
        ),
        (
            ((5, 16, 23, 3), np.uint8, "NHWC"),
            (5, 132, 15, 3),
            cvcuda.Interp.LINEAR,
        ),
        (
            ((5, 3, 16, 23), np.uint8, "NCHW"),
            (5, 3, 132, 15),
            cvcuda.Interp.LINEAR,
        ),
        (
            ((16, 23, 4), np.uint8, "HWC"),
            (132, 15, 4),
            cvcuda.Interp.CUBIC,
        ),
        (
            ((4, 16, 23), np.uint8, "CHW"),
            (4, 132, 15),
            cvcuda.Interp.CUBIC,
        ),
        (((16, 23, 1), np.uint8, "HWC"), (132, 15, 1), None),
    ],
)
def test_op_random_resized_crop(input_args, out_shape, interp):
    input = cvcuda.Tensor(*input_args)

    if interp is None:
        out = cvcuda.random_resized_crop(input, out_shape)
    else:
        out = cvcuda.random_resized_crop(
            input, out_shape, min_scale, max_scale, min_ratio, max_ratio, interp, seed
        )
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(out_shape, input.dtype, input.layout)
    if interp is None:
        tmp = cvcuda.random_resized_crop_into(out, input)
    else:
        tmp = cvcuda.random_resized_crop_into(
            out, input, min_scale, max_scale, min_ratio, max_ratio, interp, seed
        )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    if interp is None:
        out = cvcuda.random_resized_crop(src=input, shape=out_shape, stream=stream)
    else:
        out = cvcuda.random_resized_crop(
            src=input, shape=out_shape, interp=interp, stream=stream
        )
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    if interp is None:
        tmp = cvcuda.random_resized_crop_into(src=input, dst=out, stream=stream)
    else:
        tmp = cvcuda.random_resized_crop_into(
            src=input, dst=out, interp=interp, stream=stream
        )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype


@pytest.mark.parametrize(
    "max_input_size, max_output_size, interp",
    [
        ((123, 321), (321, 123), cvcuda.Interp.NEAREST),
        ((123, 321), (321, 123), cvcuda.Interp.LINEAR),
        ((123, 321), (321, 123), cvcuda.Interp.CUBIC),
        ((33, 44), (44, 55), None),
    ],
)
def test_op_random_resized_crop_varshape(max_input_size, max_output_size, interp):

    input = util.create_image_batch(
        10, cvcuda.Format.RGBA8, max_size=max_input_size, max_random=256, rng=RNG
    )

    base_output = util.create_image_batch(
        10, cvcuda.Format.RGBA8, max_size=max_output_size, max_random=256, rng=RNG
    )

    sizes = []
    for image in base_output:
        sizes.append([image.width, image.height])

    if interp is None:
        out = cvcuda.random_resized_crop(input, sizes)
    else:
        out = cvcuda.random_resized_crop(
            src=input,
            sizes=sizes,
            min_scale=min_scale,
            max_scale=max_scale,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            interp=interp,
            seed=seed,
        )

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize <= max_output_size

    stream = cvcuda.Stream()
    if interp is None:
        tmp = cvcuda.random_resized_crop_into(src=input, dst=base_output, stream=stream)
    else:
        tmp = cvcuda.random_resized_crop_into(
            src=input,
            dst=base_output,
            min_scale=min_scale,
            max_scale=max_scale,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            interp=interp,
            seed=seed,
            stream=stream,
        )
    assert tmp is base_output
    assert len(base_output) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize <= max_output_size


def _randomresizedcrop_params(dtype, layout, channels):
    return {
        "shape": cv_types.resolve_shape(layout, channels, (10, 20)),
        "interp": cvcuda.Interp.LINEAR,
    }


def _randomresizedcrop_varshape_params(dtype, layout, channels):
    return {
        "sizes": [[20, 10], [20, 10]],
        "interp": cvcuda.Interp.LINEAR,
    }


globals().update(
    cv_tools.make_op_tests(
        name="random_resized_crop",
        runner_info=[
            ("tensor", cvcuda.random_resized_crop, _randomresizedcrop_params),
            (
                "image_batch",
                cvcuda.random_resized_crop,
                _randomresizedcrop_varshape_params,
            ),
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
