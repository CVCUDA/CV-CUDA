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
from random import randint


@t.mark.parametrize(
    "input_args, per_channel",
    [
        (
            ((1, 460, 640, 3), cvcuda.Type.U8, "NHWC"),
            False,
        ),
        (
            ((5, 640, 460, 3), cvcuda.Type.U8, "NHWC"),
            True,
        ),
        (
            ((4, 1920, 1080, 3), cvcuda.Type.F32, "NHWC"),
            False,
        ),
        (
            ((2, 1000, 1000, 3), cvcuda.Type.F32, "NHWC"),
            True,
        ),
        (
            ((3, 100, 100, 1), cvcuda.Type.U16, "NHWC"),
            False,
        ),
        (
            ((5, 460, 640, 1), cvcuda.Type.U16, "NHWC"),
            True,
        ),
    ],
)
def test_op_gaussiannoise(input_args, per_channel):
    input = cvcuda.Tensor(*input_args)

    parameter_shape = (input.shape[0],)
    mu = cvcuda.Tensor(parameter_shape, cvcuda.Type.F32, "N")
    sigma = cvcuda.Tensor(parameter_shape, cvcuda.Type.F32, "N")

    seed = 12345
    out = cvcuda.gaussiannoise(input, mu, sigma, per_channel, seed)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.gaussiannoise_into(out, input, mu, sigma, per_channel, seed)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.gaussiannoise(
        src=input,
        mu=mu,
        sigma=sigma,
        per_channel=per_channel,
        seed=seed,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.gaussiannoise_into(
        src=input,
        dst=out,
        mu=mu,
        sigma=sigma,
        per_channel=per_channel,
        seed=seed,
        stream=stream,
    )
    assert tmp is out


@t.mark.parametrize(
    "num_images, format, min_size, max_size, per_channel",
    [
        (
            1,
            cvcuda.Format.RGB8,
            (460, 640),
            (480, 720),
            False,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (640, 460),
            (720, 480),
            True,
        ),
        (
            4,
            cvcuda.Format.RGBf32,
            (1920, 1080),
            (1920, 1080),
            False,
        ),
        (
            2,
            cvcuda.Format.RGBf32,
            (1000, 1000),
            (1000, 1000),
            True,
        ),
        (
            3,
            cvcuda.Format.U16,
            (100, 100),
            (100, 100),
            False,
        ),
        (
            5,
            cvcuda.Format.U16,
            (460, 640),
            (460, 640),
            True,
        ),
    ],
)
def test_op_gaussiannoise_varshape(num_images, format, min_size, max_size, per_channel):

    parameter_shape = (num_images,)
    mu = cvcuda.Tensor(parameter_shape, cvcuda.Type.F32, "N")
    sigma = cvcuda.Tensor(parameter_shape, cvcuda.Type.F32, "N")

    input = cvcuda.ImageBatchVarShape(num_images)
    output = cvcuda.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = randint(min_size[0], max_size[0])
        h = randint(min_size[1], max_size[1])
        img_in = cvcuda.Image([w, h], format)
        input.pushback(img_in)
        img_out = cvcuda.Image([w, h], format)
        output.pushback(img_out)

    seed = 12345
    tmp = cvcuda.gaussiannoise(input, mu, sigma, per_channel, seed)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.gaussiannoise_into(output, input, mu, sigma, per_channel, seed)
    assert tmp is output

    stream = cvcuda.Stream()
    tmp = cvcuda.gaussiannoise(
        src=input,
        mu=mu,
        sigma=sigma,
        per_channel=per_channel,
        seed=seed,
        stream=stream,
    )
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.gaussiannoise_into(
        src=input,
        dst=output,
        mu=mu,
        sigma=sigma,
        per_channel=per_channel,
        seed=seed,
        stream=stream,
    )
    assert tmp is output
