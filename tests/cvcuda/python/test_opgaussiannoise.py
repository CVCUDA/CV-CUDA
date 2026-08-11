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

from random import randint

import cupy as cp
import cvcuda
import cvcuda_tools as cv_tools
import pytest


@pytest.mark.parametrize(
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


@pytest.mark.parametrize(
    "shape,dtype,layout",
    [
        ((2, 17, 19, 3), cvcuda.Type.U8, "NHWC"),
        ((3, 17, 19), cvcuda.Type.U8, "CHW"),
        ((2, 3, 17, 19), cvcuda.Type.F32, "NCHW"),
        ((17, 19, 3), cvcuda.Type.F32, "HWC"),
    ],
)
@pytest.mark.parametrize("clip", [True, False])
def test_op_gaussiannoise_scalar(shape, dtype, layout, clip):
    src = cvcuda.Tensor(shape, dtype, layout)
    cp.asarray(src.cuda())[...].fill(127 if dtype == cvcuda.Type.U8 else 0.5)

    out = cvcuda.gaussiannoise(
        src,
        3.0 if dtype == cvcuda.Type.U8 else 0.01,
        25.0 if dtype == cvcuda.Type.U8 else 0.05,
        True,
        seed=12345,
        clip=clip,
    )
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert out.layout == src.layout

    dst = cvcuda.Tensor(shape, dtype, layout)
    ret = cvcuda.gaussiannoise_into(
        dst,
        src,
        3.0 if dtype == cvcuda.Type.U8 else 0.01,
        25.0 if dtype == cvcuda.Type.U8 else 0.05,
        True,
        seed=12345,
        clip=clip,
    )
    assert ret is dst
    assert cp.array_equal(cp.asarray(out.cuda()), cp.asarray(dst.cuda()))


def test_op_gaussiannoise_scalar_seed_contract():
    src = cvcuda.Tensor((1, 31, 37, 3), cvcuda.Type.F32, "NHWC")
    cp.asarray(src.cuda())[...].fill(0.5)

    first = cvcuda.gaussiannoise(src, 0.0, 0.05, True, seed=98765)
    repeated = cvcuda.gaussiannoise(src, 0.0, 0.05, True, seed=98765)
    implicit_first = cvcuda.gaussiannoise(src, 0.0, 0.05, True)
    implicit_second = cvcuda.gaussiannoise(src, 0.0, 0.05, True)

    assert cp.array_equal(cp.asarray(first.cuda()), cp.asarray(repeated.cuda()))
    assert not cp.array_equal(
        cp.asarray(implicit_first.cuda()), cp.asarray(implicit_second.cuda())
    )


def test_op_gaussiannoise_scalar_rejects_invalid_parameters():
    src = cvcuda.Tensor((1, 4, 4, 3), cvcuda.Type.U8, "NHWC")
    with pytest.raises(RuntimeError):
        cvcuda.gaussiannoise(src, 0.0, -0.1, True)
    with pytest.raises((TypeError, OverflowError)):
        cvcuda.gaussiannoise(src, 0.0, 0.1, True, seed=-1)


@pytest.mark.parametrize(
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


def _gaussiannoise_params(dtype, layout, channels):
    return {
        "mu": cvcuda.Tensor((1,), cvcuda.Type.F32, "N"),
        "sigma": cvcuda.Tensor((1,), cvcuda.Type.F32, "N"),
        "per_channel": False,
        "seed": 12345,
    }


globals().update(
    cv_tools.make_op_tests(
        name="gaussiannoise",
        runner_info=[
            ("tensor", cvcuda.gaussiannoise, _gaussiannoise_params),
            ("image_batch", cvcuda.gaussiannoise, _gaussiannoise_params),
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
