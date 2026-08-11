# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cupy

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "format,num_images,min_size,max_size,border,bvalue,basep,scalep,gscale,gshift,eps,flags,ch,dtype,layout",
    [
        (
            cvcuda.Format.RGBA8,
            1,
            (10, 10),
            (20, 20),
            cvcuda.Border.REPLICATE,
            0,
            (((1, 1, 1, 4), np.float32, "NHWC")),
            (((1, 1, 1, 4), np.float32, "NHWC")),
            1,
            2,
            3,
            cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
            4,
            np.uint8,
            "NHWC",
        ),
    ],
)
def test_op_crop_flip_normalize_reformat_tensor_out(
    format,
    num_images,
    min_size,
    max_size,
    border,
    bvalue,
    basep,
    scalep,
    gscale,
    gshift,
    eps,
    flags,
    ch,
    dtype,
    layout,
):
    base = cvcuda.Tensor(*basep)
    scale = cvcuda.Tensor(*scalep)
    input = cvcuda.ImageBatchVarShape(num_images)

    input.pushback(
        [
            cvcuda.Image(
                (
                    min_size[0] + (max_size[0] - min_size[0]) * i // num_images,
                    min_size[1] + (max_size[1] - min_size[1]) * i // num_images,
                ),
                format,
            )
            for i in range(num_images)
        ]
    )

    cropRect = cvcuda.Tensor((num_images, 1, 1, 4), np.int32, "NHWC")
    flipCode = util.create_tensor(
        (num_images, 1), np.int32, "NC", max_random=1, rng=RNG
    )
    if layout == "NHWC":
        out_shape = (num_images, max_size[0], max_size[1], ch)
    else:
        out_shape = (num_images, ch, max_size[0], max_size[1])

    out = cvcuda.crop_flip_normalize_reformat(
        input,
        out_shape,
        dtype,
        layout,
        cropRect,
        flipCode,
        base,
        scale,
        gscale,
        gshift,
        eps,
        flags,
        border,
        bvalue,
    )

    assert out.shape == out_shape
    assert out.dtype == dtype
    assert out.layout == layout

    stream = cvcuda.Stream()

    out_tensor = cvcuda.Tensor(out_shape, dtype, layout)

    tmp = cvcuda.crop_flip_normalize_reformat_into(
        dst=out_tensor,
        src=input,
        rect=cropRect,
        flip_code=flipCode,
        base=base,
        scale=scale,
        globalscale=gscale,
        globalshift=gshift,
        epsilon=eps,
        flags=flags,
        border=border,
        bvalue=bvalue,
        stream=stream,
    )

    assert tmp is out_tensor
    assert out_tensor.shape == out_shape
    assert out_tensor.dtype == dtype
    assert out_tensor.layout == layout


_supported_scalar_formats = {
    cvcuda.Format.U8,
    cvcuda.Format.U16,
    cvcuda.Format.U32,
    cvcuda.Format.S8,
    cvcuda.Format.S16,
    cvcuda.Format.S32,
    cvcuda.Format.F32,
}
_supported_interleaved_formats = {
    # 3 channels interleaved
    cvcuda.Format.RGB8,
    cvcuda.Format.BGR8,
    cvcuda.Format.RGBf32,
    cvcuda.Format.BGRf32,
    # 4 channels interleaved
    cvcuda.Format.RGBA8,
    cvcuda.Format.BGRA8,
    cvcuda.Format.RGBAf32,
    cvcuda.Format.BGRAf32,
}
_supported_planar_formats = {
    # 3 channels planar
    cvcuda.Format.RGB8p,
    cvcuda.Format.BGR8p,
    cvcuda.Format.RGBf32p,
    cvcuda.Format.BGRf32p,
    # 4 channels planar
    cvcuda.Format.RGBA8p,
    cvcuda.Format.BGRA8p,
    cvcuda.Format.RGBAf32p,
    cvcuda.Format.BGRAf32p,
}
_supported_formats = (
    _supported_scalar_formats
    | _supported_interleaved_formats
    | _supported_planar_formats
)
_supported_output_layouts = {"NHWC", "NCHW"}


def _get_channels_for_format(fmt: cvcuda.Format) -> int:
    fmt_name = fmt.name
    if "RGBA" in fmt_name or "BGRA" in fmt_name:
        return 4
    if "RGB" in fmt_name or "BGR" in fmt_name:
        return 3
    return 1


def _cropflipnormalizereformat(
    data: cvcuda.ImageBatchVarShape,
    out_layout: str = "NHWC",
) -> cvcuda.Tensor:
    num_images = len(data)
    size = data.maxsize
    channels = _get_channels_for_format(data.uniqueformat)

    crop_data = np.zeros((num_images, 1, 1, 4), dtype=np.int32)
    crop_data[..., 2] = size[0]
    crop_data[..., 3] = size[1]
    cropRect = cvcuda.as_tensor(cupy.asarray(crop_data), "NHWC")

    flipCode = util.create_tensor(
        (num_images, 1), np.int32, "NC", max_random=1, rng=RNG
    )
    base = cvcuda.Tensor((1, 1, 1, channels), np.float32, "NHWC")
    scale = cvcuda.Tensor((1, 1, 1, channels), np.float32, "NHWC")

    if out_layout == "NHWC":
        out_shape = (num_images, size[1], size[0], channels)
    else:  # NCHW
        out_shape = (num_images, channels, size[1], size[0])

    return cvcuda.crop_flip_normalize_reformat(
        data,
        out_shape=out_shape,
        out_dtype=np.float32,
        out_layout=out_layout,
        rect=cropRect,
        flip_code=flipCode,
        base=base,
        scale=scale,
        globalscale=1.0,
        globalshift=0.0,
        epsilon=0.0,
        flags=0,
        border=cvcuda.Border.CONSTANT,
        bvalue=0.0,
    )


def _cropflipnormalizereformat_params(dtype, layout, channels, out_layout="NHWC"):
    return {"out_layout": out_layout}


globals().update(
    cv_tools.make_op_tests(
        name="cropflipnormalizereformat",
        runner_info=[
            (
                "image_batch",
                _cropflipnormalizereformat,
                _cropflipnormalizereformat_params,
            )
        ],
        supported_formats=_supported_formats,
        extra_params={"out_layout": _supported_output_layouts},
    )
)
