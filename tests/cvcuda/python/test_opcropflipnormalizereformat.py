# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RNG = np.random.default_rng(0)


@t.mark.parametrize(
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
