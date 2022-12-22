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


@t.mark.parametrize(
    "format,num_images,min_size,max_size,border,bvalue,out_shape,out_layout,out_dtype",
    [
        (
            cvcuda.Format.RGBA8,
            1,
            (10, 5),
            (10, 5),
            cvcuda.Border.REPLICATE,
            0,
            [1, 5, 10, 4],
            "NHWC",
            np.uint8,
        ),
    ],
)
def test_op_padandstack(
    format,
    num_images,
    min_size,
    max_size,
    border,
    bvalue,
    out_shape,
    out_layout,
    out_dtype,
):

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

    left = cvcuda.Tensor([1, 1, num_images, 1], np.int32, "NHWC")
    top = cvcuda.Tensor([1, 1, num_images, 1], np.int32, "NHWC")

    out = cvcuda.padandstack(input, top, left)
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == out_dtype

    out = cvcuda.Tensor(out_shape, out_dtype, out_layout)
    tmp = cvcuda.padandstack_into(out, input, top, left)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.padandstack(
        src=input, left=left, top=top, border=border, bvalue=bvalue, stream=stream
    )
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == out_dtype

    tmp = cvcuda.padandstack_into(
        src=input,
        dst=out,
        left=left,
        top=top,
        border=border,
        bvalue=bvalue,
        stream=stream,
    )
    assert tmp is out
