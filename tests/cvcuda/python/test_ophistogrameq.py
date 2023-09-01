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
    "input",
    [
        (((1, 460, 640, 1), cvcuda.Type.U8, "NHWC")),
        (((2, 640, 460, 3), cvcuda.Type.U8, "NHWC")),
        (((1, 1920, 1080, 3), cvcuda.Type.U8, "NHWC")),
        (((3, 1000, 1000, 1), cvcuda.Type.U8, "NHWC")),
        (((2, 100, 100, 1), cvcuda.Type.U8, "NHWC")),
        (((2, 460, 640, 1), cvcuda.Type.U8, "NHWC")),
        (((100, 100, 3), cvcuda.Type.U8, "HWC")),
        (((460, 640, 1), cvcuda.Type.U8, "HWC")),
    ],
)
def test_op_histogrameq(input):

    inputTensor = cvcuda.Tensor(*input)

    out = cvcuda.histogrameq(inputTensor, inputTensor.dtype)
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    out = cvcuda.Tensor(inputTensor.shape, inputTensor.dtype, inputTensor.layout)
    tmp = cvcuda.histogrameq_into(out, inputTensor)

    assert tmp is out
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    stream = cvcuda.Stream()
    out = cvcuda.histogrameq(src=inputTensor, dtype=inputTensor.dtype, stream=stream)
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    tmp = cvcuda.histogrameq_into(dst=out, src=inputTensor, stream=stream)
    assert tmp is out
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype


@t.mark.parametrize(
    "num_images, format, max_size",
    [
        (
            1,
            cvcuda.Format.RGB8,
            (480, 720),
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (720, 480),
        ),
        (
            4,
            cvcuda.Format.Y8,
            (1920, 1080),
        ),
        (
            2,
            cvcuda.Format.BGR8,
            (1000, 1000),
        ),
        (
            3,
            cvcuda.Format.Y8,
            (100, 100),
        ),
        (
            5,
            cvcuda.Format.Y8,
            (460, 640),
        ),
    ],
)
def test_op_histogrameq_varshape(num_images, format, max_size):

    b_src = util.create_image_batch(num_images, format, max_size=max_size, rng=RNG)

    out = cvcuda.histogrameq(b_src)
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size

    tmp = cvcuda.histogrameq_into(out, b_src)
    assert tmp is out
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size

    stream = cvcuda.Stream()
    out = cvcuda.histogrameq(
        src=b_src,
        stream=stream,
    )
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size

    tmp = cvcuda.histogrameq_into(
        src=b_src,
        dst=out,
        stream=stream,
    )
    assert tmp is out
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size
