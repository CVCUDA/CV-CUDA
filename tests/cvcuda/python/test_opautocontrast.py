# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "input_spec",
    [
        (((5, 16, 23, 4), cvcuda.Type.U8, "NHWC")),
        (((2, 33, 21, 3), cvcuda.Type.U16, "NHWC")),
        (((16, 23, 3), cvcuda.Type.F32, "HWC")),
        (((3, 4, 12, 10), cvcuda.Type.U8, "NCHW")),
        (((1, 460, 640, 1), cvcuda.Type.U8, "NHWC")),
    ],
)
def test_op_autocontrast(input_spec):

    inputTensor = cvcuda.Tensor(*input_spec)

    out = cvcuda.autocontrast(inputTensor)
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    out = cvcuda.Tensor(inputTensor.shape, inputTensor.dtype, inputTensor.layout)
    tmp = cvcuda.autocontrast_into(out, inputTensor)
    assert tmp is out
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    stream = cvcuda.Stream()
    out = cvcuda.autocontrast(src=inputTensor, stream=stream)
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    tmp = cvcuda.autocontrast_into(dst=out, src=inputTensor, stream=stream)
    assert tmp is out
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype


@pytest.mark.parametrize(
    "num_images, img_format, max_size",
    [
        (1, cvcuda.Format.RGB8, (480, 720)),
        (5, cvcuda.Format.RGBA8, (720, 480)),
        (4, cvcuda.Format.RGBf32, (200, 200)),
        (2, cvcuda.Format.F32, (100, 100)),
    ],
)
def test_op_autocontrast_varshape(num_images, img_format, max_size):

    b_src = util.create_image_batch(num_images, img_format, max_size=max_size, rng=RNG)

    out = cvcuda.autocontrast(b_src)
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert all(
        actual <= limit for actual, limit in zip(out.maxsize, max_size, strict=True)
    )

    tmp = cvcuda.autocontrast_into(out, b_src)
    assert tmp is out
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)

    stream = cvcuda.Stream()
    out = cvcuda.autocontrast(src=b_src, stream=stream)
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)

    tmp = cvcuda.autocontrast_into(src=b_src, dst=out, stream=stream)
    assert tmp is out
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)


# Standard input-contract coverage (supported/unsupported dtype, layout, and channel
# combinations). AutoContrast takes no parameters, so the param factory is empty.
globals().update(
    cv_tools.make_op_tests(
        name="autocontrast",
        runner_info=[
            ("tensor", cvcuda.autocontrast, lambda dtype, layout, channels: {}),
            ("image_batch", cvcuda.autocontrast, None),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.U16, cvcuda.Type.F32},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
    )
)
