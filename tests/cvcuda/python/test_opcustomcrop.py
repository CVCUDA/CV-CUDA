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


@pytest.mark.parametrize(
    "input_args,rc,out_shape",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            cvcuda.RectI(x=0, y=0, width=20, height=2),
            (5, 2, 20, 4),
        ),
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            cvcuda.RectI(2, 13, 16, 2),
            (5, 2, 16, 4),
        ),
        (
            ((16, 23, 2), np.uint8, "HWC"),
            cvcuda.RectI(2, 13, 16, 2),
            (2, 16, 2),
        ),
    ],
)
def test_op_customcrop(input_args, rc, out_shape):
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.customcrop(input, rc)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.customcrop_into(out, input, rc)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.customcrop(input, rect=rc, stream=stream)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.customcrop_into(dst=out, src=input, rect=rc, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


def _customcrop_params(dtype, layout, channels):
    return {"rect": cvcuda.RectI(0, 0, 10, 10)}


globals().update(
    cv_tools.make_op_tests(
        name="customcrop",
        runner_info=[("tensor", cvcuda.customcrop, _customcrop_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.S32,
            cvcuda.Type.F16,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[(None, "NCHW", 2), (None, "CHW", 2)],
    )
)
