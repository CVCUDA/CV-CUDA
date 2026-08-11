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
    "tensor_args, crop_size, gold_shape",
    [
        (((5, 9, 9, 4), np.uint8, "NHWC"), [5, 5], (5, 5, 5, 4)),
        (((9, 9, 3), np.uint8, "HWC"), [5, 5], (5, 5, 3)),
        (((5, 21, 21, 4), np.uint8, "NHWC"), [15, 15], (5, 15, 15, 4)),
        (((21, 21, 3), np.uint8, "HWC"), [15, 15], (15, 15, 3)),
    ],
)
def test_op_center_crop(tensor_args, crop_size, gold_shape):
    input = cvcuda.Tensor(*tensor_args)
    out = cvcuda.center_crop(input, crop_size)
    assert out.layout == input.layout
    assert out.shape == gold_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.center_crop_into(
        src=input,
        dst=out,
        crop_size=crop_size,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


def _centercrop_params(dtype, layout, channels):
    return {"crop_size": [5, 5]}


globals().update(
    cv_tools.make_op_tests(
        name="centercrop",
        runner_info=[("tensor", cvcuda.center_crop, _centercrop_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.S8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F16,
            cvcuda.Type.S32,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[(None, "NCHW", 2), (None, "CHW", 2)],
    )
)
