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
    "input,out_shape,out_layout",
    [
        (cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"), [5, 4, 16, 23], "NCHW"),
        (cvcuda.Tensor([5, 16, 23, 3], np.uint8, "NHWC"), [5, 3, 16, 23], "NCHW"),
        (cvcuda.Tensor([5, 3, 16, 23], np.uint8, "NCHW"), [5, 16, 23, 3], "NHWC"),
    ],
)
def test_op_reformat(input, out_shape, out_layout):
    out = cvcuda.reformat(input, out_layout)
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(out_shape, input.dtype, out_layout)
    tmp = cvcuda.reformat_into(out, input)
    assert tmp is out
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.reformat(src=input, layout=out_layout, stream=stream)
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    tmp = cvcuda.reformat_into(src=input, dst=out, stream=stream)
    assert tmp is out
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype
