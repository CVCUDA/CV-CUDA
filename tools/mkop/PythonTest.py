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


@t.mark.parametrize(
    "input,dtype",
    [
        (cvcuda.Tensor((5, 16, 23, 4), np.uint8, "NHWC"), np.int8),
        (cvcuda.Tensor((16, 23, 2), np.uint8, "HWC"), np.int32),
    ],
)
def test_op___OPNAMELOW__(input, dtype):
    out = cvcuda.__OPNAMELOW__(input, dtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.__OPNAMELOW___into(out, input)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.__OPNAMELOW__(input, dtype)

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.__OPNAMELOW___into(out, input)

    stream = cvcuda.Stream()
    out = cvcuda.__OPNAMELOW__(src=input, dtype=dtype, stream=stream)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    tmp = cvcuda.__OPNAMELOW___into(dst=out, src=input, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    # TODO make test pass
    t.fail("Test failed intentionally")
