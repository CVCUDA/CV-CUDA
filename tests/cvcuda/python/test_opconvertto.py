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
    "input,dtype,scale,offset",
    [
        (cvcuda.Tensor([5, 16, 23, 4], np.uint8, "NHWC"), np.float32, 1.2, 10.2),
        (cvcuda.Tensor([16, 23, 2], np.uint8, "HWC"), np.int32, -1.2, -5.5),
    ],
)
def test_op_convertto(input, dtype, scale, offset):
    out = cvcuda.convertto(input, dtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.convertto_into(out, input)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.convertto(input, dtype, scale)
    out = cvcuda.convertto(input, dtype, scale, offset)

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.convertto_into(out, input, scale)
    tmp = cvcuda.convertto_into(out, input, scale, offset)

    stream = cvcuda.Stream()
    out = cvcuda.convertto(
        src=input, dtype=dtype, scale=scale, offset=offset, stream=stream
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    tmp = cvcuda.convertto_into(
        dst=out, src=input, scale=scale, offset=offset, stream=stream
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype
