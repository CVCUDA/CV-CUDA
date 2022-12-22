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
    "input, crop_size, gold_shape",
    [
        (cvcuda.Tensor([5, 9, 9, 4], np.uint8, "NHWC"), [5, 5], [5, 5, 5, 4]),
        (cvcuda.Tensor([9, 9, 3], np.uint8, "HWC"), [5, 5], [5, 5, 3]),
        (cvcuda.Tensor([5, 21, 21, 4], np.uint8, "NHWC"), [15, 15], [5, 15, 15, 4]),
        (cvcuda.Tensor([21, 21, 3], np.uint8, "HWC"), [15, 15], [15, 15, 3]),
    ],
)
def test_op_center_crop(input, crop_size, gold_shape):
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
