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


@t.mark.parametrize(
    "input, erasing_area_num, random, seed",
    [
        (cvcuda.Tensor([1, 460, 640, 3], cvcuda.Type.U8, "NHWC"), 1, False, 0),
        (cvcuda.Tensor([5, 460, 640, 3], cvcuda.Type.U8, "NHWC"), 1, True, 1),
    ],
)
def test_op_erase(input, erasing_area_num, random, seed):

    parameter_shape = [erasing_area_num]
    anchor = cvcuda.Tensor(parameter_shape, cvcuda.Type._2S32, "N")
    erasing = cvcuda.Tensor(parameter_shape, cvcuda.Type._3S32, "N")
    imgIdx = cvcuda.Tensor(parameter_shape, cvcuda.Type.S32, "N")
    values = cvcuda.Tensor(parameter_shape, cvcuda.Type.F32, "N")

    out = cvcuda.erase(input, anchor, erasing, values, imgIdx)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.erase_into(out, input, anchor, erasing, values, imgIdx)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.erase(
        src=input,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.erase_into(
        src=input,
        dst=out,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is out
