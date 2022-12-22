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
from random import randint


@t.mark.parametrize(
    "num_images, format, min_size, max_size, erasing_area_num, random, seed",
    [
        (1, cvcuda.Format.U8, (100, 100), (200, 200), 1, False, 0),
        (5, cvcuda.Format.RGB8, (100, 100), (200, 100), 1, True, 1),
    ],
)
def test_op_erase(
    num_images, format, min_size, max_size, erasing_area_num, random, seed
):

    parameter_shape = [erasing_area_num]
    anchor = cvcuda.Tensor(parameter_shape, cvcuda.Type._2S32, "N")
    erasing = cvcuda.Tensor(parameter_shape, cvcuda.Type._3S32, "N")
    imgIdx = cvcuda.Tensor(parameter_shape, cvcuda.Type.S32, "N")
    values = cvcuda.Tensor(parameter_shape, cvcuda.Type.F32, "N")

    input = cvcuda.ImageBatchVarShape(num_images)
    output = cvcuda.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = randint(min_size[0], max_size[0])
        h = randint(min_size[1], max_size[1])
        img_in = cvcuda.Image([w, h], format)
        input.pushback(img_in)
        img_out = cvcuda.Image([w, h], format)
        output.pushback(img_out)

    tmp = cvcuda.erase(input, anchor, erasing, values, imgIdx)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.erase_into(
        output, input, anchor, erasing, values, imgIdx, random=random, seed=seed
    )
    assert tmp is output

    stream = cvcuda.Stream()
    tmp = cvcuda.erase(
        src=input,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = cvcuda.erase_into(
        src=input,
        dst=output,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is output
