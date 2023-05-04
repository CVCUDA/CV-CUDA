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
    "inputp, bndboxes",
    [
        (
            (((3, 224, 224, 4), np.uint8, "NHWC")),
            cvcuda.BndBoxesI(
                numBoxes=[3, 3, 3],
                boxes=[
                    cvcuda.BndBoxI(
                        box=(10, 10, 5, 5),
                        thickness=2,
                        borderColor=(255, 255, 0),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(20, 10, 5, 5),
                        thickness=3,
                        borderColor=(0, 255, 255),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(30, 10, 5, 5),
                        thickness=3,
                        borderColor=(0, 255, 255),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(10, 20, 5, 5),
                        thickness=2,
                        borderColor=(255, 255, 0),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(20, 20, 5, 5),
                        thickness=3,
                        borderColor=(0, 255, 255),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(30, 20, 5, 5),
                        thickness=3,
                        borderColor=(0, 255, 255),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(10, 20, 5, 5),
                        thickness=2,
                        borderColor=(255, 255, 0),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(20, 20, 5, 5),
                        thickness=3,
                        borderColor=(0, 255, 255),
                        fillColor=(0, 128, 255, 128),
                    ),
                    cvcuda.BndBoxI(
                        box=(30, 20, 5, 5),
                        thickness=3,
                        borderColor=(0, 255, 255),
                        fillColor=(0, 128, 255, 128),
                    ),
                ],
            ),
        ),
    ],
)
def test_op_bndbox(inputp, bndboxes):
    input = cvcuda.Tensor(*inputp)

    out = cvcuda.bndbox(input, bndboxes)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.bndbox_into(out, input, bndboxes)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.bndbox(src=input, bboxes=bndboxes, stream=stream)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.bndbox_into(dst=out, src=input, bboxes=bndboxes, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
