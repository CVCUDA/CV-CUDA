# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "inputp, blurboxes",
    [
        (
            (((3, 224, 224, 4), np.uint8, "NHWC")),
            cvcuda.BlurBoxesI(
                boxes=[
                    [
                        cvcuda.BlurBoxI(box=(10, 10, 5, 5), kernelSize=7),
                        cvcuda.BlurBoxI(box=(50, 50, 7, 7), kernelSize=11),
                        cvcuda.BlurBoxI(box=(90, 90, 9, 9), kernelSize=17),
                    ],
                    [
                        cvcuda.BlurBoxI(box=(10, 10, 5, 5), kernelSize=7),
                        cvcuda.BlurBoxI(box=(50, 50, 7, 7), kernelSize=11),
                        cvcuda.BlurBoxI(box=(90, 90, 9, 9), kernelSize=17),
                    ],
                    [
                        cvcuda.BlurBoxI(box=(10, 10, 5, 5), kernelSize=7),
                        cvcuda.BlurBoxI(box=(50, 50, 7, 7), kernelSize=11),
                        cvcuda.BlurBoxI(box=(90, 90, 9, 9), kernelSize=17),
                    ],
                ],
            ),
        ),
    ],
)
def test_op_boxblur(inputp, blurboxes):
    input = cvcuda.Tensor(*inputp)

    out = cvcuda.boxblur(input, blurboxes)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.boxblur_into(out, input, blurboxes)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.boxblur(src=input, bboxes=blurboxes, stream=stream)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.boxblur_into(dst=out, src=input, bboxes=blurboxes, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


def _boxblur_params(dtype, layout, channels):
    boxes = [
        [
            cvcuda.BlurBoxI(box=(5, 5, 3, 3), kernelSize=3),
        ]
    ]
    return {"bboxes": cvcuda.BlurBoxesI(boxes=boxes)}


globals().update(
    cv_tools.make_op_tests(
        name="boxblur",
        runner_info=[("tensor", cvcuda.boxblur, _boxblur_params)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.S8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={3, 4},
    )
)
