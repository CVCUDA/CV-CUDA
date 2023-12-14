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
    "inputp, elements",
    [
        (
            (((2, 224, 224, 4), np.uint8, "NHWC")),
            cvcuda.Elements(
                elements=[
                    [
                        cvcuda.BndBoxI(
                            box=(10, 10, 5, 5),
                            thickness=2,
                            borderColor=(255, 255, 0),
                            fillColor=(0, 128, 255, 128),
                        )
                    ],
                    [
                        cvcuda.BndBoxI(
                            box=(10, 10, 5, 5),
                            thickness=2,
                            borderColor=(255, 255, 0),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.Label(
                            utf8Text="def",
                            fontSize=30,
                            tlPos=(50, 50),
                            fontColor=(255, 255, 0),
                            bgColor=(0, 128, 255, 128),
                        ),
                        cvcuda.Segment(
                            box=(20, 20, 30, 30),
                            thickness=1,
                            segArray=np.array(
                                [
                                    [0, 0, 0, 0, 0.2, 0.2, 0, 0, 0, 0],
                                    [0, 0, 0, 0.2, 0.3, 0.3, 0.2, 0, 0, 0],
                                    [0, 0, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0, 0],
                                    [0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0],
                                    [0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2],
                                    [0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2],
                                    [0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0],
                                    [0, 0, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0, 0],
                                    [0, 0, 0, 0.2, 0.3, 0.3, 0.2, 0, 0, 0],
                                    [0, 0, 0, 0, 0.2, 0.2, 0, 0, 0, 0],
                                ]
                            ),
                            segThreshold=0.2,
                            borderColor=(255, 255, 0),
                            segColor=(0, 128, 255, 128),
                        ),
                        cvcuda.Point(
                            centerPos=(30, 30),
                            radius=5,
                            color=(255, 255, 0),
                        ),
                        cvcuda.Line(
                            pos0=(50, 50),
                            pos1=(150, 50),
                            thickness=1,
                            color=(255, 0, 0),
                        ),
                        cvcuda.PolyLine(
                            points=np.array(
                                [
                                    [100, 100],
                                    [600, 100],
                                    [350, 300],
                                    [600, 500],
                                    [300, 500],
                                ]
                            ),
                            thickness=1,
                            isClosed=True,
                            borderColor=(255, 255, 0),
                            fillColor=(0, 128, 255, 128),
                        ),
                        cvcuda.RotatedBox(
                            centerPos=(30, 30),
                            width=5,
                            height=5,
                            yaw=0.3,
                            thickness=1,
                            borderColor=(255, 255, 0),
                            bgColor=(0, 128, 255, 128),
                        ),
                        cvcuda.Circle(
                            centerPos=(30, 30),
                            radius=5,
                            thickness=2,
                            borderColor=(255, 255, 0),
                            bgColor=(0, 128, 255, 128),
                        ),
                        cvcuda.Arrow(
                            pos0=(50, 50),
                            pos1=(150, 50),
                            arrowSize=3,
                            thickness=1,
                            color=(255, 0, 0),
                        ),
                        cvcuda.Clock(
                            clockFormat=cvcuda.ClockFormat.YYMMDD_HHMMSS,
                            time=0,
                            fontSize=10,
                            tlPos=(150, 50),
                            fontColor=(255, 255, 0),
                            bgColor=(0, 128, 255, 128),
                        ),
                    ],
                ],
            ),
        ),
    ],
)
def test_op_osd(inputp, elements):
    input = cvcuda.Tensor(*inputp)

    out = cvcuda.osd(input, elements)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.osd_into(out, input, elements)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.osd(src=input, elements=elements, stream=stream)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.osd_into(dst=out, src=input, elements=elements, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
