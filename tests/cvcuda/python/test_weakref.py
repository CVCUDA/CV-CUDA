# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import weakref

import cvcuda
import numpy as np
import pytest as t


CONCRETE_PUBLIC_CLASSES = (
    "Arrow",
    "BlurBoxI",
    "BlurBoxesI",
    "BndBoxI",
    "BndBoxesI",
    "Circle",
    "Clock",
    "Elements",
    "ExternalBuffer",
    "Format",
    "Image",
    "ImageBatchVarShape",
    "Label",
    "Line",
    "Point",
    "PolyLine",
    "RectI",
    "RotatedBox",
    "Segment",
    "Stream",
    "Tensor",
    "TensorBatch",
    "TensorLayout",
    "Type",
)


@t.fixture(scope="module")
def concrete_public_instances():
    tensor = cvcuda.Tensor((1, 2, 2, 1), cvcuda.Type.U8, "NHWC")
    return {
        "Arrow": cvcuda.Arrow((0, 0), (1, 1), 1, 1, (0, 0, 0, 0)),
        "BlurBoxI": cvcuda.BlurBoxI((0, 0, 1, 1), 1),
        "BlurBoxesI": cvcuda.BlurBoxesI([[]]),
        "BndBoxI": cvcuda.BndBoxI((0, 0, 1, 1), 1, (0, 0, 0, 0), (0, 0, 0, 0)),
        "BndBoxesI": cvcuda.BndBoxesI([[]]),
        "Circle": cvcuda.Circle((0, 0), 1, 1, (0, 0, 0, 0), (0, 0, 0, 0)),
        "Clock": cvcuda.Clock(
            cvcuda.ClockFormat.HHMMSS,
            0,
            1,
            tlPos=(0, 0),
            fontColor=(0, 0, 0, 0),
            bgColor=(0, 0, 0, 0),
        ),
        "Elements": cvcuda.Elements([[]]),
        "ExternalBuffer": tensor.cuda(),
        "Format": cvcuda.Format.U8,
        "Image": cvcuda.Image((2, 2), cvcuda.Format.U8),
        "ImageBatchVarShape": cvcuda.ImageBatchVarShape(1),
        "Label": cvcuda.Label(
            "x", 1, tlPos=(0, 0), fontColor=(0, 0, 0, 0), bgColor=(0, 0, 0, 0)
        ),
        "Line": cvcuda.Line((0, 0), (1, 1), 1, (0, 0, 0, 0)),
        "Point": cvcuda.Point((0, 0), 1, (0, 0, 0, 0)),
        "PolyLine": cvcuda.PolyLine(
            np.array([[0, 0], [1, 1]], dtype=np.int32),
            1,
            False,
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        "RectI": cvcuda.RectI(),
        "RotatedBox": cvcuda.RotatedBox(
            (0, 0), 1, 1, 0.0, 1, (0, 0, 0, 0), (0, 0, 0, 0)
        ),
        "Segment": cvcuda.Segment(
            (0, 0, 1, 1),
            1,
            np.zeros((1, 1), dtype=np.float32),
            0.0,
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        "Stream": cvcuda.Stream(),
        "Tensor": tensor,
        "TensorBatch": cvcuda.TensorBatch(1),
        "TensorLayout": cvcuda.TensorLayout.NHWC,
        "Type": cvcuda.Type(np.uint8),
    }


@t.mark.parametrize("class_name", CONCRETE_PUBLIC_CLASSES)
def test_concrete_public_class_instances_support_weak_references(
    class_name, concrete_public_instances
):
    instance = concrete_public_instances[class_name]
    reference = weakref.ref(instance)

    assert reference() is instance
