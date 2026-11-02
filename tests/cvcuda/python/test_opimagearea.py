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

import cvcuda

import pytest
import numpy as np

import cupy

height_width = {
    "NHWC": (1, 2),
    "NCHW": (2, 3),
    "HWC": (0, 1),
    "CHW": (1, 2),
}

params = [
    (((4, 13, 23, 1), np.uint8, "NHWC")),
    (((4, 13, 23, 2), np.uint8, "NHWC")),
    (((4, 13, 23, 3), np.uint8, "NHWC")),
    (((4, 13, 23, 4), np.uint8, "NHWC")),
    (((4, 2, 13, 23), np.uint8, "NCHW")),
    (((4, 3, 13, 23), np.uint8, "NCHW")),
    (((4, 3, 13, 23), np.float32, "NCHW")),
    (((13, 23, 3), np.uint8, "HWC")),
    (((3, 13, 23), np.uint8, "CHW")),
]


@pytest.mark.parametrize("input", params)
def test_op_image_area(input):
    shape, _, layout = input
    inputT = cvcuda.Tensor(*input)

    out = cvcuda.image_area(inputT)

    assert out.layout == "N"
    assert out.dtype == np.int32

    height_idx, width_idx = height_width[layout]
    expected_area = shape[height_idx] * shape[width_idx]
    expected_batch = shape[0] if len(shape) == 4 else 1
    assert out.shape == (expected_batch,)

    # Independent CPU gold: area is width * height, channels and dtype excluded.
    result = cupy.asarray(out.cuda()).get()
    assert (result == expected_area).all()


def test_op_image_area_into():
    src = cvcuda.Tensor((4, 13, 23, 3), np.uint8, "NHWC")
    dst = cvcuda.Tensor((4,), np.int32, "N")

    out = cvcuda.image_area_into(dst, src)

    assert out is dst
    assert (cupy.asarray(out.cuda()).get() == 13 * 23).all()


def test_op_image_area_rejects_mismatched_batch():
    src = cvcuda.Tensor((4, 13, 23, 3), np.uint8, "NHWC")
    dst = cvcuda.Tensor((3,), np.int32, "N")

    with pytest.raises(Exception):
        cvcuda.image_area_into(dst, src)
