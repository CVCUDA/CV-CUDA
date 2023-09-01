# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvcv
import cvcuda
import pytest as t
import numpy as np
import cvcuda_util as util

RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "shape, dtype, layout",
    [((1, 16, 23, 1), np.uint8, "NHWC"), ((1, 32, 32, 1), np.uint8, "NHWC")],
)
def test_op_find_contours(shape, dtype, layout):
    print(shape, dtype, layout)
    image = util.create_tensor(shape, dtype, layout, 1, rng=RNG)
    points = cvcuda.find_contours(image)
    assert points.shape[0] == image.shape[0]
    assert points.shape[2] == 2

    stream = cvcuda.Stream()
    points = cvcuda.Tensor(
        (image.shape[0], 1024, 2), nvcv.Type.S32, nvcv.TensorLayout.NHW
    )
    num_points = cvcuda.Tensor(
        (image.shape[0], 32), nvcv.Type.U32, nvcv.TensorLayout.NW
    )
    tmp = cvcuda.find_contours_into(
        src=image,
        points=points,
        num_points=num_points,
        stream=stream,
    )
    assert tmp is points
    assert points.shape[0] == image.shape[0]
    assert points.shape[2] == 2
