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

import cvcuda_util as util
import cvcuda_tools as cv_tools
import cupy

params = [
    (((10, 16, 23, 1), np.uint8, "NHWC")),
    (((10, 1, 16, 23), np.uint8, "NCHW")),
    (((1, 160, 3, 1), np.uint8, "NHWC")),
    (((16, 23, 1), np.uint8, "HWC")),
    (((1, 16, 23), np.uint8, "CHW")),
    (((257, 23, 1), np.uint8, "HWC")),
    (((100, 200, 3, 1), np.uint8, "NHWC")),
    (((100, 1, 200, 3), np.uint8, "NCHW")),
    (((50, 50, 2, 1), np.uint8, "NHWC")),
    (((27, 25, 1), np.uint8, "HWC")),
    (((10, 10, 1), np.uint8, "HWC")),
    (((5, 5, 1), np.uint8, "HWC")),
]


@pytest.mark.parametrize("input", params)
def test_op_histogram(input):

    inputT = cvcuda.Tensor(*input)

    out = cvcuda.histogram(inputT)

    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result = cupy.asarray(out.cuda())

    # Sum up the entries in result
    actual_sum = np.sum(result.get())
    total_entries = np.prod(input[0])
    assert actual_sum == total_entries

    rank = len(input[0])
    if rank == 3:
        # If the rank is 3, create an array of shape (1, 256)
        new_shape = ((1, 256, 1), np.int32, "HWC")
    elif rank == 4:
        new_shape = ((input[0][0], 256, 1), np.int32, "HWC")
    else:
        pytest.fail("Invalid test input")

    out = cvcuda.Tensor(*new_shape)
    tmp = cvcuda.histogram_into(histogram=out, src=inputT)

    assert tmp is out
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result = cupy.asarray(out.cuda())

    # Sum up the entries in result
    actual_sum = np.sum(result.get())
    total_entries = np.prod(input[0])
    assert actual_sum == total_entries


@pytest.mark.parametrize("input", params)
def test_op_histogram_mask(input):

    inputT = cvcuda.Tensor(*input)
    rng = np.random.default_rng(0)
    arr = rng.random(input[0])
    arr = (arr * 3).astype(np.uint8)
    maskT = util.to_cvcuda_tensor(arr, input[2])

    assert maskT.shape == inputT.shape

    out = cvcuda.histogram(inputT, maskT)
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result = cupy.asarray(out.cuda())

    # Sum up the entries in result
    actual_sum = np.sum(result.get())
    masked_entries = np.count_nonzero(arr)
    assert actual_sum == masked_entries

    rank = len(input[0])
    if rank == 3:
        # If the rank is 3, create an array of shape (1, 256)
        new_shape = ((1, 256, 1), np.int32, "HWC")
    elif rank == 4:
        new_shape = ((input[0][0], 256, 1), np.int32, "HWC")
    else:
        pytest.fail("Invalid test input")

    out = cvcuda.Tensor(*new_shape)
    tmp = cvcuda.histogram_into(histogram=out, mask=maskT, src=inputT)

    assert tmp is out
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result = cupy.asarray(out.cuda())

    # Sum up the entries in result
    actual_sum = np.sum(result.get())
    assert actual_sum == masked_entries


globals().update(
    cv_tools.make_op_tests(
        name="histogram",
        runner_info=[("tensor", cvcuda.histogram, None)],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 1),
        supported_dtypes={cvcuda.Type.U8},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1},
    )
)
