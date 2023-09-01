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
import torch
import pytest as t
import numpy as np
import cvcuda_util as util


params = [
    (((10, 16, 23, 1), np.uint8, "NHWC")),
    (((1, 160, 3, 1), np.uint8, "NHWC")),
    (((16, 23, 1), np.uint8, "HWC")),
    (((257, 23, 1), np.uint8, "HWC")),
    (((100, 200, 3, 1), np.uint8, "NHWC")),
    (((50, 50, 2, 1), np.uint8, "NHWC")),
    (((27, 25, 1), np.uint8, "HWC")),
    (((10, 10, 1), np.uint8, "HWC")),
    (((5, 5, 1), np.uint8, "HWC")),
]


@t.mark.parametrize("input", params)
def test_op_histogram(input):

    inputT = cvcuda.Tensor(*input)

    out = cvcuda.histogram(inputT)
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result_torch = torch.as_tensor(out.cuda())

    # Sum up the entries in result_torch
    actual_sum = torch.sum(result_torch)
    total_entries = np.prod(input[0])
    assert actual_sum == total_entries

    rank = len(input[0])
    if rank == 3:
        # If the rank is 3, create an array of shape (1, 256)
        new_shape = ((1, 256, 1), np.int32, "HWC")
    elif rank == 4:
        new_shape = ((input[0][0], 256, 1), np.int32, "HWC")
    else:
        t.fail("Invalid test input")

    out = cvcuda.Tensor(*new_shape)
    tmp = cvcuda.histogram_into(histogram=out, src=inputT)

    assert tmp is out
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result_torch = torch.as_tensor(out.cuda())

    # Sum up the entries in result_torch
    actual_sum = torch.sum(result_torch)
    total_entries = np.prod(input[0])
    assert actual_sum == total_entries


@t.mark.parametrize("input", params)
def test_op_histogram_mask(input):

    inputT = cvcuda.Tensor(*input)
    arr = np.random.random(input[0])
    arr = (arr * 3).astype(np.uint8)
    maskT = util.to_nvcv_tensor(arr, input[2])

    assert maskT.shape == inputT.shape

    out = cvcuda.histogram(inputT, maskT)
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result_torch = torch.as_tensor(out.cuda())

    # Sum up the entries in result_torch
    actual_sum = torch.sum(result_torch)
    masked_entries = np.count_nonzero(arr)
    assert actual_sum == masked_entries

    rank = len(input[0])
    if rank == 3:
        # If the rank is 3, create an array of shape (1, 256)
        new_shape = ((1, 256, 1), np.int32, "HWC")
    elif rank == 4:
        new_shape = ((input[0][0], 256, 1), np.int32, "HWC")
    else:
        t.fail("Invalid test input")

    out = cvcuda.Tensor(*new_shape)
    tmp = cvcuda.histogram_into(histogram=out, mask=maskT, src=inputT)

    assert tmp is out
    assert out.shape[1] == 256
    assert out.dtype == np.int32

    result_torch = torch.as_tensor(out.cuda())

    # Sum up the entries in result_torch
    actual_sum = torch.sum(result_torch)
    assert actual_sum == masked_entries
