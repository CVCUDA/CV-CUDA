# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import cvcuda
import pytest as t
from torch.nn.utils.rnn import pad_sequence


@t.mark.parametrize(
    "contourData, numPointsInContour, openCvRes",
    [
        (
            [
                [
                    845,
                    600,
                    845,
                    601,
                    847,
                    603,
                    859,
                    603,
                    860,
                    604,
                    865,
                    604,
                    866,
                    603,
                    867,
                    603,
                    868,
                    602,
                    868,
                    601,
                    867,
                    600,
                ],
                [
                    965,
                    489,
                    964,
                    490,
                    963,
                    490,
                    962,
                    491,
                    962,
                    494,
                    963,
                    495,
                    963,
                    499,
                    964,
                    500,
                    964,
                    501,
                    966,
                    503,
                    1011,
                    503,
                    1012,
                    504,
                    1013,
                    503,
                    1027,
                    503,
                    1027,
                    502,
                    1028,
                    501,
                    1028,
                    490,
                    1027,
                    489,
                ],
                [
                    1050,
                    198,
                    1049,
                    199,
                    1040,
                    199,
                    1040,
                    210,
                    1041,
                    211,
                    1040,
                    212,
                    1040,
                    214,
                    1045,
                    214,
                    1046,
                    213,
                    1049,
                    213,
                    1050,
                    212,
                    1051,
                    212,
                    1052,
                    211,
                    1053,
                    211,
                    1054,
                    210,
                    1055,
                    210,
                    1056,
                    209,
                    1058,
                    209,
                    1059,
                    208,
                    1059,
                    200,
                    1058,
                    200,
                    1057,
                    199,
                    1051,
                    199,
                ],
            ],
            [11, 18, 23],
            [
                [868.0, 604.0, 845.0, 604.0, 845.0, 600.0, 868.0, 600.0],
                [962.0, 504.0, 962.0, 489.0, 1028.0, 489.0, 1028.0, 504.0],
                [1040.0, 214.0, 1040.0, 198.0, 1059.0, 198.0, 1059.0, 214.0],
            ],
        ),
    ],
)
def test_op_minarearect(contourData, numPointsInContour, openCvRes):

    batchSize = len(contourData)
    numPointsInContour_torch = (
        torch.Tensor(numPointsInContour).type(torch.int32).unsqueeze(0)
    )
    src_torch = (
        pad_sequence([torch.Tensor(t) for t in contourData], batch_first=True)
        .type(torch.int16)
        .reshape(batchSize, -1, 2)
    )
    gold_torch = torch.Tensor(openCvRes).type(torch.float32)

    src_cvcuda = cvcuda.as_tensor(src_torch.contiguous().cuda(), "NWC")
    pointNumInContour_cvcuda = cvcuda.as_tensor(
        numPointsInContour_torch.contiguous().cuda(), "NW"
    )
    gold_cvcuda = cvcuda.as_tensor(gold_torch.contiguous().cuda(), "NW")

    result_cvcuda = cvcuda.minarearect(
        src_cvcuda, pointNumInContour_cvcuda, src_torch.shape[0]
    )
    assert result_cvcuda.layout == gold_cvcuda.layout
    assert result_cvcuda.shape == gold_cvcuda.shape
    assert result_cvcuda.dtype == gold_cvcuda.dtype
    result_torch, _ = torch.sort(
        torch.as_tensor(result_cvcuda.cuda()).reshape(batchSize, -1, 2), dim=1
    )
    gold_torch, _ = torch.sort(gold_torch.reshape(batchSize, -1, 2), dim=1)
    assert (gold_torch.cuda() - result_torch.cuda() < 5.0).all()

    stream = cvcuda.cuda.Stream()
    out = cvcuda.Tensor(gold_cvcuda.shape, gold_cvcuda.dtype, gold_cvcuda.layout)
    tmp = cvcuda.minarearect_into(
        out, src_cvcuda, pointNumInContour_cvcuda, src_torch.shape[0]
    )
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.minarearect(
        src=src_cvcuda,
        numPointsInContour=pointNumInContour_cvcuda,
        totalContours=src_torch.shape[0],
        stream=stream,
    )
    assert out.layout == gold_cvcuda.layout
    assert out.shape == gold_cvcuda.shape
    assert out.dtype == gold_cvcuda.dtype

    tmp = cvcuda.minarearect_into(
        src=src_cvcuda,
        dst=out,
        numPointsInContour=pointNumInContour_cvcuda,
        totalContours=src_torch.shape[0],
        stream=stream,
    )
    assert tmp is out
