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
from torchvision.ops import nms as torch_nms
import pytest as t


@t.mark.parametrize(
    "src,scores",
    [
        (
            [
                [93, 55, 276, 129],
                [96, 45, 281, 165],
                [92, 48, 279, 165],
                [0, 0, 20, 20],
            ],
            [0.7, 0.5, 0.5, 0.9],
        ),
        (
            [
                [106, 137, 283, 341],
                [98, 133, 282, 344],
                [92, 134, 279, 347],
                [10, 10, 50, 50],
                [15, 15, 40, 47],
            ],
            [0.85, 0.89, 0.2, 0.9, 0.65],
        ),
    ],
)
def test_op_nms(src, scores):

    score_theshold = 0.8
    iou_threshold = 0.1

    src_torch = torch.Tensor(src).type(torch.int32).unsqueeze(0)
    scores_torch = torch.Tensor(scores).type(torch.float32).unsqueeze(0)

    batch_size = 4
    src_batch_torch = torch.Tensor(src).type(torch.int32).repeat(batch_size, 1, 1)
    scores_batch_torch = torch.Tensor(scores).type(torch.float32).repeat(batch_size, 1)

    factor = 1000
    src_huge_torch = torch.Tensor(src).type(torch.int32).repeat(1, factor, 1)
    scores_huge_torch = torch.Tensor(scores).type(torch.float32).repeat(1, factor)

    # Small test
    src_cvcuda = cvcuda.as_tensor(src_torch.contiguous().cuda(), "NCW")
    scores_cvcuda = cvcuda.as_tensor(scores_torch.contiguous().cuda(), "NW")
    result_cvcuda = cvcuda.nms(src_cvcuda, scores_cvcuda, score_theshold, iou_threshold)
    assert result_cvcuda.layout == src_cvcuda.layout
    assert result_cvcuda.dtype == src_cvcuda.dtype
    assert result_cvcuda.shape == src_cvcuda.shape

    result_torch = torch.as_tensor(result_cvcuda.cuda())
    result_idx = torch.unique(torch.nonzero(result_torch).T[1])
    result_small_found = torch.index_select(result_torch, 1, result_idx)
    scores_small_found = torch.index_select(scores_torch, 1, result_idx)

    # Batch test
    src_cvcuda = cvcuda.as_tensor(src_batch_torch.contiguous().cuda(), "NCW")
    scores_cvcuda = cvcuda.as_tensor(scores_batch_torch.contiguous().cuda(), "NW")
    result_cvcuda = cvcuda.nms(src_cvcuda, scores_cvcuda, score_theshold, iou_threshold)
    assert result_cvcuda.layout == src_cvcuda.layout
    assert result_cvcuda.dtype == src_cvcuda.dtype
    assert result_cvcuda.shape == src_cvcuda.shape

    result_torch = torch.as_tensor(result_cvcuda.cuda())
    result_idx = torch.unique(torch.nonzero(result_torch).T[1])
    result_batch_found = torch.index_select(result_torch, 1, result_idx)

    # Huge test
    src_cvcuda = cvcuda.as_tensor(src_huge_torch.contiguous().cuda(), "NCW")
    scores_cvcuda = cvcuda.as_tensor(scores_huge_torch.contiguous().cuda(), "NW")
    result_cvcuda = cvcuda.nms(src_cvcuda, scores_cvcuda, score_theshold, iou_threshold)
    assert result_cvcuda.layout == src_cvcuda.layout
    assert result_cvcuda.dtype == src_cvcuda.dtype
    assert result_cvcuda.shape == src_cvcuda.shape

    result_torch = torch.as_tensor(result_cvcuda.cuda())
    result_idx = torch.unique(torch.nonzero(result_torch).T[1])
    result_huge_found = torch.index_select(result_torch, 1, result_idx)
    scores_huge_found = torch.index_select(scores_huge_torch, 1, result_idx)

    # Sanity test
    assert result_small_found.shape == result_huge_found.shape
    assert torch.isin(result_small_found, result_huge_found).all()
    assert torch.isin(scores_small_found, scores_huge_found).all()
    for i in range(batch_size):
        assert result_small_found[0, :, :].shape == result_batch_found[i, :, :].shape
        assert torch.isin(
            result_small_found[0, :, :], result_batch_found[i, :, :]
        ).all()

    # Correctness test
    torch_src = src_huge_torch
    torch_src[:, :, 2] = torch_src[:, :, 2] + torch_src[:, :, 0]
    torch_src[:, :, 3] = torch_src[:, :, 3] + torch_src[:, :, 1]
    torch_result = torch_nms(
        boxes=torch_src[0, :, :].type(torch.float),
        scores=scores_huge_torch[0, :],
        iou_threshold=iou_threshold,
    )
    torch_result_found = torch.index_select(torch_src[0, :, :], 0, torch_result)
    torch_scores_found = torch.index_select(scores_huge_torch[0, :], 0, torch_result)
    torch_result_found[:, 2] = torch_result_found[:, 2] - torch_result_found[:, 0]
    torch_result_found[:, 3] = torch_result_found[:, 3] - torch_result_found[:, 1]
    assert result_huge_found.shape[1] <= torch_result_found.shape[0]
    assert torch.isin(result_huge_found[0, :], torch_result_found).all()
    assert torch.isin(scores_huge_found[0, :], torch_scores_found).all()
