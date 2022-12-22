/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Requirements.hpp"

#include "Exception.hpp"

#include <util/Math.hpp>

#include <algorithm>
#include <cmath>

namespace nvcv::priv {

void Init(NVCVRequirements &reqs)
{
    reqs = {};
}

void Add(NVCVRequirements &reqSum, const NVCVRequirements &req)
{
    // just to make sure those have the types we expect
    const NVCVMemRequirements &reqCudaMem       = req.cudaMem;
    const NVCVMemRequirements &reqHostMem       = req.hostMem;
    const NVCVMemRequirements &reqHostPinnedMem = req.hostPinnedMem;

    for (size_t i = 0; i < sizeof(NVCVMemRequirements::numBlocks) / sizeof(NVCVMemRequirements::numBlocks[0]); ++i)
    {
        reqSum.cudaMem.numBlocks[i] += reqCudaMem.numBlocks[i];
        reqSum.hostMem.numBlocks[i] += reqHostMem.numBlocks[i];
        reqSum.hostPinnedMem.numBlocks[i] += reqHostPinnedMem.numBlocks[i];
    }
}

void AddBuffer(NVCVMemRequirements &memReq, int64_t bufSize, int64_t bufAlignment)
{
    if (bufAlignment <= 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Buffer alignment must be > 0, not %ld", bufAlignment);
    }

    if (!util::IsPowerOfTwo(bufAlignment))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Buffer alignment must be a power of two, not %ld", bufAlignment);
    }

    int log2Align = util::ILog2(bufAlignment);

    if (log2Align >= NVCV_MAX_MEM_REQUIREMENTS_LOG2_BLOCK_SIZE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Buffer alignment must be <= %ld, not %ld",
                        NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE, bufAlignment);
    }

    int64_t numBlocks = util::DivUpPowerOfTwo(std::abs(bufSize), bufAlignment);

    if (bufSize >= 0)
    {
        int64_t maxBlocks = std::numeric_limits<std::remove_reference_t<decltype(memReq.numBlocks[0])>>::max();

        if (memReq.numBlocks[log2Align] + numBlocks < memReq.numBlocks[log2Align])
        {
            throw Exception(NVCV_ERROR_OVERFLOW,
                            "Number of blocks with alignment %ld would overflow, buffer size must be <= %ld, not %ld",
                            bufAlignment, maxBlocks - memReq.numBlocks[log2Align], numBlocks);
        }

        memReq.numBlocks[log2Align] += numBlocks;
    }
    else
    {
        // Underflows are clamped to 0
        memReq.numBlocks[log2Align] = std::max((int64_t)0, memReq.numBlocks[log2Align] - numBlocks);
    }
}

int64_t CalcTotalSizeBytes(const NVCVMemRequirements &memReq)
{
    uint64_t total = 0;
    for (size_t i = 0; i < sizeof(memReq.numBlocks) / sizeof(memReq.numBlocks[0]); ++i)
    {
        uint64_t cur = memReq.numBlocks[i] * (1ull << i);
        if (total + cur < total)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Memory size overflow");
        }
        total += cur;
    }

    if (total > INT64_MAX)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Memory size overflow");
    }

    return static_cast<int64_t>(total);
}

} // namespace nvcv::priv
