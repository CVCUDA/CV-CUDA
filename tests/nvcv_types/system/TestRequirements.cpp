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

#include "Definitions.hpp"

#include <nvcv/alloc/Requirements.hpp>
#include <util/Math.hpp>

#include <algorithm>

namespace util = nvcv::util;

static void AssertEq(const NVCVRequirements &reqGold, const nvcv::Requirements &req)
{
    for (int i = 0; i < nvcv::Requirements::Memory::size(); ++i)
    {
        SCOPED_TRACE("log2 BlockSize: " + std::to_string(i));

        ASSERT_EQ(reqGold.cudaMem.numBlocks[i], req.cudaMem().numBlocks(i));
        ASSERT_EQ(reqGold.hostMem.numBlocks[i], req.hostMem().numBlocks(i));
        ASSERT_EQ(reqGold.hostPinnedMem.numBlocks[i], req.hostPinnedMem().numBlocks(i));
    }
}

TEST(Requirements, init)
{
    std::byte buf[sizeof(nvcv::Requirements)];
    memset(&buf, 123, sizeof(buf));

    auto *reqs = new (buf) nvcv::Requirements;

    NVCVRequirements reqGold = {};

    ASSERT_NO_FATAL_FAILURE(AssertEq(reqGold, *reqs));

    EXPECT_EQ(0, CalcTotalSizeBytes(reqs->cudaMem()));
    EXPECT_EQ(0, CalcTotalSizeBytes(reqs->hostMem()));
    EXPECT_EQ(0, CalcTotalSizeBytes(reqs->hostPinnedMem()));

    reqs->~Requirements();
}

TEST(MemRequirements, add_buffer)
{
    nvcv::Requirements reqs;

    int bufAlign1 = 256;
    int bufSize1  = 3 * bufAlign1 - 5;
    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(bufSize1, bufAlign1));

    int bufAlign2 = 16384;
    int bufSize2  = 6 * bufAlign2 - 17;
    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(bufSize2, bufAlign2));

    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(bufAlign2 / 3, bufAlign2));

    bufSize2 += bufAlign2 / 3;

    for (int i = 0; i < reqs.cudaMem().size(); ++i)
    {
        if (i == std::log2(bufAlign1))
        {
            EXPECT_EQ(util::DivUp(bufSize1, bufAlign1), reqs.cudaMem().numBlocks(i));
        }
        else if (i == std::log2(bufAlign2))
        {
            EXPECT_EQ(util::DivUp(bufSize2, bufAlign2), reqs.cudaMem().numBlocks(i));
        }
        else
        {
            EXPECT_EQ(0, reqs.cudaMem().numBlocks(i));
        }
    }
}

TEST(MemRequirements, sub_buffer)
{
    nvcv::Requirements reqs;

    int bufAlign1 = 256;
    int bufSize1  = 3 * bufAlign1 - 5;
    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(bufSize1, bufAlign1));

    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(-511, bufAlign1));
    bufSize1 -= 512;

    int bufAlign2 = 16384;
    int bufSize2  = 6 * bufAlign2 - 17;
    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(bufSize2, bufAlign2));

    ASSERT_NO_THROW(reqs.cudaMem().addBuffer(-bufSize2 * 5, bufAlign2));
    bufSize2 = 0;

    for (int i = 0; i < reqs.cudaMem().size(); ++i)
    {
        if (i == std::log2(bufAlign1))
        {
            EXPECT_EQ(util::DivUp(bufSize1, bufAlign1), reqs.cudaMem().numBlocks(i));
        }
        else
        {
            EXPECT_EQ(0, reqs.cudaMem().numBlocks(i));
        }
    }
}

TEST(Requirements, add)
{
    nvcv::Requirements reqSum;
    NVCVRequirements   reqGold = {};

    nvcv::Requirements reqA;
    for (int i = 0; i < nvcv::Requirements::Memory::size(); ++i)
    {
        int64_t bufAlign = ((int64_t)1) << i;

        int64_t bufSize = 3 * (i + 1) * bufAlign - 1;
        ASSERT_NO_THROW(reqA.cudaMem().addBuffer(bufSize, bufAlign));
        reqGold.cudaMem.numBlocks[i] += util::DivUp(bufSize, bufAlign);

        bufSize = 5 * (i + 1) * bufAlign - 2;
        ASSERT_NO_THROW(reqA.hostMem().addBuffer(bufSize, bufAlign));
        reqGold.hostMem.numBlocks[i] += util::DivUp(bufSize, bufAlign);

        bufSize = 7 * (i + 1) * bufAlign - 3;
        ASSERT_NO_THROW(reqA.hostPinnedMem().addBuffer(bufSize, bufAlign));
        reqGold.hostPinnedMem.numBlocks[i] += util::DivUp(bufSize, bufAlign);
    }

    reqSum += reqA;

    ASSERT_NO_FATAL_FAILURE(AssertEq(reqGold, reqSum));

    nvcv::Requirements reqB;
    for (int i = 0; i < nvcv::Requirements::Memory::size(); ++i)
    {
        int64_t bufAlign = ((int64_t)1) << i;

        int64_t bufSize = 17 * (i + 1) * bufAlign - 1;
        ASSERT_NO_THROW(reqB.cudaMem().addBuffer(bufSize, bufAlign));
        reqGold.cudaMem.numBlocks[i] += util::DivUp(bufSize, bufAlign);

        bufSize = 21 * (i + 1) * bufAlign - 2;
        ASSERT_NO_THROW(reqB.hostMem().addBuffer(bufSize, bufAlign));
        reqGold.hostMem.numBlocks[i] += util::DivUp(bufSize, bufAlign);

        bufSize = 37 * (i + 1) * bufAlign - 3;
        ASSERT_NO_THROW(reqB.hostPinnedMem().addBuffer(bufSize, bufAlign));
        reqGold.hostPinnedMem.numBlocks[i] += util::DivUp(bufSize, bufAlign);
    }

    reqSum += reqB;

    ASSERT_NO_FATAL_FAILURE(AssertEq(reqGold, reqSum));

    for (int i = 0; i < nvcv::Requirements::Memory::size(); ++i)
    {
        reqGold.cudaMem.numBlocks[i] *= 2;
        reqGold.hostMem.numBlocks[i] *= 2;
        reqGold.hostPinnedMem.numBlocks[i] *= 2;
    }

    reqSum += reqSum;

    ASSERT_NO_FATAL_FAILURE(AssertEq(reqGold, reqSum));
}

TEST(Requirements, total_size_bytes)
{
    nvcv::Requirements req;

    int64_t totSizeDevice     = 0;
    int64_t totSizeHost       = 0;
    int64_t totSizeHostPinned = 0;

    for (int i = 0; i < nvcv::Requirements::Memory::size(); ++i)
    {
        int64_t bufAlign = ((int64_t)1) << i;

        int64_t bufSize = 3 * (i + 1) * bufAlign - 1;
        req.cudaMem().addBuffer(bufSize, bufAlign);
        totSizeDevice += util::RoundUp(bufSize, bufAlign);

        bufSize = 5 * (i + 1) * bufAlign - 2;
        req.hostPinnedMem().addBuffer(bufSize, bufAlign);
        totSizeHostPinned += util::RoundUp(bufSize, bufAlign);

        bufSize = 7 * (i + 1) * bufAlign - 3;
        req.hostMem().addBuffer(bufSize, bufAlign);
        totSizeHost += util::RoundUp(bufSize, bufAlign);
    }

    EXPECT_EQ(totSizeDevice, CalcTotalSizeBytes(req.cudaMem()));
    EXPECT_EQ(totSizeHost, CalcTotalSizeBytes(req.hostMem()));
    EXPECT_EQ(totSizeHostPinned, CalcTotalSizeBytes(req.hostPinnedMem()));
}
