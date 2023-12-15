/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/priv/WorkspaceEstimator.hpp>

TEST(WorkspaceMemEstimatorTest, Add)
{
    // set the alignment to 1 to see if element alignment gets propagated to the base alignment
    cvcuda::WorkspaceMemEstimator est(0, 1);
    est.add(3);
    EXPECT_EQ(est.req.alignment, 1);
    EXPECT_EQ(est.req.size, 3);
    est.add<int32_t>(3);
    EXPECT_EQ(est.req.size, 16);
    EXPECT_EQ(est.req.alignment, 4);
    est.add<float>();
    EXPECT_EQ(est.req.size, 20);
    est.add<float>(1, 16);
    EXPECT_EQ(est.req.size, 48);
    EXPECT_EQ(est.req.alignment, 16);
}

TEST(WorkspaceEstimatorTest, Add)
{
    cvcuda::WorkspaceEstimator est;
    EXPECT_EQ(est.hostMem.req.alignment, 16);
    EXPECT_EQ(est.pinnedMem.req.alignment, 256);
    EXPECT_EQ(est.cudaMem.req.alignment, 256);

    // set the alignment to 1 to see if element alignment gets propagated to the base alignment for each memory type
    est.hostMem.req.alignment   = 1;
    est.pinnedMem.req.alignment = 1;
    est.cudaMem.req.alignment   = 1;

    est.add(true, false, true, 3);
    EXPECT_EQ(est.hostMem.req.size, 3);
    EXPECT_EQ(est.pinnedMem.req.size, 0);
    EXPECT_EQ(est.cudaMem.req.size, 3);

    // clang-format off
    est.add<char>(true, false, false, 4)
       .add<int32_t>(false, true, true, 2);
    // clang-format on

    EXPECT_EQ(est.hostMem.req.size, 7);      // 7 chars
    EXPECT_EQ(est.hostMem.req.alignment, 1); // no change

    EXPECT_EQ(est.pinnedMem.req.size, 8);      // just the 2 integers
    EXPECT_EQ(est.pinnedMem.req.alignment, 4); // alignment for int32

    EXPECT_EQ(est.cudaMem.req.size, 12);     // 3 chars, padding, 2 ints
    EXPECT_EQ(est.cudaMem.req.alignment, 4); // alignment for int32
}
