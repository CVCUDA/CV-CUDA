/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/priv/FlipPolicy.hpp>

namespace priv = cvcuda::priv;

TEST(OpFlipPolicy, UsesSixteenInterleavedVarShapeU8GroupsOnBlackwell)
{
    EXPECT_EQ(16, priv::FlipVarShapeU8GroupsPerThread(100, false));
    EXPECT_EQ(16, priv::FlipVarShapeU8GroupsPerThread(103, false));
    EXPECT_EQ(16, priv::FlipVarShapeU8GroupsPerThread(120, false));
}

TEST(OpFlipPolicy, KeepsFourVarShapeU8GroupsBelowBlackwell)
{
    for (int sm : {80, 86, 89, 90})
    {
        EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(sm, false)) << "sm " << sm;
        EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(sm, true)) << "sm " << sm;
    }
}

// Sixteen groups per thread more than doubles the planar var-shape row on Blackwell, so the
// planar path must stay at the conservative count even where the interleaved one is raised.
TEST(OpFlipPolicy, KeepsFourPlanarVarShapeU8GroupsOnBlackwell)
{
    EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(100, true));
    EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(103, true));
    EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(120, true));
}

// An unqueryable device reports 0; it must not fall into the Blackwell-only count.
TEST(OpFlipPolicy, KeepsFourVarShapeU8GroupsForAnUnknownDevice)
{
    EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(0, false));
    EXPECT_EQ(4, priv::FlipVarShapeU8GroupsPerThread(0, true));
}
