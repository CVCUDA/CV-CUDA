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

#include <cvcuda/priv/OpHQResizePolicy.hpp>

namespace hqresize = cvcuda::priv::hq_resize;

TEST(OpHQResizePolicy, UsesSeparableF32CubicContract2xOnSM75)
{
    using Path = hqresize::DirectTensorPath;
    using Desc = hqresize::DirectTensorPathDesc;

    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(75, Path::CubicContract2x, true, 1));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(75, Path::CubicContract2x, true, 3));

    const Desc cubicContract{false, true, false, 0.f, 0.f, 2.f, 2.f, 1920, 1080, 960, 540};
    EXPECT_FALSE(hqresize::ShouldUseDirectTensorPathForSM(75, cubicContract, true, 1));
    EXPECT_FALSE(hqresize::ShouldUseDirectTensorPathForSM(75, cubicContract, true, 3));
}

TEST(OpHQResizePolicy, KeepsDirectCubicContract2xForUnaffectedCases)
{
    using Path = hqresize::DirectTensorPath;

    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(75, Path::CubicContract2x, false, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(75, Path::CubicContract2x, false, 3));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(75, Path::CubicContract2x, true, 2));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(75, Path::CubicContract2x, true, 4));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(75, Path::LinearExpand2x, true, 3));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(80, Path::CubicContract2x, true, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(90, Path::CubicContract2x, true, 3));
}
