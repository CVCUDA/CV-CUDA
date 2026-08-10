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

TEST(OpHQResizePolicy, ClassifiesOnlyExactDirectTensorPaths)
{
    using Path = hqresize::DirectTensorPath;
    using Desc = hqresize::DirectTensorPathDesc;

    const Desc linearExpand{true, false, true, 0.f, 0.f, 0.5f, 0.5f, 1920, 1080, 3840, 2160};
    const Desc cubicContract{false, true, false, 0.f, 0.f, 2.f, 2.f, 1920, 1080, 960, 540};
    const Desc cubicExpand{false, true, true, 0.f, 0.f, 0.5f, 0.5f, 1920, 1080, 3840, 2160};

    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(linearExpand), Path::LinearExpand2x);
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(cubicContract), Path::CubicContract2x);
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(cubicExpand), Path::CubicExpand2x);

    Desc control   = linearExpand;
    control.scaleX = 0.6f;
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(control), Path::Other);
    control         = cubicContract;
    control.originX = 1.f;
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(control), Path::Other);
    control        = cubicContract;
    control.xFirst = true;
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(control), Path::Other);
    control           = cubicExpand;
    control.outHeight = 2159;
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(control), Path::Other);
    control        = cubicExpand;
    control.xFirst = false;
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(control), Path::Other);
    control       = cubicExpand;
    control.cubic = false; // Ineligible filter or nonzero ROI.
    EXPECT_EQ(hqresize::ClassifyDirectTensorPath(control), Path::Other);
}

TEST(OpHQResizePolicy, UsesSeparableForRegressedBlackwellTensorPaths)
{
    using Path = hqresize::DirectTensorPath;
    using Desc = hqresize::DirectTensorPathDesc;

    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(100, Path::LinearExpand2x, true, 3));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(100, Path::CubicContract2x, true, 3));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(100, Path::CubicContract2x, false, 3));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(100, Path::CubicExpand2x, true, 1));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(103, Path::LinearExpand2x, true, 3));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(103, Path::CubicContract2x, true, 3));
    EXPECT_FALSE(hqresize::UseDirectTensorPathForSM(103, Path::CubicContract2x, false, 3));

    const Desc linearExpand{true, false, true, 0.f, 0.f, 0.5f, 0.5f, 1920, 1080, 3840, 2160};
    EXPECT_FALSE(hqresize::ShouldUseDirectTensorPathForSM(100, linearExpand, true, 3));
    EXPECT_FALSE(hqresize::ShouldUseDirectTensorPathForSM(103, linearExpand, true, 3));
}

TEST(OpHQResizePolicy, KeepsDirectForUnaffectedTensorPaths)
{
    using Path = hqresize::DirectTensorPath;

    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(90, Path::LinearExpand2x, true, 3));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(103, Path::CubicExpand2x, true, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(103, Path::LinearExpand2x, true, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(103, Path::LinearExpand2x, false, 3));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(103, Path::CubicContract2x, true, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(100, Path::LinearExpand2x, true, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(100, Path::LinearExpand2x, false, 3));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(100, Path::CubicContract2x, true, 1));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(100, Path::CubicExpand2x, true, 3));
    EXPECT_TRUE(hqresize::UseDirectTensorPathForSM(100, Path::CubicExpand2x, false, 1));
}
