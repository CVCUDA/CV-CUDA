/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cvcuda/priv/PillowResizePolicy.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

void CheckPillowTileSpan(int inputHeight, int outputHeight, float radius, bool everyTile)
{
    const float scale   = static_cast<float>(inputHeight) / static_cast<float>(outputHeight);
    const float support = radius * std::max(scale, 1.0f);
    if (const int kernelSize = 2 * static_cast<int>(std::ceil(support)) + 1;
        !cvcuda::priv::PillowU8FusedTileFits(inputHeight, outputHeight, kernelSize))
        return;

    auto check = [&](int firstOutput)
    {
        const int   lastOutput  = std::min(firstOutput + cvcuda::priv::kPillowU8TileHeight, outputHeight) - 1;
        const float firstCenter = (static_cast<float>(firstOutput) + 0.5f) * scale;
        const float lastCenter  = (static_cast<float>(lastOutput) + 0.5f) * scale;
        const int   firstInput  = std::max(static_cast<int>(firstCenter - support + 0.5f), 0);
        const int   lastInput   = std::min(static_cast<int>(lastCenter + support + 0.5f), inputHeight);
        EXPECT_LE(lastInput - firstInput, cvcuda::priv::kPillowU8TileMaxRows)
            << inputHeight << " -> " << outputHeight << " at " << firstOutput << " radius=" << radius;
    };
    if (everyTile)
    {
        for (int row = 0; row < outputHeight; row += cvcuda::priv::kPillowU8TileHeight) check(row);
    }
    else
    {
        check(0);
        check((outputHeight / 2 / cvcuda::priv::kPillowU8TileHeight) * cvcuda::priv::kPillowU8TileHeight);
        check(((outputHeight - 1) / cvcuda::priv::kPillowU8TileHeight) * cvcuda::priv::kPillowU8TileHeight);
    }
}

} // namespace

TEST(OpPillowResizePolicy, SharedTileFitsOrdinaryAndNonIntegerScales)
{
    EXPECT_TRUE(cvcuda::priv::PillowU8FusedTileFits(1080, 540, 9));
    EXPECT_TRUE(cvcuda::priv::PillowU8FusedTileFits(1080, 480, 11));
    EXPECT_TRUE(cvcuda::priv::PillowU8FusedTileFits(17, 33, 5));
    EXPECT_TRUE(cvcuda::priv::PillowU8FusedTileFits(129, 65, 9));
}

TEST(OpPillowResizePolicy, SharedTileRejectsOversizedAndInvalidSpans)
{
    EXPECT_FALSE(cvcuda::priv::PillowU8FusedTileFits(257, 17, 65));
    EXPECT_FALSE(cvcuda::priv::PillowU8FusedTileFits(0, 32, 5));
    EXPECT_FALSE(cvcuda::priv::PillowU8FusedTileFits(32, 0, 5));
    EXPECT_FALSE(cvcuda::priv::PillowU8FusedTileFits(32, 32, 0));
    EXPECT_FALSE(cvcuda::priv::PillowU8FusedTileFits(std::numeric_limits<int>::max(), 1, 5));
    EXPECT_FALSE(
        cvcuda::priv::PillowU8FusedTileFits(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 5));
}

TEST(OpPillowResizePolicy, SharedTileBoundsCoverScalarWindowEnumeration)
{
    for (int inputHeight = 1; inputHeight <= 512; ++inputHeight)
        for (int outputHeight = 1; outputHeight <= 129; ++outputHeight)
            for (float radius : {0.5f, 1.0f, 2.0f, 3.0f}) CheckPillowTileSpan(inputHeight, outputHeight, radius, true);
}

TEST(OpPillowResizePolicy, SharedTileBoundsIncludeLargeCoordinateRounding)
{
    for (int inputHeight : {31, 257, 1080, 65537, 16777217})
        for (int outputHeight : {31, 65, 480, 540, 1024, 8388609})
            for (float radius : {0.5f, 1.0f, 2.0f, 3.0f}) CheckPillowTileSpan(inputHeight, outputHeight, radius, false);
}

TEST(OpPillowResizePolicy, PlaneStorageRangesRejectInvalidAndOverflowingExtents)
{
    cvcuda::priv::PillowStorageRange range;
    EXPECT_TRUE(cvcuda::priv::PillowPlaneStorageRange(1024, 3, 32, 21, range));
    EXPECT_EQ(range.begin, 1024u);
    EXPECT_EQ(range.end, 1109u);
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(0, 3, 32, 21, range));
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(1024, 0, 32, 21, range));
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(1024, 3, -32, 21, range));
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(1024, 3, 20, 21, range));
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(1024, 3, 32, 0, range));
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(std::numeric_limits<uintptr_t>::max() - 5, 1, 8, 8, range));
    EXPECT_FALSE(cvcuda::priv::PillowPlaneStorageRange(1024, std::numeric_limits<int64_t>::max(),
                                                       std::numeric_limits<int64_t>::max(), 3, range));
    EXPECT_EQ(range.begin, 0u);
    EXPECT_EQ(range.end, 0u);
}

TEST(OpPillowResizePolicy, AllPlaneGuardAcceptsDisjointTouchingAndInterleavedRanges)
{
    const std::array<cvcuda::priv::PillowStorageRange, 3> source{
        {{100, 120}, {200, 220}, {300, 320}}
    };
    for (auto output : {
             cvcuda::priv::PillowStorageRange{ 80, 100},
             {120, 140},
             {220, 300},
             {320, 340}
    })
        EXPECT_TRUE(cvcuda::priv::PillowStorageRangeDisjointFromAll(output, source.data(), source.size()));
}

TEST(OpPillowResizePolicy, AllPlaneGuardRejectsExactShiftedCrossSampleAndCrossPlaneAliases)
{
    const std::array<cvcuda::priv::PillowStorageRange, 3> source{
        {{100, 120}, {200, 220}, {300, 320}}
    };
    for (auto output : {
             cvcuda::priv::PillowStorageRange{100, 120},
             {119, 139},
             { 90, 101},
             {200, 220},
             {310, 330}
    })
        EXPECT_FALSE(cvcuda::priv::PillowStorageRangeDisjointFromAll(output, source.data(), source.size()));
    const std::array<cvcuda::priv::PillowStorageRange, 2> swapped{
        {source[1], source[0]}
    };
    for (size_t sample = 0; sample < swapped.size(); ++sample)
    {
        EXPECT_TRUE(cvcuda::priv::PillowStorageRangesDisjoint(source[sample], swapped[sample]));
        EXPECT_FALSE(cvcuda::priv::PillowStorageRangeDisjointFromAll(swapped[sample], source.data(), source.size()));
    }
    EXPECT_FALSE(cvcuda::priv::PillowStorageRangeDisjointFromAll({0, 0}, source.data(), source.size()));
    EXPECT_FALSE(cvcuda::priv::PillowStorageRangeDisjointFromAll({500, 520}, nullptr, source.size()));
    EXPECT_FALSE(cvcuda::priv::PillowStorageRangeDisjointFromAll({500, 520}, source.data(), 0));
}

TEST(OpPillowResizePolicy, GuardedSwapPreservesEveryLegalTwoCtaSchedule)
{
    const std::array<cvcuda::priv::PillowStorageRange, 2> source{
        {{100, 103}, {200, 203}}
    };
    const std::array<cvcuda::priv::PillowStorageRange, 2> swapped{
        {source[1], source[0]}
    };
    const bool disjoint = cvcuda::priv::PillowStorageRangeDisjointFromAll(swapped[0], source.data(), source.size())
                       && cvcuda::priv::PillowStorageRangeDisjointFromAll(swapped[1], source.data(), source.size());
    ASSERT_FALSE(disjoint);
    std::array<int, 4> ordering{0, 1, 2, 3};
    int                schedules          = 0;
    int                corruptedSchedules = 0;
    do
    {
        if (std::ranges::find(ordering, 0) > std::ranges::find(ordering, 1)
            || std::ranges::find(ordering, 2) > std::ranges::find(ordering, 3))
            continue;
        ++schedules;
        const std::array<int, 2> original{17, 231};
        std::array<int, 2>       unguarded = original;
        std::array<int, 2>       loaded    = original;
        for (int action : ordering)
        {
            if (action % 2 == 0)
                loaded[action / 2] = unguarded[action / 2];
            else
                unguarded[1 - action / 2] = loaded[action / 2];
        }
        if (unguarded != (std::array<int, 2>{231, 17}))
            ++corruptedSchedules;
        const std::array<int, 2> snapshot = original;
        const std::array<int, 2> guarded{snapshot[1], snapshot[0]};
        EXPECT_EQ(guarded, (std::array<int, 2>{231, 17}));
    }
    while (std::ranges::next_permutation(ordering).found);
    EXPECT_EQ(schedules, 6);
    EXPECT_EQ(corruptedSchedules, 2);
}
