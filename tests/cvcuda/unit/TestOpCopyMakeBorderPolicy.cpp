/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Definitions.hpp"

#include <cvcuda/priv/CopyMakeBorderPolicy.hpp>

#include <limits>

namespace priv = cvcuda::priv;

TEST(OpCopyMakeBorderPolicy, UsesRGB8RowCarrierForStreamingReflections)
{
    constexpr int64_t pixelsPerSample = 1080 * 1920 + 1620 * 2880;
    constexpr int     cacheBytes      = 48 * 1024 * 1024;
    EXPECT_TRUE(priv::UseRGB8Reflect101TensorRowCarrier(pixelsPerSample, 64, cacheBytes));
}

TEST(OpCopyMakeBorderPolicy, KeepsCacheResidentRGB8OnCarrier)
{
    constexpr int64_t pixelsPerSample = 1080 * 1920 + 1620 * 2880;
    for (int cacheBytes : {48 * 1024 * 1024, 96 * 1024 * 1024})
    {
        const int residentSamples = cacheBytes / (3 * pixelsPerSample);
        EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(pixelsPerSample, residentSamples, cacheBytes));
        EXPECT_TRUE(priv::UseRGB8Reflect101TensorRowCarrier(pixelsPerSample, residentSamples + 1, cacheBytes));
        EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(16, 1, 48));
        EXPECT_TRUE(priv::UseRGB8Reflect101TensorRowCarrier(17, 1, 48));
        EXPECT_TRUE(priv::UseRGB8Reflect101TensorRowCarrier(std::numeric_limits<int64_t>::max(),
                                                            std::numeric_limits<int>::max(), cacheBytes));
    }
}

TEST(OpCopyMakeBorderPolicy, PreservesRGB8CarrierWithoutUsableCacheBudget)
{
    EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(1024 * 1024, 64, 0));
    EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(1024 * 1024, 64, -1));
    EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(1024 * 1024, 0, 48 * 1024 * 1024));
    EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(1024 * 1024, -1, 48 * 1024 * 1024));
}

TEST(OpCopyMakeBorderPolicy, LegacyCacheBudgetKeepsOnlyTheShippingException)
{
    EXPECT_FALSE(priv::UseLegacyPlanarReflect101VectorForSM(89));
    EXPECT_TRUE(priv::UseLegacyPlanarReflect101VectorForSM(120));
    for (int sm : {0, 75, 80, 86, 87, 90, 100, 110, 119, 121})
    {
        EXPECT_FALSE(priv::UseLegacyPlanarReflect101VectorForSM(sm)) << "sm=" << sm;
    }
}

TEST(OpCopyMakeBorderPolicy, KeepsCacheResidentAdaPlanesOnCarrier)
{
    constexpr int64_t planeElements = 1080 * 1920 + 1620 * 2880;
    for (int cacheBytes : {48 * 1024 * 1024, 96 * 1024 * 1024})
    {
        const int residentSamples = cacheBytes / (planeElements * static_cast<int64_t>(sizeof(float)));
        EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(planeElements, residentSamples, cacheBytes));
        EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(planeElements, residentSamples + 1, cacheBytes));
        EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(planeElements, 16, cacheBytes));
    }
}

TEST(OpCopyMakeBorderPolicy, KeepsCarrierThroughCacheCapacity)
{
    constexpr int cacheBytes = 128 * 1024 * 1024;
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, 31, cacheBytes));
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, 32, cacheBytes));
    EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, 33, cacheBytes));
    EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024 + 1, 32, cacheBytes));
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1, 3, 15));
    EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(2, 3, 15));
}

TEST(OpCopyMakeBorderPolicy, PreservesCarrierWithoutUsableCacheBudget)
{
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, 64, 0));
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, 64, -1));
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, 0, 128 * 1024 * 1024));
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(1024 * 1024, -1, 128 * 1024 * 1024));
}

TEST(OpCopyMakeBorderPolicy, UsesPerChannelFootprintWithoutOverflow)
{
    constexpr int64_t planeElements = 1080 * 1920 + 1620 * 2880;
    constexpr int     cacheBytes    = 128 * 1024 * 1024;
    EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(planeElements, 4, cacheBytes));
    EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(planeElements, 5, cacheBytes));
    EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(planeElements, 16, cacheBytes));
    EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(std::numeric_limits<int64_t>::max(),
                                                      std::numeric_limits<int>::max(), cacheBytes));
}

TEST(OpCopyMakeBorderPolicy, RowCarrierAcceptsAlignedSingleFoldHorizontalSpans)
{
    EXPECT_TRUE(priv::CanUseReflect101RowCarrier(3, 1920, 2880, 960));
    EXPECT_TRUE(priv::CanUseReflect101RowCarrier(4, 1920, 2880, 960));
    EXPECT_TRUE(priv::CanUseReflect101RowCarrier(3, 1920, 2016, 32));
    EXPECT_TRUE(priv::CanUseReflect101RowCarrier(4, 1920, 2016, 0));
}

TEST(OpCopyMakeBorderPolicy, RowCarrierDeclinesMisalignmentRepeatedFoldsAndOtherWidths)
{
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(3, 1920, 1920, 0));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(3, 1919, 2880, 960));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(3, 1920, 2880, 959));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(3, 1920, 2879, 960));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(3, 1920, 3840, 1920));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(4, 1920, 3840, 0));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(4, 1920, 1920, 4));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(4, 1920, 2880, -4));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(4, 0, 16, 4));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(4, 1, 4, 0));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(1, 1920, 2880, 960));
    EXPECT_FALSE(priv::CanUseReflect101RowCarrier(12, 1920, 2880, 960));
}

TEST(OpCopyMakeBorderPolicy, UsesExactResourceBoundaryAcrossBudgetsAndBatchSizes)
{
    for (int cacheBytes : {1, 3, 15, 4096, 15361, 1048576, std::numeric_limits<int>::max()})
    {
        for (int samples : {1, 2, 3, 16, 257, std::numeric_limits<int>::max()})
        {
            const int64_t rgb8Capacity = cacheBytes / (int64_t{3} * samples);
            const int64_t f32Capacity  = cacheBytes / (static_cast<int64_t>(sizeof(float)) * samples);
            EXPECT_FALSE(priv::UseRGB8Reflect101TensorRowCarrier(rgb8Capacity, samples, cacheBytes));
            EXPECT_TRUE(priv::UseRGB8Reflect101TensorRowCarrier(rgb8Capacity + 1, samples, cacheBytes));
            EXPECT_FALSE(priv::UsePlanarRGBF32Reflect101Vector(f32Capacity, samples, cacheBytes));
            EXPECT_TRUE(priv::UsePlanarRGBF32Reflect101Vector(f32Capacity + 1, samples, cacheBytes));
        }
    }
}
