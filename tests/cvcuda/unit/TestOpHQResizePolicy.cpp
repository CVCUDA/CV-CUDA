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

#include <cvcuda/priv/OpHQResizeConfig.hpp>
#include <cvcuda/priv/OpHQResizePolicy.hpp>

#include <array>
#include <cmath>
#include <string>

namespace hqresize = cvcuda::priv::hq_resize;

TEST(OpHQResizePolicy, PreservesF32DirectLinearScalarDomain)
{
    struct Case
    {
        int  width;
        int  height;
        int  samples;
        bool vectorized;
    };

    constexpr int        maximum = std::numeric_limits<int>::max();
    constexpr std::array cases{
        Case{     64,    4096,      64, false},
        Case{     96,    4096,      64, false},
        Case{    127,    4096,      64, false},
        Case{    128,    4088,       1, false},
        Case{    128,    4089,       1,  true},
        Case{    128,       8,     511, false},
        Case{    128,       8,     512,  true},
        Case{    129,    2040,       1, false},
        Case{    129,    2041,       1,  true},
        Case{    960,     540,     192,  true},
        Case{maximum,       1,       1,  true},
        Case{maximum, maximum, maximum,  true},
        Case{      0,      64,      64, false},
        Case{    128,       0,     512, false},
        Case{    128,      64,       0, false},
    };
    for (const auto &test : cases)
    {
        SCOPED_TRACE(::testing::Message() << test.width << "x" << test.height << " samples=" << test.samples);
        EXPECT_EQ(hqresize::CanVectorizeF32DirectLinear2D(test.width, test.height, test.samples), test.vectorized);
    }
}

TEST(OpHQResizePolicy, RestrictsF32DirectLinearTwoLanesToExactContraction)
{
    struct Case
    {
        int inWidth;
        int inHeight;
        int outWidth;
        int outHeight;
        int samples;
        int lanes;
    };

    constexpr int        maximum = std::numeric_limits<int>::max();
    constexpr std::array cases{
        Case{         1920,          1080,           960,           540,     192, 2},
        Case{         1280,           720,           640,           360,     512, 2},
        Case{          258,           898,           129,           449,       9, 2},
        Case{          254,           898,           127,           449,       9, 1},
        Case{          256,           896,           128,           448,       9, 1},
        Case{          256,           898,           128,           449,       9, 2},
        Case{          864,           480,          3840,          2160,      48, 4},
        Case{         1920,          1080,           864,           480,     120, 4},
        Case{         1920,          1080,           961,           540,      64, 4},
        Case{         1920,          1080,           960,           541,      64, 4},
        Case{         1920,          1080,           960,          2160,      64, 4},
        Case{         1920,          1080,          1920,          1080,      64, 4},
        Case{          256,            16,           128,             8,     511, 1},
        Case{          256,            16,           128,             8,     512, 2},
        Case{      1 << 23,            16,       1 << 22,             8,     512, 2},
        Case{(1 << 23) + 2,            16, (1 << 22) + 1,             8,     512, 4},
        Case{          256,       1 << 23,           128,       1 << 22,     512, 2},
        Case{          256, (1 << 23) + 2,           128, (1 << 22) + 1,     512, 4},
        Case{      maximum,       maximum,       maximum,       maximum, maximum, 4},
    };
    for (const auto &test : cases)
    {
        SCOPED_TRACE(::testing::Message() << test.inWidth << "x" << test.inHeight << " -> " << test.outWidth << "x"
                                          << test.outHeight << " samples=" << test.samples);
        EXPECT_EQ(hqresize::GetF32DirectLinear2DLanes(test.inWidth, test.inHeight, test.outWidth, test.outHeight,
                                                      test.samples),
                  test.lanes);
    }
}

TEST(OpHQResizePolicy, ParsesBlockLaneOverride)
{
    const auto defaultConfig = hqresize::ParseResizeBlockLanesConfig(nullptr);
    EXPECT_EQ(defaultConfig.lanes, 8);
    EXPECT_FALSE(defaultConfig.overridden);
    EXPECT_TRUE(hqresize::AllowsDtypeSpecificTensorBlockLaneTuning(defaultConfig));

    const auto explicitDefault = hqresize::ParseResizeBlockLanesConfig("8");
    EXPECT_EQ(explicitDefault.lanes, 8);
    EXPECT_TRUE(explicitDefault.overridden);
    EXPECT_TRUE(hqresize::AllowsDtypeSpecificTensorBlockLaneTuning(explicitDefault));

    const auto wideConfig = hqresize::ParseResizeBlockLanesConfig("16");
    EXPECT_EQ(wideConfig.lanes, 16);
    EXPECT_TRUE(wideConfig.overridden);
    EXPECT_FALSE(hqresize::AllowsDtypeSpecificTensorBlockLaneTuning(wideConfig));

    EXPECT_THROW(hqresize::ParseResizeBlockLanesConfig("0"), nvcv::Exception);
    EXPECT_THROW(hqresize::ParseResizeBlockLanesConfig(""), nvcv::Exception);
    EXPECT_THROW(hqresize::ParseResizeBlockLanesConfig("bad"), nvcv::Exception);
    EXPECT_EQ(hqresize::ParseResizeBlockLanesConfig("+8").lanes, 8);
    EXPECT_THROW(hqresize::ParseResizeBlockLanesConfig("-1"), nvcv::Exception);
    EXPECT_EQ(hqresize::ParseResizeBlockLanesConfig(" 8").lanes, 8);
    EXPECT_EQ(hqresize::ParseResizeBlockLanesConfig("8 ").lanes, 8);
    EXPECT_EQ(hqresize::ParseResizeBlockLanesConfig("\t8\r\n").lanes, 8);
    EXPECT_EQ(hqresize::ParseResizeBlockLanesConfig("8x").lanes, 8);
    EXPECT_THROW(hqresize::ParseResizeBlockLanesConfig("99999999999"), nvcv::Exception);
    const std::string maximum = std::to_string(hqresize::kMaximumResizeBlockLanes);
    EXPECT_EQ(hqresize::ParseResizeBlockLanesConfig(maximum.c_str()).lanes, hqresize::kMaximumResizeBlockLanes);
    const std::string aboveMaximum = std::to_string(hqresize::kMaximumResizeBlockLanes + 1);
    EXPECT_THROW(hqresize::ParseResizeBlockLanesConfig(aboveMaximum.c_str()), nvcv::Exception);
}

TEST(OpHQResizePolicy, ClassifiesAdaptiveLayoutEligibility)
{
    constexpr int cc              = hqresize::kValidatedComputeCapability;
    constexpr int sm              = hqresize::kValidatedMultiprocessorCount;
    const auto    explicitDefault = hqresize::ParseResizeBlockLanesConfig("8");
    EXPECT_FALSE(hqresize::CanUseAdaptiveHorizontalLayout(true, explicitDefault.overridden, true, cc, sm));
    EXPECT_TRUE(hqresize::CanUseAdaptiveHorizontalLayout(true, false, true, cc, sm));
    EXPECT_FALSE(hqresize::CanUseAdaptiveHorizontalLayout(false, false, true, cc, sm));
    EXPECT_FALSE(hqresize::CanUseAdaptiveHorizontalLayout(true, true, true, cc, sm));
    EXPECT_FALSE(hqresize::CanUseAdaptiveHorizontalLayout(true, false, false, cc, sm));
    EXPECT_FALSE(hqresize::CanUseAdaptiveHorizontalLayout(true, false, true, cc - 1, sm));
    EXPECT_FALSE(hqresize::CanUseAdaptiveHorizontalLayout(true, false, true, cc, sm - 1));
}

TEST(OpHQResizePolicy, ClassifiesAdaptiveGeometry)
{
    EXPECT_TRUE(hqresize::IsValidatedAdaptiveHorizontalDevice(hqresize::kValidatedComputeCapability,
                                                              hqresize::kValidatedMultiprocessorCount));
    EXPECT_FALSE(hqresize::IsValidatedAdaptiveHorizontalDevice(hqresize::kValidatedComputeCapability - 1,
                                                               hqresize::kValidatedMultiprocessorCount));
    EXPECT_FALSE(hqresize::IsValidatedAdaptiveHorizontalDevice(hqresize::kValidatedComputeCapability,
                                                               hqresize::kValidatedMultiprocessorCount - 1));

    struct Case
    {
        int   width;
        int   height;
        int   batch;
        float scale;
        int   expectedLanes;
    };

    constexpr std::array cases{
        Case{224, 224, 16, 4.824121f,  8},
        Case{224, 224, 32, 4.824121f, 16},
        Case{224, 224, 64, 4.824121f, 32},
        Case{352, 224, 64, 4.824121f, 32},
        Case{353, 224, 64, 4.824121f,  8},
        Case{256, 256, 64, 4.824121f, 32},
        Case{256, 257, 64, 4.824121f,  8},
        Case{256, 256, 64,     3.75f,  8},
        Case{256, 256, 64,   3.7501f, 32},
        Case{224, 224, 26, 4.824121f,  8},
        Case{224, 224, 27, 4.824121f, 16},
        Case{224, 224, 53, 4.824121f, 16},
        Case{224, 224, 54, 4.824121f, 32},
        Case{  0, 224, 64, 4.824121f,  8},
        Case{224,   0, 64, 4.824121f,  8},
        Case{ -1, 224, 64, 4.824121f,  8},
        Case{224,  -1, 64, 4.824121f,  8},
        Case{224,  64, 64, 4.824121f,  8},
        Case{224, 224,  0, 4.824121f,  8},
        Case{224, 224, 64,       0.f,  8},
    };
    constexpr int multiprocessorCount = hqresize::kValidatedMultiprocessorCount;
    for (const auto &test : cases)
    {
        const hqresize::AdaptiveHorizontalWork work{test.width, test.height, 4, test.scale};
        EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes(work, test.batch, multiprocessorCount), test.expectedLanes);
    }

    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes({224, 224, 4, 4.824121f}, 64, 0),
              hqresize::kDefaultResizeBlockLanes);
    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes({224, 224, 4, 4.824121f}, 64, -1),
              hqresize::kDefaultResizeBlockLanes);
    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes({224, 224, 0, 4.824121f}, 64, multiprocessorCount),
              hqresize::kDefaultResizeBlockLanes);
    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes(
                  {224, 224, hqresize::kMaximumValidatedAdaptiveHorizontalFilterSupport + 1, 4.824121f}, 64,
                  multiprocessorCount),
              hqresize::kDefaultResizeBlockLanes);
    EXPECT_EQ(
        hqresize::GetAdaptiveHorizontalBlockLanes(
            {224, 224, hqresize::kMaximumValidatedAdaptiveHorizontalFilterSupport, 4.824121f}, 64, multiprocessorCount),
        32);
    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes({224, 224, 4, 1.0f}, 64, multiprocessorCount),
              hqresize::kDefaultResizeBlockLanes);
}

TEST(OpHQResizePolicy, MapsAdaptiveHorizontalSampleFields)
{
    struct Filter
    {
        int numCoeffs;
        int filterSupport;

        constexpr int support() const
        {
            return filterSupport;
        }
    };

    struct Pair
    {
        int x;
        int y;
    };

    struct Sample
    {
        Pair                  processingOrder;
        std::array<int, 2>    filterKind;
        std::array<Filter, 2> filter;
        std::array<Pair, 3>   shapes;

        struct
        {
            float x;
            float y;
        } scale;
    };

    Sample sample{};
    sample.processingOrder = {1, 0};
    sample.filterKind[1]   = 7;
    sample.filter[1]       = {hqresize::kCubicFilterCoefficients, 16};
    sample.shapes[2]       = {352, 224};
    sample.scale.y         = -4.824121f;
    const auto work        = hqresize::DescribeAdaptiveHorizontalSample(sample, [](int kind) { return kind == 7; });
    ASSERT_TRUE(work);
    EXPECT_EQ(work->outWidth, 352);
    EXPECT_EQ(work->outHeight, 224);
    EXPECT_EQ(work->filterSupport, 16);
    EXPECT_FLOAT_EQ(work->horizontalScale, 4.824121f);

    auto rejected              = sample;
    rejected.processingOrder.x = 0;
    EXPECT_FALSE(hqresize::DescribeAdaptiveHorizontalSample(rejected, [](int kind) { return kind == 7; }));
    rejected               = sample;
    rejected.filterKind[1] = 8;
    EXPECT_FALSE(hqresize::DescribeAdaptiveHorizontalSample(rejected, [](int kind) { return kind == 7; }));
    rejected                     = sample;
    rejected.filter[1].numCoeffs = 128;
    EXPECT_FALSE(hqresize::DescribeAdaptiveHorizontalSample(rejected, [](int kind) { return kind == 7; }));
    rejected         = sample;
    rejected.scale.y = 1.0f;
    EXPECT_FALSE(hqresize::DescribeAdaptiveHorizontalSample(rejected, [](int kind) { return kind == 7; }));
    rejected.scale.y = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(hqresize::DescribeAdaptiveHorizontalSample(rejected, [](int kind) { return kind == 7; }));
    rejected.scale.y = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(hqresize::DescribeAdaptiveHorizontalSample(rejected, [](int kind) { return kind == 7; }));
}

TEST(OpHQResizePolicy, AccumulatesHomogeneousAdaptiveWork)
{
    const hqresize::AdaptiveHorizontalWork work{224, 224, 4, 4.824121f};
    hqresize::AdaptiveHorizontalBatchWork  batch;
    hqresize::AddAdaptiveHorizontalWork(batch, work);
    auto nearlyEqualScale            = work;
    nearlyEqualScale.horizontalScale = std::nextafter(work.horizontalScale, 5.0f);
    hqresize::AddAdaptiveHorizontalWork(batch, nearlyEqualScale);
    EXPECT_TRUE(batch.homogeneous);
    EXPECT_EQ(batch.numSamples, 2);
    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes(batch, 1), 32);

    hqresize::AdaptiveHorizontalBatchWork allIneligible;
    hqresize::AddAdaptiveHorizontalWork(allIneligible, std::nullopt);
    hqresize::AddAdaptiveHorizontalWork(allIneligible, std::nullopt);
    EXPECT_TRUE(allIneligible.homogeneous);
    EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes(allIneligible, 1), hqresize::kDefaultResizeBlockLanes);

    hqresize::AdaptiveHorizontalBatchWork mixedEligibility;
    hqresize::AddAdaptiveHorizontalWork(mixedEligibility, work);
    hqresize::AddAdaptiveHorizontalWork(mixedEligibility, std::nullopt);
    EXPECT_FALSE(mixedEligibility.homogeneous);

    for (int field = 0; field < 4; ++field)
    {
        auto different = work;
        if (field == 0)
            ++different.outWidth;
        else if (field == 1)
            ++different.outHeight;
        else if (field == 2)
            ++different.filterSupport;
        else
            different.horizontalScale += 0.25f;

        hqresize::AdaptiveHorizontalBatchWork mixed;
        hqresize::AddAdaptiveHorizontalWork(mixed, work);
        hqresize::AddAdaptiveHorizontalWork(mixed, different);
        EXPECT_FALSE(mixed.homogeneous);
        EXPECT_EQ(hqresize::GetAdaptiveHorizontalBlockLanes(mixed, 1), hqresize::kDefaultResizeBlockLanes);
    }
}

TEST(OpHQResizePolicy, AppliesAdaptiveLayoutOnlyToHorizontalPass)
{
    struct Sample
    {
        std::array<int, 2> lanes{8, 8};
    };

    std::array<Sample, 2>                 samples;
    hqresize::AdaptiveHorizontalBatchWork batch;
    hqresize::AddAdaptiveHorizontalWork(batch, hqresize::AdaptiveHorizontalWork{224, 224, 4, 4.824121f});
    hqresize::AddAdaptiveHorizontalWork(batch, hqresize::AdaptiveHorizontalWork{224, 224, 4, 4.824121f});
    hqresize::SetupAdaptiveHorizontalBlockLayout(
        samples.data(), batch, false, 1, [](Sample &sample, int pass, int lanes) { sample.lanes[pass] = lanes; });

    for (const auto &sample : samples)
    {
        EXPECT_EQ(sample.lanes[0], 8);
        EXPECT_EQ(sample.lanes[1], 32);
    }

    std::array<Sample, 2> wideStrideSamples;
    hqresize::SetupAdaptiveHorizontalBlockLayout(wideStrideSamples.data(), batch, true, 1,
                                                 [](Sample &sample, int pass, int lanes)
                                                 { sample.lanes[pass] = lanes; });
    for (const auto &sample : wideStrideSamples)
    {
        EXPECT_EQ(sample.lanes[0], 8);
        EXPECT_EQ(sample.lanes[1], 8);
    }

    std::array<Sample, 2>                 defaultSamples;
    hqresize::AdaptiveHorizontalBatchWork defaultBatch;
    hqresize::AddAdaptiveHorizontalWork(defaultBatch, hqresize::AdaptiveHorizontalWork{224, 64, 4, 4.824121f});
    hqresize::AddAdaptiveHorizontalWork(defaultBatch, hqresize::AdaptiveHorizontalWork{224, 64, 4, 4.824121f});
    hqresize::SetupAdaptiveHorizontalBlockLayout(defaultSamples.data(), defaultBatch, false, 1,
                                                 [](Sample &sample, int pass, int lanes)
                                                 { sample.lanes[pass] = lanes; });
    for (const auto &sample : defaultSamples)
    {
        EXPECT_EQ(sample.lanes[0], 8);
        EXPECT_EQ(sample.lanes[1], 8);
    }
}

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
