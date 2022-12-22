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

#include "DeviceBorderWrap.hpp" // to test in the device

#include <common/BorderUtils.hpp>    // for test::ReplicateBorderIndex, etc.
#include <common/Printers.hpp>       // for stream operator, etc.
#include <common/TypedTests.hpp>     // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/Tensor.hpp>           // for Tensor, etc.
#include <nvcv/TensorDataAccess.hpp> // for TensorDataAccessStridedImagePlanar, etc.
#include <nvcv/cuda/BorderWrap.hpp>  // the object of this test
#include <nvcv/cuda/MathOps.hpp>     // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/TensorWrap.hpp>  // for Tensor3DWrap, etc.

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

// ---------------------------- Testing IsOutside ------------------------------

#define NVCV_TEST_ROW(SIZE, INDEX, GOLD) ttype::Types<ttype::Value<SIZE>, ttype::Value<INDEX>, ttype::Value<GOLD>>

NVCV_TYPED_TEST_SUITE(IsOutsideTests,
                      ttype::Types<NVCV_TEST_ROW(int{5}, int{3}, false), NVCV_TEST_ROW(int{5}, int{6}, true),
                                   NVCV_TEST_ROW(int{10000000}, int{-1}, true)>);

#undef NVCV_TEST_ROW

TYPED_TEST(IsOutsideTests, correct_boolean)
{
    constexpr auto size  = ttype::GetValue<TypeParam, 0>;
    constexpr auto coord = ttype::GetValue<TypeParam, 1>;
    constexpr auto gold  = ttype::GetValue<TypeParam, 2>;

    EXPECT_EQ(cuda::IsOutside(coord, size), gold);
}

// ----------------------- Testing GetIndexWithBorder --------------------------

#define NVCV_TEST_ROW(BORDERTYPE, BASE, INPUTSIZE, GOLDSIZE, ...)                       \
    ttype::Types<ttype::Value<BORDERTYPE>, ttype::Value<BASE>, ttype::Value<INPUTSIZE>, \
                 ttype::Value<std::array<int, GOLDSIZE>{__VA_ARGS__}>>

NVCV_TYPED_TEST_SUITE(
    GetIndexWithBorderTests,
    ttype::Types<NVCV_TEST_ROW(NVCV_BORDER_REPLICATE, -3, 3, 9, {0, 0, 0, 0, 1, 2, 2, 2, 2}),
                 NVCV_TEST_ROW(NVCV_BORDER_REPLICATE, +10000000, 5, 3, {4, 4, 4}),
                 NVCV_TEST_ROW(NVCV_BORDER_REPLICATE, -10000000, 5, 3, {0, 0, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_WRAP, -3, 3, 9, {0, 1, 2, 0, 1, 2, 0, 1, 2}),
                 NVCV_TEST_ROW(NVCV_BORDER_WRAP, +10000000, 5, 10, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}),
                 NVCV_TEST_ROW(NVCV_BORDER_WRAP, -10000000, 5, 10, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT, -3, 3, 9, {2, 1, 0, 0, 1, 2, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT, +10000000, 5, 10, {0, 1, 2, 3, 4, 4, 3, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT, -10000000, 5, 10, {0, 1, 2, 3, 4, 4, 3, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, -3, 3, 9, {1, 2, 1, 0, 1, 2, 1, 0, 1}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, +10000000, 5, 9, {0, 1, 2, 3, 4, 3, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, -10000000, 5, 9, {0, 1, 2, 3, 4, 3, 2, 1, 0})>);

#undef NVCV_TEST_ROW

TYPED_TEST(GetIndexWithBorderTests, correct_index)
{
    constexpr auto BorderType = ttype::GetValue<TypeParam, 0>;
    constexpr int  kBase      = ttype::GetValue<TypeParam, 1>;
    constexpr int  kInputSize = ttype::GetValue<TypeParam, 2>;
    constexpr auto gold       = ttype::GetValue<TypeParam, 3>;

    std::array<int, gold.size()> test;

    std::iota(test.begin(), test.end(), kBase);

    std::transform(test.cbegin(), test.cend(), test.begin(),
                   [](const int &coord) { return cuda::GetIndexWithBorder<BorderType>(coord, kInputSize); });

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing BorderWrap ------------------------------

// Shortcuts to easy write each test case

constexpr auto U8      = NVCV_IMAGE_FORMAT_U8;
constexpr auto S16     = NVCV_IMAGE_FORMAT_S16;
constexpr auto _2S16   = NVCV_IMAGE_FORMAT_2S16;
constexpr auto F32     = NVCV_IMAGE_FORMAT_F32;
constexpr auto RGB8    = NVCV_IMAGE_FORMAT_RGB8;
constexpr auto RGBA8   = NVCV_IMAGE_FORMAT_RGBA8;
constexpr auto RGBf32  = NVCV_IMAGE_FORMAT_RGBf32;
constexpr auto RGBAf32 = NVCV_IMAGE_FORMAT_RGBAf32;

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, BORDERSIZE, FORMAT, VALUETYPE, BORDERTYPE)                     \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<BORDERSIZE>, \
                 ttype::Value<FORMAT>, VALUETYPE, ttype::Value<BORDERTYPE>>

NVCV_TYPED_TEST_SUITE(BorderWrapNHWTest,
                      ttype::Types<NVCV_TEST_ROW(22, 33, 1, 0, RGBA8, uchar4, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(33, 22, 3, 3, _2S16, short2, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(11, 44, 5, 55, RGBAf32, float4, NVCV_BORDER_REPLICATE),
                                   NVCV_TEST_ROW(122, 6, 9, 7, RGBf32, float3, NVCV_BORDER_WRAP),
                                   NVCV_TEST_ROW(66, 163, 6, 9, RGB8, uchar3, NVCV_BORDER_REFLECT),
                                   NVCV_TEST_ROW(199, 99, 4, 19, S16, short1, NVCV_BORDER_REFLECT101)>);

TYPED_TEST(BorderWrapNHWTest, correct_fill)
{
    int width      = ttype::GetValue<TypeParam, 0>;
    int height     = ttype::GetValue<TypeParam, 1>;
    int batches    = ttype::GetValue<TypeParam, 2>;
    int borderSize = ttype::GetValue<TypeParam, 3>;

    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 4>};

    using ValueType            = ttype::GetType<TypeParam, 5>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 6>;
    using DimType              = int3;

    ValueType borderValue = cuda::SetAll<ValueType>(123);

    nvcv::Tensor srcTensor(batches, {width, height}, format);
    nvcv::Tensor dstTensor(batches, {width + borderSize * 2, height + borderSize * 2}, format);

    const auto *srcDev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(srcTensor.exportData());
    const auto *dstDev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(dstTensor.exportData());

    ASSERT_NE(srcDev, nullptr);
    ASSERT_NE(dstDev, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    DimType srcSize, dstSize;

    srcSize.x = srcAccess->numCols();
    dstSize.x = dstAccess->numCols();

    srcSize.y = srcAccess->numRows();
    dstSize.y = dstAccess->numRows();

    srcSize.z = srcAccess->numSamples();
    dstSize.z = dstAccess->numSamples();

    int srcSizeBytes = srcDev->stride(0) * srcSize.z;
    int dstSizeBytes = dstDev->stride(0) * dstSize.z;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    cuda::BorderWrapNHW<const ValueType, kBorderType> srcWrap(*srcDev, borderValue);
    cuda::Tensor3DWrap<ValueType>                     dstWrap(*dstDev);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunFillBorder(dstWrap, srcWrap, dstSize, srcSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    // Get test fill border
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    // Run gold fill border
    for (int z = 0; z < dstSize.z; ++z)
    {
        int2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {width, height}, kBorderType);

                *reinterpret_cast<ValueType *>(
                    &gold[z * dstDev->stride(0) + y * dstDev->stride(1) + x * dstDev->stride(2)])
                    = isInside
                        ? *reinterpret_cast<ValueType *>(&srcVec[z * srcDev->stride(0) + srcCoord.y * srcDev->stride(1)
                                                                 + srcCoord.x * srcDev->stride(2)])
                        : borderValue;
            }
        }
    }

    EXPECT_EQ(test, gold);
}

NVCV_TYPED_TEST_SUITE(BorderWrapNHWCTest,
                      ttype::Types<NVCV_TEST_ROW(33, 22, 3, 3, F32, float1, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(23, 13, 11, 33, _2S16, short1, NVCV_BORDER_REPLICATE),
                                   NVCV_TEST_ROW(77, 15, 8, 16, RGB8, uchar1, NVCV_BORDER_WRAP),
                                   NVCV_TEST_ROW(144, 33, 8, 99, U8, uchar1, NVCV_BORDER_REFLECT),
                                   NVCV_TEST_ROW(199, 99, 6, 200, RGBA8, uchar1, NVCV_BORDER_REFLECT101)>);

#undef NVCV_TEST_ROW

TYPED_TEST(BorderWrapNHWCTest, correct_fill)
{
    int width      = ttype::GetValue<TypeParam, 0>;
    int height     = ttype::GetValue<TypeParam, 1>;
    int batches    = ttype::GetValue<TypeParam, 2>;
    int borderSize = ttype::GetValue<TypeParam, 3>;

    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 4>};

    using ValueType            = ttype::GetType<TypeParam, 5>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 6>;
    using DimType              = int4;

    ValueType borderValue = cuda::SetAll<ValueType>(123);

    nvcv::Tensor srcTensor(batches, {width, height}, format);
    nvcv::Tensor dstTensor(batches, {width + borderSize * 2, height + borderSize * 2}, format);

    const auto *srcDev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(srcTensor.exportData());
    const auto *dstDev = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(dstTensor.exportData());

    ASSERT_NE(srcDev, nullptr);
    ASSERT_NE(dstDev, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    DimType srcSize, dstSize;

    srcSize.x = srcAccess->numCols();
    dstSize.x = dstAccess->numCols();

    srcSize.y = srcAccess->numRows();
    dstSize.y = dstAccess->numRows();

    srcSize.z = srcAccess->numSamples();
    dstSize.z = dstAccess->numSamples();

    srcSize.w = srcAccess->numChannels();
    dstSize.w = dstAccess->numChannels();

    int srcSizeBytes = srcDev->stride(0) * srcSize.z;
    int dstSizeBytes = dstDev->stride(0) * dstSize.z;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    cuda::BorderWrapNHWC<const ValueType, kBorderType> srcWrap(*srcDev, borderValue);
    cuda::Tensor4DWrap<ValueType>                      dstWrap(*dstDev);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunFillBorder(dstWrap, srcWrap, dstSize, srcSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    // Get test fill border
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    // Run gold fill border
    for (int z = 0; z < dstSize.z; ++z)
    {
        int2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {width, height}, kBorderType);

                for (int w = 0; w < dstSize.w; ++w)
                {
                    *reinterpret_cast<ValueType *>(&gold[z * dstDev->stride(0) + y * dstDev->stride(1)
                                                         + x * dstDev->stride(2) + w * dstDev->stride(3)])
                        = isInside ? *reinterpret_cast<ValueType *>(
                              &srcVec[z * srcDev->stride(0) + srcCoord.y * srcDev->stride(1)
                                      + srcCoord.x * srcDev->stride(2) + w * srcDev->stride(3)])
                                   : borderValue;
                }
            }
        }
    }

    EXPECT_EQ(test, gold);
}
