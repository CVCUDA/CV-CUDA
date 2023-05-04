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

#include "DeviceInterpolationWrap.hpp" // to test in the device
#include "DeviceTensorWrap.hpp"        // for PackedImage, etc.

#include <common/InterpUtils.hpp>          // for test::GoldInterp, etc.
#include <common/TypedTests.hpp>           // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/Tensor.hpp>                 // for Tensor, etc.
#include <nvcv/TensorDataAccess.hpp>       // for TensorDataAccessStridedImagePlanar, etc.
#include <nvcv/cuda/DropCast.hpp>          // for DropCast, etc.
#include <nvcv/cuda/InterpolationWrap.hpp> // the object of this test
#include <nvcv/cuda/MathOps.hpp>           // for operator == to allow EXPECT_EQ

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

constexpr auto U8      = NVCV_IMAGE_FORMAT_U8;
constexpr auto S16     = NVCV_IMAGE_FORMAT_S16;
constexpr auto _2S16   = NVCV_IMAGE_FORMAT_2S16;
constexpr auto F32     = NVCV_IMAGE_FORMAT_F32;
constexpr auto RGB8    = NVCV_IMAGE_FORMAT_RGB8;
constexpr auto RGBA8   = NVCV_IMAGE_FORMAT_RGBA8;
constexpr auto RGBf32  = NVCV_IMAGE_FORMAT_RGBf32;
constexpr auto RGBAf32 = NVCV_IMAGE_FORMAT_RGBAf32;

// -------------------- Testing GetIndexForInterpolation -----------------------

#define NVCV_TEST_ROW(INTERP_TYPE, POSITION, INPUT, GOLD) \
    ttype::Types<ttype::Value<INTERP_TYPE>, ttype::Value<POSITION>, ttype::Value<INPUT>, ttype::Value<GOLD>>

NVCV_TYPED_TEST_SUITE(
    GetIndexForInterpolationTests,
    ttype::Types<
        NVCV_TEST_ROW(NVCV_INTERP_NEAREST, 1, 1234.567f, 1234), NVCV_TEST_ROW(NVCV_INTERP_NEAREST, 2, 1234.567f, 1234),
        NVCV_TEST_ROW(NVCV_INTERP_NEAREST, 1, -3.6f, -4), NVCV_TEST_ROW(NVCV_INTERP_LINEAR, 1, 5.678f, 5),
        NVCV_TEST_ROW(NVCV_INTERP_LINEAR, 2, 5.678f, 5), NVCV_TEST_ROW(NVCV_INTERP_CUBIC, 1, -1234.567f, -1234),
        NVCV_TEST_ROW(NVCV_INTERP_CUBIC, 2, -1234.567f, -1235), NVCV_TEST_ROW(NVCV_INTERP_AREA, 1, 4.567f, 5),
        NVCV_TEST_ROW(NVCV_INTERP_AREA, 2, 4.567f, 4)>);

#undef NVCV_TEST_ROW

TYPED_TEST(GetIndexForInterpolationTests, correct_index)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;
    constexpr int  kPosition   = ttype::GetValue<TypeParam, 1>;

    const float in   = ttype::GetValue<TypeParam, 2>;
    const int   gold = ttype::GetValue<TypeParam, 3>;

    int test = cuda::GetIndexForInterpolation<kInterpType, kPosition>(in);

    EXPECT_EQ(test, gold);
}

// ---------------------- Testing InterpolationWrap 2D -------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    InterpolationWrap2DTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_INTERP_NEAREST>,
                 ttype::Value<PackedImage<float, 2, 2>{2.f, 3.f, -5.6f, 1.2f}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_LINEAR>,
                 ttype::Value<PackedImage<short3, 1, 2>{short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_CUBIC>,
                 ttype::Value<PackedImage<uchar3, 2, 1>{uchar3{1, 2, 3}, uchar3{56, 78, 0}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_AREA>,
                 ttype::Value<PackedImage<int2, 2, 2>{int2{1, -2}, int2{-34, 56}, int2{78, -9}, int2{123, 0}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_NEAREST>, ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_LINEAR>, ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_CUBIC>, ttype::Value<PackedImage<long3, 2, 3>{
        long3{0, 1234, -2345}, long3{5678, -6789, 1234}, long3{1234567, -9876543, 1},
        long3{-12345, 456789, 0}, long3{-23456, 65432, -7654321}, long3{-1234567, 7654321, 123}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_AREA>, ttype::Value<PackedImage<short2, 3, 2>{
        short2{0, 1234}, short2{5678, -6789}, short2{1234, -9876},
        short2{-1234, 4567}, short2{-2345, 6543}, short2{-1234, 7654}}>>
>);

// clang-format on

TYPED_TEST(InterpolationWrap2DTest, correct_grid_aligned_values_in_host)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    constexpr auto kBorderType = NVCV_BORDER_CONSTANT;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    using TensorWrap = cuda::TensorWrap<ValueType, -1, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, kBorderType, true, true>;
    using InterpWrap = cuda::InterpolationWrap<BorderWrap, kInterpType>;

    EXPECT_TRUE((std::is_same_v<typename InterpWrap::BorderWrapper, BorderWrap>));
    EXPECT_TRUE((std::is_same_v<typename InterpWrap::TensorWrapper, TensorWrap>));

    EXPECT_EQ(InterpWrap::BorderWrapper::kBorderType, kBorderType);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[0], true);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[1], true);
    EXPECT_EQ(InterpWrap::BorderWrapper::kNumActiveDimensions, 2);
    EXPECT_EQ(InterpWrap::kInterpolationType, kInterpType);
    EXPECT_EQ(InterpWrap::kNumDimensions, 2);
    EXPECT_EQ(InterpWrap::kCoordMap.id[0], 0);
    EXPECT_EQ(InterpWrap::kCoordMap.id[1], 1);

    const float scaleX = 1.f, scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[0], InputType::kShapes[1]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    EXPECT_TRUE(interpWrap.scaleX() == scaleX || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.scaleY() == scaleY || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.isIntegerArea() || kInterpType != NVCV_INTERP_AREA);

    const int2 shapes{InputType::kShapes[1], InputType::kShapes[0]};

    ValueType gold;

    for (int y = -2; y < shapes.y + 2; ++y)
    {
        for (int x = -2; x < shapes.x + 2; ++x)
        {
            int2   intCoord{x, y};
            float2 floatCoord = cuda::StaticCast<float>(intCoord);

            if (test::IsInside(intCoord, shapes, kBorderType))
            {
                EXPECT_TRUE(std::is_reference_v<decltype(tensorWrap[intCoord])>);

                gold = input[intCoord.y * InputType::kShapes[1] + intCoord.x];

                EXPECT_EQ(tensorWrap[intCoord], gold);
            }
            else
            {
                gold = borderValue;
            }

            EXPECT_TRUE(std::is_reference_v<decltype(borderWrap[intCoord])>);
            EXPECT_FALSE(std::is_reference_v<decltype(interpWrap[floatCoord])>);

            EXPECT_EQ(borderWrap[intCoord], gold);
            EXPECT_EQ(interpWrap[floatCoord], gold);
        }
    }
}

TYPED_TEST(InterpolationWrap2DTest, correct_grid_unaligned_values_in_host)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    constexpr auto kBorderType = NVCV_BORDER_CONSTANT;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    using TensorWrap = cuda::Tensor2DWrap<const ValueType>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, kBorderType, true, true>;
    using InterpWrap = cuda::InterpolationWrap<BorderWrap, kInterpType>;

    const float scaleX = 1.f, scaleY = 2.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[0], InputType::kShapes[1]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    const int2 shapes{InputType::kShapes[1], InputType::kShapes[0]};

    ValueType gold;

    std::default_random_engine            randEng{0};
    std::uniform_real_distribution<float> randCoord{0.f, 1.f};

    for (float y = -2; y < shapes.y + 2; ++y)
    {
        for (float x = -2; x < shapes.x + 2; ++x)
        {
            float2 floatCoord{x + randCoord(randEng), y + randCoord(randEng)};

            if (kInterpType == NVCV_INTERP_NEAREST)
            {
                int2 c = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord + .5f);

                gold = borderWrap[c];
            }
            else if (kInterpType == NVCV_INTERP_LINEAR)
            {
                int2 c1 = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord);
                int2 c2 = c1 + 1;

                auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

                out += borderWrap[int2{c1.x, c1.y}] * (c2.x - floatCoord.x) * (c2.y - floatCoord.y);
                out += borderWrap[int2{c2.x, c1.y}] * (floatCoord.x - c1.x) * (c2.y - floatCoord.y);
                out += borderWrap[int2{c1.x, c2.y}] * (c2.x - floatCoord.x) * (floatCoord.y - c1.y);
                out += borderWrap[int2{c2.x, c2.y}] * (floatCoord.x - c1.x) * (floatCoord.y - c1.y);

                gold = cuda::SaturateCast<ValueType>(out);
            }
            else if (kInterpType == NVCV_INTERP_CUBIC)
            {
                int xmin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.x - 2.f);
                int ymin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.y - 2.f);
                int xmax = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord.x + 2.f);
                int ymax = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord.y + 2.f);
                using FT = cuda::ConvertBaseTypeTo<float, ValueType>;
                auto sum = cuda::SetAll<FT>(0);

                float w, wsum = 0.f;

                for (int cy = ymin; cy <= ymax; cy++)
                {
                    for (int cx = xmin; cx <= xmax; cx++)
                    {
                        w = test::GetBicubicCoeff(floatCoord.x - cx) * test::GetBicubicCoeff(floatCoord.y - cy);
                        sum += w * borderWrap[int2{cx, cy}];
                        wsum += w;
                    }
                }

                sum = (wsum == 0.f) ? cuda::SetAll<FT>(0) : sum / wsum;

                gold = cuda::SaturateCast<ValueType>(sum);
            }
            else if (kInterpType == NVCV_INTERP_AREA)
            {
                int xmin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.x * scaleX);
                int xmax = cuda::round<cuda::RoundMode::DOWN, int>((floatCoord.x + 1) * scaleX);
                int ymin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.y * scaleY);
                int ymax = cuda::round<cuda::RoundMode::DOWN, int>((floatCoord.y + 1) * scaleY);

                auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

                for (int cy = ymin; cy < ymax; ++cy)
                {
                    for (int cx = xmin; cx < xmax; ++cx)
                    {
                        out += borderWrap[int2{cx, cy}] * (1.f / (scaleX * scaleY));
                    }
                }

                gold = cuda::SaturateCast<ValueType>(out);
            }

            EXPECT_EQ(interpWrap[floatCoord], gold);
        }
    }
}

#define NVCV_TEST_ROW(WIDTH, HEIGHT, SHIFTX, SHIFTY, SCALEX, SCALEY, FORMAT, VALUETYPE, BORDERTYPE, INTERPTYPE) \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<SHIFTX>, ttype::Value<SHIFTY>,         \
                 ttype::Value<SCALEX>, ttype::Value<SCALEY>, ttype::Value<FORMAT>, VALUETYPE,                   \
                 ttype::Value<BORDERTYPE>, ttype::Value<INTERPTYPE>>

NVCV_TYPED_TEST_SUITE(
    InterpolationWrapHWTest,
    ttype::Types<
        NVCV_TEST_ROW(21, 11, 0.f, 0.f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(33, 22, 3.f, 2.f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(43, 33, 3.33f, 2.22f, 0.f, 0.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(11, 12, 4.f, 4.f, 1.4f, 1.2f, RGB8, uchar3, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(7, 6, 5.5f, 6.5, 0.f, 0.f, RGBAf32, float4, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(8, 3, 9.25f, 4.25f, 0.f, 0.f, RGB8, uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(6, 4, 3.75f, 5.75f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(61, 41, 6.f, 6.f, 3.456f, 4.567f, _2S16, short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(12, 13, 6.f, 4.f, 0.f, 0.f, RGBf32, float3, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(19, 99, 19.789f, 91.234f, 0.f, 0.f, S16, short1, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(21, 91, 22.987f, 11.123f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(26, 37, 8.f, 9.f, 0.123f, 0.456f, RGBA8, uchar4, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA)>);

#undef NVCV_TEST_ROW

TYPED_TEST(InterpolationWrapHWTest, correct_shift_in_device)
{
    const int   width  = ttype::GetValue<TypeParam, 0>;
    const int   height = ttype::GetValue<TypeParam, 1>;
    const float shiftX = ttype::GetValue<TypeParam, 2>;
    const float shiftY = ttype::GetValue<TypeParam, 3>;
    const float scaleX = ttype::GetValue<TypeParam, 4>;
    const float scaleY = ttype::GetValue<TypeParam, 5>;

    const nvcv::ImageFormat format{ttype::GetValue<TypeParam, 6>};

    using ValueType            = ttype::GetType<TypeParam, 7>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 8>;
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 9>;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    const float2 shift{shiftX, shiftY};
    const float2 scale{scaleX, scaleY};

    nvcv::Tensor srcTensor(
        nvcv::TensorShape{
            {height, width},
            "HW"
    },
        nvcv::DataType{format});
    nvcv::Tensor dstTensor(
        nvcv::TensorShape{
            {height, width},
            "HW"
    },
        nvcv::DataType{format});

    auto srcDev = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstDev = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcDev, nvcv::NullOpt);
    ASSERT_NE(dstDev, nvcv::NullOpt);

    const long2 srcStrides{srcDev->stride(0), srcDev->stride(1)};
    const long2 dstStrides{dstDev->stride(0), dstDev->stride(1)};

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    const int2 srcSize{srcAccess->numCols(), srcAccess->numRows()};
    const int2 dstSize{dstAccess->numCols(), dstAccess->numRows()};

    const std::size_t srcSizeBytes = srcStrides.x * srcSize.y;
    const std::size_t dstSizeBytes = dstStrides.x * dstSize.y;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    using Interp2DWrapConstValueType
        = cuda::InterpolationWrap<cuda::BorderWrap<cuda::Tensor2DWrap<const ValueType>, kBorderType, true, true>,
                                  kInterpType>;

    Interp2DWrapConstValueType    srcWrap(*srcDev, borderValue, scaleX, scaleY);
    cuda::Tensor2DWrap<ValueType> dstWrap(*dstDev);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunInterpShift(dstWrap, srcWrap, dstSize, shift, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    const int2 srcSize2{srcSize.x, srcSize.y};

    // Run gold interpolation shift
    float2 srcCoord;

    for (int y = 0; y < dstSize.y; ++y)
    {
        srcCoord.y = y + shiftY;

        for (int x = 0; x < dstSize.x; ++x)
        {
            srcCoord.x = x + shiftX;

            test::ValueAt<ValueType>(gold, dstStrides, int2{x, y}) = test::GoldInterp<kInterpType, kBorderType>(
                srcVec, srcStrides, srcSize2, borderValue, scale, srcCoord);
        }
    }

    VEC_EXPECT_NEAR(test, gold, 1);
}

// ---------------------- Testing InterpolationWrap 3D -------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    InterpolationWrap3DTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_INTERP_NEAREST>,
                 ttype::Value<PackedTensor3D<int, 1, 2, 2>{2, 3, -5, 1}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_LINEAR>,
                 ttype::Value<PackedTensor3D<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_CUBIC>,
                 ttype::Value<PackedTensor3D<uchar3, 1, 2, 1>{uchar3{1, 2, 3}, uchar3{56, 78, 0}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_AREA>,
                 ttype::Value<PackedTensor3D<int2, 1, 2, 2>{int2{1, -2}, int2{-34, 56}, int2{78, -9}, int2{123, 0}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_NEAREST>,
                 ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_LINEAR>,
                 ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_CUBIC>, ttype::Value<PackedTensor3D<long3, 1, 2, 3>{
        long3{0, 1234, -2345}, long3{5678, -6789, 1234}, long3{1234567, -9876543, 1},
        long3{-12345, 456789, 0}, long3{-23456, 65432, -7654321}, long3{-1234567, 7654321, 123}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_AREA>, ttype::Value<PackedTensor3D<short2, 3, 1, 2>{
        short2{0, 1234}, short2{5678, -6789}, short2{1234, -9876},
        short2{-1234, 4567}, short2{-2345, 6543}, short2{-1234, 7654}}>>
>);

// clang-format on

TYPED_TEST(InterpolationWrap3DTest, correct_grid_aligned_values_in_host)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    constexpr auto kBorderType = NVCV_BORDER_CONSTANT;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    using TensorWrap = cuda::TensorWrap<ValueType, -1, -1, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, kBorderType, false, true, true>;
    using InterpWrap = cuda::InterpolationWrap<BorderWrap, kInterpType>;

    EXPECT_TRUE((std::is_same_v<typename InterpWrap::BorderWrapper, BorderWrap>));
    EXPECT_TRUE((std::is_same_v<typename InterpWrap::TensorWrapper, TensorWrap>));

    EXPECT_EQ(InterpWrap::BorderWrapper::kBorderType, kBorderType);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[0], false);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[1], true);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[2], true);
    EXPECT_EQ(InterpWrap::BorderWrapper::kNumActiveDimensions, 2);
    EXPECT_EQ(InterpWrap::kInterpolationType, kInterpType);
    EXPECT_EQ(InterpWrap::kNumDimensions, 3);
    EXPECT_EQ(InterpWrap::kCoordMap.id[0], 0);
    EXPECT_EQ(InterpWrap::kCoordMap.id[1], 1);
    EXPECT_EQ(InterpWrap::kCoordMap.id[2], 2);

    const float scaleX = 1.f, scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1], InputType::kStrides[2]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    EXPECT_TRUE(interpWrap.scaleX() == scaleX || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.scaleY() == scaleY || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.isIntegerArea() || kInterpType != NVCV_INTERP_AREA);

    const int3 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0]};

    ValueType gold;

    for (int z = 0; z < shapes.z; ++z)
    {
        for (int y = -2; y < shapes.y + 2; ++y)
        {
            for (int x = -2; x < shapes.x + 2; ++x)
            {
                int2   inCoord{x, y};
                int3   intCoord{x, y, z};
                float3 floatCoord = cuda::StaticCast<float>(intCoord);

                if (test::IsInside(inCoord, int2{shapes.x, shapes.y}, kBorderType))
                {
                    intCoord.x = inCoord.x;
                    intCoord.y = inCoord.y;

                    EXPECT_TRUE(std::is_reference_v<decltype(tensorWrap[intCoord])>);

                    gold = input[intCoord.z * InputType::kShapes[2] * InputType::kShapes[1]
                                 + intCoord.y * InputType::kShapes[2] + intCoord.x];

                    EXPECT_EQ(tensorWrap[intCoord], gold);
                }
                else
                {
                    gold = borderValue;
                }

                EXPECT_TRUE(std::is_reference_v<decltype(borderWrap[intCoord])>);
                EXPECT_FALSE(std::is_reference_v<decltype(interpWrap[floatCoord])>);

                EXPECT_EQ(borderWrap[intCoord], gold);
                EXPECT_EQ(interpWrap[floatCoord], gold);
            }
        }
    }
}

TYPED_TEST(InterpolationWrap3DTest, correct_grid_unaligned_values_in_host)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    constexpr auto kBorderType = NVCV_BORDER_CONSTANT;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    using TensorWrap = cuda::Tensor3DWrap<const ValueType>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, kBorderType, false, true, true>;
    using InterpWrap = cuda::InterpolationWrap<BorderWrap, kInterpType>;

    const float scaleX = 2.f, scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    const int3 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0]};

    ValueType gold;

    std::default_random_engine            randEng{0};
    std::uniform_real_distribution<float> randCoord{0.f, 1.f};

    for (int z = 0; z < shapes.z; ++z)
    {
        for (float y = -2; y < shapes.y + 2; ++y)
        {
            for (float x = -2; x < shapes.x + 2; ++x)
            {
                float3 floatCoord{x + randCoord(randEng), y + randCoord(randEng), static_cast<float>(z)};

                if (kInterpType == NVCV_INTERP_NEAREST)
                {
                    int2 c = cuda::round<cuda::RoundMode::DOWN, int>(cuda::DropCast<2>(floatCoord + .5f));

                    gold = borderWrap[int3{c.x, c.y, z}];
                }
                else if (kInterpType == NVCV_INTERP_LINEAR)
                {
                    int2 c1 = cuda::round<cuda::RoundMode::DOWN, int>(cuda::DropCast<2>(floatCoord));
                    int2 c2 = c1 + 1;

                    auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

                    out += borderWrap[int3{c1.x, c1.y, z}] * (c2.x - floatCoord.x) * (c2.y - floatCoord.y);
                    out += borderWrap[int3{c2.x, c1.y, z}] * (floatCoord.x - c1.x) * (c2.y - floatCoord.y);
                    out += borderWrap[int3{c1.x, c2.y, z}] * (c2.x - floatCoord.x) * (floatCoord.y - c1.y);
                    out += borderWrap[int3{c2.x, c2.y, z}] * (floatCoord.x - c1.x) * (floatCoord.y - c1.y);

                    gold = cuda::SaturateCast<ValueType>(out);
                }
                else if (kInterpType == NVCV_INTERP_CUBIC)
                {
                    int xmin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.x - 2.f);
                    int ymin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.y - 2.f);
                    int xmax = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord.x + 2.f);
                    int ymax = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord.y + 2.f);
                    using FT = cuda::ConvertBaseTypeTo<float, ValueType>;
                    auto sum = cuda::SetAll<FT>(0);

                    float w, wsum = 0.f;

                    for (int cy = ymin; cy <= ymax; cy++)
                    {
                        for (int cx = xmin; cx <= xmax; cx++)
                        {
                            w = test::GetBicubicCoeff(floatCoord.x - cx) * test::GetBicubicCoeff(floatCoord.y - cy);
                            sum += w * borderWrap[int3{cx, cy, z}];
                            wsum += w;
                        }
                    }

                    sum = (wsum == 0.f) ? cuda::SetAll<FT>(0) : sum / wsum;

                    gold = cuda::SaturateCast<ValueType>(sum);
                }
                else if (kInterpType == NVCV_INTERP_AREA)
                {
                    int xmin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.x * scaleX);
                    int xmax = cuda::round<cuda::RoundMode::DOWN, int>((floatCoord.x + 1) * scaleX);
                    int ymin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.y * scaleY);
                    int ymax = cuda::round<cuda::RoundMode::DOWN, int>((floatCoord.y + 1) * scaleY);

                    auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

                    for (int cy = ymin; cy < ymax; ++cy)
                    {
                        for (int cx = xmin; cx < xmax; ++cx)
                        {
                            out += borderWrap[int3{cx, cy, z}] * (1.f / (scaleX * scaleY));
                        }
                    }

                    gold = cuda::SaturateCast<ValueType>(out);
                }

                EXPECT_EQ(interpWrap[floatCoord], gold);
            }
        }
    }
}

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, SHIFTX, SHIFTY, SCALEX, SCALEY, FORMAT, VALUETYPE, BORDERTYPE,        \
                      INTERPTYPE)                                                                                   \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<SHIFTX>,            \
                 ttype::Value<SHIFTY>, ttype::Value<SCALEX>, ttype::Value<SCALEY>, ttype::Value<FORMAT>, VALUETYPE, \
                 ttype::Value<BORDERTYPE>, ttype::Value<INTERPTYPE>>

NVCV_TYPED_TEST_SUITE(
    InterpolationWrapNHWTest,
    ttype::Types<
        NVCV_TEST_ROW(71, 17, 1, 0.f, 0.f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(31, 13, 2, 2.f, 3.f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(12, 32, 3, 2.2f, 3.3f, 0.f, 0.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(13, 31, 2, 3.f, 4.f, 2.f, 2.f, RGB8, uchar3, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(52, 25, 2, 2.f, 5.f, 0.f, 0.f, RGBAf32, float4, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(26, 29, 3, 6.f, 9.f, 0.f, 0.f, RGB8, uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(24, 42, 4, 4.25f, 2.25f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(20, 20, 3, 3.f, 4.f, 1.f, 1.f, _2S16, short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(12, 87, 2, 8.123f, 9.234f, 0.f, 0.f, RGBf32, float3, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(11, 21, 3, 11.f, 22.f, 0.f, 0.f, S16, short1, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(13, 24, 4, 12.345f, 21.678f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(13, 24, 4, 13.f, 22.f, 31.12f, 23.54f, RGBA8, uchar4, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA)>);

#undef NVCV_TEST_ROW

TYPED_TEST(InterpolationWrapNHWTest, correct_shift_in_device)
{
    const int   width   = ttype::GetValue<TypeParam, 0>;
    const int   height  = ttype::GetValue<TypeParam, 1>;
    const int   batches = ttype::GetValue<TypeParam, 2>;
    const float shiftX  = ttype::GetValue<TypeParam, 3>;
    const float shiftY  = ttype::GetValue<TypeParam, 4>;
    const float scaleX  = ttype::GetValue<TypeParam, 5>;
    const float scaleY  = ttype::GetValue<TypeParam, 6>;

    const nvcv::ImageFormat format{ttype::GetValue<TypeParam, 7>};

    using ValueType            = ttype::GetType<TypeParam, 8>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 9>;
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 10>;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    const float2 shift{shiftX, shiftY};
    const float2 scale{scaleX, scaleY};

    nvcv::Tensor srcTensor(batches, {width, height}, format);
    nvcv::Tensor dstTensor(batches, {width, height}, format);

    auto srcDev = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstDev = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcDev, nvcv::NullOpt);
    ASSERT_NE(dstDev, nvcv::NullOpt);

    const long3 srcStrides{srcDev->stride(0), srcDev->stride(1), srcDev->stride(2)};
    const long3 dstStrides{dstDev->stride(0), dstDev->stride(1), dstDev->stride(2)};

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    const int3 srcSize{srcAccess->numCols(), srcAccess->numRows(), static_cast<int>(srcAccess->numSamples())};
    const int3 dstSize{dstAccess->numCols(), dstAccess->numRows(), static_cast<int>(dstAccess->numSamples())};

    const std::size_t srcSizeBytes = srcStrides.x * srcSize.z;
    const std::size_t dstSizeBytes = dstStrides.x * dstSize.z;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    auto srcWrap = cuda::CreateInterpolationWrapNHW<const ValueType, kBorderType, kInterpType>(*srcDev, borderValue,
                                                                                               scaleX, scaleY);
    auto dstWrap = cuda::CreateTensorWrapNHW<ValueType>(*dstDev);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunInterpShift(dstWrap, srcWrap, dstSize, shift, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    const int2 srcSize2{srcSize.x, srcSize.y};

    // Run gold interpolation shift
    for (int z = 0; z < dstSize.z; ++z)
    {
        float2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y + shiftY;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x + shiftX;

                test::ValueAt<ValueType>(gold, dstStrides, int3{x, y, z}) = test::GoldInterp<kInterpType, kBorderType>(
                    srcVec, srcStrides, srcSize2, borderValue, scale, srcCoord, z);
            }
        }
    }

    VEC_EXPECT_NEAR(test, gold, 1);
}

// ---------------------- Testing InterpolationWrap 4D -------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    InterpolationWrap4DTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_INTERP_NEAREST>,
                 ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_LINEAR>,
                 ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}, short3{-1, -2, -3}, short3{-567, 234, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}, short3{1, 2, 3}, short3{-56, 23, 1}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_CUBIC>,
                 ttype::Value<PackedTensor4D<uchar3, 1, 2, 1, 2>{
        uchar3{1, 2, 3}, uchar3{56, 78, 0}, uchar3{123, 21, 32}, uchar3{76, 98, 87}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_AREA>,
                 ttype::Value<PackedTensor4D<int2, 1, 2, 2, 1>{int2{1, -2}, int2{-34, 56}, int2{78, -9}, int2{123, 0}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_NEAREST>,
                 ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_LINEAR>,
                 ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_CUBIC>, ttype::Value<PackedTensor4D<long3, 1, 2, 3, 1>{
        long3{0, 1234, -2345}, long3{5678, -6789, 1234}, long3{1234567, -9876543, 1},
        long3{-12345, 456789, 0}, long3{-23456, 65432, -7654321}, long3{-1234567, 7654321, 123}}>>,
    ttype::Types<ttype::Value<NVCV_INTERP_AREA>, ttype::Value<PackedTensor4D<short2, 1, 3, 1, 2>{
        short2{0, 1234}, short2{5678, -6789}, short2{1234, -9876},
        short2{-1234, 4567}, short2{-2345, 6543}, short2{-1234, 7654}}>>
>);

// clang-format on

TYPED_TEST(InterpolationWrap4DTest, correct_grid_aligned_values_in_host)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    constexpr auto kBorderType = NVCV_BORDER_CONSTANT;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    using TensorWrap = cuda::TensorWrap<ValueType, -1, -1, -1, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, kBorderType, false, true, true, false>;
    using InterpWrap = cuda::InterpolationWrap<BorderWrap, kInterpType>;

    EXPECT_TRUE((std::is_same_v<typename InterpWrap::BorderWrapper, BorderWrap>));
    EXPECT_TRUE((std::is_same_v<typename InterpWrap::TensorWrapper, TensorWrap>));

    EXPECT_EQ(InterpWrap::BorderWrapper::kBorderType, kBorderType);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[0], false);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[1], true);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[2], true);
    EXPECT_EQ(InterpWrap::BorderWrapper::kActiveDimensions[3], false);
    EXPECT_EQ(InterpWrap::BorderWrapper::kNumActiveDimensions, 2);
    EXPECT_EQ(InterpWrap::kInterpolationType, kInterpType);
    EXPECT_EQ(InterpWrap::kNumDimensions, 4);
    EXPECT_EQ(InterpWrap::kCoordMap.id[0], 1);
    EXPECT_EQ(InterpWrap::kCoordMap.id[1], 2);
    EXPECT_EQ(InterpWrap::kCoordMap.id[2], 3);
    EXPECT_EQ(InterpWrap::kCoordMap.id[3], 0);

    const float scaleX = 1.f, scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1], InputType::kStrides[2],
                          InputType::kStrides[3]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    EXPECT_TRUE(interpWrap.scaleX() == scaleX || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.scaleY() == scaleY || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.isIntegerArea() || kInterpType != NVCV_INTERP_AREA);

    const int4 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0], InputType::kShapes[3]};

    ValueType gold;

    for (int z = 0; z < shapes.z; ++z)
    {
        for (int y = -2; y < shapes.y + 2; ++y)
        {
            for (int x = -2; x < shapes.x + 2; ++x)
            {
                for (int c = 0; c < shapes.w; ++c)
                {
                    int2   inCoord{x, y};
                    int4   intCoord{c, x, y, z};
                    float4 floatCoord = cuda::StaticCast<float>(intCoord);

                    if (test::IsInside(inCoord, int2{shapes.x, shapes.y}, kBorderType))
                    {
                        intCoord.y = inCoord.x;
                        intCoord.z = inCoord.y;

                        EXPECT_TRUE(std::is_reference_v<decltype(tensorWrap[intCoord])>);

                        gold = input[intCoord.w * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                                     + intCoord.z * InputType::kShapes[2] * InputType::kShapes[3]
                                     + intCoord.y * InputType::kShapes[3] + intCoord.x];

                        EXPECT_EQ(tensorWrap[intCoord], gold);
                    }
                    else
                    {
                        gold = borderValue;
                    }

                    EXPECT_TRUE(std::is_reference_v<decltype(borderWrap[intCoord])>);
                    EXPECT_FALSE(std::is_reference_v<decltype(interpWrap[floatCoord])>);

                    EXPECT_EQ(borderWrap[intCoord], gold);
                    EXPECT_EQ(interpWrap[floatCoord], gold);
                }
            }
        }
    }
}

TYPED_TEST(InterpolationWrap4DTest, correct_grid_unaligned_values_in_host)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    constexpr auto kBorderType = NVCV_BORDER_CONSTANT;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    using TensorWrap = cuda::Tensor4DWrap<const ValueType>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, kBorderType, false, true, true, false>;
    using InterpWrap = cuda::InterpolationWrap<BorderWrap, kInterpType>;

    const float scaleX = 2.f, scaleY = 2.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1], InputType::kStrides[2]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    const int4 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0], InputType::kShapes[3]};

    ValueType gold;

    std::default_random_engine            randEng{0};
    std::uniform_real_distribution<float> randCoord{0.f, 1.f};

    for (int z = 0; z < shapes.z; ++z)
    {
        for (float y = -2; y < shapes.y + 2; ++y)
        {
            for (float x = -2; x < shapes.x + 2; ++x)
            {
                for (int k = 0; k < shapes.w; ++k)
                {
                    float2 floatCoord{x + randCoord(randEng), y + randCoord(randEng)};

                    if (kInterpType == NVCV_INTERP_NEAREST)
                    {
                        int2 c = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord + .5f);

                        gold = borderWrap[int4{k, c.x, c.y, z}];
                    }
                    else if (kInterpType == NVCV_INTERP_LINEAR)
                    {
                        int2 c1 = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord);
                        int2 c2 = c1 + 1;

                        auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

                        out += borderWrap[int4{k, c1.x, c1.y, z}] * (c2.x - floatCoord.x) * (c2.y - floatCoord.y);
                        out += borderWrap[int4{k, c2.x, c1.y, z}] * (floatCoord.x - c1.x) * (c2.y - floatCoord.y);
                        out += borderWrap[int4{k, c1.x, c2.y, z}] * (c2.x - floatCoord.x) * (floatCoord.y - c1.y);
                        out += borderWrap[int4{k, c2.x, c2.y, z}] * (floatCoord.x - c1.x) * (floatCoord.y - c1.y);

                        gold = cuda::SaturateCast<ValueType>(out);
                    }
                    else if (kInterpType == NVCV_INTERP_CUBIC)
                    {
                        int xmin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.x - 2.f);
                        int ymin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.y - 2.f);
                        int xmax = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord.x + 2.f);
                        int ymax = cuda::round<cuda::RoundMode::DOWN, int>(floatCoord.y + 2.f);
                        using FT = cuda::ConvertBaseTypeTo<float, ValueType>;
                        auto sum = cuda::SetAll<FT>(0);

                        float w, wsum = 0.f;

                        for (int cy = ymin; cy <= ymax; cy++)
                        {
                            for (int cx = xmin; cx <= xmax; cx++)
                            {
                                w = test::GetBicubicCoeff(floatCoord.x - cx) * test::GetBicubicCoeff(floatCoord.y - cy);
                                sum += w * borderWrap[int4{k, cx, cy, z}];
                                wsum += w;
                            }
                        }

                        sum = (wsum == 0.f) ? cuda::SetAll<FT>(0) : sum / wsum;

                        gold = cuda::SaturateCast<ValueType>(sum);
                    }
                    else if (kInterpType == NVCV_INTERP_AREA)
                    {
                        int xmin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.x * scaleX);
                        int xmax = cuda::round<cuda::RoundMode::DOWN, int>((floatCoord.x + 1) * scaleX);
                        int ymin = cuda::round<cuda::RoundMode::UP, int>(floatCoord.y * scaleY);
                        int ymax = cuda::round<cuda::RoundMode::DOWN, int>((floatCoord.y + 1) * scaleY);

                        auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

                        for (int cy = ymin; cy < ymax; ++cy)
                        {
                            for (int cx = xmin; cx < xmax; ++cx)
                            {
                                out += borderWrap[int4{k, cx, cy, z}] * (1.f / (scaleX * scaleY));
                            }
                        }

                        gold = cuda::SaturateCast<ValueType>(out);
                    }

                    float4 floatCoord4{static_cast<float>(k), floatCoord.x, floatCoord.y, static_cast<float>(z)};

                    EXPECT_EQ(interpWrap[floatCoord4], gold);
                }
            }
        }
    }
}

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, SHIFTX, SHIFTY, SCALEX, SCALEY, FORMAT, VALUETYPE, BORDERTYPE,        \
                      INTERPTYPE)                                                                                   \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<SHIFTX>,            \
                 ttype::Value<SHIFTY>, ttype::Value<SCALEX>, ttype::Value<SCALEY>, ttype::Value<FORMAT>, VALUETYPE, \
                 ttype::Value<BORDERTYPE>, ttype::Value<INTERPTYPE>>

NVCV_TYPED_TEST_SUITE(
    InterpolationWrapNHWCTest,
    ttype::Types<
        NVCV_TEST_ROW(22, 33, 1, 0.f, 0.f, 0.f, 0.f, RGBA8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(33, 22, 3, 4.f, 5.f, 0.f, 0.f, _2S16, short1, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(31, 21, 4, 4.4f, 3.6f, 0.f, 0.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(30, 20, 2, 4.f, 6.f, 2.f, 2.f, RGB8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(11, 44, 3, 7.25f, 8.25f, 0.f, 0.f, RGBAf32, float1, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(66, 16, 2, 8.f, 7.f, 0.f, 0.f, RGB8, uchar1, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(12, 4, 3, 2.75f, 3.75f, 0.f, 0.f, RGBA8, uchar1, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(16, 14, 2, 3.f, 3.f, 2.f, 1.f, _2S16, short1, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(12, 33, 3, 4.123f, 3.21f, 0.f, 0.f, RGBf32, float1, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(19, 29, 2, 16.f, 23.f, 0.f, 0.f, S16, short1, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(13, 33, 3, 12.5f, 25.5f, 0.f, 0.f, _2S16, short1, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(13, 30, 2, 1.f, 2.f, 11.f, 22.f, RGBA8, uchar1, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA)>);

#undef NVCV_TEST_ROW

TYPED_TEST(InterpolationWrapNHWCTest, correct_shift_in_device)
{
    const int   width   = ttype::GetValue<TypeParam, 0>;
    const int   height  = ttype::GetValue<TypeParam, 1>;
    const int   batches = ttype::GetValue<TypeParam, 2>;
    const float shiftX  = ttype::GetValue<TypeParam, 3>;
    const float shiftY  = ttype::GetValue<TypeParam, 4>;
    const float scaleX  = ttype::GetValue<TypeParam, 5>;
    const float scaleY  = ttype::GetValue<TypeParam, 6>;

    const nvcv::ImageFormat format{ttype::GetValue<TypeParam, 7>};

    using ValueType            = ttype::GetType<TypeParam, 8>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 9>;
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 10>;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    const float2 shift{shiftX, shiftY};
    const float2 scale{scaleX, scaleY};

    nvcv::Tensor srcTensor(batches, {width, height}, format);
    nvcv::Tensor dstTensor(batches, {width, height}, format);

    auto srcDev = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstDev = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcDev, nvcv::NullOpt);
    ASSERT_NE(dstDev, nvcv::NullOpt);

    const long4 srcStrides{srcDev->stride(0), srcDev->stride(1), srcDev->stride(2), srcDev->stride(3)};
    const long4 dstStrides{dstDev->stride(0), dstDev->stride(1), dstDev->stride(2), srcDev->stride(3)};

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    const int4 srcSize{srcAccess->numCols(), srcAccess->numRows(), static_cast<int>(srcAccess->numSamples()),
                       srcAccess->numChannels()};
    const int4 dstSize{dstAccess->numCols(), dstAccess->numRows(), static_cast<int>(dstAccess->numSamples()),
                       dstAccess->numChannels()};

    const std::size_t srcSizeBytes = srcStrides.x * srcSize.z;
    const std::size_t dstSizeBytes = dstStrides.x * dstSize.z;

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    auto srcWrap = cuda::CreateInterpolationWrapNHWC<const ValueType, kBorderType, kInterpType>(*srcDev, borderValue,
                                                                                                scaleX, scaleY);
    auto dstWrap = cuda::CreateTensorWrapNHWC<ValueType>(*dstDev);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunInterpShift(dstWrap, srcWrap, dstSize, shift, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    const int2 srcSize2{srcSize.x, srcSize.y};

    // Run gold interpolation shift
    for (int z = 0; z < dstSize.z; ++z)
    {
        float2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y + shiftY;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x + shiftX;

                for (int k = 0; k < dstSize.w; ++k)
                {
                    test::ValueAt<ValueType>(gold, dstStrides, int4{k, x, y, z})
                        = test::GoldInterp<kInterpType, kBorderType>(srcVec, srcStrides, srcSize2, borderValue, scale,
                                                                     srcCoord, z, k);
                }
            }
        }
    }

    VEC_EXPECT_NEAR(test, gold, 1);
}
