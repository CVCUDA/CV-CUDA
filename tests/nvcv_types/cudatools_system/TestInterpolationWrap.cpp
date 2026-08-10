/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/InterpUtils.hpp>                  // for test::GoldInterp, etc.
#include <common/TypedTests.hpp>                   // for NVCV_TYPED_TEST_SUITE, etc.
#include <cvcuda/cuda_tools/DropCast.hpp>          // for DropCast, etc.
#include <cvcuda/cuda_tools/InterpolationWrap.hpp> // the object of this test
#include <cvcuda/cuda_tools/MathOps.hpp>           // for operator == to allow EXPECT_EQ
#include <nvcv/Tensor.hpp>                         // for Tensor, etc.
#include <nvcv/TensorDataAccess.hpp>               // for TensorDataAccessStridedImagePlanar, etc.

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

template<int N, typename BorderWrap>
auto BorderValueAt(BorderWrap &borderWrap, int x, int y, int z = 0, int k = 0)
{
    return borderWrap[test::GetCoord<N>(x, y, z, k)];
}

template<int N, typename ValueType, typename BorderWrap>
ValueType HostGoldInterpCubic(BorderWrap &borderWrap, float2 coord, int z = 0, int k = 0)
{
    int ix = cuda::round<cuda::RoundMode::DOWN, int>(coord.x);
    int iy = cuda::round<cuda::RoundMode::DOWN, int>(coord.y);

    using FT = cuda::ConvertBaseTypeTo<float, ValueType>;
    auto sum = cuda::SetAll<FT>(0);

    std::array<float, 4> wx;
    test::GetBicubicCoeffs(coord.x - static_cast<float>(ix), wx[0], wx[1], wx[2], wx[3]);
    std::array<float, 4> wy;
    test::GetBicubicCoeffs(coord.y - static_cast<float>(iy), wy[0], wy[1], wy[2], wy[3]);

    for (int i = 0; i < 16; ++i)
    {
        int cx = i % 4 - 1;
        int cy = i / 4 - 1;
        sum += BorderValueAt<N>(borderWrap, ix + cx, iy + cy, z, k) * (wx[cx + 1] * wy[cy + 1]);
    }

    return cuda::SaturateCast<ValueType>(sum);
}

template<int N, typename ValueType, typename BorderWrap>
ValueType HostGoldInterpArea(BorderWrap &borderWrap, float2 scale, float2 coord, int z = 0, int k = 0)
{
    int xmin = cuda::round<cuda::RoundMode::UP, int>(coord.x * scale.x);
    int xmax = cuda::round<cuda::RoundMode::DOWN, int>((coord.x + 1) * scale.x);
    int ymin = cuda::round<cuda::RoundMode::UP, int>(coord.y * scale.y);
    int ymax = cuda::round<cuda::RoundMode::DOWN, int>((coord.y + 1) * scale.y);

    auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

    int width  = std::max(0, xmax - xmin);
    int height = std::max(0, ymax - ymin);

    for (int i = 0; i < width * height; ++i)
    {
        int cx = xmin + i % width;
        int cy = ymin + i / width;
        out += BorderValueAt<N>(borderWrap, cx, cy, z, k) * (1.f / (scale.x * scale.y));
    }

    return cuda::SaturateCast<ValueType>(out);
}

template<int N, NVCVInterpolationType I, typename ValueType, typename BorderWrap>
ValueType HostGoldInterp(BorderWrap &borderWrap, float2 scale, float2 coord, int z = 0, int k = 0)
{
    if constexpr (I == NVCV_INTERP_NEAREST)
    {
        int2 c = cuda::round<cuda::RoundMode::DOWN, int>(coord + .5f);
        return BorderValueAt<N>(borderWrap, c.x, c.y, z, k);
    }
    else if constexpr (I == NVCV_INTERP_LINEAR)
    {
        int2 c1 = cuda::round<cuda::RoundMode::DOWN, int>(coord);
        int2 c2 = c1 + 1;

        auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

        const auto c1x = static_cast<float>(c1.x);
        const auto c1y = static_cast<float>(c1.y);
        const auto c2x = static_cast<float>(c2.x);
        const auto c2y = static_cast<float>(c2.y);

        out += BorderValueAt<N>(borderWrap, c1.x, c1.y, z, k) * (c2x - coord.x) * (c2y - coord.y);
        out += BorderValueAt<N>(borderWrap, c2.x, c1.y, z, k) * (coord.x - c1x) * (c2y - coord.y);
        out += BorderValueAt<N>(borderWrap, c1.x, c2.y, z, k) * (c2x - coord.x) * (coord.y - c1y);
        out += BorderValueAt<N>(borderWrap, c2.x, c2.y, z, k) * (coord.x - c1x) * (coord.y - c1y);

        return cuda::SaturateCast<ValueType>(out);
    }
    else if constexpr (I == NVCV_INTERP_CUBIC)
    {
        return HostGoldInterpCubic<N, ValueType>(borderWrap, coord, z, k);
    }
    else if constexpr (I == NVCV_INTERP_AREA)
    {
        return HostGoldInterpArea<N, ValueType>(borderWrap, scale, coord, z, k);
    }
}

template<NVCVBorderType B, typename TensorWrap, typename BorderWrap, typename InterpWrap, typename InputType,
         typename ValueType>
void ExpectGridAligned3D(TensorWrap &tensorWrap, BorderWrap &borderWrap, InterpWrap &interpWrap, const InputType &input,
                         const ValueType &borderValue, int3 shapes, int x, int y, int z)
{
    int2   inCoord{x, y};
    int3   intCoord{x, y, z};
    float3 floatCoord = cuda::StaticCast<float>(intCoord);

    ValueType gold = borderValue;

    if (test::IsInside(inCoord, int2{shapes.x, shapes.y}, B))
    {
        intCoord.x = inCoord.x;
        intCoord.y = inCoord.y;

        EXPECT_TRUE(std::is_reference_v<decltype(tensorWrap[intCoord])>);

        gold = input[intCoord.z * InputType::kShapes[2] * InputType::kShapes[1] + intCoord.y * InputType::kShapes[2]
                     + intCoord.x];

        EXPECT_EQ(tensorWrap[intCoord], gold);
    }

    EXPECT_TRUE(std::is_reference_v<decltype(borderWrap[intCoord])>);
    EXPECT_FALSE(std::is_reference_v<decltype(interpWrap[floatCoord])>);

    EXPECT_EQ(borderWrap[intCoord], gold);
    EXPECT_EQ(interpWrap[floatCoord], gold);
}

template<NVCVBorderType B, typename TensorWrap, typename BorderWrap, typename InterpWrap, typename InputType,
         typename ValueType>
void ExpectGridAligned4D(TensorWrap &tensorWrap, BorderWrap &borderWrap, InterpWrap &interpWrap, const InputType &input,
                         const ValueType &borderValue, int4 shapes, int x, int y, int z, int c)
{
    int2   inCoord{x, y};
    int4   intCoord{c, x, y, z};
    float4 floatCoord = cuda::StaticCast<float>(intCoord);

    ValueType gold = borderValue;

    if (test::IsInside(inCoord, int2{shapes.x, shapes.y}, B))
    {
        intCoord.y = inCoord.x;
        intCoord.z = inCoord.y;

        EXPECT_TRUE(std::is_reference_v<decltype(tensorWrap[intCoord])>);

        gold = input[intCoord.w * InputType::kShapes[1] * InputType::kShapes[2] * InputType::kShapes[3]
                     + intCoord.z * InputType::kShapes[2] * InputType::kShapes[3] + intCoord.y * InputType::kShapes[3]
                     + intCoord.x];

        EXPECT_EQ(tensorWrap[intCoord], gold);
    }

    EXPECT_TRUE(std::is_reference_v<decltype(borderWrap[intCoord])>);
    EXPECT_FALSE(std::is_reference_v<decltype(interpWrap[floatCoord])>);

    EXPECT_EQ(borderWrap[intCoord], gold);
    EXPECT_EQ(interpWrap[floatCoord], gold);
}

// -------------------- Testing GetIndexForInterpolation -----------------------

#define NVCV_TEST_ROW(INTERP_TYPE, POSITION, INPUT, GOLD) \
    ttype::Types<ttype::Value<INTERP_TYPE>, ttype::Value<POSITION>, ttype::Value<INPUT>, ttype::Value<GOLD>>

NVCV_TYPED_TEST_SUITE(
    GetIndexForInterpolationTests,
    ttype::Types<
        NVCV_TEST_ROW(NVCV_INTERP_NEAREST, 1, 1234.567f, 1234), NVCV_TEST_ROW(NVCV_INTERP_NEAREST, 2, 1234.567f, 1234),
        NVCV_TEST_ROW(NVCV_INTERP_NEAREST, 1, -3.6f, -4), NVCV_TEST_ROW(NVCV_INTERP_LINEAR, 1, 5.678f, 5),
        NVCV_TEST_ROW(NVCV_INTERP_LINEAR, 2, 5.678f, 5), NVCV_TEST_ROW(NVCV_INTERP_CUBIC, 1, -1234.567f, -1235),
        NVCV_TEST_ROW(NVCV_INTERP_CUBIC, 2, -1234.567f, -1235), NVCV_TEST_ROW(NVCV_INTERP_AREA, 1, 4.567f, 5),
        NVCV_TEST_ROW(NVCV_INTERP_AREA, 2, 4.567f, 4)>);

#undef NVCV_TEST_ROW

TYPED_TEST(GetIndexForInterpolationTests, correct_index)
{
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 0>;
    constexpr int  kPosition   = ttype::GetValue<TypeParam, 1>;

    const float in   = ttype::GetValue<TypeParam, 2>;
    const int   gold = ttype::GetValue<TypeParam, 3>;

    auto test = static_cast<int>(cuda::GetIndexForInterpolation<kInterpType, kPosition>(in));

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

    const float scaleX = 1.f;
    const float scaleY = 1.f;

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

    const float scaleX = 1.f;
    const float scaleY = 2.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[0], InputType::kShapes[1]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    const int2 shapes{InputType::kShapes[1], InputType::kShapes[0]};

    std::default_random_engine            randEng{0};
    std::uniform_real_distribution<float> randCoord{0.f, 1.f};
    const float2                          scale{scaleX, scaleY};

    for (int y = -2; y < shapes.y + 2; ++y)
    {
        for (int x = -2; x < shapes.x + 2; ++x)
        {
            float2 floatCoord{static_cast<float>(x) + randCoord(randEng), static_cast<float>(y) + randCoord(randEng)};
            ValueType gold = HostGoldInterp<2, kInterpType, ValueType>(borderWrap, scale, floatCoord);

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
        format.planeDataType(0));
    nvcv::Tensor dstTensor(
        nvcv::TensorShape{
            {height, width},
            "HW"
    },
        format.planeDataType(0));

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
    std::ranges::generate(srcVec, [&srcRand, &randEng]() { return srcRand(randEng); });

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
        srcCoord.y = static_cast<float>(y) + shiftY;

        for (int x = 0; x < dstSize.x; ++x)
        {
            srcCoord.x = static_cast<float>(x) + shiftX;

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

    const float scaleX = 1.f;
    const float scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1], InputType::kStrides[2]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    EXPECT_TRUE(interpWrap.scaleX() == scaleX || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.scaleY() == scaleY || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.isIntegerArea() || kInterpType != NVCV_INTERP_AREA);

    const int3 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0]};

    const int xBegin = -2;
    const int xCount = shapes.x + 4;
    const int yBegin = -2;
    const int yCount = shapes.y + 4;

    for (int i = 0; i < shapes.z * yCount * xCount; ++i)
    {
        int x = xBegin + i % xCount;
        int y = yBegin + (i / xCount) % yCount;
        int z = i / (xCount * yCount);

        ExpectGridAligned3D<kBorderType>(tensorWrap, borderWrap, interpWrap, input, borderValue, shapes, x, y, z);
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

    const float scaleX = 2.f;
    const float scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    const int3 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0]};

    std::default_random_engine            randEng{0};
    std::uniform_real_distribution<float> randCoord{0.f, 1.f};
    const float2                          scale{scaleX, scaleY};

    for (int z = 0; z < shapes.z; ++z)
    {
        for (int y = -2; y < shapes.y + 2; ++y)
        {
            for (int x = -2; x < shapes.x + 2; ++x)
            {
                float3    floatCoord{static_cast<float>(x) + randCoord(randEng),
                                  static_cast<float>(y) + randCoord(randEng), static_cast<float>(z)};
                ValueType gold
                    = HostGoldInterp<3, kInterpType, ValueType>(borderWrap, scale, cuda::DropCast<2>(floatCoord), z);

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
    std::ranges::generate(srcVec, [&srcRand, &randEng]() { return srcRand(randEng); });

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
            srcCoord.y = static_cast<float>(y) + shiftY;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = static_cast<float>(x) + shiftX;

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

    const float scaleX = 1.f;
    const float scaleY = 1.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1], InputType::kStrides[2],
                          InputType::kStrides[3]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    EXPECT_TRUE(interpWrap.scaleX() == scaleX || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.scaleY() == scaleY || kInterpType != NVCV_INTERP_AREA);
    EXPECT_TRUE(interpWrap.isIntegerArea() || kInterpType != NVCV_INTERP_AREA);

    const int4 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0], InputType::kShapes[3]};

    const int xBegin = -2;
    const int xCount = shapes.x + 4;
    const int yBegin = -2;
    const int yCount = shapes.y + 4;

    for (int i = 0; i < shapes.z * yCount * xCount * shapes.w; ++i)
    {
        int c = i % shapes.w;
        int x = xBegin + (i / shapes.w) % xCount;
        int y = yBegin + (i / (shapes.w * xCount)) % yCount;
        int z = i / (shapes.w * xCount * yCount);

        ExpectGridAligned4D<kBorderType>(tensorWrap, borderWrap, interpWrap, input, borderValue, shapes, x, y, z, c);
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

    const float scaleX = 2.f;
    const float scaleY = 2.f;

    TensorWrap tensorWrap(input.data(), InputType::kStrides[0], InputType::kStrides[1], InputType::kStrides[2]);
    BorderWrap borderWrap(tensorWrap, borderValue, InputType::kShapes[1], InputType::kShapes[2]);
    InterpWrap interpWrap(borderWrap, scaleX, scaleY);

    const int4 shapes{InputType::kShapes[2], InputType::kShapes[1], InputType::kShapes[0], InputType::kShapes[3]};

    std::default_random_engine            randEng{0};
    std::uniform_real_distribution<float> randCoord{0.f, 1.f};
    const float2                          scale{scaleX, scaleY};

    const int xBegin = -2;
    const int xCount = shapes.x + 4;
    const int yBegin = -2;
    const int yCount = shapes.y + 4;

    for (int i = 0; i < shapes.z * yCount * xCount * shapes.w; ++i)
    {
        int       k = i % shapes.w;
        auto      x = static_cast<float>(xBegin + (i / shapes.w) % xCount);
        auto      y = static_cast<float>(yBegin + (i / (shapes.w * xCount)) % yCount);
        int       z = i / (shapes.w * xCount * yCount);
        float2    floatCoord{x + randCoord(randEng), y + randCoord(randEng)};
        float4    floatCoord4{static_cast<float>(k), floatCoord.x, floatCoord.y, static_cast<float>(z)};
        ValueType gold = HostGoldInterp<4, kInterpType, ValueType>(borderWrap, scale, floatCoord, z, k);

        EXPECT_EQ(interpWrap[floatCoord4], gold);
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

    const long4_16a srcStrides{srcDev->stride(0), srcDev->stride(1), srcDev->stride(2), srcDev->stride(3)};
    const long4_16a dstStrides{dstDev->stride(0), dstDev->stride(1), dstDev->stride(2), srcDev->stride(3)};

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
    std::ranges::generate(srcVec, [&srcRand, &randEng]() { return srcRand(randEng); });

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

    for (int i = 0; i < dstSize.z * dstSize.y * dstSize.x * dstSize.w; ++i)
    {
        int    k        = i % dstSize.w;
        int    x        = (i / dstSize.w) % dstSize.x;
        int    y        = (i / (dstSize.w * dstSize.x)) % dstSize.y;
        int    z        = i / (dstSize.w * dstSize.x * dstSize.y);
        float2 srcCoord = {x + shiftX, y + shiftY};

        test::ValueAt<ValueType>(gold, dstStrides, int4{k, x, y, z}) = test::GoldInterp<kInterpType, kBorderType>(
            srcVec, srcStrides, srcSize2, borderValue, scale, srcCoord, z, k);
    }

    VEC_EXPECT_NEAR(test, gold, 1);
}
