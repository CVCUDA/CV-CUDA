/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceMathWrappers.hpp" // to test in the device

#include <common/TypedTests.hpp>      // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/cuda/MathOps.hpp>      // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/MathWrappers.hpp> // the object of this test

namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

template<typename T>
constexpr T epsilon = std::numeric_limits<T>::epsilon();

using schar = signed char;
using uchar = unsigned char;
using uint  = unsigned int;

// ----------------------------- Testing round ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersRoundSameTypeTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{123}>, ttype::Value<uchar{123}>, ttype::Value<cuda::RoundMode::UP>>,
    ttype::Types<ttype::Value<int{-123456}>, ttype::Value<int{-123456}>, ttype::Value<cuda::RoundMode::DOWN>>,
    ttype::Types<ttype::Value<float{-1.23456f}>, ttype::Value<float{-1.f}>, ttype::Value<cuda::RoundMode::DEFAULT>>,
    ttype::Types<ttype::Value<double{1.23456}>, ttype::Value<double{1.0}>, ttype::Value<cuda::RoundMode::NEAREST>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{123}>, ttype::Value<char1{123}>, ttype::Value<cuda::RoundMode::UP>>,
    ttype::Types<ttype::Value<uint2{0, 123456}>, ttype::Value<uint2{0, 123456}>, ttype::Value<cuda::RoundMode::DOWN>>,
    ttype::Types<ttype::Value<float3{-1.23456f, 0.789f, 3.456f}>,
                 ttype::Value<float3{-1.f, 1.f, 3.f}>, ttype::Value<cuda::RoundMode::DEFAULT>>,
    ttype::Types<ttype::Value<double4{1.23456, 6.789, epsilon<double>, -4.567}>,
                 ttype::Value<double4{1.0, 7.0, 0.0, -5.0}>, ttype::Value<cuda::RoundMode::NEAREST>>,
    // unusual round modes
    ttype::Types<ttype::Value<float2{-1.23456f, 0.789f}>, ttype::Value<float2{-1.f, 1.f}>, ttype::Value<cuda::RoundMode::UP>>,
    ttype::Types<ttype::Value<double2{1.789, -6.123}>, ttype::Value<double2{1.0, -7.0}>, ttype::Value<cuda::RoundMode::DOWN>>,
    ttype::Types<ttype::Value<float3{-0.999f, -0.23456f, 0.789f}>,
                 ttype::Value<float3{0.f, 0.f, 0.f}>, ttype::Value<cuda::RoundMode::ZERO>>,
    ttype::Types<ttype::Value<float4{-.5f, -1.5f, 2.5f, -3.5f}>,
                 ttype::Value<float4{0.f, -2.f, 2.f, -4.f}>, ttype::Value<cuda::RoundMode::NEAREST>>
    >);

// clang-format on

TYPED_TEST(MathWrappersRoundSameTypeTest, correct_output_in_host)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    constexpr cuda::RoundMode RM = ttype::GetValue<TypeParam, 2>;

    auto test = (RM == cuda::RoundMode::DEFAULT) ? cuda::round(input) : cuda::round<RM>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersRoundSameTypeTest, correct_output_in_device)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    constexpr cuda::RoundMode RM = ttype::GetValue<TypeParam, 2>;

    auto test = DeviceRunRoundSameType<RM>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersRoundDiffTypeTest, ttype::Types<
    // regular C types passing different type
    ttype::Types<int, ttype::Value<float{-1.23456f}>, ttype::Value<int{-1}>, ttype::Value<cuda::RoundMode::UP>>,
    ttype::Types<uint, ttype::Value<double{1.23456}>, ttype::Value<uint{1}>, ttype::Value<cuda::RoundMode::DOWN>>,
    // CUDA compound types passing different type
    ttype::Types<int, ttype::Value<float3{-1.23456f, 0.789f, 3.456f}>,
                      ttype::Value<int3{-1, 1, 3}>, ttype::Value<cuda::RoundMode::NEAREST>>,
    ttype::Types<long, ttype::Value<double4{1.23456, 6.789, epsilon<double>, -4.567}>,
                       ttype::Value<long4{1, 7, 0, -5}>, ttype::Value<cuda::RoundMode::DEFAULT>>,
    // regular C types and CUDA compound types passing same type
    ttype::Types<schar, ttype::Value<schar{123}>, ttype::Value<schar{123}>, ttype::Value<cuda::RoundMode::UP>>,
    ttype::Types<float, ttype::Value<float2{-4.56f, 7.89f}>,
                        ttype::Value<float2{-5.f, 8.f}>, ttype::Value<cuda::RoundMode::DEFAULT>>,
    ttype::Types<uint, ttype::Value<uint1{123456}>, ttype::Value<uint1{123456}>, ttype::Value<cuda::RoundMode::DOWN>>,
    ttype::Types<double2, ttype::Value<double2{1.23, epsilon<double>}>,
                          ttype::Value<double2{1.0, 0.0}>, ttype::Value<cuda::RoundMode::NEAREST>>,
    // unusual round modes
    ttype::Types<int2, ttype::Value<float2{-1.789f, 0.123f}>, ttype::Value<int2{-1, 1}>, ttype::Value<cuda::RoundMode::UP>>,
    ttype::Types<long long int, ttype::Value<double2{6.789, -1.123}>,
                                ttype::Value<longlong2{6, -2}>, ttype::Value<cuda::RoundMode::DOWN>>,
    ttype::Types<short, ttype::Value<float3{0.999f, -0.789f, 0.123f}>,
                        ttype::Value<short3{0, 0, 0}>, ttype::Value<cuda::RoundMode::ZERO>>
    >);

// clang-format on

TYPED_TEST(MathWrappersRoundDiffTypeTest, correct_output_in_host)
{
    using TargetType = ttype::GetType<TypeParam, 0>;
    auto input       = ttype::GetValue<TypeParam, 1>;
    auto gold        = ttype::GetValue<TypeParam, 2>;

    constexpr cuda::RoundMode RM = ttype::GetValue<TypeParam, 3>;

    auto test = (RM == cuda::RoundMode::DEFAULT) ? cuda::round<TargetType>(input) : cuda::round<RM, TargetType>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersRoundDiffTypeTest, correct_output_in_device)
{
    using TargetType = ttype::GetType<TypeParam, 0>;
    auto input       = ttype::GetValue<TypeParam, 1>;
    auto gold        = ttype::GetValue<TypeParam, 2>;

    constexpr cuda::RoundMode RM = ttype::GetValue<TypeParam, 3>;

    using SourceDataType = decltype(input);
    using TargetBaseType = cuda::BaseType<TargetType>;
    using TargetDataType = cuda::ConvertBaseTypeTo<TargetBaseType, SourceDataType>;

    auto test = DeviceRunRoundDiffType<RM, TargetDataType>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------- Testing min ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersMinTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{1}>, ttype::Value<uchar{2}>, ttype::Value<uchar{1}>>,
    ttype::Types<ttype::Value<int{2}>, ttype::Value<int{-2}>, ttype::Value<int{-2}>>,
    ttype::Types<ttype::Value<float{-1.23f}>, ttype::Value<float{4.56f}>, ttype::Value<float{-1.23f}>>,
    ttype::Types<ttype::Value<double{1.2}>, ttype::Value<double{3.4}>, ttype::Value<double{1.2}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{1}>, ttype::Value<char1{2}>, ttype::Value<char1{1}>>,
    ttype::Types<ttype::Value<uint2{2, 3}>, ttype::Value<uint2{3, 2}>, ttype::Value<uint2{2, 2}>>,
    ttype::Types<ttype::Value<float3{1.23f, 2.34f, 3.45f}>, ttype::Value<float3{2.34f, -1.23f, 4.56f}>,
                 ttype::Value<float3{1.23f, -1.23f, 3.45f}>>,
    ttype::Types<ttype::Value<double4{1.2, 2.3, -3.4, 5.6}>, ttype::Value<double4{2.3, 3.4, -4.5, 0.1}>,
                 ttype::Value<double4{1.2, 2.3, -4.5, 0.1}>>,
    // CUDA compound types subject to SIMD
    ttype::Types<ttype::Value<short2{1234, -4567}>, ttype::Value<short2{-1234, 4567}>,
                 ttype::Value<short2{-1234, -4567}>>,
    ttype::Types<ttype::Value<char4{12, -34, 56, -78}>, ttype::Value<char4{-12, 34, -56, 78}>,
                 ttype::Value<char4{-12, -34, -56, -78}>>,
    ttype::Types<ttype::Value<ushort2{1234, 4567}>, ttype::Value<ushort2{4567, 1234}>,
                 ttype::Value<ushort2{1234, 1234}>>,
    ttype::Types<ttype::Value<uchar4{12, 34, 56, 78}>, ttype::Value<uchar4{21, 23, 61, 67}>,
                 ttype::Value<uchar4{12, 23, 56, 67}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersMinTest, correct_output_in_host)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::min(input1, input2);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersMinTest, correct_output_in_device)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test = DeviceRunMin(input1, input2);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------- Testing max ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersMaxTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{1}>, ttype::Value<uchar{2}>, ttype::Value<uchar{2}>>,
    ttype::Types<ttype::Value<int{2}>, ttype::Value<int{-2}>, ttype::Value<int{2}>>,
    ttype::Types<ttype::Value<float{-1.23f}>, ttype::Value<float{4.56f}>, ttype::Value<float{4.56f}>>,
    ttype::Types<ttype::Value<double{1.2}>, ttype::Value<double{3.4}>, ttype::Value<double{3.4}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{1}>, ttype::Value<char1{2}>, ttype::Value<char1{2}>>,
    ttype::Types<ttype::Value<uint2{2, 3}>, ttype::Value<uint2{3, 2}>, ttype::Value<uint2{3, 3}>>,
    ttype::Types<ttype::Value<float3{1.23f, 2.34f, 3.45f}>, ttype::Value<float3{2.34f, -1.23f, 4.56f}>,
                 ttype::Value<float3{2.34f, 2.34f, 4.56f}>>,
    ttype::Types<ttype::Value<double4{1.2, 2.3, -3.4, 5.6}>, ttype::Value<double4{2.3, 3.4, -4.5, 0.1}>,
                 ttype::Value<double4{2.3, 3.4, -3.4, 5.6}>>,
    // CUDA compound types subject to SIMD
    ttype::Types<ttype::Value<short2{1234, -4567}>, ttype::Value<short2{-1234, 4567}>,
                 ttype::Value<short2{1234, 4567}>>,
    ttype::Types<ttype::Value<char4{12, -34, 56, -78}>, ttype::Value<char4{-12, 34, -56, 78}>,
                 ttype::Value<char4{12, 34, 56, 78}>>,
    ttype::Types<ttype::Value<ushort2{1234, 4567}>, ttype::Value<ushort2{4567, 1234}>,
                 ttype::Value<ushort2{4567, 4567}>>,
    ttype::Types<ttype::Value<uchar4{12, 34, 56, 78}>, ttype::Value<uchar4{21, 23, 61, 67}>,
                 ttype::Value<uchar4{21, 34, 61, 78}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersMaxTest, correct_output_in_host)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::max(input1, input2);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersMaxTest, correct_output_in_device)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test = DeviceRunMax(input1, input2);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------- Testing pow ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersPowTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{2}>, ttype::Value<uchar{0}>, ttype::Value<uchar{1}>>,
    ttype::Types<ttype::Value<uchar{2}>, ttype::Value<int{4}>, ttype::Value<uchar{16}>>,
    ttype::Types<ttype::Value<float{-2.f}>, ttype::Value<float{3.f}>, ttype::Value<float{-8.f}>>,
    ttype::Types<ttype::Value<double{3.5}>, ttype::Value<uchar{2}>, ttype::Value<double{3.5 * 3.5}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{1}>, ttype::Value<char1{126}>, ttype::Value<char1{1}>>,
    ttype::Types<ttype::Value<uint2{2, 3}>, ttype::Value<uint2{3, 2}>, ttype::Value<uint2{8, 9}>>,
    ttype::Types<ttype::Value<float3{0.f, 1.f, 2.f}>, ttype::Value<int{2}>, ttype::Value<float3{0.f, 1.f, 4.f}>>,
    ttype::Types<ttype::Value<double4{1.0, -2.0, 4.0, -6.0}>, ttype::Value<float4{2.789f, 2.f, 1.f, 0.f}>,
                 ttype::Value<double4{1.0, 4.0, 4.0, 1.0}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersPowTest, correct_output_in_host)
{
    auto value = ttype::GetValue<TypeParam, 0>;
    auto power = ttype::GetValue<TypeParam, 1>;
    auto gold  = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::pow(value, power);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersPowTest, correct_output_in_device)
{
    auto value = ttype::GetValue<TypeParam, 0>;
    auto power = ttype::GetValue<TypeParam, 1>;
    auto gold  = ttype::GetValue<TypeParam, 2>;

    auto test = DeviceRunPow(value, power);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------- Testing exp ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersExpTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{1}>, ttype::Value<uchar{2}>>,
    ttype::Types<ttype::Value<int{2}>, ttype::Value<int{7}>>,
    ttype::Types<ttype::Value<float{0.f}>, ttype::Value<float{1.f}>>,
    ttype::Types<ttype::Value<double{0.0}>, ttype::Value<double{1.0}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{1}>, ttype::Value<char1{2}>>,
    ttype::Types<ttype::Value<uint2{2, 3}>, ttype::Value<uint2{7, 20}>>,
    ttype::Types<ttype::Value<float3{0.f, -0.f, 0.f}>, ttype::Value<float3{1.f, 1.f, 1.f}>>,
    ttype::Types<ttype::Value<double4{0.0, -0.0, 0.0, 0.0}>, ttype::Value<double4{1.0, 1.0, 1.0, 1.0}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersExpTest, correct_output_in_host)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = cuda::exp(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersExpTest, correct_output_in_device)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = DeviceRunExp(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------ Testing sqrt ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersSqrtTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{4}>, ttype::Value<uchar{2}>>,
    ttype::Types<ttype::Value<int{1}>, ttype::Value<int{1}>>,
    ttype::Types<ttype::Value<float{9.f}>, ttype::Value<float{3.f}>>,
    ttype::Types<ttype::Value<double{16.0}>, ttype::Value<double{4.0}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{4}>, ttype::Value<char1{2}>>,
    ttype::Types<ttype::Value<uint2{1, 4}>, ttype::Value<uint2{1, 2}>>,
    ttype::Types<ttype::Value<float3{4.f, 16.f, 25.f}>, ttype::Value<float3{2.f, 4.f, 5.f}>>,
    ttype::Types<ttype::Value<double4{36.0, 49.0, 64.0, 81.0}>, ttype::Value<double4{6.0, 7.0, 8.0, 9.0}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersSqrtTest, correct_output_in_host)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = cuda::sqrt(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersSqrtTest, correct_output_in_device)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = DeviceRunSqrt(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------- Testing abs ---------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersAbsTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{1}>, ttype::Value<uchar{1}>>,
    ttype::Types<ttype::Value<int{-1}>, ttype::Value<int{1}>>,
    ttype::Types<ttype::Value<float{-2.f}>, ttype::Value<float{2.f}>>,
    ttype::Types<ttype::Value<double{-3.0}>, ttype::Value<double{3.0}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{-1}>, ttype::Value<char1{1}>>,
    ttype::Types<ttype::Value<uint2{1, 2}>, ttype::Value<uint2{1, 2}>>,
    ttype::Types<ttype::Value<float3{-1.f, 2.f, -3.f}>, ttype::Value<float3{1.f, 2.f, 3.f}>>,
    ttype::Types<ttype::Value<double4{-4.0, -5.0, -6.0, 7.0}>, ttype::Value<double4{4.0, 5.0, 6.0, 7.0}>>,
    // CUDA compound types subject to SIMD
    ttype::Types<ttype::Value<short2{-1234, -4567}>, ttype::Value<short2{1234, 4567}>>,
    ttype::Types<ttype::Value<char4{-12, -34, -56, -78}>, ttype::Value<char4{12, 34, 56, 78}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersAbsTest, correct_output_in_host)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = cuda::abs(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersAbsTest, correct_output_in_device)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = DeviceRunAbs(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// ------------------------------ Testing clamp --------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    MathWrappersClampTest, ttype::Types<
    // regular C types
    ttype::Types<ttype::Value<uchar{1}>, ttype::Value<uchar{0}>, ttype::Value<uchar{6}>, ttype::Value<uchar{1}>>,
    ttype::Types<ttype::Value<int{-1}>, ttype::Value<uchar{0}>, ttype::Value<uchar{7}>, ttype::Value<int{0}>>,
    ttype::Types<ttype::Value<float{9.f}>, ttype::Value<ushort{0}>, ttype::Value<ushort{8}>, ttype::Value<float{8.f}>>,
    ttype::Types<ttype::Value<double{-0.123f}>, ttype::Value<double{0}>, ttype::Value<double{1}>, ttype::Value<double{0}>>,
    // CUDA compound types
    ttype::Types<ttype::Value<char1{-1}>, ttype::Value<char1{0}>, ttype::Value<char1{1}>, ttype::Value<char1{0}>>,
    ttype::Types<ttype::Value<uint2{2, 3}>, ttype::Value<uint2{0, 1}>, ttype::Value<uint2{1, 4}>, ttype::Value<uint2{1, 3}>>,
    ttype::Types<ttype::Value<float3{-1.f, 2.f, -3.f}>, ttype::Value<short{0}>, ttype::Value<short{1}>,
                 ttype::Value<float3{0.f, 1.f, 0.f}>>,
    ttype::Types<ttype::Value<double4{4.0, 5.0, 6.0, 7.0}>, ttype::Value<int4{-4, 6, -6, 0}>,
                 ttype::Value<int4{2, 7, 6, 8}>, ttype::Value<double4{2.0, 6.0, 6.0, 7.0}>>
    >);

// clang-format on

TYPED_TEST(MathWrappersClampTest, correct_output_in_host)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto low   = ttype::GetValue<TypeParam, 1>;
    auto high  = ttype::GetValue<TypeParam, 2>;
    auto gold  = ttype::GetValue<TypeParam, 3>;

    auto test = cuda::clamp(input, low, high);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(MathWrappersClampTest, correct_output_in_device)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto low   = ttype::GetValue<TypeParam, 1>;
    auto high  = ttype::GetValue<TypeParam, 2>;
    auto gold  = ttype::GetValue<TypeParam, 3>;

    auto test = DeviceRunClamp(input, low, high);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}
