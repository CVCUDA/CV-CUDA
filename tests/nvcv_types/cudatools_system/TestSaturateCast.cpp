/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceSaturateCast.hpp" // to test in the device

#include <common/TypedTests.hpp>              // for NVCV_TYPED_TEST_SUITE, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>      // for operator == to allow EXPECT_EQ
#include <cvcuda/cuda_tools/SaturateCast.hpp> // the object of this test

#include <cmath> // for std::isinf, etc.

namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

template<typename T>
constexpr T min = std::numeric_limits<T>::min();
template<typename T>
constexpr T max = std::numeric_limits<T>::max();

using schar = signed char;

// -------------------------- Testing SaturateCast -----------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    SaturateCastTest, ttype::Types<
    // identity regular C types do not change
    ttype::Types<schar, ttype::Value<schar{123}>, ttype::Value<schar{123}>>,
    ttype::Types<short, ttype::Value<short{-1234}>, ttype::Value<short{-1234}>>,
    ttype::Types<int, ttype::Value<int{-123456}>, ttype::Value<int{-123456}>>,
    ttype::Types<float, ttype::Value<float{-1.23456f}>, ttype::Value<float{-1.23456f}>>,
    ttype::Types<double, ttype::Value<double{1.23456}>, ttype::Value<double{1.23456}>>,
    // float -> float
    ttype::Types<float3, ttype::Value<double3{-1.5, 0.0, 1.5}>, ttype::Value<float3{-1.5f, 0.f, 1.5f}>>,
    ttype::Types<double, ttype::Value<float3{-1.5f, 0.f, 1.5f}>, ttype::Value<double3{-1.5, 0.0, 1.5}>>,
    // int -> float
    ttype::Types<float4, ttype::Value<char4{-128, -127, 0, 127}>, ttype::Value<float4{-128.f, -127.f, 0.f, 127.f}>>,
    ttype::Types<float, ttype::Value<ushort3{0, 123, 456}>, ttype::Value<float3{0.f, 123.f, 456.f}>>,
    ttype::Types<double2, ttype::Value<uchar2{0, 255}>, ttype::Value<double2{0, 255.0}>>,
    ttype::Types<double, ttype::Value<int2{-1234, 1234}>, ttype::Value<double2{-1234.0, 1234.0}>>,
    // float -> int
    ttype::Types<signed char, ttype::Value<float2{-345.67f, 456.78f}>, ttype::Value<char2{min<signed char>, max<signed char>}>>,
    ttype::Types<unsigned short, ttype::Value<float2{-0.1f, 1.6f}>, ttype::Value<ushort2{0, 2}>>,
    ttype::Types<int2, ttype::Value<float2{-1.1f, 0.6f}>, ttype::Value<int2{-1, 1}>>,
    ttype::Types<unsigned int, ttype::Value<float2{-3.3f, 1.4f}>, ttype::Value<uint2{0, 1}>>,
    ttype::Types<unsigned char, ttype::Value<double2{-0.3, 256.1}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<double2{-0.7, 345.67}>, ttype::Value<char2{-1, max<signed char>}>>,
    ttype::Types<short2, ttype::Value<double2{-1.4, 1234567.8}>, ttype::Value<short2{-1, max<short>}>>,
    // int -> int, from small to big and equal
    ttype::Types<short1, ttype::Value<char1{123}>, ttype::Value<short1{123}>>,
    ttype::Types<unsigned long long, ttype::Value<ulong2{0, max<unsigned long>}>, ttype::Value<ulonglong2{0, max<unsigned long>}>>,
    ttype::Types<long long, ttype::Value<long2{-1234567, 1234567}>, ttype::Value<longlong2{-1234567, 1234567}>>,
    ttype::Types<unsigned short, ttype::Value<char3{-128, 0, 127}>, ttype::Value<ushort3{0, 0, 127}>>,
    ttype::Types<short2, ttype::Value<uchar2{0, 255}>, ttype::Value<short2{0, 255}>>,
    ttype::Types<unsigned char, ttype::Value<char4{-128, -127, 0, 127}>, ttype::Value<uchar4{0, 0, 0, 127}>>,
    ttype::Types<signed char, ttype::Value<uchar3{0, 1, 255}>, ttype::Value<char3{0, 1, max<signed char>}>>,
    // int -> int, from big to small
    ttype::Types<short1, ttype::Value<int1{1234567}>, ttype::Value<short1{max<short>}>>,
    ttype::Types<short2, ttype::Value<uint2{0, 1234567}>, ttype::Value<short2{0, max<short>}>>,
    ttype::Types<unsigned short, ttype::Value<int3{-1234, 0, 1234567}>, ttype::Value<ushort3{0, 0, max<unsigned short>}>>,
    ttype::Types<unsigned char, ttype::Value<int2{-1234, 1234}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<uint2{0, 1234567}>, ttype::Value<char2{0, max<signed char>}>>,
    ttype::Types<unsigned char, ttype::Value<ulonglong2{0, 123456789}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<long2{-1234567, 1234567}>, ttype::Value<char2{-128, 127}>>
    >);

// clang-format on

TYPED_TEST(SaturateCastTest, correct_output_in_host)
{
    using TargetType = ttype::GetType<TypeParam, 0>;
    auto input       = ttype::GetValue<TypeParam, 1>;
    auto gold        = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::SaturateCast<TargetType>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(SaturateCastTest, correct_output_in_device)
{
    using TargetType     = ttype::GetType<TypeParam, 0>;
    auto input           = ttype::GetValue<TypeParam, 1>;
    auto gold            = ttype::GetValue<TypeParam, 2>;
    using InputType      = decltype(input);
    using TargetDataType = cuda::ConvertBaseTypeTo<cuda::BaseType<TargetType>, InputType>;

    auto test = DeviceRunSaturateCast<TargetDataType>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

// --------------------- Testing SaturateCast for __half -----------------------

// __half and its vector types (half1/__half2/half3/half4) are not structural types, so they
// cannot be used in ttype::Value<> non-type template parameters as in the typed suite above;
// the fp16 coverage below uses plain TESTs with values built at run time.

TEST(SaturateCastHalfTest, half_to_uchar_rounds_to_nearest_even_and_saturates)
{
    // Device uses cvt.rni.sat.u8.f16 PTX and host goes through float; both must round to
    // nearest even and then saturate to the target range, giving identical results
    half4  input{__float2half(1.5f), __float2half(2.5f), __float2half(-2.f), __float2half(300.f)};
    uchar4 gold{2, 2, 0, 255};

    auto host = cuda::SaturateCast<uchar4>(input);
    auto dev  = DeviceRunSaturateCast<uchar4>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), uchar4>));
    EXPECT_EQ(host, gold);
    EXPECT_EQ(dev, gold);
}

TEST(SaturateCastHalfTest, half_to_schar_rounds_to_nearest_even_and_saturates)
{
    half3 input{__float2half(2.5f), __float2half(127.5f), __float2half(-200.f)};
    char3 gold{2, 127, -128};

    auto host = cuda::SaturateCast<char3>(input);
    auto dev  = DeviceRunSaturateCast<char3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), char3>));
    EXPECT_EQ(host, gold);
    EXPECT_EQ(dev, gold);
}

TEST(SaturateCastHalfTest, half_to_ushort_rounds_to_nearest_even_and_saturates)
{
    half3   input{__float2half(1.5f), __float2half(-2.f), cuda::HalfMax()};
    ushort3 gold{2, 0, 65504};

    auto host = cuda::SaturateCast<ushort3>(input);
    auto dev  = DeviceRunSaturateCast<ushort3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), ushort3>));
    EXPECT_EQ(host, gold);
    EXPECT_EQ(dev, gold);
}

TEST(SaturateCastHalfTest, half_to_short_rounds_to_nearest_even_and_saturates)
{
    half3  input{__float2half(1.5f), __float2half(-1.5f), cuda::HalfMax()};
    short3 gold{2, -2, 32767};

    auto host = cuda::SaturateCast<short3>(input);
    auto dev  = DeviceRunSaturateCast<short3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), short3>));
    EXPECT_EQ(host, gold);
    EXPECT_EQ(dev, gold);
}

TEST(SaturateCastHalfTest, half_to_int_goes_through_float)
{
    // __half -> int has no PTX specialization; both sides hop through float (lossless) and
    // use the round-to-nearest-even plus clamp path
    half3 input{__float2half(-1.5f), __float2half(2.5f), cuda::HalfMax()};
    int3  gold{-2, 2, 65504};

    auto host = cuda::SaturateCast<int3>(input);
    auto dev  = DeviceRunSaturateCast<int3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), int3>));
    EXPECT_EQ(host, gold);
    EXPECT_EQ(dev, gold);
}

TEST(SaturateCastHalfTest, half_to_float_is_exact)
{
    half3  input{__float2half(-1.5f), __float2half(0.25f), cuda::HalfMax()};
    float3 gold{-1.5f, 0.25f, 65504.f};

    auto host = cuda::SaturateCast<float>(input);
    auto dev  = DeviceRunSaturateCast<float3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), float3>));
    EXPECT_EQ(host, gold);
    EXPECT_EQ(dev, gold);
}

TEST(SaturateCastHalfTest, float_to_half_is_plain_conversion_without_saturation)
{
    // any -> any-float reduces to no saturation: float -> __half is a single round-to-nearest
    // conversion, so out-of-range values correctly become infinity instead of clamping
    float3 input{1.e9f, -1.e9f, 0.5f};

    auto host = cuda::SaturateCast<half3>(input);
    auto dev  = DeviceRunSaturateCast<half3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), half3>));
    EXPECT_TRUE(std::isinf(__half2float(host.x)) && __half2float(host.x) > 0.f);
    EXPECT_TRUE(std::isinf(__half2float(host.y)) && __half2float(host.y) < 0.f);
    EXPECT_EQ(__half2float(host.z), 0.5f);
    EXPECT_EQ(dev, host);
}

TEST(SaturateCastHalfTest, double_to_half_is_plain_conversion_without_saturation)
{
    double3 input{0.1, -2.5, 1.e9};

    auto host = cuda::SaturateCast<__half>(input);
    auto dev  = DeviceRunSaturateCast<half3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), half3>));
    EXPECT_EQ(__half2float(host.x), __half2float(__double2half(0.1)));
    EXPECT_EQ(__half2float(host.y), -2.5f);
    EXPECT_TRUE(std::isinf(__half2float(host.z)) && __half2float(host.z) > 0.f);
    EXPECT_EQ(dev, host);
}

TEST(SaturateCastHalfTest, int_to_half_is_plain_conversion_without_saturation)
{
    int3 input{-3, 0, 70000};

    auto host = cuda::SaturateCast<half3>(input);
    auto dev  = DeviceRunSaturateCast<half3>(input);

    EXPECT_TRUE((std::is_same_v<decltype(host), half3>));
    EXPECT_EQ(__half2float(host.x), -3.f);
    EXPECT_EQ(__half2float(host.y), 0.f);
    EXPECT_TRUE(std::isinf(__half2float(host.z)) && __half2float(host.z) > 0.f);
    EXPECT_EQ(dev, host);
}

TEST(SaturateCastHalfTest, half_to_half_is_identity)
{
    auto test = cuda::SaturateCast<__half>(__float2half(1.5f));

    EXPECT_TRUE((std::is_same_v<decltype(test), __half>));
    EXPECT_EQ(__half2float(test), 1.5f);
}
