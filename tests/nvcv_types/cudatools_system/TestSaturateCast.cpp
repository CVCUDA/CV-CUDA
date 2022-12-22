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

#include "DeviceSaturateCast.hpp" // to test in the device

#include <common/TypedTests.hpp>      // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/cuda/MathOps.hpp>      // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/SaturateCast.hpp> // the object of this test

namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

template<typename T>
constexpr T min = std::numeric_limits<T>::min();
template<typename T>
constexpr T max = std::numeric_limits<T>::max();

// -------------------------- Testing SaturateCast -----------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    SaturateCastTest, ttype::Types<
    // identity regular C types do not change
    ttype::Types<char, ttype::Value<char{123}>, ttype::Value<char{123}>>,
    ttype::Types<short, ttype::Value<short{-1234}>, ttype::Value<short{-1234}>>,
    ttype::Types<int, ttype::Value<int{-123456}>, ttype::Value<int{-123456}>>,
    ttype::Types<float, ttype::Value<float{-1.23456f}>, ttype::Value<float{-1.23456f}>>,
    ttype::Types<double, ttype::Value<double{1.23456}>, ttype::Value<double{1.23456}>>,
    // float -> float
    ttype::Types<float, ttype::Value<double3{-1.5, 0.0, 1.5}>, ttype::Value<float3{-1.5f, 0.f, 1.5f}>>,
    ttype::Types<double, ttype::Value<float3{-1.5f, 0.f, 1.5f}>, ttype::Value<double3{-1.5, 0.0, 1.5}>>,
    // int -> float
    ttype::Types<float, ttype::Value<char4{-128, -127, 0, 127}>, ttype::Value<float4{-128.f, -127.f, 0.f, 127.f}>>,
    ttype::Types<float, ttype::Value<ushort3{0, 123, 456}>, ttype::Value<float3{0.f, 123.f, 456.f}>>,
    ttype::Types<double, ttype::Value<uchar2{0, 255}>, ttype::Value<double2{0, 255.0}>>,
    ttype::Types<double, ttype::Value<int2{-1234, 1234}>, ttype::Value<double2{-1234.0, 1234.0}>>,
    // float -> int
    ttype::Types<signed char, ttype::Value<float2{-345.67f, 456.78f}>, ttype::Value<char2{min<signed char>, max<signed char>}>>,
    ttype::Types<unsigned short, ttype::Value<float2{-0.1f, 1.6f}>, ttype::Value<ushort2{0, 2}>>,
    ttype::Types<int, ttype::Value<float2{-1.1f, 0.6f}>, ttype::Value<int2{-1, 1}>>,
    ttype::Types<unsigned int, ttype::Value<float2{-3.3f, 1.4f}>, ttype::Value<uint2{0, 1}>>,
    ttype::Types<unsigned char, ttype::Value<double2{-0.3, 256.1}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<double2{-0.7, 345.67}>, ttype::Value<char2{-1, max<signed char>}>>,
    ttype::Types<short, ttype::Value<double2{-1.4, 1234567.8}>, ttype::Value<short2{-1, max<short>}>>,
    // int -> int, from small to big and equal
    ttype::Types<short, ttype::Value<char1{123}>, ttype::Value<short1{123}>>,
    ttype::Types<unsigned long long, ttype::Value<ulong2{0, max<unsigned long>}>, ttype::Value<ulonglong2{0, max<unsigned long>}>>,
    ttype::Types<long long, ttype::Value<long2{-1234567, 1234567}>, ttype::Value<longlong2{-1234567, 1234567}>>,
    ttype::Types<unsigned short, ttype::Value<char3{-128, 0, 127}>, ttype::Value<ushort3{0, 0, 127}>>,
    ttype::Types<short, ttype::Value<uchar2{0, 255}>, ttype::Value<short2{0, 255}>>,
    ttype::Types<unsigned char, ttype::Value<char4{-128, -127, 0, 127}>, ttype::Value<uchar4{0, 0, 0, 127}>>,
    ttype::Types<signed char, ttype::Value<uchar3{0, 1, 255}>, ttype::Value<char3{0, 1, max<signed char>}>>,
    // int -> int, from big to small
    ttype::Types<short, ttype::Value<int1{1234567}>, ttype::Value<short1{max<short>}>>,
    ttype::Types<short, ttype::Value<uint2{0, 1234567}>, ttype::Value<short2{0, max<short>}>>,
    ttype::Types<unsigned short, ttype::Value<int3{-1234, 0, 1234567}>, ttype::Value<ushort3{0, 0, max<unsigned short>}>>,
    ttype::Types<unsigned char, ttype::Value<int2{-1234, 1234}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<uint2{0, 1234567}>, ttype::Value<char2{0, max<signed char>}>>,
    ttype::Types<unsigned char, ttype::Value<ulonglong2{0, 123456789}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<long2{-1234567, 1234567}>, ttype::Value<char2{-128, 127}>>
    >);

// clang-format on

TYPED_TEST(SaturateCastTest, correct_output_in_host)
{
    using TargetBaseType = ttype::GetType<TypeParam, 0>;
    auto input           = ttype::GetValue<TypeParam, 1>;
    auto gold            = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::SaturateCast<TargetBaseType>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}

TYPED_TEST(SaturateCastTest, correct_output_in_device)
{
    using TargetBaseType = ttype::GetType<TypeParam, 0>;
    auto input           = ttype::GetValue<TypeParam, 1>;
    auto gold            = ttype::GetValue<TypeParam, 2>;
    using InputType      = decltype(input);
    using TargetDataType = cuda::ConvertBaseTypeTo<TargetBaseType, InputType>;

    auto test = DeviceRunSaturateCast<TargetDataType>(input);

    EXPECT_TRUE((std::is_same_v<decltype(test), decltype(gold)>));
    EXPECT_EQ(test, gold);
}
