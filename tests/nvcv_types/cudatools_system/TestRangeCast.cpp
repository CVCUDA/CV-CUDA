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

#include <common/TypedTests.hpp>   // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/cuda/MathOps.hpp>   // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/RangeCast.hpp> // the object of this test

#include <cmath>  // for std::round, etc.
#include <limits> // for std::numeric_limits, etc.

namespace t     = ::testing;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

template<typename T>
constexpr T min = std::numeric_limits<T>::min();
template<typename T>
constexpr T max = std::numeric_limits<T>::max();

// ---------------------------- Testing RangeCast ------------------------------

// clang-format off

NVCV_TYPED_TEST_SUITE(
    RangeCastTest, ttype::Types<
    // identity regular C types do not change for non-float, and clamps for float
    ttype::Types<char, ttype::Value<char{123}>, ttype::Value<char{123}>>,
    ttype::Types<short, ttype::Value<int{-1234}>, ttype::Value<short{-1234}>>,
    ttype::Types<int, ttype::Value<int{-123456}>, ttype::Value<int{-123456}>>,
    ttype::Types<float, ttype::Value<float{-1.23456f}>, ttype::Value<float{-1.23456f}>>,
    ttype::Types<double, ttype::Value<double{1.23456}>, ttype::Value<double{1.23456}>>,
    // float -> float
    ttype::Types<float, ttype::Value<double4{-max<double>, -0.5, 1.5, max<double>}>,
                        ttype::Value<float4{-max<float>, -.5f, 1.5f, max<float>}>>,
    ttype::Types<double, ttype::Value<float4{-max<float>, 0.f, 1.5f, max<float>}>,
                         ttype::Value<double4{-max<float>, 0.0, 1.5, max<float>}>>,
    // int -> float
    ttype::Types<float, ttype::Value<char4{-128, -127, 0, 127}>, ttype::Value<float4{-1.f, -1.f, 0.f, 1.f}>>,
    ttype::Types<float, ttype::Value<char2{-12, 123}>, ttype::Value<float2{-12.f/127, 123.f/127}>>,
    ttype::Types<float, ttype::Value<ushort4{0, 123, 234, max<unsigned short>}>,
                        ttype::Value<float4{0.f, 123.f/max<unsigned short>, 234.f/max<unsigned short>, 1.f}>>,
    ttype::Types<double, ttype::Value<uchar2{0, max<unsigned char>}>, ttype::Value<double2{0.0, 1.0}>>,
    ttype::Types<double, ttype::Value<int2{min<int>, max<int>}>, ttype::Value<double2{-1.0, 1.0}>>,
    // float -> int
    ttype::Types<signed char, ttype::Value<float4{-max<float>, -3.456f, 2.34f, max<float>}>,
                              ttype::Value<char4{-max<signed char>, -max<signed char>, max<signed char>, max<signed char>}>>,
    ttype::Types<signed char, ttype::Value<float2{-0.123f, 0.456f}>,
                              ttype::Value<char2{static_cast<signed char>(std::round(-0.123f*max<signed char>)),
                                                 static_cast<signed char>(std::round(0.456f*max<signed char>))}>>,
    ttype::Types<unsigned short, ttype::Value<float4{min<float>, -0.123f, 0.456f, 1.2f}>,
                                 ttype::Value<ushort4{0, 0, static_cast<unsigned short>(std::round(0.456f*max<unsigned short>)),
                                                      max<unsigned short>}>>,
    ttype::Types<int, ttype::Value<float2{-1.23f+min<int>, 123.456f+max<int>}>, ttype::Value<int2{-max<int>, max<int>}>>,
    ttype::Types<unsigned int, ttype::Value<float2{-3.3f, max<float>}>, ttype::Value<uint2{0, max<unsigned int>}>>,
    ttype::Types<unsigned char, ttype::Value<double2{-0.3, 256.1}>, ttype::Value<uchar2{0, max<unsigned char>}>>,
    ttype::Types<signed char, ttype::Value<double2{-1.7, 133.3}>, ttype::Value<char2{-max<signed char>, max<signed char>}>>,
    ttype::Types<short, ttype::Value<double2{-1.4, 1234567.8}>, ttype::Value<short2{-max<short>, max<short>}>>,
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

TYPED_TEST(RangeCastTest, correct_output)
{
    using TargetBaseType = ttype::GetType<TypeParam, 0>;

    auto input = ttype::GetValue<TypeParam, 1>;
    auto gold  = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::RangeCast<TargetBaseType>(input);

    using TestType = decltype(test);
    using GoldType = decltype(gold);

    EXPECT_TRUE((std::is_same_v<TestType, GoldType>));
    EXPECT_EQ(test, gold);
}

// -------------------------- Testing corner cases -----------------------------

TEST(RangeCastCornerCasesTest, float_to_char)
{
#if CHAR_MIN == 0 // char with no sign?
    EXPECT_EQ((unsigned char)0, cuda::RangeCast<unsigned char>(0.f));
    EXPECT_EQ((unsigned char)0, cuda::RangeCast<unsigned char>(-1.f));
    EXPECT_EQ((unsigned char)128, cuda::RangeCast<unsigned char>(0.5f));
    EXPECT_EQ((unsigned char)255, cuda::RangeCast<unsigned char>(1.f));
    EXPECT_EQ((unsigned char)255, cuda::RangeCast<unsigned char>(1.1f));
#else
    EXPECT_EQ((char)0, cuda::RangeCast<char>(0.f));
    EXPECT_EQ((char)64, cuda::RangeCast<char>(0.5f));
    EXPECT_EQ((char)127, cuda::RangeCast<char>(1.f));
    EXPECT_EQ((char)127, cuda::RangeCast<char>(1.1f));
    EXPECT_EQ((char)-64, cuda::RangeCast<char>(-0.5f));
    EXPECT_EQ((char)-127, cuda::RangeCast<char>(-1.f));
    EXPECT_EQ((char)-127, cuda::RangeCast<char>(-1.1f));
#endif
}

TEST(RangeCastCornerCasesTest, char_to_float)
{
#if CHAR_MIN == 0 // char with no sign?
    EXPECT_EQ(0.f, cuda::RangeCast<float>((unsigned char)0));
    EXPECT_EQ(0.50196078431f, cuda::RangeCast<float>((unsigned char)128));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((unsigned char)255));
#else
    EXPECT_EQ(0.f, cuda::RangeCast<float>((char)0));
    EXPECT_EQ(0.503937f, cuda::RangeCast<float>((char)64));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((char)127));
    EXPECT_EQ(-0.503937f, cuda::RangeCast<float>((char)-64));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((char)-127));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((char)-128));
#endif
}

TEST(RangeCastCornerCasesTest, float_to_signed_char)
{
    EXPECT_EQ((signed char)0, cuda::RangeCast<signed char>(0.f));
    EXPECT_EQ((signed char)64, cuda::RangeCast<signed char>(0.5f));
    EXPECT_EQ((signed char)127, cuda::RangeCast<signed char>(1.f));
    EXPECT_EQ((signed char)127, cuda::RangeCast<signed char>(1.1f));
    EXPECT_EQ((signed char)-64, cuda::RangeCast<signed char>(-0.5f));
    EXPECT_EQ((signed char)-127, cuda::RangeCast<signed char>(-1.f));
    EXPECT_EQ((signed char)-127, cuda::RangeCast<signed char>(-1.1f));
}

TEST(RangeCastCornerCasesTest, signed_char_to_float)
{
    EXPECT_EQ(0.f, cuda::RangeCast<float>((signed char)0));
    EXPECT_EQ(0.503937f, cuda::RangeCast<float>((signed char)64));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((signed char)127));
    EXPECT_EQ(-0.503937f, cuda::RangeCast<float>((signed char)-64));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((signed char)-127));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((signed char)-128));
}

TEST(RangeCastCornerCasesTest, float_to_unsigned_char)
{
    EXPECT_EQ((unsigned char)0, cuda::RangeCast<unsigned char>(0.f));
    EXPECT_EQ((unsigned char)0, cuda::RangeCast<unsigned char>(-1.f));
    EXPECT_EQ((unsigned char)128, cuda::RangeCast<unsigned char>(0.5f));
    EXPECT_EQ((unsigned char)255, cuda::RangeCast<unsigned char>(1.f));
    EXPECT_EQ((unsigned char)255, cuda::RangeCast<unsigned char>(1.1f));
}

TEST(RangeCastCornerCasesTest, unsigned_char_to_float)
{
    EXPECT_EQ(0.f, cuda::RangeCast<float>((unsigned char)0));
    EXPECT_EQ(0.50196078431f, cuda::RangeCast<float>((unsigned char)128));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((unsigned char)255));
}

TEST(RangeCastCornerCasesTest, float_to_short)
{
    EXPECT_EQ((short)0, cuda::RangeCast<short>(0.f));
    EXPECT_EQ((short)16384, cuda::RangeCast<short>(0.5f));
    EXPECT_EQ((short)32767, cuda::RangeCast<short>(1.f));
    EXPECT_EQ((short)32767, cuda::RangeCast<short>(1.1f));
    EXPECT_EQ((short)-16384, cuda::RangeCast<short>(-0.5f));
    EXPECT_EQ((short)-32767, cuda::RangeCast<short>(-1.f));
    EXPECT_EQ((short)-32767, cuda::RangeCast<short>(-1.1f));
}

TEST(RangeCastCornerCasesTest, short_to_float)
{
    EXPECT_EQ(0.f, cuda::RangeCast<float>((short)0));
    EXPECT_EQ(0.5000152592f, cuda::RangeCast<float>((short)16384));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((short)32767));
    EXPECT_EQ(-0.5000152592f, cuda::RangeCast<float>((short)-16384));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((short)-32767));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((short)-32768));
}

TEST(RangeCastCornerCasesTest, float_to_unsigned_short)
{
    EXPECT_EQ((unsigned short)0, cuda::RangeCast<unsigned short>(0.f));
    EXPECT_EQ((unsigned short)0, cuda::RangeCast<unsigned short>(-1.f));
    EXPECT_EQ((unsigned short)32768, cuda::RangeCast<unsigned short>(0.5f));
    EXPECT_EQ((unsigned short)65535, cuda::RangeCast<unsigned short>(1.f));
    EXPECT_EQ((unsigned short)65535, cuda::RangeCast<unsigned short>(1.1f));
}

TEST(RangeCastCornerCasesTest, unsigned_short_to_float)
{
    EXPECT_EQ(0.f, cuda::RangeCast<float>((unsigned short)0));
    EXPECT_EQ(0.5000076295109f, cuda::RangeCast<float>((unsigned short)32768));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((unsigned short)65535));
}

TEST(RangeCastCornerCasesTest, float_to_int)
{
    EXPECT_EQ((int)0, cuda::RangeCast<int>(0.f));
    EXPECT_EQ((int)1073741824, cuda::RangeCast<int>(0.5f));
    EXPECT_EQ((int)2147483647, cuda::RangeCast<int>(1.f));
    EXPECT_EQ((int)2147483647, cuda::RangeCast<int>(1.1f));
    EXPECT_EQ((int)-1073741824, cuda::RangeCast<int>(-0.5f));
    EXPECT_EQ((int)-2147483647, cuda::RangeCast<int>(-1.f));
    EXPECT_EQ((int)-2147483647, cuda::RangeCast<int>(-1.1f));
}

TEST(RangeCastCornerCasesTest, int_to_float)
{
    EXPECT_EQ(0.f, cuda::RangeCast<float>((int)0));
    EXPECT_EQ(0.5f, cuda::RangeCast<float>((int)1073741824));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((int)2147483647));
    EXPECT_EQ(-0.5f, cuda::RangeCast<float>((int)-1073741824));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((int)-2147483647));
    EXPECT_EQ(-1.f, cuda::RangeCast<float>((int)-2147483648));
}

TEST(RangeCastCornerCasesTest, float_to_unsigned_int)
{
    EXPECT_EQ((unsigned int)0, cuda::RangeCast<unsigned int>(0.f));
    EXPECT_EQ((unsigned int)0, cuda::RangeCast<unsigned int>(-1.f));
    EXPECT_EQ((unsigned int)2147483648, cuda::RangeCast<unsigned int>(0.5f));
    EXPECT_EQ((unsigned int)4294967295, cuda::RangeCast<unsigned int>(1.f));
    EXPECT_EQ((unsigned int)4294967295, cuda::RangeCast<unsigned int>(1.1f));
}

TEST(RangeCastCornerCasesTest, unsigned_int_to_float)
{
    EXPECT_EQ(0.f, cuda::RangeCast<float>((unsigned int)0));
    EXPECT_EQ(0.5f, cuda::RangeCast<float>((unsigned int)2147483648));
    EXPECT_EQ(1.f, cuda::RangeCast<float>((unsigned int)4294967295));
}

TEST(RangeCastCornerCasesTest, composite_types)
{
    EXPECT_EQ(make_char2(0, 0), cuda::RangeCast<signed char>(make_float2(0, 0)));
    EXPECT_EQ(make_char2(127, -127), cuda::RangeCast<signed char>(make_float2(1, -1)));
}

TEST(RangeCastCornerCasesTest, identity_values)
{
    EXPECT_EQ(make_float2(142, 23), cuda::RangeCast<float>(make_float2(142, 23)));
    EXPECT_EQ(make_int2(142, 23), cuda::RangeCast<int>(make_int2(142, 23)));
}
