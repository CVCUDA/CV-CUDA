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

#include <common/TypedTests.hpp>         // for NVCV_TYPED_TEST_SUITE_F, etc.
#include <cvcuda/cuda_tools/MathOps.hpp> // the object of this test

namespace t     = ::testing;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

template<typename T>
constexpr T min = std::numeric_limits<T>::min();
template<typename T>
constexpr T max = std::numeric_limits<T>::max();

// ------------------------ Testing IsSameCompound -----------------------------

TEST(IsSameCompoundTest, is_false)
{
    EXPECT_FALSE((cuda::detail::IsSameCompound<int, int>));
    EXPECT_FALSE((cuda::detail::IsSameCompound<int2, int3>));
    EXPECT_FALSE((cuda::detail::IsSameCompound<uint3, int>));
}

TEST(IsSameCompoundTest, is_true)
{
    EXPECT_TRUE((cuda::detail::IsSameCompound<float3, int3>));
}

// ------------------------ Testing OneIsCompound ------------------------------

TEST(OneIsCompoundTest, is_false)
{
    EXPECT_FALSE((cuda::detail::OneIsCompound<int, int>));
    EXPECT_FALSE((cuda::detail::OneIsCompound<int2, int3>));
}

TEST(OneIsCompoundTest, is_true)
{
    EXPECT_TRUE((cuda::detail::OneIsCompound<uint3, int>));
    EXPECT_TRUE((cuda::detail::OneIsCompound<float3, int3>));
}

// -------------------------- Testing IsIntegral -------------------------------

TEST(IsIntegralTest, is_false)
{
    EXPECT_FALSE((cuda::detail::IsIntegral<float>));
    EXPECT_FALSE((cuda::detail::IsIntegral<double2>));
}

TEST(IsIntegralTest, is_true)
{
    EXPECT_TRUE((cuda::detail::IsIntegral<unsigned char>));
    EXPECT_TRUE((cuda::detail::IsIntegral<int2>));
}

// ---------------- Testing OneIsCompoundAndBothAreIntegral --------------------

TEST(OneIsCompoundAndBothAreIntegralTest, is_false)
{
    EXPECT_FALSE((cuda::detail::OneIsCompoundAndBothAreIntegral<int, int>));
    EXPECT_FALSE((cuda::detail::OneIsCompoundAndBothAreIntegral<int1, float>));
    EXPECT_FALSE((cuda::detail::OneIsCompoundAndBothAreIntegral<int2, int3>));
    EXPECT_FALSE((cuda::detail::OneIsCompoundAndBothAreIntegral<float3, int3>));
}

TEST(OneIsCompoundAndBothAreIntegralTest, is_true)
{
    EXPECT_TRUE((cuda::detail::OneIsCompoundAndBothAreIntegral<uint3, int>));
    EXPECT_TRUE((cuda::detail::OneIsCompoundAndBothAreIntegral<uint3, int3>));
    EXPECT_TRUE((cuda::detail::OneIsCompoundAndBothAreIntegral<short, uchar4>));
}

// ---------------------- Testing IsIntegralCompound ---------------------------

TEST(IsIntegralCompoundTest, is_false)
{
    EXPECT_FALSE((cuda::detail::IsIntegralCompound<float>));
    EXPECT_FALSE((cuda::detail::IsIntegralCompound<int>));
}

TEST(IsIntegralCompoundTest, is_true)
{
    EXPECT_TRUE((cuda::detail::IsIntegralCompound<uchar1>));
    EXPECT_TRUE((cuda::detail::IsIntegralCompound<short2>));
    EXPECT_TRUE((cuda::detail::IsIntegralCompound<int3>));
    EXPECT_TRUE((cuda::detail::IsIntegralCompound<uint4>));
}

// --------------------- Testing binary operators ==, != -----------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsEqualityTest, ttype::Types<
    ttype::Types<ttype::Value<short1{-2}>, ttype::Value<short1{-2}>, ttype::Value<true>>,
    ttype::Types<ttype::Value<short1{-1}>, ttype::Value<short1{1}>, ttype::Value<false>>,
    ttype::Types<ttype::Value<uchar2{1, 2}>, ttype::Value<uchar2{1, 2}>, ttype::Value<true>>,
    ttype::Types<ttype::Value<uchar2{2, 1}>, ttype::Value<uchar2{2, 3}>, ttype::Value<false>>,
    ttype::Types<ttype::Value<int3{-1, 0, 1}>, ttype::Value<long3{-1, 0, 1}>, ttype::Value<true>>,
    ttype::Types<ttype::Value<int3{-2, 0, 2}>, ttype::Value<long3{-2, 1, 2}>, ttype::Value<false>>,
    ttype::Types<ttype::Value<float4{0.f, 123.f, 234.f, 345.f}>, ttype::Value<ulong4_16a{0, 123, 234, 345}>, ttype::Value<true>>,
    ttype::Types<ttype::Value<float4{0.123f, 12.f, 23.f, 34.f}>, ttype::Value<ulong4_16a{0, 12, 23, 34}>, ttype::Value<false>>
>);

// clang-format on

TYPED_TEST(MathOpsEqualityTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    EXPECT_EQ(input1 == input2, gold);
    EXPECT_EQ(input1 != input2, !gold);
}

// --------------------- Testing unary operators -, +, ~ -----------------------

#define EXPECT_SAME_EQ(a, b)                                 \
    EXPECT_TRUE((std::is_same_v<decltype(a), decltype(b)>)); \
    EXPECT_EQ(a, b)

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsUnaryMinusTest, ttype::Types<
    ttype::Types<ttype::Value<short1{-1}>, ttype::Value<int1{1}>>,
    ttype::Types<ttype::Value<uchar2{1, 2}>, ttype::Value<int2{-1, -2}>>,
    ttype::Types<ttype::Value<int3{-1, 0, 1}>, ttype::Value<int3{1, 0, -1}>>,
    ttype::Types<ttype::Value<float4{-1.23f, 0.12f, 1.23f}>, ttype::Value<float4{1.23f, -0.12f, -1.23f}>>
>);

// clang-format on

TYPED_TEST(MathOpsUnaryMinusTest, correct_output)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = -input;

    EXPECT_SAME_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsUnaryPlusTest, ttype::Types<
    ttype::Types<ttype::Value<short1{-1}>, ttype::Value<int1{-1}>>,
    ttype::Types<ttype::Value<uchar2{1, 2}>, ttype::Value<int2{1, 2}>>,
    ttype::Types<ttype::Value<int3{-1, 0, 1}>, ttype::Value<int3{-1, 0, 1}>>,
    ttype::Types<ttype::Value<float4{-1.23f, 0.12f, 1.23f}>, ttype::Value<float4{-1.23f, 0.12f, 1.23f}>>
>);

// clang-format on

TYPED_TEST(MathOpsUnaryPlusTest, correct_output)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = +input;

    EXPECT_SAME_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsUnaryBitwiseNOTTest, ttype::Types<
    ttype::Types<ttype::Value<ushort1{0}>, ttype::Value<int1{-1}>>,
    ttype::Types<ttype::Value<uchar2{0, 0xFF}>, ttype::Value<int2{-1, static_cast<int>(0xFFFFFF00)}>>,
    ttype::Types<ttype::Value<uint3{0x77778888, 0, 0x88887777}>, ttype::Value<uint3{0x88887777, max<unsigned int>, 0x77778888}>>,
    ttype::Types<ttype::Value<long4_16a{0, 1, 2, max<long>}>, ttype::Value<long4_16a{-1, -2, -3, min<long>}>>
>);

// clang-format on

TYPED_TEST(MathOpsUnaryBitwiseNOTTest, correct_output)
{
    auto input = ttype::GetValue<TypeParam, 0>;
    auto gold  = ttype::GetValue<TypeParam, 1>;

    auto test = ~input;

    EXPECT_SAME_EQ(test, gold);
}

// ---------- Testing binary operators (with assignment) -, +, *, / ------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinarySubtractionTest, ttype::Types<
    ttype::Types<ttype::Value<int1{-2}>, ttype::Value<short1{2}>, ttype::Value<int1{-4}>>,
    ttype::Types<ttype::Value<int2{1, 2}>, ttype::Value<ushort2{2, 3}>, ttype::Value<int2{-1, -1}>>,
    ttype::Types<ttype::Value<uint3{1, 2, 3}>, ttype::Value<dim3{1, 1, 1}>, ttype::Value<uint3{0, 1, 2}>>,
    ttype::Types<ttype::Value<long3{-1, 0, 1}>, ttype::Value<long3{1, 2, 3}>, ttype::Value<long3{-2, -2, -2}>>,
    ttype::Types<ttype::Value<float4{0.f, -1.23f, -2.34f, -3.45f}>, ttype::Value<ulong4_16a{0, 12, 23, 34}>,
                 ttype::Value<float4{0, -13.23f, -25.34f, -37.45f}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinarySubtractionTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 - input2;
    auto test2 = input1;
    test2 -= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryAdditionTest, ttype::Types<
    ttype::Types<ttype::Value<int1{-2}>, ttype::Value<short1{2}>, ttype::Value<int1{0}>>,
    ttype::Types<ttype::Value<uint2{1, 2}>, ttype::Value<ushort2{2, 3}>, ttype::Value<uint2{3, 5}>>,
    ttype::Types<ttype::Value<uint3{1, 2, 3}>, ttype::Value<dim3{1, 1, 1}>, ttype::Value<uint3{2, 3, 4}>>,
    ttype::Types<ttype::Value<long3{-1, 0, 1}>, ttype::Value<long3{1, 2, 3}>, ttype::Value<long3{0, 2, 4}>>,
    ttype::Types<ttype::Value<float4{0.f, 1.23f, 2.34f, 3.45f}>, ttype::Value<ulong4_16a{0, 12, 23, 34}>,
                 ttype::Value<float4{0, 13.23f, 25.34f, 37.45f}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryAdditionTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 + input2;
    auto test2 = input1;
    test2 += input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryMultiplicationTest, ttype::Types<
    ttype::Types<ttype::Value<int1{-2}>, ttype::Value<short1{2}>, ttype::Value<int1{-4}>>,
    ttype::Types<ttype::Value<uint2{1, 2}>, ttype::Value<ushort2{2, 3}>, ttype::Value<uint2{2, 6}>>,
    ttype::Types<ttype::Value<uint3{1, 2, 3}>, ttype::Value<dim3{2, 2, 2}>, ttype::Value<uint3{2, 4, 6}>>,
    ttype::Types<ttype::Value<long3{-1, 0, 1}>, ttype::Value<long3{1, 2, 3}>, ttype::Value<long3{-1, 0, 3}>>,
    ttype::Types<ttype::Value<float4{0.f, 0.5f, 1.25f, 2.75f}>, ttype::Value<ulong4_16a{0, 12, 20, 24}>,
                 ttype::Value<float4{0, 6.f, 25.f, 66.f}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryMultiplicationTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 * input2;
    auto test2 = input1;
    test2 *= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryDivisionTest, ttype::Types<
    ttype::Types<ttype::Value<int1{-2}>, ttype::Value<short1{2}>, ttype::Value<int1{-1}>>,
    ttype::Types<ttype::Value<uint2{2, 4}>, ttype::Value<ushort2{1, 2}>, ttype::Value<uint2{2, 2}>>,
    ttype::Types<ttype::Value<uint3{2, 4, 6}>, ttype::Value<dim3{2, 2, 2}>, ttype::Value<uint3{1, 2, 3}>>,
    ttype::Types<ttype::Value<long3{-1, 0, 2}>, ttype::Value<int3{1, 2, 1}>, ttype::Value<long3{-1, 0, 2}>>,
    ttype::Types<ttype::Value<float4{0.f, 12.f, 21.f, 28.5f}>, ttype::Value<ulong4_16a{1, 2, 3, 2}>,
                 ttype::Value<float4{0, 6.f, 7.f, 14.25f}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryDivisionTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 / input2;
    auto test2 = input1;
    test2 /= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// ------ Testing binary operators (with assignment) %, &, |, ^, <<, >> --------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryModuloTest, ttype::Types<
    ttype::Types<ttype::Value<int1{2}>, ttype::Value<short1{2}>, ttype::Value<int1{0}>>,
    ttype::Types<ttype::Value<uint2{2, 4}>, ttype::Value<ushort2{1, 3}>, ttype::Value<uint2{0, 1}>>,
    ttype::Types<ttype::Value<uint3{1, 2, 3}>, ttype::Value<dim3{2, 2, 2}>, ttype::Value<uint3{1, 0, 1}>>,
    ttype::Types<ttype::Value<long3{1, 0, 5}>, ttype::Value<int3{1, 2, 3}>, ttype::Value<long3{0, 0, 2}>>,
    ttype::Types<ttype::Value<ulonglong4_16a{0, 11, 20, 31}>, ttype::Value<ulong4_16a{1, 2, 3, 7}>,
                 ttype::Value<ulonglong4_16a{0, 1, 2, 3}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryModuloTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 % input2;
    auto test2 = input1;
    test2 %= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryBitwiseANDTest, ttype::Types<
    ttype::Types<ttype::Value<int1{0x1}>, ttype::Value<short1{0x2}>, ttype::Value<int1{0}>>,
    ttype::Types<ttype::Value<uint2{0x2, 0x4}>, ttype::Value<ushort2{0x3, 0x5}>, ttype::Value<uint2{0x2, 0x4}>>,
    ttype::Types<ttype::Value<long3{0x1, 0, 0x5}>, ttype::Value<int3{0x1, 0x2, 0x3}>, ttype::Value<long3{0x1, 0, 0x1}>>,
    ttype::Types<ttype::Value<ulonglong4_16a{0, 0xF, 0xFF, 0xAA}>, ttype::Value<ulong4_16a{0x1, 0x9, 0xA5, 0x55}>,
                 ttype::Value<ulonglong4_16a{0, 0x9, 0xA5, 0}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryBitwiseANDTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 & input2;
    auto test2 = input1;
    test2 &= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryBitwiseORTest, ttype::Types<
    ttype::Types<ttype::Value<int1{0x1}>, ttype::Value<short1{0x2}>, ttype::Value<int1{0x3}>>,
    ttype::Types<ttype::Value<uint2{0x2, 0x4}>, ttype::Value<ushort2{0x3, 0x5}>, ttype::Value<uint2{0x3, 0x5}>>,
    ttype::Types<ttype::Value<long3{0x1, 0, 0x5}>, ttype::Value<int3{0x1, 0x2, 0x3}>, ttype::Value<long3{0x1, 0x2, 0x7}>>,
    ttype::Types<ttype::Value<ulonglong4_16a{0, 0xF, 0xFF, 0xAA}>, ttype::Value<ulong4_16a{0x1, 0x9, 0xA5, 0x55}>,
                 ttype::Value<ulonglong4_16a{0x1, 0xF, 0xFF, 0xFF}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryBitwiseORTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 | input2;
    auto test2 = input1;
    test2 |= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryBitwiseXORTest, ttype::Types<
    ttype::Types<ttype::Value<int1{0x1}>, ttype::Value<short1{0x2}>, ttype::Value<int1{0x3}>>,
    ttype::Types<ttype::Value<uint2{0x2, 0x4}>, ttype::Value<ushort2{0x3, 0x5}>, ttype::Value<uint2{0x1, 0x1}>>,
    ttype::Types<ttype::Value<long3{0x1, 0, 0x5}>, ttype::Value<int3{0x1, 0x2, 0x3}>, ttype::Value<long3{0, 0x2, 0x6}>>,
    ttype::Types<ttype::Value<ulonglong4_16a{0, 0xF, 0xFF, 0xAA}>, ttype::Value<ulong4_16a{0x1, 0x7, 0xAA, 0x55}>,
                 ttype::Value<ulonglong4_16a{0x1, 0x8, 0x55, 0xFF}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryBitwiseXORTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 ^ input2;
    auto test2 = input1;
    test2 ^= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryBitwiseLSHTest, ttype::Types<
    ttype::Types<ttype::Value<int1{0x1}>, ttype::Value<short1{2}>, ttype::Value<int1{0x4}>>,
    ttype::Types<ttype::Value<uint2{0x2, 0x4}>, ttype::Value<ushort2{2, 1}>, ttype::Value<uint2{0x8, 0x8}>>,
    ttype::Types<ttype::Value<long3{0x1, 0, 0x5}>, ttype::Value<int3{1, 2, 1}>, ttype::Value<long3{0x2, 0, 0xA}>>,
    ttype::Types<ttype::Value<ulonglong4_16a{0, 0x8, 0x3, 0xA}>, ttype::Value<ulong4_16a{1, 1, 2, 1}>,
                 ttype::Value<ulonglong4_16a{0, 0x10, 0xC, 0x14}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryBitwiseLSHTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 << input2;
    auto test2 = input1;
    test2 <<= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsBinaryBitwiseRSHTest, ttype::Types<
    ttype::Types<ttype::Value<int1{0x4}>, ttype::Value<short1{2}>, ttype::Value<int1{0x1}>>,
    ttype::Types<ttype::Value<uint2{0x4, 0x8}>, ttype::Value<ushort2{1, 1}>, ttype::Value<uint2{0x2, 0x4}>>,
    ttype::Types<ttype::Value<long3{0x3, 0, 0x5}>, ttype::Value<int3{1, 2, 1}>, ttype::Value<long3{0x1, 0, 0x2}>>,
    ttype::Types<ttype::Value<ulonglong4_16a{0, 0xF, 0x7, 0xA}>, ttype::Value<ulong4_16a{1, 1, 2, 1}>,
                 ttype::Value<ulonglong4_16a{0, 0x7, 0x1, 0x5}>>
>);

// clang-format on

TYPED_TEST(MathOpsBinaryBitwiseRSHTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test1 = input1 >> input2;
    auto test2 = input1;
    test2 >>= input2;

    EXPECT_SAME_EQ(test1, test2);
    EXPECT_SAME_EQ(test1, gold);
}

// --------------------------- Testing dot product -----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    MathOpsDotProductTest, ttype::Types<
    ttype::Types<ttype::Value<int1{2}>, ttype::Value<short1{1}>, ttype::Value<int{2}>>,
    ttype::Types<ttype::Value<uint2{2, 4}>, ttype::Value<ushort2{1, 3}>, ttype::Value<uint{14}>>,
    ttype::Types<ttype::Value<uchar3{1, 2, 3}>, ttype::Value<dim3{2, 2, 2}>, ttype::Value<uint{12}>>,
    ttype::Types<ttype::Value<float4{0.f, 1.f, 2.f, 3.f}>, ttype::Value<long4_16a{1, 2, 3, 4}>, ttype::Value<float{20.f}>>
>);

// clang-format on

TYPED_TEST(MathOpsDotProductTest, correct_output)
{
    auto input1 = ttype::GetValue<TypeParam, 0>;
    auto input2 = ttype::GetValue<TypeParam, 1>;
    auto gold   = ttype::GetValue<TypeParam, 2>;

    auto test = cuda::dot(input1, input2);

    EXPECT_SAME_EQ(test, gold);
}

// ---------------------- Testing operators for __half -------------------------

// __half and its vector types (half1/__half2/half3/half4) are not structural types, so they
// cannot be used in ttype::Value<> non-type template parameters as in the typed suites above;
// the fp16 coverage below uses plain host-side TESTs with values built at run time.  These
// tests also prove the cuda_fp16.h host-side __half operators (CUDA >= 12.2) that the
// MathOps compound operators rely on.

TEST(MathOpsHalfTest, half3_arithmetic_operators)
{
    // all values and results below are exactly representable in half
    half3 a{__float2half(1.f), __float2half(2.f), __float2half(3.f)};
    half3 b{__float2half(0.5f), __float2half(0.25f), __float2half(2.f)};

    auto add = a + b;
    auto sub = a - b;
    auto mul = a * b;
    auto div = a / b;

    static_assert(std::is_same_v<decltype(add), half3>);
    static_assert(std::is_same_v<decltype(sub), half3>);
    static_assert(std::is_same_v<decltype(mul), half3>);
    static_assert(std::is_same_v<decltype(div), half3>);

    EXPECT_SAME_EQ(add, (half3{__float2half(1.5f), __float2half(2.25f), __float2half(5.f)}));
    EXPECT_SAME_EQ(sub, (half3{__float2half(0.5f), __float2half(1.75f), __float2half(1.f)}));
    EXPECT_SAME_EQ(mul, (half3{__float2half(0.5f), __float2half(0.5f), __float2half(6.f)}));
    EXPECT_SAME_EQ(div, (half3{__float2half(2.f), __float2half(8.f), __float2half(1.5f)}));

    half3 c = a;
    c += b;
    EXPECT_SAME_EQ(c, add);
    c = a;
    c -= b;
    EXPECT_SAME_EQ(c, sub);
    c = a;
    c *= b;
    EXPECT_SAME_EQ(c, mul);
    c = a;
    c /= b;
    EXPECT_SAME_EQ(c, div);

    auto neg = -a;
    static_assert(std::is_same_v<decltype(neg), half3>);
    EXPECT_SAME_EQ(neg, (half3{__float2half(-1.f), __float2half(-2.f), __float2half(-3.f)}));
}

TEST(MathOpsHalfTest, half4_and_half2_arithmetic_operators)
{
    half4 a{__float2half(1.f), __float2half(2.f), __float2half(3.f), __float2half(4.f)};
    half4 b{__float2half(0.5f), __float2half(0.5f), __float2half(0.5f), __float2half(0.5f)};

    auto add = a + b;
    static_assert(std::is_same_v<decltype(add), half4>);
    EXPECT_SAME_EQ(add, (half4{__float2half(1.5f), __float2half(2.5f), __float2half(3.5f), __float2half(4.5f)}));

    auto mul = a * b;
    static_assert(std::is_same_v<decltype(mul), half4>);
    EXPECT_SAME_EQ(mul, (half4{__float2half(0.5f), __float2half(1.f), __float2half(1.5f), __float2half(2.f)}));

    // __half2 addition resolves to the cuda_fp16.h host operator (CUDA >= 12.2)
    __half2 c{__float2half(1.f), __float2half(2.f)};
    __half2 d{__float2half(0.25f), __float2half(0.5f)};

    auto sum2 = c + d;
    static_assert(std::is_same_v<decltype(sum2), __half2>);
    EXPECT_EQ(__half2float(sum2.x), 1.25f);
    EXPECT_EQ(__half2float(sum2.y), 2.5f);
}

TEST(MathOpsHalfTest, half3_equality_operators)
{
    half3 a{__float2half(1.f), __float2half(2.f), __float2half(3.f)};
    half3 b{__float2half(1.f), __float2half(2.f), __float2half(-3.f)};

    EXPECT_TRUE(a == a);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST(MathOpsHalfTest, half_mixed_operators_promote_through_float)
{
    half3 a{__float2half(1.f), __float2half(2.f), __float2half(3.f)};

    // __half OP arithmetic-scalar promotes through float, so half3 * float yields float3
    auto scaled = a * 0.5f;
    static_assert(std::is_same_v<decltype(scaled), float3>);
    EXPECT_SAME_EQ(scaled, (float3{0.5f, 1.f, 1.5f}));

    auto doubled = 2 * a;
    static_assert(std::is_same_v<decltype(doubled), float3>);
    EXPECT_SAME_EQ(doubled, (float3{2.f, 4.f, 6.f}));

    // compound assignment casts the float3 result back to half3
    half3 c = a;
    c *= 2.f;
    EXPECT_SAME_EQ(c, (half3{__float2half(2.f), __float2half(4.f), __float2half(6.f)}));

    // mixed scalar comparisons also promote through float
    EXPECT_TRUE(__float2half(1.f) > 0.5f);
    EXPECT_TRUE(0.5f < __float2half(1.f));
    EXPECT_TRUE(__float2half(0.5f) == 0.5f);
    EXPECT_TRUE(__float2half(0.5f) != 0.25f);
    EXPECT_TRUE(__float2half(0.5f) <= 0.5);
    EXPECT_TRUE(__float2half(0.5f) >= 0.25);
}

#undef EXPECT_SAME_EQ
