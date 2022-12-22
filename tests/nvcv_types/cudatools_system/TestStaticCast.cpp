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

#include <common/TypedTests.hpp>    // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/cuda/StaticCast.hpp> // the object of this test

namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

// --------------------------- Testing StaticCast ------------------------------

// clang-format off

// Target type, input values, expected (gold) output values
NVCV_TYPED_TEST_SUITE(
    StaticCastTest, ttype::Types<
    // identity
    ttype::Types<signed short, ttype::Value<short{-1}>, ttype::Value<short{-1}>>,
    ttype::Types<unsigned short, ttype::Value<ushort1{1}>, ttype::Value<ushort1{1}>>,
    ttype::Types<signed char, ttype::Value<char2{-1, 2}>, ttype::Value<char2{-1, 2}>>,
    ttype::Types<unsigned char, ttype::Value<uchar3{1, 2, 3}>, ttype::Value<uchar3{1, 2, 3}>>,
    ttype::Types<int, ttype::Value<int4{123, 234, 345, 456}>, ttype::Value<int4{123, 234, 345, 456}>>,
    // same size
    ttype::Types<unsigned char, ttype::Value<short2{-1, 234}>, ttype::Value<uchar2{255, 234}>>,
    ttype::Types<signed char, ttype::Value<ushort2{123, 255}>, ttype::Value<char2{123, -1}>>,
    ttype::Types<unsigned int, ttype::Value<int3{123, 234, 345}>, ttype::Value<uint3{123, 234, 345}>>,
    ttype::Types<int, ttype::Value<uint3{123, 234, 345}>, ttype::Value<int3{123, 234, 345}>>,
    // different sizes
    ttype::Types<double, ttype::Value<float4{-1.5f, 0.f, 1.5f, 2.5f}>, ttype::Value<double4{-1.5, 0.0, 1.5, 2.5}>>,
    ttype::Types<float, ttype::Value<double3{-1.5, 0.0, 1.5f}>, ttype::Value<float3{-1.5f, 0.f, 1.5f}>>,
    ttype::Types<unsigned long, ttype::Value<long2{1234567, 2345678}>, ttype::Value<ulong2{1234567, 2345678}>>,
    ttype::Types<long, ttype::Value<longlong2{-1234, -2345}>, ttype::Value<long2{-1234, -2345}>>
    >);

// clang-format on

TYPED_TEST(StaticCastTest, correct_output)
{
    using TargetBaseType = ttype::GetType<TypeParam, 0>;
    constexpr auto input = ttype::GetValue<TypeParam, 1>;
    constexpr auto gold  = ttype::GetValue<TypeParam, 2>;

    const auto test = cuda::StaticCast<TargetBaseType>(input);

    using TestType = decltype(test);
    using GoldType = decltype(gold);

    EXPECT_TRUE((std::is_same_v<TestType, GoldType>));

    for (int e = 0; e < cuda::NumElements<TestType>; ++e)
    {
        EXPECT_EQ(cuda::GetElement(test, e), cuda::GetElement(gold, e));
    }
}
