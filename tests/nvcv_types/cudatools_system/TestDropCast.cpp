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

#include <common/TypedTests.hpp>  // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/cuda/DropCast.hpp> // the object of this test

namespace t     = ::testing;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

// --------------------------- Testing DropCast ------------------------------

// clang-format off

// Target number of dimension, input values, expected (gold) output values
NVCV_TYPED_TEST_SUITE(
    DropCastTest, ttype::Types<
    // identity
    ttype::Types<ttype::Value<0>, ttype::Value<short{-1}>, ttype::Value<short{-1}>>,
    ttype::Types<ttype::Value<1>, ttype::Value<ushort1{1}>, ttype::Value<ushort1{1}>>,
    ttype::Types<ttype::Value<2>, ttype::Value<char2{-1, 2}>, ttype::Value<char2{-1, 2}>>,
    ttype::Types<ttype::Value<3>, ttype::Value<uchar3{1, 2, 3}>, ttype::Value<uchar3{1, 2, 3}>>,
    ttype::Types<ttype::Value<4>, ttype::Value<int4{123, 234, 345, 456}>, ttype::Value<int4{123, 234, 345, 456}>>,
    // reducing by one
    ttype::Types<ttype::Value<0>, ttype::Value<short2{-123, 234}>, ttype::Value<short{-123}>>,
    ttype::Types<ttype::Value<1>, ttype::Value<uchar2{123, 234}>, ttype::Value<uchar1{123}>>,
    ttype::Types<ttype::Value<2>, ttype::Value<int3{-12, 0, 123}>, ttype::Value<int2{-12, 0}>>,
    ttype::Types<ttype::Value<3>, ttype::Value<float4{-1.2f, 0.f, 1.2f, 2.3f}>, ttype::Value<float3{-1.2f, 0.f, 1.2f}>>,
    ttype::Types<ttype::Value<2>, ttype::Value<double3{-1.3, 0.0, 1.3f}>, ttype::Value<double2{-1.3, 0.0}>>,
    ttype::Types<ttype::Value<1>, ttype::Value<ulong2{1234567, 2345678}>, ttype::Value<ulong1{1234567}>>,
    ttype::Types<ttype::Value<0>, ttype::Value<int2{-1234, 2345}>, ttype::Value<int{-1234}>>,
    // reducing more
    ttype::Types<ttype::Value<0>, ttype::Value<short4{-123, -234, 345, 456}>, ttype::Value<short{-123}>>,
    ttype::Types<ttype::Value<1>, ttype::Value<char3{-127, 0, 127}>, ttype::Value<char1{-127}>>,
    ttype::Types<ttype::Value<2>, ttype::Value<uint4{0, 123, 234, 345}>, ttype::Value<uint2{0, 123}>>,
    ttype::Types<ttype::Value<1>, ttype::Value<double3{-1.3, 0.0, 1.3}>, ttype::Value<double1{-1.3}>>,
    ttype::Types<ttype::Value<0>, ttype::Value<long3{1234567, 2345678, 3456789}>, ttype::Value<long{1234567}>>
    >);

// clang-format on

TYPED_TEST(DropCastTest, correct_output)
{
    const int  NC    = ttype::GetValue<TypeParam, 0>;
    const auto input = ttype::GetValue<TypeParam, 1>;
    const auto gold  = ttype::GetValue<TypeParam, 2>;

    const auto test = cuda::DropCast<NC>(input);

    using TestType = decltype(test);
    using GoldType = decltype(gold);

    EXPECT_EQ(cuda::NumComponents<TestType>, NC);

    EXPECT_TRUE((std::is_same_v<TestType, GoldType>));

    for (int e = 0; e < cuda::NumElements<TestType>; ++e)
    {
        EXPECT_EQ(cuda::GetElement(test, e), cuda::GetElement(gold, e));
    }
}
