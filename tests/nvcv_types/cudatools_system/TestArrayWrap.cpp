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

#include "DeviceTensorWrap.hpp" // to test in device

#include <common/HashUtils.hpp>  // for NVCV_INSTANTIATE_TEST_SUITE_P, etc.
#include <common/TypedTests.hpp> // for NVCV_TYPED_TEST_SUITE, etc.
#include <common/ValueTests.hpp> // for StringLiteral
#include <nvcv/Array.hpp>
#include <nvcv/ArrayDataAccess.hpp>
#include <nvcv/cuda/ArrayWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>

#include <limits>

namespace t     = ::testing;
namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

// clang-format off
NVCV_TYPED_TEST_SUITE(
    ArrayWrapTest, ttype::Types<
    ttype::Types<ttype::Value<Array<int, 2>{-5, 1}>>,
    ttype::Types<ttype::Value<Array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<Array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<Array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(ArrayWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::ArrayWrap<const ValueType> wrap(input.data(), input.m_data.size());

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(0))>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(0))>>);

    EXPECT_EQ(wrap.ptr(0), input.data());

    for (int i = 0; i < InputType::kShapes[0]; ++i)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(i))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(i))>>);

        EXPECT_EQ(wrap.ptr(i), &input[i]);
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    ArrayWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<Array<int, 2>{}>, ttype::Value<Array<int, 2>{
       -5, 1}>>,
    ttype::Types<ttype::Value<Array<short3, 2>{}>, ttype::Value<Array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<Array<float1, 4>{}>, ttype::Value<Array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<Array<uchar4, 3>{}>, ttype::Value<Array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(ArrayWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::ArrayWrap<ValueType> wrap(test.data(), test.m_data.size());

    ASSERT_EQ(InputType::kShapes[0], decltype(gold)::kShapes[0]);

    for (int i = 0; i < InputType::kShapes[0]; ++i)
    {
        wrap[i] = gold[i];
    }

    EXPECT_EQ(test, gold);
}
