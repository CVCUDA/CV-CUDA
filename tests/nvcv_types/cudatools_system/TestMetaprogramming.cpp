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

#include <common/TypedTests.hpp>                        // for NVCV_TYPED_TEST_SUITE_F, etc.
#include <cvcuda/cuda_tools/detail/Metaprogramming.hpp> // the object of this test

namespace t      = ::testing;
namespace test   = nvcv::test;
namespace detail = nvcv::cuda::detail;

// ------------------------- Testing CopyConstness_t ---------------------------

template<typename T>
class CopyConstnessTest : public t::Test
{
public:
    using SourceType = test::type::GetType<T, 0>;
    using TargetType = test::type::GetType<T, 1>;
};

NVCV_TYPED_TEST_SUITE_F(CopyConstnessTest,
                        test::type::Zip<t::Types<const float, const unsigned long long>, t::Types<int, double>>);

TYPED_TEST(CopyConstnessTest, is_const)
{
    using ConstType = detail::CopyConstness_t<typename TestFixture::SourceType, typename TestFixture::TargetType>;

    EXPECT_TRUE(std::is_const_v<ConstType>);
}

TYPED_TEST(CopyConstnessTest, correct_type)
{
    using ConstType = detail::CopyConstness_t<typename TestFixture::SourceType, typename TestFixture::TargetType>;

    EXPECT_TRUE((std::is_same_v<typename std::remove_const_t<ConstType>, typename TestFixture::TargetType>));
}

// ----------------------------- Testing IsHalfV --------------------------------

TEST(IsHalfTest, is_false)
{
    EXPECT_FALSE(detail::IsHalfV<float>);
    EXPECT_FALSE(detail::IsHalfV<double>);
    EXPECT_FALSE(detail::IsHalfV<unsigned short>);
    // only the __half base type is half; half vector types are not
    EXPECT_FALSE(detail::IsHalfV<half1>);
    EXPECT_FALSE(detail::IsHalfV<__half2>);
    EXPECT_FALSE(detail::IsHalfV<half3>);
    EXPECT_FALSE(detail::IsHalfV<half4>);
}

TEST(IsHalfTest, is_true)
{
    EXPECT_TRUE(detail::IsHalfV<__half>);
    EXPECT_TRUE(detail::IsHalfV<const __half>);
    EXPECT_TRUE(detail::IsHalfV<volatile __half>);
    EXPECT_TRUE(detail::IsHalfV<const volatile __half>);
}

// ------------------------ Testing IsFloatingPointV ----------------------------

TEST(IsFloatingPointTest, is_false)
{
    EXPECT_FALSE(detail::IsFloatingPointV<int>);
    EXPECT_FALSE(detail::IsFloatingPointV<unsigned char>);
    EXPECT_FALSE(detail::IsFloatingPointV<half3>);
}

TEST(IsFloatingPointTest, is_true)
{
    EXPECT_TRUE(detail::IsFloatingPointV<float>);
    EXPECT_TRUE(detail::IsFloatingPointV<double>);
    // std::is_floating_point does not cover the extended type __half; IsFloatingPointV must
    EXPECT_TRUE(detail::IsFloatingPointV<__half>);
    EXPECT_TRUE(detail::IsFloatingPointV<const __half>);
}
