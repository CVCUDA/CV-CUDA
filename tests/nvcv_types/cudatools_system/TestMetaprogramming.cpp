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

#include <common/TypedTests.hpp>                // for NVCV_TYPED_TEST_SUITE_F, etc.
#include <nvcv/cuda/detail/Metaprogramming.hpp> // the object of this test

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
