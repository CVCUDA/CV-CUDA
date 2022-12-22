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

#include "Definitions.hpp"

#include <common/MixTypedTests.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

// clang-format off
NVCV_MIXTYPED_TEST_SUITE(TypedMixedTest,
    test::type::Combine<test::Types<test::type::Value<0>, test::type::Value<2>, test::type::Value<3>>,
                                    test::Types<int,float,char>,
                                    test::Types<test::type::Value<'a'>, test::type::Value<'b'>>
    >);

// clang-format on

NVCV_MIXTYPED_TEST(TypedMixedTest, test)
{
    const int A  = GetValue<0>();
    using T      = GetType<1>;
    const char B = GetValue<2>();

    EXPECT_THAT(A, t::AnyOf(0, 2, 3));
    EXPECT_TRUE((std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, char>));
    EXPECT_THAT(B, t::AnyOf('a', 'b'));
}
