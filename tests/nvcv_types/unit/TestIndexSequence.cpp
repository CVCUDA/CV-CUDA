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

#include <common/TypedTests.hpp>
#include <nvcv/detail/IndexSequence.hpp>

namespace test = nvcv::test::type;

// clang-format off
NVCV_TYPED_TEST_SUITE(IndexSequence,
    test::Types<
        test::Types<test::Value<0>, nvcv::detail::IndexSequence<>>,
        test::Types<test::Value<1>, nvcv::detail::IndexSequence<0>>,
        test::Types<test::Value<2>, nvcv::detail::IndexSequence<0,1>>,
        test::Types<test::Value<5>, nvcv::detail::IndexSequence<0,1,2,3,4>>
    >);

// clang-format on

TYPED_TEST(IndexSequence, make_index_sequence)
{
    constexpr int N = test::GetValue<TypeParam, 0>;
    using GOLD      = test::GetType<TypeParam, 1>;

    EXPECT_TRUE((std::is_same_v<GOLD, nvcv::detail::MakeIndexSequence<N>>));
}
