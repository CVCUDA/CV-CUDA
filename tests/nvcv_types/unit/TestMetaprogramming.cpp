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
#include <util/Metaprogramming.hpp>

namespace ttest = nvcv::test::type;
namespace util  = nvcv::util;

// clang-format off
NVCV_TYPED_TEST_SUITE(MetaprogrammingTypeIdentityTest,
                      ttest::Types<
                        ttest::Types<int, int>,
                        ttest::Types<char & , char &>,
                        ttest::Types<const short * , const short *>,
                        ttest::Types<volatile long *, volatile long *>,
                        ttest::Types<volatile long *, volatile long *>,
                        ttest::Types<const int*[3], const int*[3]>>);

// clang-format on

TYPED_TEST(MetaprogrammingTypeIdentityTest, works)
{
    using IN   = ttest::GetType<TypeParam, 0>;
    using GOLD = ttest::GetType<TypeParam, 1>;

    EXPECT_TRUE((std::is_same_v<util::TypeIdentity<IN>, GOLD>));
}
