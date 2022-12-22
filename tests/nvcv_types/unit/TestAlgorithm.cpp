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

#include <util/Algorithm.hpp>

namespace util = nvcv::util;
namespace test = nvcv::test;

TEST(AlgorithmMaxTest, works)
{
    EXPECT_EQ(2, util::Max(2));

    EXPECT_EQ(5, util::Max(2, 5));
    EXPECT_EQ(5, util::Max(5, 2));

    EXPECT_EQ(double('a'), util::Max(2, 3, 5.0, 'a'));
    EXPECT_EQ(7.6, util::Max(5, 7.6, 2, 3.0));
    EXPECT_EQ(5.0, util::Max(5, 2, 3.0));
    EXPECT_EQ(5.0, util::Max(5u, -2, 3.0));
}

template<int I>
struct Value
{
    constexpr static int value = I;
};

TEST(AlgorithmMaxTest, constexpr_works)
{
    Value<util::Max(3, 8)> v;
    EXPECT_EQ(8, v.value);
}
