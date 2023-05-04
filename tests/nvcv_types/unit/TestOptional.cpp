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

#include "Definitions.hpp"

#include <nvcv/Optional.hpp>

#include <vector>

TEST(Optional, default_no_value)
{
    nvcv::Optional<int> opt;

    EXPECT_FALSE(opt.hasValue());
    EXPECT_FALSE(opt && true);

    EXPECT_THROW(*opt, std::runtime_error);
}

TEST(Optional, ctor_with_value)
{
    nvcv::Optional<int> opt(5);

    EXPECT_TRUE(opt.hasValue());
    EXPECT_TRUE(opt && true);

    EXPECT_EQ(5, *opt);
}

TEST(Optional, assignment)
{
    std::vector<int>                 test_value  = {1, 2, 3, 4, 42, 666, 31337};
    std::vector<int>                 test_value2 = {10, 9, 8, 7};
    nvcv::Optional<std::vector<int>> opt(test_value);

    ASSERT_TRUE(opt.hasValue());
    ASSERT_EQ(*opt, test_value);

    nvcv::Optional<std::vector<int>> o1, o2;
    o1 = opt;
    EXPECT_TRUE(o1.hasValue());
    EXPECT_EQ(o1.value(), test_value);
    EXPECT_TRUE(opt.hasValue());
    EXPECT_EQ(opt.value(), test_value);

    o2 = std::move(opt);
    EXPECT_TRUE(o1.hasValue());
    EXPECT_EQ(o1.value(), test_value);
    EXPECT_TRUE(opt.hasValue()) << "A moved-out optional still has a value.";
    EXPECT_TRUE(opt.value().empty()) << "The value wasn't moved out properly";

    opt = nvcv::NullOpt;
    EXPECT_FALSE(opt.hasValue());

    opt = test_value2;
    EXPECT_TRUE(opt.hasValue());
    EXPECT_EQ(opt.value(), test_value2);

    nvcv::Optional<char> c = 42;
    nvcv::Optional<int>  i;
    i = c;
    EXPECT_TRUE(i.hasValue());
    EXPECT_EQ(i.value(), 42);
}

TEST(Optional, equality)
{
    nvcv::Optional<int> optA(5);
    nvcv::Optional<int> optB(5);
    EXPECT_TRUE(optA == optB);
    EXPECT_FALSE(optA != optB);

    EXPECT_TRUE(optA == 5);
    EXPECT_FALSE(optA != 5);

    EXPECT_TRUE(5 == optA);
    EXPECT_FALSE(5 != optA);

    EXPECT_FALSE(optA == nullptr);
    EXPECT_TRUE(optA != nullptr);

    EXPECT_FALSE(optA == nvcv::NullOpt);
    EXPECT_TRUE(optA != nvcv::NullOpt);

    EXPECT_FALSE(nullptr == optA);
    EXPECT_TRUE(nullptr != optA);

    EXPECT_FALSE(nvcv::NullOpt == optA);
    EXPECT_TRUE(nvcv::NullOpt != optA);
}

// TODO need way more tests.
// We're not writing them now because if we can upgrade public API to c++17,
// we won't need our Optional.
