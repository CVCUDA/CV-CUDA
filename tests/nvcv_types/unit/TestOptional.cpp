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

#include <nvcv/detail/Optional.hpp>

namespace d = nvcv::detail;

TEST(Optional, default_no_value)
{
    d::Optional<int> opt;

    EXPECT_FALSE(opt.hasValue());
    EXPECT_FALSE(opt && true);

    EXPECT_THROW(*opt, std::runtime_error);
}

TEST(Optional, ctor_with_value)
{
    d::Optional<int> opt(5);

    EXPECT_TRUE(opt.hasValue());
    EXPECT_TRUE(opt && true);

    EXPECT_EQ(5, *opt);
}

TEST(Optional, equality)
{
    d::Optional<int> optA(5);
    d::Optional<int> optB(5);
    EXPECT_TRUE(optA == optB);
    EXPECT_FALSE(optA != optB);

    EXPECT_TRUE(optA == 5);
    EXPECT_FALSE(optA != 5);

    EXPECT_TRUE(5 == optA);
    EXPECT_FALSE(5 != optA);

    EXPECT_FALSE(optA == nullptr);
    EXPECT_TRUE(optA != nullptr);

    EXPECT_FALSE(optA == d::NullOpt);
    EXPECT_TRUE(optA != d::NullOpt);

    EXPECT_FALSE(nullptr == optA);
    EXPECT_TRUE(nullptr != optA);

    EXPECT_FALSE(d::NullOpt == optA);
    EXPECT_TRUE(d::NullOpt != optA);
}

// TODO need way more tests.
// We're not writing them now because if we can upgrade public API to c++17,
// we won't need our Optional.
