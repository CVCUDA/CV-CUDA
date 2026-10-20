/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <nvcv/RoundMode.h>
#include <nvcv/detail/TypeTraits.hpp>
namespace d = nvcv::detail;

static_assert(d::Conjunction<>::value, "Conjunction tail should evaluate to true");
static_assert(d::Conjunction<std::true_type>::value, "Sanity check failed");
static_assert(!d::Conjunction<std::false_type>::value, "Sanity check failed");
static_assert(!d::Conjunction<std::true_type, std::true_type, std::false_type, std::true_type>::value,
              "Sanity check failed");
static_assert(d::Conjunction<std::true_type, std::true_type, std::true_type>::value, "Sanity check failed");

static_assert(!d::Disjunction<>::value, "Disjunction tail should evaluate to false");
static_assert(d::Disjunction<std::true_type>::value, "Sanity check failed");
static_assert(!d::Disjunction<std::false_type>::value, "Sanity check failed");
static_assert(d::Disjunction<std::false_type, std::false_type, std::true_type, std::false_type>::value,
              "Sanity check failed");
static_assert(!d::Disjunction<std::false_type, std::false_type, std::false_type>::value, "Sanity check failed");

TEST(RoundModeTests, preserves_unsigned_int_abi)
{
    EXPECT_EQ(sizeof(NVCVRoundMode), sizeof(unsigned int));
    EXPECT_EQ(alignof(NVCVRoundMode), alignof(unsigned int));
    EXPECT_TRUE((std::is_same<std::underlying_type<NVCVRoundMode>::type, unsigned int>::value));
    EXPECT_EQ(NVCV_ROUND_NEAREST, 0);
    EXPECT_EQ(NVCV_ROUND_TRUNCATE, 1);
    EXPECT_EQ(NVCV_ROUND_NEAREST_WRAP, 2);
    EXPECT_EQ(NVCV_ROUND_TRUNCATE_WRAP, 3);
}

class RoundModeInvalidValueTests : public ::testing::TestWithParam<int>
{
};

TEST_P(RoundModeInvalidValueTests, preserves_value_for_validation)
{
    volatile auto roundMode = static_cast<NVCVRoundMode>(GetParam());
    EXPECT_EQ(static_cast<unsigned int>(roundMode), static_cast<unsigned int>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(InvalidRoundModes, RoundModeInvalidValueTests, ::testing::Values(255, 99, -1));
