/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceAtomics.hpp" // to test in the device

#include <common/TypedTests.hpp> // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/cuda/Atomics.hpp> // the object of this test
#include <nvcv/cuda/MathOps.hpp> // for operator == to allow EXPECT_EQ

#include <numeric> // for std::iota

namespace t = ::testing;

// ---------------------------- Testing Atomics --------------------------------

template<typename T>
class AtomicsTest : public t::Test
{
public:
    using Type = T;
};

// clang-format off

using SupportedTypes = t::Types<unsigned int, int, unsigned long long int, long long int, float, double>;

TYPED_TEST_SUITE(AtomicsTest, SupportedTypes);

// clang-format on

TYPED_TEST(AtomicsTest, correct_minimum_in_device)
{
    using DataType = typename TestFixture::Type;

    std::vector<DataType> input(100);

    std::iota(input.begin(), input.end(), 1);

    auto test = DeviceRunAtomicMin(input);

    EXPECT_EQ(test, 1);
}

TYPED_TEST(AtomicsTest, correct_maximum_in_device)
{
    using DataType = typename TestFixture::Type;

    std::vector<DataType> input(100);

    std::iota(input.begin(), input.end(), 1);

    auto test = DeviceRunAtomicMax(input);

    EXPECT_EQ(test, 100);
}
