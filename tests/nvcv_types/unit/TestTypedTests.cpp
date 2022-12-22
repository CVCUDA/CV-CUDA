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

namespace t    = ::testing;
namespace test = nvcv::test;

struct Foo
{
};

struct Bar
{
};

NVCV_TYPED_TEST_SUITE(TypedTest, test::type::Combine<test::Types<Foo, Bar>, test::Values<1, 2, 3>>);

TYPED_TEST(TypedTest, test)
{
    // For now we're concerned if typed tests will compile.
    // TODO: How to test if the tests were correctly generated?
}
