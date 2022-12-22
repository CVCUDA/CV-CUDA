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

#ifndef NVCV_TEST_COMMON_TYPEDTESTS_HPP
#define NVCV_TEST_COMMON_TYPEDTESTS_HPP

#include "TypeList.hpp"

#include <gtest/gtest.h>

// Helper to be able to define typed test cases by defining the
// types inline (no worries about commas in macros)

#define NVCV_TYPED_TEST_SUITE_F(TEST, ...) \
    using TEST##_Types = __VA_ARGS__;      \
    TYPED_TEST_SUITE(TEST, TEST##_Types)

#define NVCV_TYPED_TEST_SUITE(TEST, ...) \
    template<class T>                    \
    class TEST : public ::testing::Test  \
    {                                    \
    };                                   \
    NVCV_TYPED_TEST_SUITE_F(TEST, __VA_ARGS__)

#define NVCV_INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, ...) \
    using TEST##INSTNAME##_Types = __VA_ARGS__;                  \
    INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, TEST##INSTNAME##_Types)

#endif // NVCV_TEST_COMMON_TYPEDTESTS_HPP
