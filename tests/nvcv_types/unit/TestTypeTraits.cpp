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

#include <nvcv/detail/TypeTraits.hpp>
//#include "Definitions.hpp"

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
