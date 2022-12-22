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

/**
 * @file Size.hpp
 *
 * @brief Declaration of NVCV C++ Size class and its operators.
 */

#ifndef NVCV_SIZE_HPP
#define NVCV_SIZE_HPP

#include <cassert>
#include <iostream>
#include <tuple>

namespace nvcv {

/**
 * @defgroup NVCV_CPP_CORE_SIZE Size Operator
 * @{
*/

struct Size2D
{
    int w, h;
};

inline bool operator==(const Size2D &a, const Size2D &b)
{
    return std::tie(a.w, a.h) == std::tie(b.w, b.h);
}

inline bool operator!=(const Size2D &a, const Size2D &b)
{
    return !(a == b);
}

inline bool operator<(const Size2D &a, const Size2D &b)
{
    return std::tie(a.w, a.h) < std::tie(b.w, b.h);
}

inline std::ostream &operator<<(std::ostream &out, const Size2D &size)
{
    return out << size.w << "x" << size.h;
}

/**@}*/

} // namespace nvcv

#endif // NVCV_SIZE_HPP
