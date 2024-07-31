/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Size.h"

#include <cassert>
#include <iostream>
#include <tuple>

namespace nvcv {

/**
 * @defgroup NVCV_CPP_CORE_SIZE Size Operator
 * @{
*/

struct Size2D : NVCVSize2D
{
    using NVCVSize2D::NVCVSize2D;

    constexpr Size2D(int32_t w, int32_t h)
        : NVCVSize2D{w, h}
    {
    }

    constexpr Size2D(const NVCVSize2D &s)
        : NVCVSize2D{s.w, s.h}
    {
    }

    inline Size2D &operator=(const NVCVSize2D &s)
    {
        static_cast<NVCVSize2D &>(*this) = s;
        return *this;
    }

    /**
     * @brief Compares two Size2D structures for equality.
     *
     * @param a First size to compare.
     * @param b Second size to compare.
     * @return true if both width and height of `a`  and `b` are equal, otherwise false.
     */
    constexpr bool operator==(const Size2D &rhs) const
    {
        return w == rhs.w && h == rhs.h;
    }

    /**
     * @brief Compares two Size2D structures for inequality.
     *
     * @param a First size to compare.
     * @param b Second size to compare.
     * @return true if width or height of `a` and `b` are not equal, otherwise false.
     */
    constexpr bool operator!=(const Size2D &rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * @brief Compares two Size2D structures.
     *
     * The comparison is based on the width first, and then the height.
     *
     * @param a First size to compare.
     * @param b Second size to compare.
     * @return true if `a` is less than `b`, otherwise false.
     */
    inline bool operator<(const nvcv::Size2D &rhs) const
    {
        return std::tie(w, h) < std::tie(rhs.w, rhs.h);
    }
};

/**
 * @brief Computes the maximum size in each dimension
 *
 * @param a First size to compare.
 * @param b Second size to compare.
 * @return The size with `w` and `h` computed as a maximum of the respective fields in `a` and `b`.
 */
constexpr Size2D MaxSize(const Size2D &a, const Size2D &b)
{
    return {b.w > a.w ? b.w : a.w, b.h > a.h ? b.h : a.h};
}

/**
 * @brief Overloads the stream insertion operator for Size2D.
 *
 * This allows for easy printing of Size2D structures in the format "width x height".
 *
 * @param out Output stream to which the size string will be written.
 * @param size Size2D structure to be output.
 * @return Reference to the modified output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const nvcv::Size2D &size)
{
    return out << size.w << "x" << size.h;
}

/**@}*/

} // namespace nvcv

#endif // NVCV_SIZE_HPP
