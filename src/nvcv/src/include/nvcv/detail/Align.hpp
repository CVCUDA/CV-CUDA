/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_DETAIL_ALIGN_HPP
#define NVCV_DETAIL_ALIGN_HPP

#include <cstdint>
#include <type_traits>

namespace nvcv { namespace detail {

/**
 * @brief Aligns the @p value down to a multiple of @p alignment_pow2
 *
 * The function operates by masking the least significant bits of the value.
 * If the alignment is not a power of two, the behavior is undefined.
 *
 * @remark Negative values are aligned down, not towards zero.
 *
 * @tparam T                an integral type
 * @param value             a value to align
 * @param alignment_pow2    the alignment, must be a positive power of 2
 * @return constexpr T      the value aligned down to a multiple of @p alignment_pow2
 */
template<typename T>
constexpr T AlignDown(T value, T alignment_pow2)
{
    static_assert(std::is_integral<T>::value, "Cannot align a value of a non-integral type");
    // Explanation:
    // When alignmnent_pow2 is a power of 2, (for example 16) it has a form:
    // 00010000
    // Negating it in U2 gives:
    // 11110000
    // We can use this as a mask to align a number _down_.

    // NOTE: This is much more efficient than (value/alignment) * alignment for run-time alignment values, where
    //       the compiler cannot replace the division/multiplication with bit shifts.
    return value & -alignment_pow2;
}

/**
 * @brief Aligns the @p value up to a multiple of @p alignment_pow2
 *
 * The function operates by adding alignment-1 to the value and masking the least significant bits.
 * If the alignment is not a power of two, the behavior is undefined.
 *
 * @remark Negative values are aligned up, that is, towards zero.
 *
 * @tparam T                an integral type
 * @param value             a value to align
 * @param alignment_pow2    the alignment, must be a positive power of 2
 * @return constexpr T      the value aligned up to a multiple of @p alignment_pow2
 */
template<typename T>
constexpr T AlignUp(T value, T alignment_pow2)
{
    static_assert(std::is_integral<T>::value, "Cannot align a value of a non-integral type");
    return AlignDown(value + (alignment_pow2 - 1), alignment_pow2);
}

/**
 * @brief Checks if the value is a multiple of alignment
 *
 * @tparam T                an integral type
 * @param value             the value whose alignment is checked
 * @param alignment_pow2    the alignment, must be a power of 2
 * @return true             if value is a multiple of alignment_pow2
 * @return false            otherwise
 */
template<typename T>
constexpr bool IsAligned(T value, T alignment_pow2)
{
    static_assert(std::is_integral<T>::value, "Cannot check alignment of a value of a non-integral type");
    return (value & (alignment_pow2 - 1)) == 0;
}

/**
 * @brief Checks if a pointer is aligned to a multiple of @p alignment_pow2 bytes.
 *
 * @param ptr               the pointer whose alignment is checked
 * @param alignment_pow2    the alignment, must be a power of 2
 * @return true             if value is a multiple of alignment_pow2
 * @return false            otherwise
 */
inline bool IsAligned(const void *ptr, uintptr_t alignment_pow2)
{
    return IsAligned((uintptr_t)ptr, alignment_pow2);
}

}} // namespace nvcv::detail

#endif // NVCV_DETAIL_ALIGN_HPP
