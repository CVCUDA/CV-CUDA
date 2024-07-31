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

#ifndef NVCV_TENSOR_LAYOUT_HPP
#define NVCV_TENSOR_LAYOUT_HPP

#include "TensorLayout.h"
#include "detail/CheckError.hpp"
#include "detail/Concepts.hpp"

#include <cassert>
#include <iostream>

/**
 * @brief Compares two NVCVTensorLayouts for equality.
 *
 * @param lhs The left-hand side tensor layout to compare.
 * @param rhs The right-hand side tensor layout to compare.
 * @return true if both tensor layouts are equal, false otherwise.
 */
inline bool operator==(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs, rhs) == 0;
}

/**
 * @brief Compares two NVCVTensorLayouts for inequality.
 *
 * @param lhs The left-hand side tensor layout to compare.
 * @param rhs The right-hand side tensor layout to compare.
 * @return true if both tensor layouts are not equal, false otherwise.
 */
inline bool operator!=(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return !operator==(lhs, rhs);
}

/**
 * @brief Compares two NVCVTensorLayouts to determine if one is less than the other.
 *
 * @param lhs The left-hand side tensor layout to compare.
 * @param rhs The right-hand side tensor layout to compare.
 * @return true if lhs is less than rhs, false otherwise.
 */
inline bool operator<(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs, rhs) < 0;
}

/**
 * @brief Outputs the name of an NVCVTensorLayout to a stream.
 *
 * @param out The output stream.
 * @param layout The tensor layout whose name will be output.
 * @return The output stream.
 */
inline std::ostream &operator<<(std::ostream &out, const NVCVTensorLayout &layout)
{
    return out << nvcvTensorLayoutGetName(&layout);
}

namespace nvcv {

enum TensorLabel : char
{
    LABEL_BATCH   = NVCV_TLABEL_BATCH,
    LABEL_CHANNEL = NVCV_TLABEL_CHANNEL,
    LABEL_FRAME   = NVCV_TLABEL_FRAME,
    LABEL_DEPTH   = NVCV_TLABEL_DEPTH,
    LABEL_HEIGHT  = NVCV_TLABEL_HEIGHT,
    LABEL_WIDTH   = NVCV_TLABEL_WIDTH
};

/**
 * @brief Represents the layout of a tensor.
 *
 * This class wraps around the NVCVTensorLayout structure and provides additional
 * functionality for handling and manipulating tensor layouts. The class allows
 * for easy construction from character descriptions or from other tensor layout
 * structures. It also supports a range of operations including subsetting and
 * checking for prefixes or suffixes.
 */
class TensorLayout final
{
public:
    using const_iterator = const char *;
    using iterator       = const_iterator;
    using value_type     = char;

    TensorLayout() = default;

    /**
     * @brief Constructs a TensorLayout from an NVCVTensorLayout.
     * @param layout The NVCVTensorLayout to wrap.
     */
    constexpr TensorLayout(const NVCVTensorLayout &layout)
        : m_layout(layout)
    {
    }

    /**
     * @brief Constructs a TensorLayout from a character description.
     * @param descr The character description of the layout.
     */
    explicit TensorLayout(const char *descr)
    {
        detail::CheckThrow(nvcvTensorLayoutMake(descr, &m_layout));
    }

    /**
     * @brief Constructs a TensorLayout from a range of iterators.
     * @tparam IT The type of the iterators.
     * @param itbeg The beginning of the range.
     * @param itend The end of the range.
     */
    template<class IT, class = detail::IsRandomAccessIterator<IT>>
    explicit TensorLayout(IT itbeg, IT itend)
    {
        detail::CheckThrow(nvcvTensorLayoutMakeRange(&*itbeg, &*itend, &m_layout));
    }

    /**
     * @brief Fetches the character representation of a dimension at a specified index.
     * @param idx The index of the dimension.
     * @return The character representation of the dimension.
     */
    constexpr char operator[](int idx) const;

    /**
     * @brief Gets the rank (number of dimensions) of the tensor layout.
     * @return The rank of the tensor layout.
     */
    constexpr int rank() const;

    /**
     * @brief Finds the index of the first occurrence of a specified dimension label.
     * @param dimLabel The dimension label to search for.
     * @param start The starting index for the search.
     * @return The index of the first occurrence of the dimension label, or -1 if not found.
     */
    int find(char dimLabel, int start = 0) const;

    /**
     * @brief Checks if the current layout starts with a specified layout.
     * @param test The layout to check against.
     * @return true if the current layout starts with the test layout, false otherwise.
     */
    bool startsWith(const TensorLayout &test) const
    {
        return nvcvTensorLayoutStartsWith(m_layout, test.m_layout) != 0;
    }

    /**
     * @brief Checks if the current layout ends with a specified layout.
     * @param test The layout to check against.
     * @return true if the current layout ends with the test layout, false otherwise.
     */
    bool endsWith(const TensorLayout &test) const
    {
        return nvcvTensorLayoutEndsWith(m_layout, test.m_layout) != 0;
    }

    /**
     * @brief Creates a sub-layout from a specified range.
     * @param beg The starting index of the range.
     * @param end The ending index of the range.
     * @return A new TensorLayout representing the sub-range.
     */
    TensorLayout subRange(int beg, int end) const
    {
        TensorLayout out;
        detail::CheckThrow(nvcvTensorLayoutMakeSubRange(m_layout, beg, end, &out.m_layout));
        return out;
    }

    /**
     * @brief Creates a sub-layout consisting of the first n dimensions.
     * @param n The number of dimensions to include.
     * @return A new TensorLayout representing the first n dimensions.
     */
    TensorLayout first(int n) const
    {
        TensorLayout out;
        detail::CheckThrow(nvcvTensorLayoutMakeFirst(m_layout, n, &out.m_layout));
        return out;
    }

    /**
     * @brief Creates a sub-layout consisting of the last n dimensions.
     * @param n The number of dimensions to include.
     * @return A new TensorLayout representing the last n dimensions.
     */
    TensorLayout last(int n) const
    {
        TensorLayout out;
        detail::CheckThrow(nvcvTensorLayoutMakeLast(m_layout, n, &out.m_layout));
        return out;
    }

    friend bool operator==(const TensorLayout &a, const TensorLayout &b);
    bool        operator!=(const TensorLayout &that) const;
    bool        operator<(const TensorLayout &that) const;

    constexpr const_iterator begin() const;
    constexpr const_iterator end() const;
    constexpr const_iterator cbegin() const;
    constexpr const_iterator cend() const;

    constexpr operator const NVCVTensorLayout &() const;

    /**
     * @brief Outputs the TensorLayout to a stream.
     * @param out The output stream.
     * @param that The TensorLayout to output.
     * @return The output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const TensorLayout &that);

    // Public so that class is trivial but still the
    // implicit ctors do the right thing
    NVCVTensorLayout m_layout;
};

#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) constexpr const TensorLayout TENSOR_##LAYOUT{NVCV_TENSOR_##LAYOUT};
NVCV_DETAIL_DEF_TLAYOUT(NONE)
#include "TensorLayoutDef.inc"
#undef NVCV_DETAIL_DEF_TLAYOUT

/**
 * @brief Retrieves the default tensor layout based on the rank (number of dimensions).
 *
 * This function maps commonly used tensor ranks to their typical tensor layouts.
 * For example, a rank of 4 typically corresponds to the NCHW layout (Batch, Channel, Height, Width).
 *
 * @param rank The rank (number of dimensions) of the tensor.
 * @return The corresponding default tensor layout. Returns TENSOR_NONE for unsupported ranks.
 */
constexpr const TensorLayout &GetImplicitTensorLayout(int rank)
{
    // clang-format off
    return rank == 1
            ? TENSOR_W
            : (rank == 2
                ? TENSOR_HW
                : (rank == 3
                    ? TENSOR_NHW
                    : (rank == 4
                        ? TENSOR_NCHW
                        : (rank == 5
                            ? TENSOR_NCDHW
                            : (rank == 6
                                ? TENSOR_NCFDHW
                                : TENSOR_NONE
                              )
                          )
                      )
                  )
              );
    // clang-format on
}

constexpr char TensorLayout::operator[](int idx) const
{
    return nvcvTensorLayoutGetLabel(m_layout, idx);
}

constexpr int TensorLayout::rank() const
{
    return nvcvTensorLayoutGetNumDim(m_layout);
}

inline int TensorLayout::find(char dimLabel, int start) const
{
    return nvcvTensorLayoutFindDimIndex(m_layout, dimLabel, start);
}

constexpr TensorLayout::operator const NVCVTensorLayout &() const
{
    return m_layout;
}

inline bool operator==(const TensorLayout &a, const TensorLayout &b)
{
    return a.m_layout == b.m_layout;
}

inline bool TensorLayout::operator!=(const TensorLayout &that) const
{
    return !(*this == that);
}

inline bool TensorLayout::operator<(const TensorLayout &that) const
{
    return m_layout < that.m_layout;
}

constexpr auto TensorLayout::begin() const -> const_iterator
{
    return nvcvTensorLayoutGetName(&m_layout);
}

constexpr inline auto TensorLayout::end() const -> const_iterator
{
    return this->begin() + this->rank();
}

constexpr auto TensorLayout::cbegin() const -> const_iterator
{
    return this->begin();
}

constexpr auto TensorLayout::cend() const -> const_iterator
{
    return this->end();
}

inline std::ostream &operator<<(std::ostream &out, const TensorLayout &that)
{
    return out << that.m_layout;
}

// For disambiguation
inline bool operator==(const TensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs.m_layout, rhs) == 0;
}

inline bool operator!=(const TensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return !operator==(lhs, rhs);
}

inline bool operator<(const TensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs.m_layout, rhs) < 0;
}

} // namespace nvcv

#endif // NVCV_TENSOR_LAYOUT_HPP
