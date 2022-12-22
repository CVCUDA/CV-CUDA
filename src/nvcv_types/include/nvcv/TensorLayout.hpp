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

#ifndef NVCV_TENSOR_LAYOUT_HPP
#define NVCV_TENSOR_LAYOUT_HPP

#include "TensorLayout.h"
#include "detail/CheckError.hpp"
#include "detail/Concepts.hpp"

#include <cassert>
#include <iostream>

inline bool operator==(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs, rhs) == 0;
}

inline bool operator!=(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return !operator==(lhs, rhs);
}

inline bool operator<(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs, rhs) < 0;
}

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

class TensorLayout final
{
public:
    using const_iterator = const char *;
    using iterator       = const_iterator;
    using value_type     = char;

    TensorLayout() = default;

    constexpr TensorLayout(const NVCVTensorLayout &layout)
        : m_layout(layout)
    {
    }

    explicit TensorLayout(const char *descr)
    {
        detail::CheckThrow(nvcvTensorLayoutMake(descr, &m_layout));
    }

    template<class IT, class = detail::IsRandomAccessIterator<IT>>
    explicit TensorLayout(IT itbeg, IT itend)
    {
        detail::CheckThrow(nvcvTensorLayoutMakeRange(&*itbeg, &*itend, &m_layout));
    }

    constexpr char operator[](int idx) const;
    constexpr int  rank() const;

    int find(char dimLabel, int start = 0) const;

    bool startsWith(const TensorLayout &test) const
    {
        return nvcvTensorLayoutStartsWith(m_layout, test.m_layout) != 0;
    }

    bool endsWith(const TensorLayout &test) const
    {
        return nvcvTensorLayoutEndsWith(m_layout, test.m_layout) != 0;
    }

    TensorLayout subRange(int beg, int end) const
    {
        TensorLayout out;
        detail::CheckThrow(nvcvTensorLayoutMakeSubRange(m_layout, beg, end, &out.m_layout));
        return out;
    }

    TensorLayout first(int n) const
    {
        TensorLayout out;
        detail::CheckThrow(nvcvTensorLayoutMakeFirst(m_layout, n, &out.m_layout));
        return out;
    }

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

    friend std::ostream &operator<<(std::ostream &out, const TensorLayout &that);

    // Public so that class is trivial but still the
    // implicit ctors do the right thing
    NVCVTensorLayout m_layout;

#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) static const TensorLayout LAYOUT;

    NVCV_DETAIL_DEF_TLAYOUT(NONE)
#include "TensorLayoutDef.inc"
#undef NVCV_DETAIL_DEF_TLAYOUT
};

#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) constexpr const TensorLayout TensorLayout::LAYOUT{NVCV_TENSOR_##LAYOUT};
NVCV_DETAIL_DEF_TLAYOUT(NONE)
#include "TensorLayoutDef.inc"
#undef NVCV_DETAIL_DEF_TLAYOUT

constexpr const TensorLayout &GetImplicitTensorLayout(int rank)
{
    // clang-format off
    return rank == 1
            ? TensorLayout::W
            : (rank == 2
                ? TensorLayout::HW
                : (rank == 3
                    ? TensorLayout::NHW
                    : (rank == 4
                        ? TensorLayout::NCHW
                        : (rank == 5
                            ? TensorLayout::NCDHW
                            : (rank == 6
                                ? TensorLayout::NCFDHW
                                : TensorLayout::NONE
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
