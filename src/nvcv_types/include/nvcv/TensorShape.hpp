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

#ifndef NVCV_TENSORSHAPE_HPP
#define NVCV_TENSORSHAPE_HPP

#include "Shape.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.h"
#include "detail/Concepts.hpp"

namespace nvcv {

class TensorShape
{
public:
    using DimType                 = int64_t;
    using ShapeType               = Shape<DimType, NVCV_TENSOR_MAX_RANK>;
    constexpr static int MAX_RANK = ShapeType::MAX_RANK;

    TensorShape() = default;

    TensorShape(ShapeType shape, TensorLayout layout)
        : m_shape(std::move(shape))
        , m_layout(std::move(layout))
    {
        if (m_layout != TensorLayout::NONE && m_shape.rank() != m_layout.rank())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Layout dimensions must match shape dimensions");
        }
    }

    TensorShape(int size, TensorLayout layout)
        : TensorShape(ShapeType(size), std::move(layout))
    {
    }

    explicit TensorShape(TensorLayout layout)
        : TensorShape(layout.rank(), std::move(layout))
    {
    }

    TensorShape(const DimType *data, int32_t size, TensorLayout layout)
        : TensorShape(ShapeType(data, size), std::move(layout))
    {
    }

    TensorShape(const DimType *data, int32_t size, const char *layout)
        : TensorShape(ShapeType(data, size), TensorLayout{layout})
    {
    }

    TensorShape(ShapeType shape, const char *layout)
        : TensorShape(std::move(shape), TensorLayout{layout})
    {
    }

    const ShapeType &shape() const
    {
        return m_shape;
    }

    const TensorLayout &layout() const
    {
        return m_layout;
    }

    const DimType &operator[](int i) const
    {
        return m_shape[i];
    }

    int rank() const
    {
        return m_shape.rank();
    }

    int size() const
    {
        return m_shape.size();
    }

    bool empty() const
    {
        return m_shape.empty();
    }

    bool operator==(const TensorShape &that) const
    {
        return std::tie(m_shape, m_layout) == std::tie(that.m_shape, that.m_layout);
    }

    bool operator!=(const TensorShape &that) const
    {
        return !(*this == that);
    }

    bool operator<(const TensorShape &that) const
    {
        return std::tie(m_shape, m_layout) < std::tie(that.m_shape, that.m_layout);
    }

    friend std::ostream &operator<<(std::ostream &out, const TensorShape &ts)
    {
        if (ts.m_layout == TensorLayout::NONE)
        {
            return out << ts.m_shape;
        }
        else
        {
            return out << ts.m_layout << '{' << ts.m_shape << '}';
        }
    }

private:
    ShapeType    m_shape;
    TensorLayout m_layout;
};

inline TensorShape Permute(const TensorShape &src, TensorLayout dstLayout)
{
    TensorShape::ShapeType dst(dstLayout.rank());
    detail::CheckThrow(nvcvTensorShapePermute(src.layout(), &src[0], dstLayout, &dst[0]));

    return {std::move(dst), std::move(dstLayout)};
}

} // namespace nvcv

#endif // NVCV_TENSORSHAPE_HPP
