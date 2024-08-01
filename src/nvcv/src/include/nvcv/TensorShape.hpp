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

#ifndef NVCV_TENSORSHAPE_HPP
#define NVCV_TENSORSHAPE_HPP

#include "Shape.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.h"
#include "detail/Concepts.hpp"

#include <tuple>

namespace nvcv {
/**
 * @brief The TensorShape class represents the shape and layout of a tensor.
 */
class TensorShape
{
public:
    using DimType                 = int64_t;
    using ShapeType               = Shape<DimType, NVCV_TENSOR_MAX_RANK>;
    constexpr static int MAX_RANK = ShapeType::MAX_RANK;

    /**
     * @brief Default constructor.
     */
    TensorShape() = default;

    /**
     * @brief Constructs a TensorShape with the given shape and layout.
     *
     * @param shape Shape of the tensor.
     * @param layout Layout of the tensor.
     */
    TensorShape(ShapeType shape, TensorLayout layout)
        : m_shape(std::move(shape))
        , m_layout(std::move(layout))
    {
        if (m_layout != TENSOR_NONE && m_shape.rank() != m_layout.rank())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Layout dimensions must match shape dimensions");
        }
    }

    /**
     * @brief Constructs a TensorShape with the given size and layout.
     *
     * @param size Size of the tensor.
     * @param layout Layout of the tensor.
     */
    TensorShape(int size, TensorLayout layout)
        : TensorShape(ShapeType(size), std::move(layout))
    {
    }

    /**
     * @brief Constructs a TensorShape with the given layout.
     *
     * @param layout Layout of the tensor.
     */
    explicit TensorShape(TensorLayout layout)
        : TensorShape(layout.rank(), std::move(layout))
    {
    }

    /**
     * @brief Constructs a TensorShape with the given data, size, and layout.
     *
     * @param data Pointer to the array of tensor dimensions.
     * @param size Size of the tensor.
     * @param layout Layout of the tensor.
     */
    TensorShape(const DimType *data, int32_t size, TensorLayout layout)
        : TensorShape(ShapeType(data, size), std::move(layout))
    {
    }

    /**
     * @brief Constructs a TensorShape with the given data, size, and layout.
     *
     * @param data Pointer to the array of tensor dimensions.
     * @param size Size of the tensor.
     * @param layout Layout string of the tensor.
     */
    TensorShape(const DimType *data, int32_t size, const char *layout)
        : TensorShape(ShapeType(data, size), TensorLayout{layout})
    {
    }

    /**
     * @brief Constructs a TensorShape with the given shape and layout.
     *
     * @param shape Shape of the tensor.
     * @param layout Layout string of the tensor.
     */
    TensorShape(ShapeType shape, const char *layout)
        : TensorShape(std::move(shape), TensorLayout{layout})
    {
    }

    /**
     * @brief Returns the shape of the tensor.
     *
     * @return The shape of the tensor.
     */
    const ShapeType &shape() const
    {
        return m_shape;
    }

    /**
     * @brief Returns the layout of the tensor.
     *
     * @return The layout of the tensor.
     */
    const TensorLayout &layout() const
    {
        return m_layout;
    }

    /**
     * @brief Returns the dimension size at the given index.
     *
     * @param i Index of the dimension to return.
     * @return The dimension size at the given index.
     */
    const DimType &operator[](int i) const
    {
        return m_shape[i];
    }

    /**
     * @brief Returns the rank (number of dimensions) of the tensor.
     *
     * @return The rank of the tensor.
     */
    int rank() const
    {
        return m_shape.rank();
    }

    /**
     * @brief Returns the size (total number of elements) of the tensor.
     *
     * @return The size of the tensor.
     */
    int size() const
    {
        return m_shape.size();
    }

    /**
     * @brief Checks if the tensor is empty.
     *
     * @return True if the tensor is empty, false otherwise.
     */
    bool empty() const
    {
        return m_shape.empty();
    }

    /**
     * @brief Equality operator.
     *
     * @param that The TensorShape to compare with.
     * @return True if this TensorShape is equal to `that`, false otherwise.
     */
    bool operator==(const TensorShape &that) const
    {
        return std::tie(m_shape, m_layout) == std::tie(that.m_shape, that.m_layout);
    }

    /**
     * @brief Inequality operator.
     *
     * @param that The TensorShape to compare with.
     * @return True if this TensorShape is not equal to `that`, false otherwise.
     */
    bool operator!=(const TensorShape &that) const
    {
        return !(*this == that);
    }

    /**
     * @brief Less than operator.
     *
     * @param that The TensorShape to compare with.
     * @return True if this TensorShape is less than `that`, false otherwise.
     */
    bool operator<(const TensorShape &that) const
    {
        return std::tie(m_shape, m_layout) < std::tie(that.m_shape, that.m_layout);
    }

    /**
     * @brief Overload of the << operator for pretty printing.
     *
     * @param out The output stream.
     * @param ts The TensorShape to print.
     * @return The output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const TensorShape &ts)
    {
        if (ts.m_layout == TENSOR_NONE)
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

/**
 * @brief Function to permute the dimensions of a tensor to a new layout.
 *
 * This function rearranges the dimensions of the tensor according to a new layout.
 * It can be used to change the order of dimensions, for example, from NHWC (channel-last)
 * to NCHW (channel-first) and vice versa.
 *
 * @param src The original tensor shape.
 * @param dstLayout The desired layout after permutation.
 * @return The new TensorShape with permuted dimensions according to the desired layout.
 */
inline TensorShape Permute(const TensorShape &src, TensorLayout dstLayout)
{
    TensorShape::ShapeType dst(dstLayout.rank());
    detail::CheckThrow(nvcvTensorShapePermute(src.layout(), &src[0], dstLayout, &dst[0]));

    return {std::move(dst), std::move(dstLayout)};
}

} // namespace nvcv

#endif // NVCV_TENSORSHAPE_HPP
