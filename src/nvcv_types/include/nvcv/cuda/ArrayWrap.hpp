/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_CUDA_ARRAY_WRAP_HPP
#define NVCV_CUDA_ARRAY_WRAP_HPP

#include "TypeTraits.hpp"

#include <assert.h>
#include <nvcv/ArrayData.hpp>
#include <nvcv/ArrayDataAccess.hpp>
#include <util/Assert.h>

#include <utility>

namespace nvcv::cuda {

template<typename ValueType>
class ArrayWrap
{
    static_assert(HasTypeTraits<ValueType>,
                  "The type T is not a valid NVCV Array element type. It must be a fundamental C type, "
                  "a CUDA vector type or a type for which TypeTraits are otherwise defined.");

    using itr_traits = std::iterator_traits<ValueType *>;

public:
    using size_type       = int32_t;
    using value_type      = typename itr_traits::value_type;
    using difference_type = typename itr_traits::difference_type;
    using pointer         = typename itr_traits::pointer;
    using reference       = typename itr_traits::reference;

    ArrayWrap() = default;

    explicit __host__ __device__ ArrayWrap(value_type *data, size_type length,
                                           size_type stride = static_cast<size_type>(sizeof(value_type)))
        : m_data{reinterpret_cast<std::byte *>(data)}
        , m_length{length}
        , m_stride{stride}
    {
        assert(length > 0 && stride >= static_cast<size_type>(sizeof(value_type)));
    }

    __host__ ArrayWrap(const ArrayData &data)
        : m_data{reinterpret_cast<std::byte *>(data.basePtr())}
        , m_length{static_cast<size_type>(data.capacity())}
        , m_stride{static_cast<size_type>(data.stride())}
    {
    }

    __host__ __device__ size_type length() const
    {
        return m_length;
    }

    __host__ __device__ size_type size() const
    {
        return this->length();
    }

    inline __host__ __device__ value_type &operator[](size_type c) const
    {
        return *doGetPtr(c);
    }

    inline __host__ __device__ pointer ptr(size_type c) const
    {
        return doGetPtr(c);
    }

    inline __host__ __device__ operator pointer() const
    {
        return reinterpret_cast<pointer>(m_data);
    }

protected:
    inline __host__ __device__ pointer doGetPtr(size_type c) const
    {
        assert(0 <= c && c < m_length);

        difference_type offset = c * m_stride;

        return reinterpret_cast<pointer>(m_data + offset);
    }

private:
    std::byte *m_data{nullptr};
    size_type  m_length{0};
    size_type  m_stride;
};

} // namespace nvcv::cuda

#endif // NVCV_CUDA_ARRAY_WRAP_HPP
