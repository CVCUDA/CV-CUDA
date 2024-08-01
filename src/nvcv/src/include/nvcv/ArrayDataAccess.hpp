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

#ifndef NVCV_ARRAYDATAACESSOR_HPP
#define NVCV_ARRAYDATAACESSOR_HPP

#include "ArrayData.hpp"

#include <type_traits>

namespace nvcv {

namespace detail {
#ifdef __cpp_lib_is_invocable
template<typename TypeExpression>
struct invoke_result : public std::invoke_result<TypeExpression>
{
};
#else  // __cpp_lib_is_invocable
template<typename TypeExpression>
struct invoke_result : public std::result_of<TypeExpression>
{
};
#endif // __cpp_lib_is_invocable

template<typename ArrayDataType,
         typename = typename std::enable_if<std::is_base_of<ArrayData, ArrayDataType>::value>::type>
class ArrayDataAccessImpl
{
    using traits = std::pointer_traits<Byte *>;

public:
    using ArrayType = ArrayDataType;

    using pointer         = typename traits::pointer;
    using difference_type = typename traits::difference_type;

    ArrayDataAccessImpl() = delete;

    int64_t length() const
    {
        return m_length;
    }

    DataType dtype() const
    {
        return m_data.dtype();
    }

    int64_t stride() const
    {
        return m_data.stride();
    }

    NVCVArrayBufferType kind() const
    {
        return m_data.kind();
    }

    pointer sampleData(int64_t n) const
    {
        auto result = m_data.basePtr();

        if ((n + m_idxShift) >= m_length)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Requested index is out of bounds.");
        }

        result += m_data.stride() * m_idxShift;
        result += m_memShift;

        return result;
    }

    pointer ptr() const
    {
        auto result = m_data.basePtr();

        if (m_idxShift > 0)
        {
            if (m_idxShift >= m_length)
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "Requested index is out of bounds.");
            }

            result += m_data.stride() * m_idxShift;
            result += m_memShift;
        }

        return result;
    }

protected:
    ArrayType m_data;

    ArrayDataAccessImpl(const ArrayType &data)
        : m_data{data}
        , m_length{data.length()}
        , m_idxShift{0}
        , m_memShift{0}
    {
    }

    ArrayDataAccessImpl(const ArrayType &data, int64_t _length, const pointer _start)
        : ArrayDataAccessImpl{data}
    {
        auto length = _length == 0 ? m_data.length() : _length;
        auto start  = _start == nullptr ? m_data.basePtr() : _start;

        auto memLineRange = start - m_data.basePtr();
        auto itrEnd       = m_data.basePtr();
        itrEnd += m_data.stride() * m_data.capacity();

        if (start && m_data.basePtr() <= start && start < itrEnd)
        {
            m_memShift = memLineRange % m_data.stride();
        }
        else
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Requested start address is out of bounds.");
        }

        auto itrAt = memLineRange / m_data.stride();
        if ((itrAt + length) <= m_data.capacity())
        {
            m_length   = length;
            m_idxShift = itrAt;
        }
        else
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Requested array length is out of bounds.");
        }
    }

private:
    int64_t         m_length;
    int64_t         m_idxShift;
    difference_type m_memShift;
};
} // namespace detail

class ArrayDataAccess : public detail::ArrayDataAccessImpl<ArrayData>
{
    using Base = detail::ArrayDataAccessImpl<ArrayData>;

public:
    static bool IsCompatible(const ArrayData &data)
    {
        return data.IsCompatible<ArrayData>();
    }

    static Optional<ArrayDataAccess> Create(const ArrayData &data, int64_t length = 0, const pointer start = nullptr)
    {
        auto castData = data.cast<ArrayData>();
        if (castData)
        {
            return ArrayDataAccess{castData.value(), length, start};
        }
        else
        {
            return NullOpt;
        }
    }

private:
    ArrayDataAccess(const ArrayData &data, int64_t length, const pointer start)
        : Base{data, length, start}
    {
    }
};

class ArrayDataAccessHost : public detail::ArrayDataAccessImpl<ArrayDataHost>
{
    using Base = detail::ArrayDataAccessImpl<ArrayDataHost>;

public:
    static bool IsCompatible(const ArrayData &data)
    {
        return data.IsCompatible<ArrayDataHost>();
    }

    static Optional<ArrayDataAccessHost> Create(const ArrayData &data, int64_t length = 0,
                                                const pointer start = nullptr)
    {
        auto castData = data.cast<ArrayDataHost>();
        if (castData)
        {
            return ArrayDataAccessHost{castData.value(), length, start};
        }
        else
        {
            return NullOpt;
        }
    }

private:
    ArrayDataAccessHost(const ArrayDataHost &data, int64_t length, const pointer start)
        : Base{data, length, start}
    {
    }
};

class ArrayDataAccessHostPinned : public detail::ArrayDataAccessImpl<ArrayDataHostPinned>
{
    using Base = detail::ArrayDataAccessImpl<ArrayDataHostPinned>;

public:
    static bool IsCompatible(const ArrayData &data)
    {
        return data.IsCompatible<ArrayDataHostPinned>();
    }

    static Optional<ArrayDataAccessHostPinned> Create(const ArrayData &data, int64_t length = 0,
                                                      const pointer start = nullptr)
    {
        auto castData = data.cast<ArrayDataHostPinned>();
        if (castData)
        {
            return ArrayDataAccessHostPinned{castData.value(), length, start};
        }
        else
        {
            return NullOpt;
        }
    }

private:
    ArrayDataAccessHostPinned(const ArrayDataHostPinned &data, int64_t length, const pointer start)
        : Base{data, length, start}
    {
    }
};

class ArrayDataAccessCuda : public detail::ArrayDataAccessImpl<ArrayDataCuda>
{
    using Base = detail::ArrayDataAccessImpl<ArrayDataCuda>;

public:
    static bool IsCompatible(const ArrayData &data)
    {
        return data.IsCompatible<ArrayDataCuda>();
    }

    static Optional<ArrayDataAccessCuda> Create(const ArrayData &data, int64_t length = 0,
                                                const pointer start = nullptr)
    {
        auto castData = data.cast<ArrayDataCuda>();
        if (castData)
        {
            return ArrayDataAccessCuda{castData.value(), length, start};
        }
        else
        {
            return NullOpt;
        }
    }

private:
    ArrayDataAccessCuda(const ArrayDataCuda &data, int64_t length, const pointer start)
        : Base{data, length, start}
    {
    }
};

} // namespace nvcv

#endif // NVCV_ARRAYDATAACESSOR_HPP
