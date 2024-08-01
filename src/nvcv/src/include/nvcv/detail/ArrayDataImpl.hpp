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

#ifndef NVCV_ARRAYDATA_IMPL_HPP
#define NVCV_ARRAYDATA_IMPL_HPP

#ifndef NVCV_ARRAYDATA_HPP
#    error "You must not include this header directly"
#endif

#include <algorithm>

namespace nvcv {

// Implementation - ArrayData -----------------------------

inline ArrayData::ArrayData(const NVCVArrayData &data)
    : m_data(data)
{
}

inline int ArrayData::rank() const
{
    return 1;
}

inline int64_t ArrayData::length() const
{
    return this->cdata().length;
}

inline int64_t ArrayData::capacity() const
{
    return this->cdata().capacity;
}

inline DataType ArrayData::dtype() const
{
    const NVCVArrayData &data = this->cdata();
    return DataType{data.dtype};
}

inline NVCVArrayBufferType ArrayData::kind() const
{
    return this->cdata().bufferType;
}

inline const NVCVArrayData &ArrayData::cdata() const &
{
    return m_data;
}

inline NVCVArrayData &ArrayData::data() &
{
    return m_data;
}

template<typename Derived>
bool ArrayData::IsCompatible() const
{
    return Derived::IsCompatibleKind(this->cdata().bufferType);
}

template<typename Derived>
inline Optional<Derived> ArrayData::cast() const
{
    static_assert(std::is_base_of<ArrayData, Derived>::value, "Cannot cast ArrayData to an unrelated type");

    static_assert(sizeof(Derived) == sizeof(ArrayData), "The derived type must not add new data members.");

    if (IsCompatible<Derived>())
    {
        return Derived{this->cdata()};
    }
    else
    {
        return NullOpt;
    }
}

inline Byte *ArrayData::basePtr() const
{
    const NVCVArrayBufferStrided &buffer = this->cdata().buffer.strided;
    return reinterpret_cast<Byte *>(buffer.basePtr);
}

inline int64_t ArrayData::stride() const
{
    return this->cdata().buffer.strided.stride;
}

// Derived implementation -----------------------

inline ArrayDataCuda::ArrayDataCuda(const NVCVArrayData &data)
{
    if (!ArrayDataCuda::IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }

    this->data() = data;
}

inline ArrayDataCuda::ArrayDataCuda(int64_t length, const DataType &dtype, const Buffer &buffer)
{
    auto &data = this->data();

    data.length   = length;
    data.capacity = length;
    data.dtype    = dtype;

    data.bufferType     = NVCV_ARRAY_BUFFER_CUDA;
    data.buffer.strided = buffer;
}

inline ArrayDataHost::ArrayDataHost(const NVCVArrayData &data)
{
    if (!ArrayDataHost::IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }

    this->data() = data;
}

inline ArrayDataHost::ArrayDataHost(int64_t length, const DataType &dtype, const Buffer &buffer)
{
    auto &data = this->data();

    data.length   = length;
    data.capacity = length;
    data.dtype    = dtype;

    data.bufferType     = NVCV_ARRAY_BUFFER_HOST;
    data.buffer.strided = buffer;
}

inline ArrayDataHostPinned::ArrayDataHostPinned(const NVCVArrayData &data)
{
    if (!ArrayDataHostPinned::IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }

    this->data() = data;
}

inline ArrayDataHostPinned::ArrayDataHostPinned(int64_t length, const DataType &dtype, const Buffer &buffer)
{
    auto &data = this->data();

    data.length   = length;
    data.capacity = length;
    data.dtype    = dtype;

    data.bufferType     = NVCV_ARRAY_BUFFER_HOST_PINNED;
    data.buffer.strided = buffer;
}

} // namespace nvcv

#endif // NVCV_ARRAYDATA_IMPL_HPP
