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

#ifndef NVCV_ARRAYDATA_HPP
#define NVCV_ARRAYDATA_HPP

#include "ArrayData.h"
#include "Optional.hpp"

#include <nvcv/DataType.hpp>

namespace nvcv {

// Interface hierarchy of array contents
class ArrayData
{
public:
    ArrayData(const NVCVArrayData &data);

    int     rank() const;
    int64_t length() const;
    int64_t capacity() const;

    DataType            dtype() const;
    NVCVArrayBufferType kind() const;

    Byte   *basePtr() const;
    int64_t stride() const;

    const NVCVArrayData &cdata() const &;

    NVCVArrayData cdata() &&
    {
        return this->cdata();
    }

    static bool IsCompatibleKind(NVCVArrayBufferType kind)
    {
        return kind != NVCV_ARRAY_BUFFER_NONE;
    }

    template<typename DerivedArrayData>
    Optional<DerivedArrayData> cast() const;

    template<typename Derived>
    bool IsCompatible() const;

protected:
    ArrayData() = default;

    NVCVArrayData &data() &;

private:
    NVCVArrayData m_data{};
};

class ArrayDataCuda : public ArrayData
{
public:
    using Buffer = NVCVArrayBufferStrided;

    ArrayDataCuda(const NVCVArrayData &data);
    ArrayDataCuda(int64_t length, const DataType &dtype, const Buffer &buffer);

    static bool IsCompatibleKind(NVCVArrayBufferType kind)
    {
        return static_cast<bool>(kind & NVCV_ARRAY_BUFFER_CUDA);
    }
};

class ArrayDataHost : public ArrayData
{
public:
    using Buffer = NVCVArrayBufferStrided;

    ArrayDataHost(const NVCVArrayData &data);
    ArrayDataHost(int64_t length, const DataType &dtype, const Buffer &buffer);

    static bool IsCompatibleKind(NVCVArrayBufferType kind)
    {
        return static_cast<bool>(kind & NVCV_ARRAY_BUFFER_HOST);
    }
};

class ArrayDataHostPinned : public ArrayData
{
public:
    using Buffer = NVCVArrayBufferStrided;

    ArrayDataHostPinned(const NVCVArrayData &data);
    ArrayDataHostPinned(int64_t length, const DataType &dtype, const Buffer &buffer);

    static bool IsCompatibleKind(NVCVArrayBufferType kind)
    {
        return kind == NVCV_ARRAY_BUFFER_HOST_PINNED;
    }
};

} // namespace nvcv

#include "detail/ArrayDataImpl.hpp"

#endif // NVCV_ARRAYDATA_HPP
