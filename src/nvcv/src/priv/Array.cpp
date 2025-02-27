/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Array.hpp"

#include "DataLayout.hpp"
#include "DataType.hpp"
#include "IAllocator.hpp"
#include "Requirements.hpp"

#include <cuda_runtime.h>
#include <nvcv/Array.h>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nvcv::priv {

namespace detail {
constexpr NVCVArrayBufferType ResourceToBufferType(NVCVResourceType target)
{
    NVCVArrayBufferType result = NVCV_ARRAY_BUFFER_NONE;

    switch (target)
    {
    case NVCV_RESOURCE_MEM_CUDA:
        result = NVCV_ARRAY_BUFFER_CUDA;
        break;
    case NVCV_RESOURCE_MEM_HOST:
        result = NVCV_ARRAY_BUFFER_HOST;
        break;
    case NVCV_RESOURCE_MEM_HOST_PINNED:
        result = NVCV_ARRAY_BUFFER_HOST_PINNED;
        break;
    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Unknown Resource type " << target;
    }

    return result;
}
} // namespace detail

// Array implementation -------------------------------------------

NVCVArrayRequirements Array::CalcRequirements(int64_t capacity, const DataType &dtype, int32_t alignment,
                                              NVCVResourceType target)
{
    NVCVArrayRequirements reqs;

    reqs.dtype = dtype.value();

    reqs.capacity = capacity;

    reqs.mem = {};

    int dev;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    // Calculate alignment
    int align;
    {
        if (alignment == 0)
        {
            switch (target)
            {
            case NVCV_RESOURCE_MEM_CUDA:
            case NVCV_RESOURCE_MEM_HOST:
            case NVCV_RESOURCE_MEM_HOST_PINNED:
                align = dtype.alignment();
                break;
            default:
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Unknown Resource type " << target;
            }

            align = std::lcm(align, util::RoundUpNextPowerOfTwo(dtype.strideBytes()));
        }
        else
        {
            if (!util::IsPowerOfTwo(alignment))
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Invalid alignment of " << alignment << ", it must be a power-of-two";
            }
            // must at least satisfy dtype alignment
            align = std::lcm(alignment, dtype.alignment());
        }
    }
    reqs.alignBytes = align;

    reqs.stride = dtype.strideBytes();

    switch (target)
    {
    case NVCV_RESOURCE_MEM_CUDA:
        AddBuffer(reqs.mem.cudaMem, reqs.stride * reqs.capacity, reqs.alignBytes);
        break;
    case NVCV_RESOURCE_MEM_HOST:
        AddBuffer(reqs.mem.hostMem, reqs.stride * reqs.capacity, reqs.alignBytes);
        break;
    case NVCV_RESOURCE_MEM_HOST_PINNED:
        AddBuffer(reqs.mem.hostPinnedMem, reqs.stride * reqs.capacity, reqs.alignBytes);
        break;
    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Unknown Resource type " << target;
    }

    return reqs;
}

Array::Array(NVCVArrayRequirements reqs, IAllocator &alloc, NVCVResourceType target)
    : m_alloc{alloc}
    , m_reqs{std::move(reqs)}
    , m_target{target}
{
    // Assuming reqs are already validated during its creation

    int64_t bufSize;
    switch (m_target)
    {
    case NVCV_RESOURCE_MEM_CUDA:
        bufSize     = CalcTotalSizeBytes(m_reqs.mem.cudaMem);
        m_memBuffer = m_alloc->allocCudaMem(bufSize, m_reqs.alignBytes);
        break;
    case NVCV_RESOURCE_MEM_HOST:
        bufSize     = CalcTotalSizeBytes(m_reqs.mem.hostMem);
        m_memBuffer = m_alloc->allocHostMem(bufSize, m_reqs.alignBytes);
        break;
    case NVCV_RESOURCE_MEM_HOST_PINNED:
        bufSize     = CalcTotalSizeBytes(m_reqs.mem.hostPinnedMem);
        m_memBuffer = m_alloc->allocHostPinnedMem(bufSize, m_reqs.alignBytes);
        break;
    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Unknown Resource type " << m_target;
        break;
    }

    NVCV_ASSERT(m_memBuffer != nullptr);

    this->exportData(m_data);
    m_data.length = 0;
}

Array::~Array()
{
    switch (m_target)
    {
    case NVCV_RESOURCE_MEM_CUDA:
        m_alloc->freeCudaMem(m_memBuffer, CalcTotalSizeBytes(m_reqs.mem.cudaMem), m_reqs.alignBytes);
        break;
    case NVCV_RESOURCE_MEM_HOST:
        m_alloc->freeHostMem(m_memBuffer, CalcTotalSizeBytes(m_reqs.mem.hostMem), m_reqs.alignBytes);
        break;
    case NVCV_RESOURCE_MEM_HOST_PINNED:
        m_alloc->freeHostPinnedMem(m_memBuffer, CalcTotalSizeBytes(m_reqs.mem.hostPinnedMem), m_reqs.alignBytes);
        break;
    default:
        break;
    }
}

int32_t Array::rank() const
{
    return 1;
}

int64_t Array::capacity() const
{
    return m_reqs.capacity;
}

int64_t Array::length() const
{
    return m_data.length;
}

DataType Array::dtype() const
{
    return DataType{m_reqs.dtype};
}

SharedCoreObj<IAllocator> Array::alloc() const
{
    return m_alloc;
}

NVCVResourceType Array::target() const
{
    return m_target;
}

void Array::exportData(NVCVArrayData &data) const
{
    data.bufferType = detail::ResourceToBufferType(m_target);

    data.dtype = m_reqs.dtype;

    data.capacity = m_reqs.capacity;
    data.length   = this->length();

    auto &buf = data.buffer.strided;
    {
        buf.stride  = m_reqs.stride;
        buf.basePtr = reinterpret_cast<NVCVByte *>(m_memBuffer);
    }
}

void Array::resize(int64_t length)
{
    if (length <= this->capacity())
    {
        m_data.length = length;
    }
}

} // namespace nvcv::priv
