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

#include "ArrayWrapData.hpp"

#include "DataType.hpp"
#include "IAllocator.hpp"
#include "Requirements.hpp"

#include <cuda_runtime.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nvcv::priv {

static NVCVResourceType ValidateArrayBuffer(const NVCVArrayData &data)
{
    NVCVResourceType resource = NVCV_RESOURCE_MEM_CUDA;

    const auto &buffer = data.buffer.strided;
    if (buffer.basePtr == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Memory buffer must not be NULL";
    }

    {
        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, reinterpret_cast<void *>(data.buffer.strided.basePtr));
#if CUDART_VERSION >= 11000
        if (err == cudaSuccess && attr.type == cudaMemoryTypeUnregistered)
#else  // CUDART_VERSION < 11.0
        if (err == cudaErrorInvalidValue)
#endif // CUDART_VERSION
        {
            resource = NVCV_RESOURCE_MEM_HOST;
        }
        else if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged))
        {
            resource = NVCV_RESOURCE_MEM_CUDA;
        }
        else if (err == cudaSuccess && attr.type == cudaMemoryTypeHost)
        {
            resource = NVCV_RESOURCE_MEM_HOST_PINNED;
        }
        else
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Unknown buffer allocation.";
        }
    }

    return resource;
}

ArrayWrapData::ArrayWrapData(const NVCVArrayData &data, NVCVArrayDataCleanupFunc cleanup, void *ctxCleanup)
    : m_data{data}
    , m_cleanup{cleanup}
    , m_ctxCleanup{ctxCleanup}
{
    m_target = ValidateArrayBuffer(data);
}

ArrayWrapData::~ArrayWrapData()
{
    if (m_cleanup)
    {
        m_cleanup(m_ctxCleanup, &m_data);
    }
}

int32_t ArrayWrapData::rank() const
{
    return 1;
}

int64_t ArrayWrapData::capacity() const
{
    return m_data.capacity;
}

int64_t ArrayWrapData::length() const
{
    return m_data.length;
}

DataType ArrayWrapData::dtype() const
{
    return DataType{m_data.dtype};
}

SharedCoreObj<IAllocator> ArrayWrapData::alloc() const
{
    return GetDefaultAllocator();
}

NVCVResourceType ArrayWrapData::target() const
{
    return m_target;
}

void ArrayWrapData::exportData(NVCVArrayData &data) const
{
    data = m_data;
}

} // namespace nvcv::priv
