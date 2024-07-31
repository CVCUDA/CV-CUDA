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

#include "TensorBatch.hpp"

#include "Requirements.hpp"
#include "TensorBatchManager.hpp"

#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

namespace nvcv::priv {

TensorBatch::TensorBatch(const NVCVTensorBatchRequirements &reqs, IAllocator &alloc)
    : m_alloc(alloc)
    , m_reqs(reqs)
    , m_dirtyBegin(0)
    , m_dirtyEnd(0)
    , m_dtype(NVCV_DATA_TYPE_NONE)
    , m_layout(NVCV_TENSOR_LAYOUT_MAKE(""))
    , m_rank(-1)
    , m_userPointer(nullptr)
{
    m_evPostFence         = nullptr;
    m_devTensorsBuffer    = nullptr;
    m_pinnedTensorsBuffer = nullptr;
    m_Tensors             = nullptr;

    int64_t bufferSize = m_reqs.capacity * sizeof(BatchElement);

    try
    {
        m_devTensorsBuffer = static_cast<BatchElement *>(m_alloc->allocCudaMem(bufferSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devTensorsBuffer != nullptr);

        m_pinnedTensorsBuffer = static_cast<BatchElement *>(m_alloc->allocHostPinnedMem(bufferSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_pinnedTensorsBuffer != nullptr);

        m_Tensors = static_cast<NVCVTensorHandle *>(m_alloc->allocHostMem(bufferSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_Tensors != nullptr);

        NVCV_CHECK_THROW(cudaEventCreateWithFlags(&m_evPostFence, cudaEventDisableTiming));
    }
    catch (...)
    {
        cleanUp();
        throw;
    }
}

NVCVTensorBatchRequirements TensorBatch::CalcRequirements(int32_t capacity)
{
    NVCVTensorBatchRequirements reqs;
    reqs.capacity = capacity;
    reqs.mem      = {};

    reqs.alignBytes = alignof(BatchElement);
    reqs.alignBytes = util::RoundUpNextPowerOfTwo(reqs.alignBytes);

    if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                        NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
    }

    AddBuffer(reqs.mem.cudaMem, capacity * sizeof(BatchElement), reqs.alignBytes);
    AddBuffer(reqs.mem.hostPinnedMem, capacity * sizeof(BatchElement), reqs.alignBytes);
    AddBuffer(reqs.mem.hostMem, capacity * sizeof(BatchElement), reqs.alignBytes);

    return reqs;
}

TensorBatch::~TensorBatch()
{
    cleanUp();
}

void TensorBatch::cleanUp()
{
    if (m_evPostFence)
    {
        NVCV_CHECK_LOG(cudaEventDestroy(m_evPostFence));
    }

    for (int i = 0; i < m_numTensors; ++i)
    {
        CoreObjectDecRef(m_Tensors[i]);
    }

    int64_t bufferSize = m_reqs.capacity * sizeof(BatchElement);

    m_alloc->freeCudaMem(m_devTensorsBuffer, bufferSize, m_reqs.alignBytes);
    m_alloc->freeHostPinnedMem(m_pinnedTensorsBuffer, bufferSize, m_reqs.alignBytes);
    m_alloc->freeHostMem(m_Tensors, bufferSize, m_reqs.alignBytes);
}

void TensorBatch::exportData(CUstream stream, NVCVTensorBatchData &data)
{
    if (m_dirtyBegin < m_dirtyEnd)
    {
        // Block until the previous call to exportData finishes the buffer copy.
        NVCV_CHECK_THROW(cudaEventSynchronize(m_evPostFence));

        for (auto i = m_dirtyBegin; i < m_dirtyEnd; ++i)
        {
            auto          &t = ToStaticRef<ITensor>(m_Tensors[i]);
            NVCVTensorData tdata;
            t.exportData(tdata);
            auto &element = m_pinnedTensorsBuffer[i];
            element.data  = tdata.buffer.strided.basePtr;
            for (int d = 0; d < tdata.rank; ++d)
            {
                element.shape[d]  = tdata.shape[d];
                element.stride[d] = tdata.buffer.strided.strides[d];
            }
        }

        int64_t copySize = (m_dirtyEnd - m_dirtyBegin) * sizeof(BatchElement);
        NVCV_CHECK_THROW(cudaMemcpyAsync(m_devTensorsBuffer + m_dirtyBegin, m_pinnedTensorsBuffer + m_dirtyBegin,
                                         copySize, cudaMemcpyHostToDevice, stream));

        // Signal the buffer copy is finished.
        NVCV_CHECK_THROW(cudaEventRecord(m_evPostFence, stream));
        m_dirtyBegin = m_dirtyEnd;
    }
    NVCVTensorBatchBuffer buffer;
    buffer.strided  = NVCVTensorBatchBufferStrided{m_devTensorsBuffer};
    data.buffer     = buffer;
    data.type       = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    data.rank       = m_rank;
    data.dtype      = m_dtype;
    data.layout     = m_layout;
    data.numTensors = m_numTensors;
}

void TensorBatch::validateTensors(const NVCVTensorHandle *tensors, int32_t numTensors)
{
    for (int32_t i = 0; i < numTensors; ++i)
    {
        auto &t = ToStaticRef<ITensor>(tensors[i]);
        if (m_rank != -1 && t.rank() != m_rank)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                            "Trying to add a tensor to a tensor batch with an inconsistent rank.");
        }
        if (t.dtype().value() != m_dtype)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                            "Trying to add a tensor to a tensor batch with an inconsistent type.");
        }
        if (nvcvTensorLayoutCompare(t.layout(), m_layout) != 0)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                            "Trying to add a tensor to a tensor batch with an inconsistent layout.");
        }
    }
}

void TensorBatch::setLayoutAndDType(const NVCVTensorHandle *tensors, int32_t numTensors)
{
    if (numTensors > 0 && m_numTensors == 0)
    {
        auto &t  = ToStaticRef<ITensor>(tensors[0]);
        m_rank   = t.rank();
        m_dtype  = t.dtype().value();
        m_layout = t.layout();
    }
}

void TensorBatch::pushTensors(const NVCVTensorHandle *tensors, int32_t numTensors)
{
    if (numTensors == 0)
    {
        return;
    }
    if (numTensors < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Number of tensors cannot be nagative: " << numTensors;
    }
    if (m_numTensors + numTensors > capacity())
    {
        throw Exception(NVCV_ERROR_OVERFLOW)
            << "Adding " << numTensors << " tensors to a tensor batch would exceed its capacity (" << capacity()
            << ") by " << m_numTensors + numTensors - capacity();
    }
    setLayoutAndDType(tensors, numTensors);
    validateTensors(tensors, numTensors);
    for (int32_t i = 0; i < numTensors; ++i)
    {
        CoreObjectIncRef(tensors[i]);
        m_Tensors[m_numTensors + i] = tensors[i];
    }
    if (m_dirtyEnd == m_dirtyBegin)
    {
        m_dirtyBegin = m_numTensors;
    }
    m_numTensors += numTensors;
    m_dirtyEnd = m_numTensors;
}

void TensorBatch::popTensors(int32_t numTensors)
{
    if (numTensors < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Trying to pop a negative number of tensors: " << numTensors;
    }
    if (numTensors > m_numTensors)
    {
        throw Exception(NVCV_ERROR_UNDERFLOW)
            << "Trying to pop " << numTensors << " tensors from a tensor batch with " << m_numTensors << " tensors.";
    }
    for (int i = m_numTensors - numTensors; i < m_numTensors; ++i)
    {
        CoreObjectDecRef(m_Tensors[i]);
    }
    m_numTensors -= numTensors;
    m_dirtyEnd   = std::min(m_dirtyEnd, m_numTensors);
    m_dirtyBegin = std::min(m_dirtyBegin, m_dirtyEnd);
    if (m_numTensors == 0)
    {
        m_dtype  = NVCV_DATA_TYPE_NONE;
        m_layout = NVCV_TENSOR_LAYOUT_MAKE("");
        m_rank   = -1;
    }
}

void TensorBatch::getTensors(int32_t index, NVCVTensorHandle *tensors, int32_t numTensors) const
{
    if (index + numTensors > m_numTensors)
    {
        throw Exception(NVCV_ERROR_OVERFLOW) << "Trying to get a tensor on index " << index + numTensors
                                             << " while the tensor batch contains only " << m_numTensors << " tensors.";
    }
    if (index < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Trying to get a tensor with negative index: " << index;
    }
    std::copy(m_Tensors + index, m_Tensors + index + numTensors, tensors);
    for (int i = 0; i < numTensors; ++i)
    {
        CoreObjectIncRef(tensors[i]);
    }
}

void TensorBatch::setTensors(int32_t index, const NVCVTensorHandle *tensors, int32_t numTensors)
{
    if (index + numTensors > m_numTensors)
    {
        throw Exception(NVCV_ERROR_OVERFLOW) << "Trying to set a tensor on index " << index + numTensors
                                             << " while the tensor batch contains only " << m_numTensors << " tensors.";
    }
    if (index < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Trying to set a tensor with negative index: " << index;
    }
    validateTensors(tensors, numTensors);
    for (int32_t idx = 0; idx < numTensors; ++idx)
    {
        CoreObjectDecRef(m_Tensors[idx + index]);
        CoreObjectIncRef(tensors[idx]);
        m_Tensors[idx + index] = tensors[idx];
    }
    if (m_dirtyBegin != m_dirtyEnd)
    {
        m_dirtyBegin = std::min(m_dirtyBegin, index);
        m_dirtyEnd   = std::max(m_dirtyEnd, index + numTensors);
    }
    else
    {
        m_dirtyBegin = index;
        m_dirtyEnd   = m_dirtyBegin + numTensors;
    }
}

SharedCoreObj<IAllocator> TensorBatch::alloc() const
{
    return m_alloc;
}

int32_t TensorBatch::capacity() const
{
    return m_reqs.capacity;
}

int32_t TensorBatch::rank() const
{
    return m_rank;
}

int32_t TensorBatch::numTensors() const
{
    return m_numTensors;
}

NVCVDataType TensorBatch::dtype() const
{
    return m_dtype;
}

NVCVTensorLayout TensorBatch::layout() const
{
    return m_layout;
}

NVCVTensorBufferType TensorBatch::type() const
{
    return BUFFER_TYPE;
}

void TensorBatch::clear()
{
    for (int i = 0; i < m_numTensors; ++i)
    {
        CoreObjectDecRef(m_Tensors[i]);
    }
    m_numTensors = 0;
    m_dirtyBegin = 0;
    m_dirtyEnd   = 0;
    m_dtype      = NVCV_DATA_TYPE_NONE;
    m_layout     = NVCV_TENSOR_LAYOUT_MAKE("");
    m_rank       = -1;
}

} // namespace nvcv::priv
