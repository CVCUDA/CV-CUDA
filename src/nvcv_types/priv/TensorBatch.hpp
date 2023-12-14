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

#ifndef NVCV_CORE_PRIV_TENSORBATCH_HPP
#define NVCV_CORE_PRIV_TENSORBATCH_HPP

#include "DataType.hpp"
#include "IAllocator.hpp"
#include "ITensorBatch.hpp"
#include "SharedCoreObj.hpp"
#include "Tensor.hpp"

#include <cuda_runtime.h>

namespace nvcv::priv {

class TensorBatch final : public CoreObjectBase<ITensorBatch>
{
public:
    using BatchElement                            = NVCVTensorBatchElementStrided;
    static const NVCVTensorBufferType BUFFER_TYPE = NVCV_TENSOR_BUFFER_STRIDED_CUDA;

    static NVCVTensorBatchRequirements CalcRequirements(int32_t capacity);

    TensorBatch(const NVCVTensorBatchRequirements &reqs, IAllocator &alloc);

    ~TensorBatch();

    SharedCoreObj<IAllocator> alloc() const override;

    int32_t capacity() const override;

    NVCVDataType dtype() const override;

    NVCVTensorLayout layout() const override;

    int32_t numTensors() const override;

    NVCVTensorBufferType type() const override;

    int32_t rank() const override;

    void clear() override;

    void exportData(CUstream stream, NVCVTensorBatchData &data) override;

    void pushTensors(const NVCVTensorHandle *tensors, int32_t numTensors) override;

    void popTensors(int32_t numTensors) override;

    void getTensors(int32_t index, NVCVTensorHandle *tensors, int32_t numTensors) const override;

    void setTensors(int32_t index, const NVCVTensorHandle *tensors, int32_t numTensors) override;

private:
    SharedCoreObj<IAllocator>   m_alloc;
    NVCVTensorBatchRequirements m_reqs;

    // Dirty begin and end describe a range containing all the tensors that have been modified
    // since the previous exportData call and thus should be updated in the exported buffer.
    int32_t m_dirtyBegin;
    int32_t m_dirtyEnd;

    int32_t m_numTensors = 0;

    NVCVTensorHandle              *m_Tensors; // host buffer for tensor handles
    // Pinned buffer for the tensor data descriptors
    // It's updated every time the user updates the tensor batch.
    // Changes are tracked with the m_dirty flags.
    NVCVTensorBatchElementStrided *m_pinnedTensorsBuffer;
    // Device buffer for the tensor data descriptors.
    // It's updated and returned when the exportData method is called.
    NVCVTensorBatchElementStrided *m_devTensorsBuffer;

    NVCVDataType     m_dtype;
    NVCVTensorLayout m_layout;
    int32_t          m_rank;

    // TODO: must be retrieved from the resource allocator;
    cudaEvent_t m_evPostFence;

    void *m_userPointer;

    void cleanUp();

    void validateTensors(const NVCVTensorHandle *tensors, int32_t numTensors);

    void setLayoutAndDType(const NVCVTensorHandle *tensors, int32_t numTensors);
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TENSORBATCH_HPP
