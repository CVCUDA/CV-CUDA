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

#ifndef NVCV_CORE_PRIV_ITENSORBATCH_HPP
#define NVCV_CORE_PRIV_ITENSORBATCH_HPP

#include "ICoreObject.hpp"
#include "SharedCoreObj.hpp"

#include <nvcv/TensorBatch.h>

namespace nvcv::priv {

class IAllocator;

class ITensorBatch : public ICoreObjectHandle<ITensorBatch, NVCVTensorBatchHandle>
{
public:
    virtual int32_t              capacity() const   = 0;
    virtual int32_t              rank() const       = 0;
    virtual NVCVDataType         dtype() const      = 0;
    virtual int32_t              numTensors() const = 0;
    virtual NVCVTensorLayout     layout() const     = 0;
    virtual NVCVTensorBufferType type() const       = 0;

    virtual SharedCoreObj<IAllocator> alloc() const = 0;

    virtual void clear() = 0;

    virtual void pushTensors(const NVCVTensorHandle *tensors, int32_t numTensors) = 0;

    virtual void popTensors(int32_t numTensors) = 0;

    virtual void getTensors(int32_t index, NVCVTensorHandle *tensors, int32_t numTensors) const = 0;

    virtual void setTensors(int32_t index, const NVCVTensorHandle *tensors, int32_t numTensors) = 0;

    virtual void exportData(CUstream stream, NVCVTensorBatchData &data) = 0;
};

template<>
class CoreObjManager<NVCVTensorBatchHandle> : public HandleManager<ITensorBatch>
{
    using Base = HandleManager<ITensorBatch>;

public:
    using Base::Base;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TENSORBATCH_HPP
