/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/TensorBatch.hpp"

#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"
#include "priv/TensorBatchManager.hpp"

#include <nvcv/TensorBatch.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchCalcRequirements,
                (int32_t capacity, NVCVTensorBatchRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            *reqs = priv::TensorBatch::CalcRequirements(capacity);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchConstruct,
                (const NVCVTensorBatchRequirements *reqs, NVCVAllocatorHandle halloc, NVCVTensorBatchHandle *outHandle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to requirements must not be NULL");
            }
            if (outHandle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);
            *outHandle              = priv::CreateCoreObject<priv::TensorBatch>(*reqs, alloc);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchClear, (NVCVTensorBatchHandle handle))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.clear();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchPushTensors,
                (NVCVTensorBatchHandle handle, const NVCVTensorHandle *tensors, int32_t numTensors))
{
    return priv::ProtectCall(
        [&]
        {
            if (tensors == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to tensors must not be NULL");
            }
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.pushTensors(tensors, numTensors);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchDecRef, (NVCVTensorBatchHandle handle, int32_t *newRefCount))
{
    return priv::ProtectCall(
        [&]
        {
            int32_t newRef = priv::CoreObjectDecRef(handle);
            if (newRefCount)
                *newRefCount = newRef;
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchIncRef, (NVCVTensorBatchHandle handle, int32_t *newRefCount))
{
    return priv::ProtectCall(
        [&]
        {
            int32_t refCount = priv::CoreObjectIncRef(handle);
            if (newRefCount)
                *newRefCount = refCount;
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchRefCount, (NVCVTensorBatchHandle handle, int32_t *outRefCount))
{
    return priv::ProtectCall(
        [&]
        {
            if (outRefCount == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to reference count must not be NULL");
            }
            *outRefCount = priv::CoreObjectRefCount(handle);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetCapacity, (NVCVTensorBatchHandle handle, int32_t *outCapacityPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outCapacityPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to capacity must not be NULL");
            }
            auto &tb        = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outCapacityPtr = tb.capacity();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetRank, (NVCVTensorBatchHandle handle, int32_t *outRankPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outRankPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to rank must not be NULL");
            }
            auto &tb    = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outRankPtr = tb.rank();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetDType, (NVCVTensorBatchHandle handle, NVCVDataType *outDTypePtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outDTypePtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type must not be NULL");
            }
            auto &tb     = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outDTypePtr = tb.dtype();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetLayout,
                (NVCVTensorBatchHandle handle, NVCVTensorLayout *outLayoutPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outLayoutPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to layout must not be NULL");
            }
            auto &tb      = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outLayoutPtr = tb.layout();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetType,
                (NVCVTensorBatchHandle handle, NVCVTensorBufferType *outTypePtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outTypePtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to buffer type must not be NULL");
            }
            auto &tb    = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outTypePtr = tb.type();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetNumTensors,
                (NVCVTensorBatchHandle handle, int32_t *outNumTensorsPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumTensorsPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to tensors number must not be NULL");
            }
            auto &tb          = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outNumTensorsPtr = tb.numTensors();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetAllocator,
                (NVCVTensorBatchHandle handle, NVCVAllocatorHandle *outAllocatorPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outAllocatorPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to allocator must not be NULL");
            }
            auto &tb         = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outAllocatorPtr = tb.alloc().release()->handle();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchExportData,
                (NVCVTensorBatchHandle handle, CUstream stream, NVCVTensorBatchData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to tensor batch data must not be NULL");
            }
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.exportData(stream, *data);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchPopTensors, (NVCVTensorBatchHandle handle, int32_t numTensors))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.popTensors(numTensors);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetTensors,
                (NVCVTensorBatchHandle handle, int32_t index, NVCVTensorHandle *outTensors, int32_t numTensors))
{
    return priv::ProtectCall(
        [&]
        {
            if (outTensors == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output tensors must not be NULL");
            }
            if (index < 0)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Index cannot be negative");
            }
            if (numTensors < 0)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Number of tensors cannot be negative");
            }
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.getTensors(index, outTensors, numTensors);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchSetTensors,
                (NVCVTensorBatchHandle handle, int32_t index, const NVCVTensorHandle *tensors, int32_t numTensors))
{
    return priv::ProtectCall(
        [&]
        {
            if (tensors == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to tensors must not be NULL");
            }
            if (index < 0)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Index cannot be negative");
            }
            if (numTensors < 0)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Number of tensors cannot be negative");
            }
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.setTensors(index, tensors, numTensors);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchSetUserPointer, (NVCVTensorBatchHandle handle, void *userPointer))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tb = priv::ToStaticRef<priv::ITensorBatch>(handle);
            tb.setUserPointer(userPointer);
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvTensorBatchGetUserPointer, (NVCVTensorBatchHandle handle, void **outUserPointer))
{
    return priv::ProtectCall(
        [&]
        {
            if (outUserPointer == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to user poniter must not be NULL");
            }
            auto &tb        = priv::ToStaticRef<priv::ITensorBatch>(handle);
            *outUserPointer = tb.userPointer();
        });
}
