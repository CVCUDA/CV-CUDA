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

#include "priv/Array.hpp"

#include "priv/AllocatorManager.hpp"
#include "priv/ArrayManager.hpp"
#include "priv/ArrayWrapData.hpp"
#include "priv/DataType.hpp"
#include "priv/Exception.hpp"
#include "priv/IAllocator.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Array.hpp>

#include <algorithm>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayCalcRequirements,
                (int64_t capacity, NVCVDataType dtype, int32_t alignment, NVCVArrayRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::DataType type{dtype};

            *reqs = priv::Array::CalcRequirements(capacity, type, alignment);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayCalcRequirementsWithTarget,
                (int64_t capacity, NVCVDataType dtype, int32_t alignment, NVCVResourceType target,
                 NVCVArrayRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::DataType type{dtype};

            *reqs = priv::Array::CalcRequirements(capacity, type, alignment, target);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayConstruct,
                (const NVCVArrayRequirements *reqs, NVCVAllocatorHandle halloc, NVCVArrayHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to array requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::Array>(*reqs, alloc, NVCV_RESOURCE_MEM_CUDA);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayConstructWithTarget,
                (const NVCVArrayRequirements *reqs, NVCVAllocatorHandle halloc, NVCVResourceType target,
                 NVCVArrayHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to array requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::Array>(*reqs, alloc, target);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayWrapDataConstruct,
                (const NVCVArrayData *data, NVCVArrayDataCleanupFunc cleanup, void *ctxCleanup,
                 NVCVArrayHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            *handle = priv::CreateCoreObject<priv::ArrayWrapData>(*data, cleanup, ctxCleanup);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayDecRef, (NVCVArrayHandle handle, int *newRefCount))
{
    return priv::ProtectCall(
        [&]
        {
            int newRef = priv::CoreObjectDecRef(handle);
            if (newRefCount)
                *newRefCount = newRef;
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayIncRef, (NVCVArrayHandle handle, int *newRefCount))
{
    return priv::ProtectCall(
        [&]
        {
            int newRef = priv::CoreObjectIncRef(handle);
            if (newRefCount)
                *newRefCount = newRef;
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayRefCount, (NVCVArrayHandle handle, int *refCount))
{
    return priv::ProtectCall([&] { *refCount = priv::CoreObjectRefCount(handle); });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArraySetUserPointer, (NVCVArrayHandle handle, void *userPtr))
{
    return priv::ProtectCall(
        [&]
        {
            auto &array = priv::ToStaticRef<priv::IArray>(handle);
            array.setUserPointer(userPtr);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayGetUserPointer, (NVCVArrayHandle handle, void **outUserPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outUserPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output user pointer cannot be NULL");
            }

            auto &array = priv::ToStaticRef<const priv::IArray>(handle);

            *outUserPtr = array.userPointer();
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayGetDataType, (NVCVArrayHandle handle, NVCVDataType *dtype))
{
    return priv::ProtectCall(
        [&]
        {
            if (dtype == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }

            auto &array = priv::ToStaticRef<const priv::IArray>(handle);
            *dtype      = array.dtype().value();
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayGetAllocator, (NVCVArrayHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &array = priv::ToStaticRef<const priv::IArray>(handle);

            *halloc = array.alloc().release()->handle();
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayExportData, (NVCVArrayHandle handle, NVCVArrayData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output data cannot be NULL");
            }

            auto &array = priv::ToStaticRef<const priv::IArray>(handle);
            array.exportData(*data);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayGetLength, (NVCVArrayHandle handle, int64_t *length))
{
    return priv::ProtectCall(
        [&]
        {
            auto &array = priv::ToStaticRef<const priv::IArray>(handle);

            if (length == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input pointer to length cannot be NULL");
            }

            *length = array.length();
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayGetCapacity, (NVCVArrayHandle handle, int64_t *capacity))
{
    return priv::ProtectCall(
        [&]
        {
            auto &array = priv::ToStaticRef<const priv::IArray>(handle);

            if (capacity == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input pointer to capacity cannot be NULL");
            }

            *capacity = array.capacity();
        });
}

NVCV_DEFINE_API(0, 5, NVCVStatus, nvcvArrayResize, (NVCVArrayHandle handle, int64_t length))
{
    return priv::ProtectCall(
        [&]
        {
            auto &array = priv::ToStaticRef<priv::IArray>(handle);

            if (length > array.capacity())
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Cannot resize array to input length because greater than capacity");
            }

            array.resize(length);
        });
}

NVCV_DEFINE_API(0, 4, NVCVStatus, nvcvArrayGetTarget, (NVCVArrayHandle handle, NVCVResourceType *target))
{
    return priv::ProtectCall(
        [&]
        {
            auto &array = priv::ToStaticRef<const priv::IArray>(handle);

            if (target == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input pointer to target cannot be NULL");
            }

            *target = array.target();
        });
}
