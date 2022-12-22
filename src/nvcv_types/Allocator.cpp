/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/AllocatorManager.hpp"
#include "priv/CustomAllocator.hpp"
#include "priv/DefaultAllocator.hpp"
#include "priv/Exception.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/alloc/Allocator.h>
#include <util/Assert.h>

#include <memory>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorConstructCustom,
                (const NVCVCustomAllocator *customAllocators, int32_t numCustomAllocators, NVCVAllocatorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            if (numCustomAllocators != 0)
            {
                *handle = priv::CreateCoreObject<priv::CustomAllocator>(customAllocators, numCustomAllocators);
            }
            else
            {
                *handle = priv::CreateCoreObject<priv::DefaultAllocator>();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorDestroy, (NVCVAllocatorHandle halloc))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(halloc); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorSetUserPointer, (NVCVAllocatorHandle handle, void *userPtr))
{
    return priv::ProtectCall(
        [&]
        {
            auto &img = priv::ToStaticRef<priv::IAllocator>(handle);

            if (priv::MustProvideHiddenFunctionality(handle))
            {
                img.setCXXObject(userPtr);
            }
            else
            {
                img.setUserPointer(userPtr);
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorGetUserPointer, (NVCVAllocatorHandle handle, void **outUserPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outUserPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output user pointer cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IAllocator>(handle);

            if (priv::MustProvideHiddenFunctionality(handle))
            {
                img.getCXXObject(outUserPtr);
            }
            else
            {
                *outUserPtr = img.userPointer();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorAllocHostMemory,
                (NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            *ptr = priv::ToStaticRef<priv::IAllocator>(halloc).allocHostMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorFreeHostMemory,
                (NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                priv::ToStaticRef<priv::IAllocator>(halloc).freeHostMem(ptr, sizeBytes, alignBytes);
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorAllocHostPinnedMemory,
                (NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            *ptr = priv::ToStaticRef<priv::IAllocator>(halloc).allocHostPinnedMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorFreeHostPinnedMemory,
                (NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                priv::ToStaticRef<priv::IAllocator>(halloc).freeHostPinnedMem(ptr, sizeBytes, alignBytes);
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorAllocCudaMemory,
                (NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            *ptr = priv::ToStaticRef<priv::IAllocator>(halloc).allocCudaMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorFreeCudaMemory,
                (NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                priv::ToStaticRef<priv::IAllocator>(halloc).freeCudaMem(ptr, sizeBytes, alignBytes);
            }
        });
}
