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

#ifndef NVCV_CORE_PRIV_IALLOCATOR_HPP
#define NVCV_CORE_PRIV_IALLOCATOR_HPP

#include "ICoreObject.hpp"

#include <nvcv/alloc/Fwd.h>

#include <memory>

namespace nvcv::priv {

class IAllocator : public ICoreObjectHandle<IAllocator, NVCVAllocatorHandle>
{
public:
    void *allocHostMem(int64_t size, int32_t align);
    void  freeHostMem(void *ptr, int64_t size, int32_t align) noexcept;

    void *allocHostPinnedMem(int64_t size, int32_t align);
    void  freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept;

    void *allocCudaMem(int64_t size, int32_t align);
    void  freeCudaMem(void *ptr, int64_t size, int32_t align) noexcept;

private:
    // NVI idiom
    virtual void *doAllocHostMem(int64_t size, int32_t align)                    = 0;
    virtual void  doFreeHostMem(void *ptr, int64_t size, int32_t align) noexcept = 0;

    virtual void *doAllocHostPinnedMem(int64_t size, int32_t align)                    = 0;
    virtual void  doFreeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept = 0;

    virtual void *doAllocCudaMem(int64_t size, int32_t align)                    = 0;
    virtual void  doFreeCudaMem(void *ptr, int64_t size, int32_t align) noexcept = 0;
};

template<class T, class... ARGS>
std::unique_ptr<T> AllocHostObj(IAllocator &alloc, ARGS &&...args)
{
    void *arena = alloc.allocHostMem(sizeof(T), alignof(T));
    try
    {
        return std::unique_ptr<T>{new (arena) T{std::forward<ARGS>(args)...}};
    }
    catch (...)
    {
        alloc.freeHostMem(arena, sizeof(T), alignof(T));
        throw;
    }
}

template<class T>
void FreeHostObj(IAllocator &alloc, T *ptr) noexcept
{
    if (ptr != nullptr)
    {
        ptr->~T();
        alloc.freeHostMem(ptr, sizeof(T), alignof(T));
    }
}

priv::IAllocator &GetAllocator(NVCVAllocatorHandle handle);
priv::IAllocator &GetDefaultAllocator();

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_IALLOCATOR_HPP
