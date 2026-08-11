/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PRIV_CORE_HANDLE_MANAGER_IMPL_HPP
#define NVCV_PRIV_CORE_HANDLE_MANAGER_IMPL_HPP

#include "Exception.hpp"
#include "LockFreeStack.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace nvcv::priv {

static const char *const LEAK_DETECTION_ENVVAR = "NVCV_LEAK_DETECTION";

template<typename Node>
class ManagedLockFreeStack : protected LockFreeStack<Node>
{
    using Base = LockFreeStack<Node>;

public:
    using Base::pop;
    using Base::push;
    using Base::pushStack;
    using Base::top;

    template<typename... Args>
    Node *emplace(Args &&...args)
    {
        std::unique_ptr<Node> n   = std::make_unique<Node>(std::forward<Args>(args)...);
        Node                 *out = n.get();
        push(n.release());
        return out;
    }

    ~ManagedLockFreeStack()
    {
        clear();
    }

    void clear()
    {
        if (Node *h = this->release())
        {
            std::unique_ptr<Node> node(h);
            while (node)
            {
                std::unique_ptr<Node> next(node->next);
                node = std::move(next);
            }
        }
    }
};

template<typename Interface>
HandleManager<Interface>::ResourceBase::ResourceBase()
{
    generation = 0;
}

template<typename Interface>
HandleManager<Interface>::ResourceBase::~ResourceBase()
{
    assert(m_ptrObj == nullptr && "Internal error - the object must be destroyed by ~Resource");
}

template<typename Interface>
void HandleManager<Interface>::ResourceBase::destroyObject()
{
    if (m_ptrObj)
    {
        std::destroy_at(m_ptrObj);
        m_ptrObj = nullptr;
    }

    NVCV_ASSERT(!this->live());
}

template<class Interface>
std::byte *HandleManager<Interface>::ResourceBase::getStorage()
{
    using Resource = typename HandleManager<Interface>::Impl::Resource;
    return static_cast<Resource *>(this)->storage();
}

template<typename Interface>
struct HandleManager<Interface>::Impl
{
    using Storage = typename ResourceStorage<Interface>::type;

    class Resource : public ResourceBase
    {
    public:
        ~Resource()
        {
            this->destroyObject();
        }

        std::byte *storage()
        {
            return m_storage.data();
        }

    private:
        alignas(Storage) std::array<std::byte, sizeof(Storage)> m_storage;
    };

    static constexpr int kMinHandles = 1024;

    std::mutex mtxAlloc;

    struct ResourcePool
    {
        explicit ResourcePool(size_t count)
            : resources(count)
        {
        }

        ResourcePool         *next = nullptr;
        std::vector<Resource> resources;
    };

    void allocate(size_t count)
    {
        auto     *res_block = resourceStack.emplace(count);
        Resource *data      = res_block->resources.data();

        // Turn the newly allocated resources into a forward_list
        if (count > 1)
            for (size_t i = 0; i < count - 1; i++)
            {
                data[i].next = &data[i + 1];
            }

        // Add them to the list of free resources.
        freeResources.pushStack(data, data + count - 1);
    }

    // Store the resources' buffer.
    // For efficient validation, it's divided into several pools,
    // it's easy to check if a given resource belong to a pool, O(1).
    ManagedLockFreeStack<ResourcePool> resourceStack;

    // All the free resources we have
    LockFreeStack<ResourceBase> freeResources;

    static_assert(std::atomic<ResourceBase *>::is_always_lock_free);

    bool            hasFixedSize  = false;
    int             totalCapacity = 0;
    std::atomic_int usedCount     = 0;
    const char     *name;
};

template<typename Interface>
HandleManager<Interface>::HandleManager(const char *name)
    : pimpl(std::make_unique<Impl>())
{
    pimpl->name = name;
}

template<typename Interface>
HandleManager<Interface>::~HandleManager()
{
    this->clear();
}

template<typename Interface>
int HandleManager<Interface>::decRef(HandleType handle)
{
    if (!handle)
        return 0; // "destruction" of a null handle is a no-op

    if (ResourceBase *res = this->getValidResource(handle))
    {
        int ref = res->decRef();
        if (ref == 0)
        {
            res->destroyObject();
            doReturnResource(res);
        }
        return ref;
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "The handle is invalid.");
    }
}

template<typename Interface>
int HandleManager<Interface>::incRef(HandleType handle)
{
    if (ResourceBase *res = this->getValidResource(handle))
    {
        return res->incRef();
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "The handle is invalid.");
    }
}

template<typename Interface>
int HandleManager<Interface>::refCount(HandleType handle)
{
    if (ResourceBase *res = this->getValidResource(handle))
    {
        return res->refCount();
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "The handle is invalid.");
    }
}

template<typename Interface>
bool HandleManager<Interface>::isManagedResource(ResourceBase *r) const
{
    for (auto *range = pimpl->resourceStack.top(); range; range = range->next)
    {
        auto *b = range->resources.data();
        auto *e = b + range->resources.size();
        if (r >= b && r < e)
        {
            return true;
        }
    }
    return false;
}

template<typename Interface>
Interface *HandleManager<Interface>::validate(HandleType handle) const
{
    // Suppress false positive array bounds warning from g++-12-14
#if defined(__GNUC__) && __GNUC__ >= 12
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    if (auto *res = getValidResource(handle))
    {
        return res->obj();
    }
    else
    {
        return nullptr;
    }
#if defined(__GNUC__) && __GNUC__ >= 12
#    pragma GCC diagnostic pop
#endif
}

template<typename Interface>
auto HandleManager<Interface>::getValidResource(HandleType handle) const -> ResourceBase *
{
    if (!handle)
    {
        return nullptr;
    }

    ResourceBase *res = doGetResourceFromHandle(handle);

    // Suppress false positive array bounds warning from g++-12-14
#if defined(__GNUC__) && __GNUC__ >= 12
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    // Add explicit null check before accessing res members to avoid false positive warning in g++-12-14
    if (res && this->isManagedResource(res) && res->live() && res->generation == doGetHandleGeneration(handle))
    {
        return res;
    }
    else
    {
        return nullptr;
    }
#if defined(__GNUC__) && __GNUC__ >= 12
#    pragma GCC diagnostic pop
#endif
}

template<typename Interface>
void HandleManager<Interface>::setFixedSize(int32_t maxSize)
{
    std::lock_guard lock(pimpl->mtxAlloc);
    if (pimpl->usedCount > 0)
    {
        throw Exception(NVCV_ERROR_INVALID_OPERATION,
                        "Cannot change the size policy while there are still %d live %s handles", (int)pimpl->usedCount,
                        pimpl->name);
    }

    if (pimpl->totalCapacity >= maxSize)
    {
        return;
    }

    this->clear();

    pimpl->hasFixedSize = true;
    doAllocate(maxSize - pimpl->totalCapacity);
}

template<typename Interface>
void HandleManager<Interface>::setDynamicSize(int32_t minSize)
{
    std::lock_guard lock(pimpl->mtxAlloc);

    pimpl->hasFixedSize = false;
    if (pimpl->totalCapacity < minSize)
    {
        doAllocate(minSize);
    }
}

template<typename Interface>
void HandleManager<Interface>::clear()
{
    if (pimpl->usedCount > 0)
    {
        // nosemgrep: flawfinder.getenv-1.curl_getenv-1
        const char *leakDetection = getenv(LEAK_DETECTION_ENVVAR);
#ifndef NDEBUG
        // On debug builds, report leaks by default
        if (leakDetection == nullptr)
        {
            leakDetection = "warn";
        }
#endif

        if (leakDetection != nullptr)
        {
            bool doAbort = false;
            if (strcmp(leakDetection, "warn") == 0)
            {
                std::cerr << "WARNING: ";
            }
            else if (strcmp(leakDetection, "abort") == 0)
            {
                std::cerr << "ERROR: ";
                doAbort = true;
            }
            else
            {
                std::cerr << "Invalid value '" << leakDetection << " for " << LEAK_DETECTION_ENVVAR
                          << " environment variable. It must be either not defiled or '0' "
                             "(to disable), 'warn' or 'abort'";
                abort();
            }

            std::cerr << pimpl->name << " leak detection: " << pimpl->usedCount << " handle"
                      << (pimpl->usedCount > 1 ? "s" : "") << " still in use" << std::endl;
            if (doAbort)
            {
                abort();
            }
        }
    }

    pimpl->freeResources.clear();
    pimpl->resourceStack.clear();
}

template<typename Interface>
void HandleManager<Interface>::doAllocate(size_t count)
{
    NVCV_ASSERT(count > 0);
    pimpl->allocate(count);
}

template<typename Interface>
void HandleManager<Interface>::doGrow()
{
    if (pimpl->hasFixedSize)
    {
        throw Exception(NVCV_ERROR_OUT_OF_MEMORY, "%s handle manager pool exhausted under fixed size policy",
                        pimpl->name);
    }

    std::lock_guard lock(pimpl->mtxAlloc);
    if (!pimpl->freeResources.top())
    {
        doAllocate(pimpl->totalCapacity ? pimpl->totalCapacity : pimpl->kMinHandles);
    }
}

template<typename Interface>
auto HandleManager<Interface>::doFetchFreeResource() -> ResourceBase *
{
    for (;;)
    {
        if (auto *r = pimpl->freeResources.pop())
        {
            pimpl->usedCount++;
            r->incRef();
            assert(r->refCount() == 1);
            return r;
        }
        else
        {
            doGrow();
        }
    }
}

template<typename Interface>
void HandleManager<Interface>::doReturnResource(ResourceBase *r)
{
    pimpl->freeResources.push(r);
    pimpl->usedCount--;
}

template<typename Interface>
uint8_t HandleManager<Interface>::doGetHandleGeneration(HandleType handle) const noexcept
{
    return ((uintptr_t)handle & (kResourceAlignment - 1));
}

template<typename Interface>
auto HandleManager<Interface>::doGetHandleFromResource(ResourceBase *r) const noexcept -> HandleType
{
    if (r)
    {
        // generation corresponds to 4 LSBs -> max 16 generations
        return reinterpret_cast<HandleType>((uintptr_t)r | r->generation);
    }
    else
    {
        return {};
    }
}

template<typename Interface>
auto HandleManager<Interface>::doGetResourceFromHandle(HandleType handle) const noexcept -> ResourceBase *
{
    if (handle)
    {
        return reinterpret_cast<ResourceBase *>((uintptr_t)handle & -kResourceAlignment);
    }
    else
    {
        return nullptr;
    }
}

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_HANDLE_MANAGER_IMPL_HPP
