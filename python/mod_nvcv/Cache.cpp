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

#include "Cache.hpp"

#include "Definitions.hpp"
#include "ThreadScope.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>

#include <atomic>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace nvcvpy::priv {

struct HashKey
{
    size_t operator()(const IKey *k) const
    {
        NVCV_ASSERT(k != nullptr);
        return k->hash();
    }
};

struct KeyEqual
{
    size_t operator()(const IKey *k1, const IKey *k2) const
    {
        NVCV_ASSERT(k1 != nullptr);
        NVCV_ASSERT(k2 != nullptr);
        return *k1 == *k2;
    }
};

CacheItem::CacheItem()
{
    static std::atomic_uint16_t idnext = 0;

    m_id = idnext++;
}

uint64_t CacheItem::id() const
{
    return m_id;
}

std::shared_ptr<CacheItem> CacheItem::shared_from_this()
{
    return std::dynamic_pointer_cast<CacheItem>(Object::shared_from_this());
}

std::shared_ptr<const CacheItem> CacheItem::shared_from_this() const
{
    return std::dynamic_pointer_cast<const CacheItem>(Object::shared_from_this());
}

bool CacheItem::isInUse() const
{
    std::shared_ptr<const CacheItem> sthis = this->shared_from_this();

    // Return true if it is being used anywhere apart from cache and sthis
    return sthis.use_count() > 2;
}

using Items = std::unordered_multimap<const IKey *, std::shared_ptr<CacheItem>, HashKey, KeyEqual>;

struct Cache::Impl
{
    Items                    items;
    inline static std::mutex mtx;
    inline static int64_t    cache_limit_inbytes;
    inline static int64_t    current_size_inbytes;
};

Cache::Cache()
    : pimpl(new Impl())
{
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    instances.insert(this);
}

Cache::~Cache()
{
    {
        std::lock_guard<std::mutex> lk(pimpl->mtx);
        instances.erase(this);
        // It might not be safe to call destructors here, decrease the size manually
        for (const auto &node : pimpl->items)
        {
            pimpl->current_size_inbytes -= node.second->GetSizeInBytes();
        }
    }

    Impl *pimpl = this->pimpl.release();
    try
    {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 13
        if (Py_IsInitialized() && !Py_IsFinalizing())
#else
        if (Py_IsInitialized())
#endif
        {
            // Make sure that the main thread doesn't finalize the interpreter until all objects have been destroyed
            py::gil_scoped_acquire acq;
            delete pimpl;
        }
    }
    catch (const std::exception &)
    {
        // Leak intentionally if the Python runtime is not available anymore.
        // See https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
    }
}

void Cache::add(CacheItem &item)
{
    Items savedItems;
    {
        std::unique_lock<std::mutex> lk(pimpl->mtx);
        if (item.GetSizeInBytes() > doGetCacheLimit())
        {
            return;
        }

        if (item.GetSizeInBytes() + doGetCurrentSizeInBytes() > doGetCacheLimit())
        {
            // we clear the cache: all pimpl->items will be dtor'ed at the end of scope of savedItems and cache size will be reset to 0
            savedItems                  = std::move(pimpl->items);
            pimpl->current_size_inbytes = 0;
        }

        pimpl->items.emplace(&item.key(), item.shared_from_this());
        pimpl->current_size_inbytes += item.GetSizeInBytes();
    }
}

void Cache::removeAllNotInUseMatching(const IKey &key)
{
    // When we're removing items, we don't want their
    // refcount getting to 0 while the mutex is locked, as
    // deleting the object might recursively call removeAllNotInUseMatching,
    // leading to a dead lock.
    //
    // Instead, we gather the removed objects in the vector below, which will
    // be destroyed after the mutex is unlocked. When this happens, the items'
    // refcount will be decremented, and any object destruction will happen
    // after the mutex is unlocked. Recursion can happen in this case, but won't
    // lead to deadlocks
    std::vector<std::shared_ptr<CacheItem>> holdItemsUntilMtxUnlocked;

    {
        std::unique_lock<std::mutex> lk(pimpl->mtx);

        auto itrange = pimpl->items.equal_range(&key);

        int numItems = std::distance(itrange.first, itrange.second);

        auto it = itrange.first;
        for (int i = 0; i < numItems; ++i)
        {
            if (!it->second->isInUse())
            {
                holdItemsUntilMtxUnlocked.push_back(it->second);
                pimpl->current_size_inbytes -= it->second->GetSizeInBytes();
                pimpl->items.erase(it++);
            }
            else
            {
                ++it;
            }
        }
    }
}

std::vector<std::shared_ptr<CacheItem>> Cache::fetch(const IKey &key) const
{
    std::vector<std::shared_ptr<CacheItem>> v;

    std::unique_lock<std::mutex> lk(pimpl->mtx);

    auto itrange = pimpl->items.equal_range(&key);

    v.reserve(distance(itrange.first, itrange.second));

    for (auto it = itrange.first; it != itrange.second; ++it)
    {
        if (!it->second->isInUse())
        {
            v.emplace_back(it->second);
        }
    }

    return v;
}

#ifndef NDEBUG
void Cache::dbgPrintCacheForKey(const IKey &key, const std::string &prefix)
{
    std::vector<std::shared_ptr<CacheItem>> v;
    std::unique_lock<std::mutex>            lk(pimpl->mtx);
    auto                                    itrange = pimpl->items.equal_range(&key);

    for (auto it = itrange.first; it != itrange.second; ++it)
    {
        std::cerr << prefix << typeid(*(it->second)).name() << " - " << it->second.use_count() << std::endl;
    }
}
#endif

std::shared_ptr<CacheItem> Cache::fetchOne(const IKey &key) const
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);

    auto itrange = pimpl->items.equal_range(&key);

    for (auto it = itrange.first; it != itrange.second; ++it)
    {
        if (!it->second->isInUse())
        {
            return it->second;
        }
    }

    return {};
}

void Cache::clear()
{
    pimpl->items.clear();
}

size_t Cache::size() const
{
    return pimpl->items.size();
}

void Cache::setCacheLimit(int64_t new_cache_limit_inbytes)
{
    if (new_cache_limit_inbytes < 0)
    {
        throw std::invalid_argument("Cache limit must be non-negative.");
    }

    size_t free_mem, total_mem;
    util::CheckThrow(cudaMemGetInfo(&free_mem, &total_mem));

    if (static_cast<int64_t>(total_mem) < new_cache_limit_inbytes)
    {
        // Cache is not device aware, so in a multi-gpu scenario it could be ok to have a cache limit larger
        // than the total mem of the current device, but we should notify the user about this.
        std::cerr << "WARNING: new_cache_limit=" << new_cache_limit_inbytes
                  << " is more than total available memory on current device: " << total_mem << std::endl;
    }

    Items savedItems;
    {
        std::unique_lock<std::mutex> lk(pimpl->mtx);
        if (doGetCurrentSizeInBytes() > new_cache_limit_inbytes)
        {
            // we clear the cache: all pimpl->items will be dtor'ed at the end of scope of savedItems and cache size will be reset to 0
            savedItems                  = std::move(pimpl->items);
            pimpl->current_size_inbytes = 0;
        }
        pimpl->cache_limit_inbytes = new_cache_limit_inbytes;
    }
}

int64_t Cache::getCacheLimit() const
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);
    return doGetCacheLimit();
}

int64_t Cache::doGetCacheLimit() const
{
    return pimpl->cache_limit_inbytes;
}

int64_t Cache::getCurrentSizeInBytes()
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);
    return doGetCurrentSizeInBytes();
}

int64_t Cache::doGetCurrentSizeInBytes() const
{
    return pimpl->current_size_inbytes;
}

void Cache::doIterateThroughItems(const std::function<void(CacheItem &item)> &fn) const
{
    // To avoid keeping mutex locked for too long, let's first gather all items
    // into a vector, unlock the mutex, and then iterate through them.
    std::vector<std::shared_ptr<CacheItem>> v;

    {
        std::unique_lock<std::mutex> lk(pimpl->mtx);
        v.reserve(pimpl->items.size());

        for (auto it = pimpl->items.begin(); it != pimpl->items.end(); ++it)
        {
            v.push_back(it->second);
        }
    }

    for (const std::shared_ptr<CacheItem> &item : v)
    {
        fn(*item);
    }
}

Cache &Cache::Instance()
{
    thread_local Cache cache;
    return cache;
}

void Cache::ClearAll()
{
    Items savedItems;
    {
        std::lock_guard<std::mutex> lk(Cache::Impl::mtx);
        std::for_each(instances.begin(), instances.end(),
                      [&](Cache *instance) { savedItems.merge(instance->pimpl->items); });
        Cache::Impl::current_size_inbytes = 0;
    }
}

size_t Cache::TotalSize()
{
    std::lock_guard<std::mutex> lk(Cache::Impl::mtx);
    return std::accumulate(instances.cbegin(), instances.cend(), static_cast<size_t>(0),
                           [](size_t sum, const Cache *instance) { return sum + instance->size(); });
}

void Cache::Export(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<CacheItem, std::shared_ptr<CacheItem>>(nullptr, "CacheItem", py::module_local());

    py::class_<ExternalCacheItem, CacheItem, std::shared_ptr<ExternalCacheItem>>(nullptr, "ExternalCacheItem",
                                                                                 py::module_local());

    // Initialy set cache limit to half the size of the GPU memory
    size_t free_mem, total_mem;
    util::CheckThrow(cudaMemGetInfo(&free_mem, &total_mem));
    Cache::Instance().setCacheLimit(total_mem / 2);

    // Make sure cache is cleared up when script ends.
    util::RegisterCleanup(m, Cache::ClearAll);

    m.def(
        "clear_cache",
        [](ThreadScope scope)
        {
            switch (scope)
            {
            case ThreadScope::GLOBAL:
                Cache::ClearAll();
                break;
            case ThreadScope::LOCAL:
                Cache::Instance().clear();
                break;
            }
        },
        "scope"_a = ThreadScope::GLOBAL, R"pbdoc(
        Clears the NVCV Python cache

        Args:
            scope (nvcv.ThreadScope): Thread scope that must be either ``nvcv.ThreadScope.GLOBAL`` or ``nvcv.ThreadScope.LOCAL``.
    )pbdoc");

    m.def(
        "cache_size",
        [](ThreadScope scope)
        {
            switch (scope)
            {
            case ThreadScope::GLOBAL:
                return Cache::TotalSize();
            case ThreadScope::LOCAL:
                return Cache::Instance().size();
            }

            // Should be unreachable, especially from Python
            throw std::invalid_argument("Invalid scope");
        },
        "scope"_a = ThreadScope::GLOBAL, R"pbdoc(
        Returns the quantity of items in the NVCV Python cache

        Args:
            scope (nvcv.ThreadScope): Thread scope that must be either ``nvcv.ThreadScope.GLOBAL`` or ``nvcv.ThreadScope.LOCAL``.
    )pbdoc");

    m.def(
        "get_cache_limit_inbytes", [] { return Cache::Instance().getCacheLimit(); },
        "Returns the current cache limit [in bytes]");
    m.def(
        "set_cache_limit_inbytes",
        [](int64_t new_cache_limit_inbytes) { Cache::Instance().setCacheLimit(new_cache_limit_inbytes); },
        "Sets the current cache limit [in bytes]");

    m.def(
        "current_cache_size_inbytes", [] { return Cache::Instance().getCurrentSizeInBytes(); },
        "Returns the current cache size [in bytes]");

    py::module_ internal = m.attr(INTERNAL_SUBMODULE_NAME);
    internal.def("nbytes_in_cache", [](const CacheItem &item) { return item.GetSizeInBytes(); });

    // Just to check if fetchAll compiles, it's harmless
    Cache::Instance().fetchAll<Cache>();
}

} // namespace nvcvpy::priv
