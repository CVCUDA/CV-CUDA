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

#include "Cache.hpp"

#include "Definitions.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>

#include <mutex>
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
    static uint64_t idnext = 0;

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
    std::mutex mtx;
    Items      items;
    int64_t    cache_limit_inbytes;
};

Cache::Cache()
    : pimpl(new Impl())
{
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

        if (item.GetSizeInBytes() + doCurrentSizeInBytes() > doGetCacheLimit())
        {
            savedItems = std::move(pimpl->items);
        }

        pimpl->items.emplace(&item.key(), item.shared_from_this());
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
    Items                        savedItems;
    std::unique_lock<std::mutex> lk(pimpl->mtx);
    savedItems = std::move(pimpl->items);
    lk.unlock();
    savedItems.clear();
}

size_t Cache::size() const
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);
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
        if (doCurrentSizeInBytes() > new_cache_limit_inbytes)
        {
            savedItems = std::move(pimpl->items);
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
    return doCurrentSizeInBytes();
}

int64_t Cache::doCurrentSizeInBytes() const
{
    int64_t current_size_inbytes = 0;

    for (auto it = pimpl->items.begin(); it != pimpl->items.end(); ++it)
    {
        current_size_inbytes += it->second->GetSizeInBytes();
    }

    return current_size_inbytes;
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
    static Cache cache;
    return cache;
}

void Cache::Export(py::module &m)
{
    py::class_<CacheItem, std::shared_ptr<CacheItem>>(nullptr, "CacheItem", py::module_local());

    py::class_<ExternalCacheItem, CacheItem, std::shared_ptr<ExternalCacheItem>>(nullptr, "ExternalCacheItem",
                                                                                 py::module_local());

    // Initialy set cache limit to half the size of the GPU memory
    size_t free_mem, total_mem;
    util::CheckThrow(cudaMemGetInfo(&free_mem, &total_mem));
    Cache::Instance().setCacheLimit(total_mem / 2);

    util::RegisterCleanup(m,
                          []
                          {
                              // Make sure cache is cleared up when script ends.
                              Cache::Instance().clear();
                          });

    m.def(
        "clear_cache", [] { Cache::Instance().clear(); }, "Clears the NVCV Python cache");

    m.def(
        "cache_size", [] { return Cache::Instance().size(); },
        "Returns the quantity of items in the NVCV Python cache");

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
