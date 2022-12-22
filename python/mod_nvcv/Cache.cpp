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

#include "Cache.hpp"

#include <common/Assert.hpp>
#include <common/PyUtil.hpp>

#include <mutex>
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

struct Cache::Impl
{
    std::mutex                                                                           mtx;
    std::unordered_multimap<const IKey *, std::shared_ptr<CacheItem>, HashKey, KeyEqual> items;
};

Cache::Cache()
    : pimpl(new Impl())
{
}

void Cache::add(CacheItem &item)
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);

    pimpl->items.emplace(&item.key(), item.shared_from_this());
}

void Cache::removeAllNotInUseMatching(const IKey &key)
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);

    auto itrange = pimpl->items.equal_range(&key);

    for (auto it = itrange.first; it != itrange.second;)
    {
        if (!it->second->isInUse())
        {
            pimpl->items.erase(it++);
        }
        else
        {
            ++it;
        }
    }
}

std::vector<std::shared_ptr<CacheItem>> Cache::fetch(const IKey &key) const
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);

    auto itrange = pimpl->items.equal_range(&key);

    std::vector<std::shared_ptr<CacheItem>> v;
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

void Cache::clear()
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);
    pimpl->items.clear();
}

void Cache::doIterateThroughItems(const std::function<void(CacheItem &item)> &fn) const
{
    std::unique_lock<std::mutex> lk(pimpl->mtx);
    for (auto it = pimpl->items.begin(); it != pimpl->items.end(); ++it)
    {
        fn(*it->second);
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

    util::RegisterCleanup(m,
                          []
                          {
                              // Make sure cache is cleared up when script ends.
                              Cache::Instance().clear();
                          });

    m.def("clear_cache", [] { Cache::Instance().clear(); });

    // Just to check if fetchAll compiles, it's harmless
    Cache::Instance().fetchAll<Cache>();
}

} // namespace nvcvpy::priv
