/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_SIMPLE_CACHE_HPP
#define NVCV_UTIL_SIMPLE_CACHE_HPP

#include <memory>
#include <mutex>
#include <optional>

namespace nvcv::util {

/** A simple cache that stores objects of type T
 *
 * The cache stores objects of type T. Upon a call to `Get`, an object
 * is moved out from the cache and returned as `optional<T>`. If the cache
 * is empty, `nullopt` is returned.
 * `GetOrCreate` alwas returns an object (unless it throws) - if the cache is empty,
 * a user-provided factory function is invoked and a new object is returned.
 * Objects can be placed in the cache with a call to `Put` or `Emplace`.
 *
 * The cache is guarded with a lockable object (by default std::mutex).
 *
 * The cache is implemented as a unidirectional list of entries.
 * Each entry holds an optional instance of type T.
 * Once an object is removed, the cache entry is stored for reuse in an auxiliary list,
 * reducing the number of dynamic allocations.
 *
 * @tparam T        The type of itmes held in the cache
 * @tparam LockType A lockable object
 */
template<typename T, typename LockType = std::mutex>
class SimpleCache
{
public:
    std::optional<T> get()
    {
        if (m_items)
        {
            std::lock_guard lg(m_lock);
            if (m_items)
            {
                auto tmp  = std::move(m_items);
                m_items   = std::move(tmp->next);
                auto obj  = std::move(tmp->payload);
                tmp->next = std::move(m_empty);
                m_empty   = std::move(tmp);
                return obj;
            }
        }
        return std::nullopt;
    }

    template<typename CreateFunc>
    T getOrCreate(CreateFunc &&create)
    {
        auto cached = get();
        if (cached.has_value())
            return std::move(cached).value();
        else
            return create();
    }

    void put(T &&payload)
    {
        emplace(std::move(payload));
    }

    template<typename... Args>
    void emplace(Args &&...args)
    {
        std::lock_guard lg(m_lock);

        std::unique_ptr<CacheItem> item;
        if (m_empty)
        {
            item    = std::move(m_empty);
            m_empty = std::move(item->next);
        }
        else
        {
            item = std::make_unique<CacheItem>();
        }
        item->payload.emplace(std::forward<Args>(args)...);

        item->next = std::move(m_items);
        m_items    = std::move(item);
    }

    void purge()
    {
        std::lock_guard lg(m_lock);

        m_items.reset();
        m_empty.reset();
    }

private:
    struct CacheItem
    {
        ~CacheItem()
        {
            // Iterate through all subsequent elements to avoid deep recursion
            while (next)
            {
                // detach the chain from the `next`
                auto tmp = std::move(next->next);
                // this will delete the next
                next = std::move(tmp);
            }
        }

        std::unique_ptr<CacheItem> next;
        std::optional<T>           payload;
    };

    std::unique_ptr<CacheItem> m_items, m_empty;
    LockType                   m_lock;
};

} // namespace nvcv::util

#endif // NVCV_UTIL_SIMPLE_CACHE_HPP
