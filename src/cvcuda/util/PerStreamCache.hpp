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

#ifndef NVCV_UTIL_PER_STREAM_CACHE_HPP
#define NVCV_UTIL_PER_STREAM_CACHE_HPP

#include "Event.hpp"
#include "SimpleCache.hpp"
#include "StreamId.hpp"

#include <nvcv/util/CheckError.hpp>

#include <cassert>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace nvcv::util {

class EventCache : public nvcv::util::SimpleCache<CudaEvent>
{
public:
    CudaEvent get()
    {
        return getOrCreate([]() { return CudaEvent::Create(); });
    }
};

template<typename Payload>
auto StreamCachePayloadReady(const Payload &payload)
{
    return payload.ready;
}

template<typename Payload>
auto StreamCachePayloadSize(const Payload &payload)
{
    return payload.size;
}

template<typename Payload>
auto StreamCachePayloadAlignment(const Payload &payload)
{
    return payload.alignment;
}

namespace detail {

/** Cache item - stores a payload in a bidirectional list item.
 *
 * @tparam Payload   The payload of the cache item. It must have the followin fields:
 *                      size_t      size
 *                      cudaEvent_t ready
 */
template<typename Payload>
struct StreamCacheItem
{
    StreamCacheItem *next = nullptr, *prev = nullptr;

    mutable bool wasReady = false;

    Payload payload{};

    /** Gets a CUDA event that signifies that the payload is ready.
     */
    cudaEvent_t readyEvent() const
    {
        return StreamCachePayloadReady(payload);
    }

    size_t payloadSize() const
    {
        return StreamCachePayloadSize(payload);
    }

    bool isReady() const
    {
        if (wasReady)
            return true;
        if (auto ev = readyEvent())
        {
            auto err = cudaEventQuery(ev);
            if (err == cudaErrorNotReady)
                return false;
            NVCV_CHECK_THROW(err);
        }
        wasReady = true;
        return true;
    }
};

template<typename Payload, typename Item = StreamCacheItem<Payload>>
class StreamCacheItemAllocator
{
public:
    using item_t = Item;

    ~StreamCacheItemAllocator()
    {
        assert(m_allocated == 0);
        while (m_head)
        {
            auto *next = m_head->next;
            delete m_head;
            m_head = next;
        }
    }

    item_t *allocate()
    {
        if (auto *p = m_head)
        {
            m_head  = p->next;
            p->next = nullptr;
            assert(!p->prev);
            m_allocated++;
            m_free--;

            *p = {}; // clear the object
            return p;
        }

        auto *p = new item_t();
        m_allocated++;
        return p;
    }

    void deallocate(item_t *item)
    {
        if (!item)
            return;

        assert(!item->next && !item->prev && "The item is still linked");
        item->payload = {};

        item->next = m_head;
        m_head     = item;
        m_allocated--;
        m_free++;
    }

private:
    item_t *m_head = nullptr;

    size_t m_allocated = 0, m_free = 0;
};

template<typename Payload, typename Item = StreamCacheItem<Payload>>
class StreamOrderedCache
{
public:
    using item_t = Item;

    explicit StreamOrderedCache(StreamCacheItemAllocator<Payload, Item> *itemAlloc)
        : m_itemAlloc(itemAlloc)
    {
    }

    ~StreamOrderedCache()
    {
        waitAndPurge();
    }

    void waitAndPurge();

    template<typename PayloadCallback>
    void removeAllReady(PayloadCallback callback);

    item_t *findNewestReady();

    void put(Payload &&payload);

    bool empty() const
    {
        return m_bySize.empty();
    }

    template<typename Predicate>
    std::optional<Payload> getIf(size_t minSize, Predicate &&pred);

    std::optional<Payload> get(size_t minSize, size_t minAlignment)
    {
        return getIf(
            minSize, [=](const Payload &p)
            { return StreamCachePayloadSize(p) >= minSize && StreamCachePayloadAlignment(p) >= minAlignment; });
    }

private:
    void insert(item_t *item);

    void remove(size_t payloadSize, item_t *item) noexcept;

    StreamCacheItemAllocator<Payload, item_t> *m_itemAlloc;

    std::set<std::pair<size_t, item_t *>> m_bySize;

    item_t *m_head = nullptr, *m_tail = nullptr;
};

} // namespace detail

template<typename Payload, typename Item = detail::StreamCacheItem<Payload>>
class PerStreamCache
{
    using StreamOrderedCache = detail::StreamOrderedCache<Payload, Item>;

public:
    template<typename Predicate>
    std::optional<Payload> getIf(size_t minSize, Predicate &&pred, std::optional<cudaStream_t> stream);

    auto get(size_t minSize, size_t minAlignment, std::optional<cudaStream_t> stream)
    {
        return getIf(
            minSize,
            [=](const Payload &p)
            { return StreamCachePayloadSize(p) >= minSize && StreamCachePayloadAlignment(p) >= minAlignment; },
            stream);
    }

    void put(Payload &&payload, std::optional<cudaStream_t> stream);

    void purge()
    {
        std::lock_guard g(m_lock);
        for (auto &[k, v] : m_perStreamCache) v.waitAndPurge();
        m_globalCache.clear();
    }

private:
    template<typename Predicate>
    std::optional<Payload> tryGetPerStream(size_t minSize, Predicate &&pred, cudaStream_t stream);

    template<typename Predicate>
    std::optional<Payload> tryGetGlobal(size_t minSize, Predicate &&pred);

    int moveReadyToGlobal();

    detail::StreamCacheItemAllocator<Payload, Item> m_cacheItemAlloc;

    std::unordered_map<uint64_t, StreamOrderedCache> m_perStreamCache;

    std::multimap<size_t, Payload> m_globalCache;

    std::mutex m_lock;
};

} // namespace nvcv::util

#include "PerStreamCacheImpl.hpp"

#endif // NVCV_UTIL_PER_STREAM_CACHE_HPP
