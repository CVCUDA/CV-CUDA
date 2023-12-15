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

#ifndef NVCV_UTIL_PER_STREAM_CACHE_IMPL_HPP
#define NVCV_UTIL_PER_STREAM_CACHE_IMPL_HPP

#ifndef NVCV_UTIL_PER_STREAM_CACHE_HPP
#    error "This file must not be included directly. Include StreamOrderedCache.hpp."
#endif

namespace nvcv::util {
namespace detail {

template<typename Payload, typename Item>
void StreamOrderedCache<Payload, Item>::waitAndPurge()
{
    bool   ready  = false;
    size_t erased = 0;
    while (m_tail)
    {
        if (!ready && m_tail->readyEvent())
        {
            auto err = cudaEventSynchronize(m_tail->readyEvent());
            if (err != cudaErrorCudartUnloading)
                NVCV_CHECK_THROW(err);
            ready = true;
        }
        auto *curr = m_tail;
        m_tail     = m_tail->prev;
        curr->prev = nullptr;
        if (m_tail)
            m_tail->next = nullptr;
        m_itemAlloc->deallocate(curr);
        erased++;
    }
    assert(erased == m_bySize.size());
    m_head = nullptr;
    m_bySize.clear();
}

template<typename Payload, typename Item>
template<typename PayloadCallback>
void StreamOrderedCache<Payload, Item>::removeAllReady(PayloadCallback callback)
{
    if (nvcv::util::IsCudaStreamIdHintUnambiguous())
    {
        // If the stream id hint is unambiguous, we can find the newest item
        // and all older items will naturally be ready.

        item_t *item = findNewestReady();
        // This item and all older items are ready
        while (item)
        {
            item_t *prev        = item->prev;
            size_t  payloadSize = item->payloadSize();
            callback(std::move(item->payload));
            remove(payloadSize, item);
            item = prev;
        }
    }
    else
    {
        // The system's stream id hint is ambiguous, so we may have a mixture
        // of items actually coming from different streams. We need to
        // chek them one by one, since the readiness order may be lost.

        item_t *item = m_tail;
        while (item)
        {
            item_t *prev = item->prev;
            if (item->isReady())
            {
                size_t payloadSize = item->payloadSize();
                callback(std::move(item->payload));
                remove(payloadSize, item);
            }
            item = prev;
        }
    }
}

template<typename Payload, typename Item>
auto StreamOrderedCache<Payload, Item>::findNewestReady() -> item_t *
{
    constexpr int kMaxItemsOnStack = 256;
    item_t       *tmp[kMaxItemsOnStack];
    item_t       *sectionStart = m_tail;
    // Process the items in blocks of up to kMaxItemsOnStack. On each block, a binary search is performed.
    while (sectionStart)
    {
        if (sectionStart->isReady())
            return sectionStart; // everything elese is newer, hence also ready

        item_t *it = sectionStart->prev; // no point in re-checking the section start
        int     hi = 0;
        for (; it && hi < kMaxItemsOnStack; hi++, it = it->prev) tmp[hi] = it;

        if (hi == 0)
            return nullptr;

        // There are no ready elements in this range - move on
        if (!tmp[hi - 1]->isReady())
        {
            sectionStart = it;
            continue;
        }

        int lo = 0, m = (lo + hi) >> 1;
        // After this loop, `m` is going to contain the index of the newest ready element
        while (lo < hi) // exclusive upper bound
        {
            // halfway element is ready - maybe there are newer ones that are ready, too?
            if (tmp[m]->isReady())
            {
                hi = m; // exclusive upper bound
                m  = (lo + hi) >> 1;
            }
            else // halfway element isn't ready - move to `m+1` as a potential lower bound
            {
                lo = m + 1;
                m  = (lo + hi) >> 1;
            }
        }
        assert(0 <= m && m <= hi);
        assert(tmp[m]->wasReady);
        return tmp[m];
    }
    return nullptr;
}

template<typename Payload, typename Item>
void StreamOrderedCache<Payload, Item>::put(Payload &&payload)
{
    item_t *item  = m_itemAlloc->allocate();
    item->payload = std::move(payload);
    payload       = {};
    try
    {
        insert(item);
    }
    catch (...)
    {
        m_itemAlloc->deallocate(item);
        throw;
    }
}

template<typename Payload, typename Item>
template<typename Predicate>
std::optional<Payload> StreamOrderedCache<Payload, Item>::getIf(size_t minSize, Predicate &&pred)
{
    auto it = m_bySize.lower_bound({minSize, nullptr});
    for (; it != m_bySize.end(); ++it)
    {
        auto *item = it->second;
        if (pred(item->payload))
        {
            size_t  payloadSize = item->payloadSize();
            Payload ret         = std::move(item->payload);
            remove(payloadSize, item);
            return ret;
        }
    }
    return std::nullopt;
}

template<typename Payload, typename Item>
void StreamOrderedCache<Payload, Item>::insert(item_t *item)
{
    auto inserted = m_bySize.insert({item->payloadSize(), item});
#ifdef NDEBUG
    (void)inserted;
#endif
    assert(inserted.second);

    if (!m_tail)
    {
        assert(!m_head);
        m_head = m_tail = item;
    }
    else
    {
        assert(!m_tail->next);
        item->prev   = m_tail;
        m_tail->next = item;
        m_tail       = item;
    }
}

template<typename Payload, typename Item>
void StreamOrderedCache<Payload, Item>::remove(size_t payloadSize, item_t *item) noexcept
{
    if (item == m_head)
        m_head = m_head->next;
    if (item == m_tail)
        m_tail = m_tail->prev;

    if (item->prev)
        item->prev->next = item->next;
    if (item->next)
        item->next->prev = item->prev;
    item->prev = item->next = nullptr;

    size_t erased = m_bySize.erase({payloadSize, item});
#ifdef NDEBUG
    (void)erased;
#endif
    assert(erased == 1);

    m_itemAlloc->deallocate(item);
}

} // namespace detail

template<typename Payload, typename Item>
template<typename Predicate>
std::optional<Payload> PerStreamCache<Payload, Item>::getIf(size_t minSize, Predicate &&pred,
                                                            std::optional<cudaStream_t> stream)
{
    std::optional<Payload> ret;

    std::lock_guard guard(m_lock);

    if (stream)
    {
        ret = tryGetPerStream(minSize, pred, *stream);
        if (ret)
            return ret;
    }

    do
    {
        ret = tryGetGlobal(minSize, pred);
        if (ret)
            return ret;
    }
    while (moveReadyToGlobal());

    return std::nullopt;
}

template<typename Payload, typename Item>
template<typename Predicate>
std::optional<Payload> PerStreamCache<Payload, Item>::tryGetPerStream(size_t size, Predicate &&pred,
                                                                      cudaStream_t stream)
{
    uint64_t streamId = GetCudaStreamIdHint(stream);
    auto     it       = m_perStreamCache.find(streamId);
    if (it == m_perStreamCache.end())
        return std::nullopt;
    return it->second.getIf(size, std::forward<Predicate>(pred));
}

template<typename Payload, typename Item>
template<typename Predicate>
std::optional<Payload> PerStreamCache<Payload, Item>::tryGetGlobal(size_t size, Predicate &&pred)
{
    for (auto it = m_globalCache.lower_bound(size); it != m_globalCache.end(); ++it)
    {
        if (pred(it->second))
        {
            Payload ret = std::move(it->second);
            m_globalCache.erase(it);
            return ret;
        }
    }
    return std::nullopt;
}

template<typename Payload, typename Item>
int PerStreamCache<Payload, Item>::moveReadyToGlobal()
{
    int moved = 0;
    for (auto it = m_perStreamCache.begin(); it != m_perStreamCache.end();)
    {
        it->second.removeAllReady(
            [&](Payload &&payload)
            {
                m_globalCache.emplace(StreamCachePayloadSize(payload), std::move(payload));
                moved++;
            });
        if (it->second.empty())
            it = m_perStreamCache.erase(it);
        else
            ++it;
    }
    return moved;
}

template<typename Payload, typename Item>
void PerStreamCache<Payload, Item>::put(Payload &&payload, std::optional<cudaStream_t> stream)
{
    cudaEvent_t readyEvent = StreamCachePayloadReady(payload);
    bool        per_stream = readyEvent != nullptr && cudaEventQuery(readyEvent) == cudaErrorNotReady;

    std::lock_guard guard(m_lock);

    if (per_stream)
    {
        uint64_t id      = stream ? GetCudaStreamIdHint(*stream) : (uint64_t)-1ll;
        auto     cacheIt = m_perStreamCache.find(id);
        if (cacheIt == m_perStreamCache.end())
            cacheIt = m_perStreamCache.emplace(id, &m_cacheItemAlloc).first;

        cacheIt->second.put(std::move(payload));
    }
    else
    {
        size_t size = StreamCachePayloadSize(payload);
        m_globalCache.emplace(size, std::move(payload));
    }
}

} // namespace nvcv::util

#endif // NVCV_UTIL_PER_STREAM_CACHE_IMPL_HPP
