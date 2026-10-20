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

#ifndef NVCV_PYTHON_PRIV_CACHE_HPP
#define NVCV_PYTHON_PRIV_CACHE_HPP

#include "Object.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <common/Hash.hpp>
#include <nvcv/python/Cache.hpp>

#include <atomic>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nvcvpy::priv {

namespace py = nanobind;

class Container;
class Image;
class Resource;
class Stream;

class NB_EXPORT CacheItem : public Object
{
public:
    virtual const IKey &key() const = 0;

    std::shared_ptr<CacheItem>       shared_from_this();
    std::shared_ptr<const CacheItem> shared_from_this() const;

    virtual int64_t GetSizeInBytes() const = 0;

    virtual bool isInUse() const;

    /** Return the object whose availability this metadata tracks. */
    virtual std::shared_ptr<Object> cachedObject();

    /** Borrow the resource while this item is alive; null for non-resource entries. */
    virtual Resource *cachedResource() const noexcept
    {
        return nullptr;
    }

    /** Acquire the cached object and close the reuse claim held by fetchOne(). */
    std::shared_ptr<Object> acquireClaimedObject();

    /** Reserve a pressure-reuse candidate while its ownership is validated. */
    bool tryAcquireReuseClaim();
    void releaseReuseClaim();

private:
    friend class Cache;
    // Protected by the cache mutex; transferring an index node keeps its charge.
    bool             m_indexed = false;
    std::atomic_bool m_reuseClaimed{false};
};

class ExternalCacheItem : public CacheItem
{
public:
    explicit ExternalCacheItem(std::shared_ptr<nvcvpy::ICacheItem> obj_)
        : m_obj(std::move(obj_))
        , m_size_inbytes(doComputeSizeInBytes())
    {
    }

    int64_t GetSizeInBytes() const override
    {
        // m_size_inbytes == -1 indicates failure case and value has not been computed yet
        NVCV_ASSERT(m_size_inbytes != -1
                    && "ExternalCacheItem has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
        return m_size_inbytes;
    }

    nvcvpy::ICacheItem *obj() const
    {
        return m_obj.get();
    }

    const IKey &key() const override
    {
        return m_obj->key();
    }

private:
    int64_t doComputeSizeInBytes() const
    {
        // ExternalCacheItems (CacheItems outside of nvcv, eg. operators from cvcuda) will not pollute the
        // Cache, thus for now we say they've no impact on the Cache
        return 0;
    }

    std::shared_ptr<nvcvpy::ICacheItem> m_obj;
    int64_t                             m_size_inbytes = -1;
};

class NB_EXPORT Cache
{
public:
    using ItemPreference  = bool (*)(const CacheItem &candidate, const CacheItem &selected);
    using CandidateFilter = bool (*)(const CacheItem &candidate, const IKey &requested);
    using ReusePredicate  = std::function<bool(Resource &candidate, bool pending)>;
    using RequestedBytes  = std::function<int64_t()>;

    static void Export(py::module_ &m);
    static void ExportTestHooks(py::module_ &m);

    static Cache                     &Instance();
    static void                       ClearAll();
    static void                       PruneToLimits();
    static void                       IndexPending(Resource &resource) noexcept;
    static void                       IndexPending(const std::vector<std::shared_ptr<Image>> &images) noexcept;
    static std::shared_ptr<CacheItem> PinItem(const Resource &resource);
    static size_t                     TotalSize();

    void add(CacheItem &item);
    void add(Container &container);
    void add(Stream &stream);
    void removeAllNotInUseMatching(const IKey &key);

    // Every reusable item under `key`. Only for callers that genuinely choose
    // among the candidates -- the wrapper-vector path in Image, and the
    // operators that specialize their static fetch() to pick a compatible
    // payload. It walks the whole key range, calling isInUse()
    // on every item and collecting the free ones, so its cost grows with
    // the number of same-key items cached.
    std::vector<std::shared_ptr<CacheItem>> fetch(const IKey &key) const;

    // The first reusable item under `key`, or null. Prefer this whenever the
    // caller just wants one -- it stops at the first free item instead of
    // collecting all of them to use only the first.
    // A metadata-only filter narrows broad cache keys. Idle thread-local items
    // apply it under the cache mutex; pending candidates apply it after an
    // exclusive claim and ownership proof. It must never inspect a live
    // descriptor being rebound outside the mutex, or reenter the cache. A known
    // consumer allows pending Tensor reuse on that CUDA queue without
    // introducing a dependency between otherwise independent streams.
    // A predicate validates unchanged composite storage after its exclusive
    // claim, outside the cache mutex. It replaces the default resource policy.
    // Dimension-independent keys supply the requested bytes lazily for pressure
    // checks; an optional preference orders candidates before claiming a match.
    std::shared_ptr<CacheItem> fetchOne(const IKey &key, const CandidateFilter &filter = {}, Stream *consumer = nullptr,
                                        const ReusePredicate &predicate = {}, const RequestedBytes &requestedBytes = {},
                                        ItemPreference prefer = nullptr) const;

    // Metadata-only lookup for operator state pools. It deliberately skips
    // isInUse(): asynchronous state ownership is tracked by native shared_ptrs
    // retained in the global pending-release queue.
    std::shared_ptr<CacheItem> fetchReusableNoPoll(const IKey &key, ItemPreference prefer = nullptr) const;

#ifndef NDEBUG
    // Make this function available only in Debug builds
    void dbgPrintCacheForKey(const IKey &key, const std::string &prefix = "");
#endif

    template<class T>
    std::vector<std::shared_ptr<T>> fetchAll() const
    {
        std::vector<std::shared_ptr<T>> out;

        for (const std::shared_ptr<CacheItem> &item : doSnapshotItems())
        {
            if (auto titem = std::dynamic_pointer_cast<T>(item->cachedObject()))
            {
                out.emplace_back(std::move(titem));
            }
        }
        return out;
    }

    void   clear();
    size_t size() const;

    void    setCacheLimit(int64_t new_cache_limit) const;
    int64_t getCacheLimit() const;
    int64_t getCurrentSizeInBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    // Keeps this cache reachable from the owning thread's Python state, and
    // tells a capsule that outlives it to do nothing. Shared because the
    // capsule can be destroyed by the finalizing thread, not just the owner.
    struct Anchor;
    std::shared_ptr<Anchor> m_anchor;

    Cache();
    ~Cache() noexcept;

    // Registers this cache with the calling thread's Python state, so its items
    // are released while the GIL is held and the interpreter is alive. No-op on
    // a thread without one, which then falls back to leaking at thread exit.
    bool anchorToPythonThreadState();
    void releaseItemsUnderGil();
    void detachItemsForDestruction();

    std::vector<std::shared_ptr<CacheItem>> doSnapshotItems() const;
    int64_t                                 doGetDeviceSize(int dev) const;
    int64_t                                 doGetDeviceLimit(int dev) const;
    static size_t                           instanceItemCount();
    static void evictReadyItems(int device, int64_t targetSize, std::vector<std::shared_ptr<CacheItem>> &savedItems);
    static void registerItem(const std::shared_ptr<CacheItem> &item);
    static void indexPendingLocked(Container &container);
    static void unregisterItem(CacheItem &item);
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CACHE_HPP
