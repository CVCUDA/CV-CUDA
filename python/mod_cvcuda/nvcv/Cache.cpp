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

#include "Cache.hpp"

#include "Container.hpp"
#include "Definitions.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"
#include "ThreadScope.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace nvcvpy::priv {

namespace {

class CacheTestHookError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Suspends a worker's Cache destructor so a test can hold the GIL across it.
// This is the only way to make that race deterministic, but it puts a pause
// button in teardown that ships in the wheel, so it is inert unless the process
// opts in through CVCUDA_CACHE_TLS_TEST_HOOK: without it arming is refused and
// enter() is a bool read, so a stray cvcuda._test call cannot stall a thread.
// Every wait is bounded for the same reason -- a coordination mistake must
// surface as a failing test, not as a thread that never finishes exiting.
// The fail-if-destroyed mode makes a regression to destructible storage
// observable from a subprocess.
class CacheDestructionTestHook
{
public:
    // Read once: the destructor path must not call getenv on every thread exit,
    // and a later setenv must not change the answer for a hook already in use.
    static bool enabled()
    {
        static const bool enabled = std::getenv("CVCUDA_CACHE_TLS_TEST_HOOK") != nullptr;
        return enabled;
    }

    ~CacheDestructionTestHook()
    {
        const char *mode = std::getenv("CVCUDA_CACHE_TLS_TEST_HOOK");
        if (mode != nullptr && std::strcmp(mode, "fail-if-destroyed") == 0)
        {
            std::fputs("Cache destruction test hook was destroyed at exit\n", stderr);
            std::fflush(stderr);
            std::abort();
        }
    }

    void arm()
    {
        if (!enabled())
        {
            throw CacheTestHookError(
                "Cache destruction test hook is disabled; set CVCUDA_CACHE_TLS_TEST_HOOK to enable it");
        }
        std::lock_guard lock(m_mutex);
        if (m_armed)
        {
            throw CacheTestHookError("Cache destruction test hook is already armed");
        }
        m_armed     = true;
        m_entered   = false;
        m_released  = false;
        m_completed = false;
    }

    bool enter()
    {
        if (!enabled())
        {
            return false;
        }
        std::unique_lock lock(m_mutex);
        if (!m_armed)
        {
            return false;
        }
        m_entered = true;
        m_condition.notify_all();
        if (!m_condition.wait_for(lock, kTimeout, [this] { return m_released; }))
        {
            // Nothing is coming to release us. Disarm and let the destructor run
            // its normal course so the thread can still exit.
            m_armed = false;
            return false;
        }
        return true;
    }

    void waitUntilEntered()
    {
        std::unique_lock lock(m_mutex);
        if (!m_condition.wait_for(lock, kTimeout, [this] { return m_entered; }))
        {
            m_armed = false;
            throw CacheTestHookError("Timed out waiting for a worker cache destructor to reach the test hook");
        }
    }

    void releaseAndWaitUntilCompleted()
    {
        std::unique_lock lock(m_mutex);
        m_released = true;
        m_condition.notify_all();
        bool completed = m_condition.wait_for(lock, kTimeout, [this] { return m_completed; });
        m_armed        = false;
        if (!completed)
        {
            throw CacheTestHookError("Timed out waiting for a worker cache destructor to leave the test hook");
        }
    }

    void complete()
    {
        std::lock_guard lock(m_mutex);
        m_completed = true;
        m_condition.notify_all();
    }

private:
    // Pure thread hand-off, so anything near this is already a failure. Kept
    // under the test's subprocess timeout so the error names the actual cause.
    static constexpr std::chrono::seconds kTimeout{30};

    std::mutex              m_mutex;
    std::condition_variable m_condition;
    bool                    m_armed     = false;
    bool                    m_entered   = false;
    bool                    m_released  = false;
    bool                    m_completed = false;
};

constexpr char kAnchorCapsuleName[] = "cvcuda.CacheAnchor";
constexpr char kAnchorDictKey[]     = "__cvcuda_cache_anchor__";

// Anchoring failing is not observable otherwise: the destructor would quietly
// leak instead, and every test would still pass. Counting installs gives the
// regression test something to assert on.
std::atomic<int> &anchorsInstalled()
{
    static std::atomic<int> count{0};
    return count;
}

// Py_IsFinalizing only became public API in 3.13. `_Py_IsFinalizing` reports
// the same state and has been exported since 3.7, so use it below that: this
// predicate guards every teardown path that touches Python objects, and
// Py_IsInitialized alone still reports true throughout finalization. Answering
// "available" in that window lets ~Cache decref into a half-torn-down
// interpreter, which segfaults.
bool pythonRuntimeAvailable()
{
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsInitialized() && !Py_IsFinalizing();
#else
    return Py_IsInitialized() && !_Py_IsFinalizing();
#endif
}

CacheDestructionTestHook &cacheDestructionTestHook()
{
    // Immortal, for the same reason as the cache registry and the Impl statics.
    // The comment this replaces claimed the opposite -- that glibc runs
    // thread_local destructors before static ones, so every ~Cache consulting
    // the hook had already finished. That is not true when the destructor runs
    // on a worker thread: process exit does not wait for it, so ~Cache can be
    // inside enter() while __cxa_finalize is destroying this object. It is
    // reachable only with the hook armed, which is why it was never the crash
    // that motivated leaking the others -- but the reasoning behind leaving it
    // destructible was wrong, so it goes the same way.
    static auto *hook = new CacheDestructionTestHook(); // NOSONAR: immortal by design
    return *hook;
}

} // namespace

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

template<class T>
class CachedObjectItem final : public CacheItem
{
public:
    explicit CachedObjectItem(std::shared_ptr<T> object)
        : m_object(std::move(object))
    {
    }

    const IKey &key() const override
    {
        return m_object->key();
    }

    int64_t GetSizeInBytes() const override
    {
        return m_object->GetSizeInBytes();
    }

    bool isInUse() const override
    {
        return m_object.use_count() > 1 || CacheItem::isInUse();
    }

    std::shared_ptr<Object> cachedObject() override
    {
        return m_object;
    }

    Resource *cachedResource() const noexcept override
    {
        if constexpr (std::is_base_of_v<Resource, T>)
            return m_object.get();
        else
            return nullptr;
    }

private:
    std::shared_ptr<T> m_object;
};

std::shared_ptr<CacheItem> CacheItem::shared_from_this()
{
    return std::static_pointer_cast<CacheItem>(Object::shared_from_this());
}

std::shared_ptr<const CacheItem> CacheItem::shared_from_this() const
{
    return std::static_pointer_cast<const CacheItem>(Object::shared_from_this());
}

bool CacheItem::isInUse() const
{
    // Inspect ownership without acquiring and releasing another strong reference.
    return m_reuseClaimed.load(std::memory_order_acquire) || weak_from_this().use_count() > 1;
}

std::shared_ptr<Object> CacheItem::cachedObject()
{
    return Object::shared_from_this();
}

std::shared_ptr<Object> CacheItem::acquireClaimedObject()
{
    try
    {
        std::shared_ptr<Object> object = cachedObject();
        releaseReuseClaim();
        return object;
    }
    catch (...)
    {
        releaseReuseClaim();
        throw;
    }
}

bool CacheItem::tryAcquireReuseClaim()
{
    bool expected = false;
    return m_reuseClaimed.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
}

void CacheItem::releaseReuseClaim()
{
    m_reuseClaimed.store(false, std::memory_order_release);
}

using Items = std::unordered_multimap<const IKey *, std::shared_ptr<CacheItem>, HashKey, KeyEqual>;

namespace {

bool HasPendingReleaseOwners(const std::shared_ptr<CacheItem> &item)
{
    const auto *resource = item->cachedResource();
    return resource != nullptr && resource->hasPendingReleaseOwners();
}

bool AllowsReadyReuse(const std::shared_ptr<CacheItem> &item)
{
    const auto *resource = item->cachedResource();
    return resource == nullptr || resource->allowsCacheReuse(false);
}

std::shared_ptr<CacheItem> FindReadyOrCollect(const Items &items, const IKey &key, const Cache::CandidateFilter &filter,
                                              bool claimReady, std::vector<std::shared_ptr<CacheItem>> &candidates)
{
    const auto [firstItem, lastItem] = items.equal_range(&key);
    for (auto it = firstItem; it != lastItem; ++it)
    {
        if (!it->second->isInUse())
        {
            if (filter && !filter(*it->second, key))
                continue;
            if (claimReady && AllowsReadyReuse(it->second) && it->second->tryAcquireReuseClaim())
                return it->second;
        }
        candidates.emplace_back(it->second);
    }
    return {};
}

// Composite predicates may pin child cache entries, so callers must release
// the cache mutex before validation while the candidate claim preserves exclusivity.
bool TryAcceptCandidate(const std::shared_ptr<CacheItem> &candidate, const IKey &requested, Stream *consumer,
                        const Cache::CandidateFilter &filter, const Cache::ReusePredicate &predicate,
                        bool wouldExceedLimit)
{
    if (!candidate->tryAcquireReuseClaim())
        return false;
    try
    {
        std::shared_ptr<Object> object   = candidate->cachedObject();
        auto                   *resource = candidate->cachedResource();
        const bool              ready    = predicate && object.use_count() == 2;
        if (resource != nullptr && (ready || Stream::IsPendingOnly(object))
            && (!filter || filter(*candidate, requested))
            && (predicate ? predicate(*resource, !ready) : resource->allowsCacheReuse(true))
            && (ready || wouldExceedLimit || (consumer && resource->isOrderedOn(*consumer))))
            return true;
    }
    catch (...)
    {
        candidate->releaseReuseClaim();
        throw;
    }
    candidate->releaseReuseClaim();
    return false;
}

template<class Predicate>
bool TryAcceptClaimedReadyCandidate(const std::shared_ptr<CacheItem> &candidate, const Predicate &predicate)
{
    try
    {
        Resource *resource = candidate->cachedResource();
        if (resource != nullptr && predicate(*resource, false))
            return true;
    }
    catch (...)
    {
        candidate->releaseReuseClaim();
        throw;
    }
    candidate->releaseReuseClaim();
    return false;
}

// Every piece of shared cache state a Cache destructor touches, in one block
// that is never destroyed. ~Cache() runs from a worker thread's thread_local
// teardown, which nothing waits for -- Thread.join() returns once the
// interpreter drops the thread state, well before pthread_exit gets to those
// destructors -- so it can still be locking the mutex, debiting the size map,
// and erasing from the registry while the main thread is inside
// __cxa_finalize. Destroying any of this there frees it under that thread.
struct SharedState
{
    std::mutex                       mtx;
    std::unordered_map<int, int64_t> cacheLimitInBytes;
    std::unordered_map<int, int64_t> currentSizeInBytes;
    std::unordered_set<Cache *>      instances;
    Items                            pendingItems;
    std::atomic_bool                 deferredPrune{false};
    std::atomic_bool                 failNextPruneAllocation{false};
};

SharedState &sharedState()
{
    static auto *state = new SharedState; // NOSONAR: deliberately immortal
    return *state;
}

} // namespace

struct Cache::Impl
{
    Items items;

    // Bound to sharedState() so the spelling of every use site is unchanged.
    // A reference has no destructor, so these add nothing to teardown.
    inline static std::mutex                       &mtx                  = sharedState().mtx;
    inline static std::unordered_map<int, int64_t> &cache_limit_inbytes  = sharedState().cacheLimitInBytes;
    inline static std::unordered_map<int, int64_t> &current_size_inbytes = sharedState().currentSizeInBytes;
};

struct Cache::Anchor
{
    static void DestroyCapsule(PyObject *capsule);

    std::mutex mtx;
    Cache     *cache = nullptr;
};

Cache::Cache()
{
    pimpl = std::make_unique<Impl>();
    std::lock_guard lk(Impl::mtx);
    sharedState().instances.insert(this);
}

void Cache::registerItem(const std::shared_ptr<CacheItem> &item)
{
    NVCV_ASSERT(!item->m_indexed);
    // The caller creates the accounting node before inserting the cache node.
    const int64_t bytes = item->GetSizeInBytes();
    Impl::current_size_inbytes.at(item->key().deviceId()) += bytes;
    item->m_indexed = true;
    // Zero-byte wrappers never enter the pending allocation index.
    if (bytes != 0)
        if (auto *resource = item->cachedResource())
            resource->m_cacheItem = item.get();
}

void Cache::unregisterItem(CacheItem &item)
{
    NVCV_ASSERT(item.m_indexed);
    Impl::current_size_inbytes.at(item.key().deviceId()) -= item.GetSizeInBytes();
    item.m_indexed = false;
    if (auto *resource = item.cachedResource(); resource && resource->m_cacheItem == &item)
        resource->m_cacheItem = nullptr;
}

size_t Cache::instanceItemCount()
{
    return std::accumulate(sharedState().instances.cbegin(), sharedState().instances.cend(), static_cast<size_t>(0),
                           [](size_t count, const Cache *instance) { return count + instance->pimpl->items.size(); });
}

void Cache::evictReadyItems(int device, int64_t targetSize, std::vector<std::shared_ptr<CacheItem>> &savedItems)
{
    for (const Cache *instance : sharedState().instances)
    {
        auto &items = instance->pimpl->items;
        for (auto it = items.begin(); it != items.end() && instance->doGetDeviceSize(device) > targetSize;)
        {
            if (it->first->deviceId() == device && !HasPendingReleaseOwners(it->second))
            {
                savedItems.emplace_back(std::move(it->second));
                unregisterItem(*savedItems.back());
                it = items.erase(it);
            }
            else
            {
                ++it;
            }
        }
        if (instance->doGetDeviceSize(device) <= targetSize)
            return;
    }
}

void Cache::detachItemsForDestruction()
{
    std::lock_guard lk(Impl::mtx);
    sharedState().instances.erase(this);
    sharedState().pendingItems.reserve(sharedState().pendingItems.size() + pimpl->items.size());
    for (auto it = pimpl->items.begin(); it != pimpl->items.end();)
    {
        if (HasPendingReleaseOwners(it->second))
            sharedState().pendingItems.insert(pimpl->items.extract(it++));
        else
            ++it;
    }
}

Cache::~Cache() noexcept
{
    std::unique_ptr<Impl> localPimpl;

    // Before anything else: a capsule destroyed after this point must not reach
    // a Cache that no longer exists. The interpreter can clear another thread's
    // state from the finalizing thread, so this races with Anchor::DestroyCapsule
    // and the lock is what orders them.
    if (m_anchor)
    {
        std::lock_guard lk(m_anchor->mtx);
        m_anchor->cache = nullptr;
    }

    try
    {
        detachItemsForDestruction();

        localPimpl          = std::move(this->pimpl);
        bool testHookActive = cacheDestructionTestHook().enter();

        // The anchor already released the Python-owned items, on a thread that
        // held the GIL with the interpreter alive, so there is nothing left here
        // that needs either. A thread that could not be anchored still reaches
        // this with items; destroying them would need the GIL, which this
        // context cannot get, so leaking is the only safe answer.
        if (localPimpl->items.empty() || (pythonRuntimeAvailable() && util::HasPythonThreadState()))
        {
            {
                std::lock_guard lk(Impl::mtx);
                for (const auto &[key, item] : localPimpl->items) unregisterItem(*item);
            }
            localPimpl.reset();
        }
        else
        {
            localPimpl.release();
        }

        if (testHookActive)
        {
            cacheDestructionTestHook().complete();
        }
    }
    catch (...)
    {
        // Teardown can run without a usable Python runtime or GIL. Leaking is
        // safer than releasing Python-owned cache items in that state.
        if (localPimpl)
        {
            localPimpl.release();
        }
        if (pimpl)
        {
            pimpl.release();
        }
    }
}

void Cache::add(CacheItem &item)
{
    std::vector<std::shared_ptr<CacheItem>> savedItems;
    {
        std::unique_lock lk(Impl::mtx);
        const int        dev       = item.key().deviceId();
        const int64_t    itemBytes = item.GetSizeInBytes();
        const int64_t    limit     = doGetDeviceLimit(dev);

        if (itemBytes > limit)
        {
            return;
        }

        // Publish the accounting node before the cache entry. If allocation
        // fails here, no retained item can become invisible to the byte limit.
        Impl::current_size_inbytes.try_emplace(dev, 0);

        const int64_t targetSize = limit - itemBytes;
        if (doGetDeviceSize(dev) > targetSize)
        {
            // Other threads' ready entries can own Python objects. A native
            // caller cannot evict them; leave this new item uncached instead.
            if (!util::HasPythonThreadState())
                return;
            savedItems.reserve(instanceItemCount());

            // Pending entries are the only index pressure reuse can query.
            // Keep them registered: detaching one merely hides memory that the
            // release queue must still retain and can cause repeated allocation
            // of the same key. Ready entries are safe eviction candidates.
            evictReadyItems(dev, targetSize, savedItems);
        }

        // All remaining room may be occupied by in-flight entries. Do not
        // displace their reuse index or exceed the configured retained-cache
        // budget; the new object remains caller-owned and uncached.
        if (doGetDeviceSize(dev) > targetSize)
        {
            return;
        }
        auto inserted = pimpl->items.emplace(&item.key(), item.shared_from_this());
        registerItem(inserted->second);
    }
}

void Cache::add(Container &container)
{
    auto item = std::make_shared<CachedObjectItem<Container>>(SharedContainerFrom(container));
    add(*item);
}

void Cache::add(Stream &stream)
{
    auto item = std::make_shared<CachedObjectItem<Stream>>(stream.sharedStream());
    add(*item);
}

void Cache::indexPendingLocked(Container &container)
{
    if (container.m_cacheItem)
        return;
    auto item = std::make_shared<CachedObjectItem<Container>>(SharedContainerFrom(container));
    Impl::current_size_inbytes.try_emplace(item->key().deviceId(), 0);
    sharedState().pendingItems.emplace(&item->key(), item);
    registerItem(item);
}

void Cache::IndexPending(const std::vector<std::shared_ptr<Image>> &images) noexcept
{
    try
    {
        std::lock_guard lock(Impl::mtx);
        for (const auto &image : images)
            if (image->GetSizeInBytes() != 0)
                indexPendingLocked(*image);
    }
    catch (...)
    {
        std::fputs("[cvcuda] Failed to index pending image allocations; continuing without reuse\n", stderr);
    }
}

void Cache::IndexPending(Resource &resource) noexcept
{
    try
    {
        auto *container = dynamic_cast<Container *>(&resource);
        if (container == nullptr || container->GetSizeInBytes() == 0
            || (dynamic_cast<Tensor *>(container) == nullptr && dynamic_cast<Image *>(container) == nullptr
                && dynamic_cast<ImageBatchVarShape *>(container) == nullptr))
            return;

        std::lock_guard lk(Impl::mtx);
        indexPendingLocked(*container);
    }
    catch (...)
    {
        // The release ledger already guarantees lifetime. Losing this
        // best-effort reuse index under host allocation pressure is safe.
        std::fputs("[cvcuda] Failed to index a pending cache resource; continuing without reuse\n", stderr);
    }
}

std::shared_ptr<CacheItem> Cache::PinItem(const Resource &resource)
{
    std::lock_guard lock(Impl::mtx);
    return resource.m_cacheItem ? resource.m_cacheItem->shared_from_this() : nullptr;
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
        std::unique_lock lk(Impl::mtx);

        auto [firstItem, lastItem] = pimpl->items.equal_range(&key);

        auto numItems = std::distance(firstItem, lastItem);

        auto it = firstItem;
        for (decltype(numItems) i = 0; i < numItems; ++i)
        {
            if (!it->second->isInUse())
            {
                holdItemsUntilMtxUnlocked.push_back(it->second);
                unregisterItem(*it->second);
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

    std::unique_lock lk(Impl::mtx);

    auto [firstItem, lastItem] = pimpl->items.equal_range(&key);

    v.reserve(distance(firstItem, lastItem));

    for (auto it = firstItem; it != lastItem; ++it)
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
    std::unique_lock                        lk(Impl::mtx);
    auto                                    itrange = pimpl->items.equal_range(&key);

    for (auto it = itrange.first; it != itrange.second; ++it)
    {
        std::cerr << prefix << typeid(*(it->second)).name() << " - " << it->second.use_count() << std::endl;
    }
}
#endif

std::shared_ptr<CacheItem> Cache::fetchOne(const IKey &key, const CandidateFilter &filter, Stream *consumer,
                                           const ReusePredicate &predicate, const RequestedBytes &requestedBytes,
                                           ItemPreference prefer) const
{
    std::vector<std::shared_ptr<CacheItem>> pendingCandidates;
    bool                                    wouldExceedLimit = false;
    {
        std::unique_lock lk(Impl::mtx);

        // Avoid discovering the end of the full equal-key group on the common
        // ready hit. find() may return any member, which is sufficient here.
        auto       local = pimpl->items.find(&key);
        const bool localMatches
            = local != pimpl->items.end() && !local->second->isInUse() && (!filter || filter(*local->second, key));
        if (!prefer && !predicate && localMatches && AllowsReadyReuse(local->second)
            && local->second->tryAcquireReuseClaim())
            return local->second;
        if (!prefer && predicate && localMatches && local->second->tryAcquireReuseClaim())
        {
            auto candidate = local->second;
            lk.unlock();
            if (TryAcceptClaimedReadyCandidate(candidate, predicate))
                return candidate;
            candidate.reset();
            lk.lock();
        }
        else if (!prefer && !predicate && local != pimpl->items.end() && consumer != nullptr)
        {
            auto candidate = local->second;
            lk.unlock();
            if (TryAcceptCandidate(candidate, key, consumer, filter, predicate, wouldExceedLimit))
                return candidate;
            candidate.reset();
            lk.lock();
        }

        if (auto item = FindReadyOrCollect(pimpl->items, key, filter, !prefer && !predicate, pendingCandidates))
            return item;
        // A completed item in the global pending index can be rebound by
        // another thread. Validate its descriptor only after the claim and
        // ownership checks below exclude that in-progress mutation.
        FindReadyOrCollect(sharedState().pendingItems, key, {}, false, pendingCandidates);
        if (pendingCandidates.empty())
            return {};

        const int     dev       = key.deviceId();
        const int64_t itemBytes = requestedBytes ? requestedBytes() : pendingCandidates.front()->GetSizeInBytes();
        const int64_t limit     = doGetDeviceLimit(dev);
        const int64_t current   = doGetDeviceSize(dev);
        wouldExceedLimit        = itemBytes > 0 && (itemBytes > limit || current > limit - itemBytes);
        if (!wouldExceedLimit && consumer == nullptr && !predicate)
        {
            return {};
        }
    }

    if (prefer)
        std::sort(pendingCandidates.begin(), pendingCandidates.end(),
                  [prefer](const auto &a, const auto &b) { return prefer(*a, *b); });
    for (const std::shared_ptr<CacheItem> &candidate : pendingCandidates)
    {
        if (TryAcceptCandidate(candidate, key, consumer, filter, predicate, wouldExceedLimit))
            return candidate;
    }
    return {};
}

std::shared_ptr<CacheItem> Cache::fetchReusableNoPoll(const IKey &key, ItemPreference prefer) const
{
    std::shared_ptr<CacheItem> selected;
    std::unique_lock           lk(Impl::mtx);

    auto [firstItem, lastItem] = pimpl->items.equal_range(&key);
    for (auto it = firstItem; it != lastItem; ++it)
    {
        if (selected == nullptr || (prefer != nullptr && prefer(*it->second, *selected)))
        {
            selected = it->second;
        }
    }
    return selected;
}

void Cache::clear()
{
    // The byte counter is shared by every thread's cache, so dropping items
    // without crediting them back leaves it reporting memory that is already
    // gone, and the next add() on any thread evicts against that stale total.
    Items savedItems;
    {
        std::lock_guard lk(Impl::mtx);
        for (const auto &[nodeKey, node] : pimpl->items)
        {
            unregisterItem(*node);
        }
        savedItems.swap(pimpl->items);
    }
    // Destroyed after the lock is dropped, like every other bulk release here:
    // item destructors run arbitrary code, including paths back into the cache.
}

size_t Cache::size() const
{
    std::lock_guard lk(Impl::mtx);
    return pimpl->items.size();
}

void Cache::setCacheLimit(int64_t new_cache_limit_inbytes) const
{
    if (new_cache_limit_inbytes < 0)
    {
        throw std::invalid_argument("Cache limit must be non-negative.");
    }

    int dev = 0;
    util::CheckThrow(cudaGetDevice(&dev));

    size_t free_mem;
    size_t total_mem;
    util::CheckThrow(cudaMemGetInfo(&free_mem, &total_mem));

    if (static_cast<int64_t>(total_mem) < new_cache_limit_inbytes)
    {
        std::cerr << "WARNING: new_cache_limit=" << new_cache_limit_inbytes
                  << " is more than total available memory on device " << dev << ": " << total_mem << std::endl;
    }

    std::vector<std::shared_ptr<CacheItem>> savedItems;
    {
        std::unique_lock lk(Impl::mtx);
        if (doGetDeviceSize(dev) > new_cache_limit_inbytes)
        {
            savedItems.reserve(instanceItemCount());
        }

        Impl::cache_limit_inbytes[dev] = new_cache_limit_inbytes;
        if (doGetDeviceSize(dev) > new_cache_limit_inbytes)
        {
            evictReadyItems(dev, new_cache_limit_inbytes, savedItems);
        }
    }
}

int64_t Cache::getCacheLimit() const
{
    int dev = 0;
    util::CheckThrow(cudaGetDevice(&dev));
    std::unique_lock lk(Impl::mtx);
    return doGetDeviceLimit(dev);
}

int64_t Cache::doGetDeviceLimit(int dev) const
{
    auto it = Impl::cache_limit_inbytes.find(dev);
    return it != Impl::cache_limit_inbytes.end() ? it->second : 0;
}

int64_t Cache::getCurrentSizeInBytes() const
{
    int dev = 0;
    util::CheckThrow(cudaGetDevice(&dev));
    std::unique_lock lk(Impl::mtx);
    return doGetDeviceSize(dev);
}

int64_t Cache::doGetDeviceSize(int dev) const
{
    auto it = Impl::current_size_inbytes.find(dev);
    return it != Impl::current_size_inbytes.end() ? it->second : 0;
}

std::vector<std::shared_ptr<CacheItem>> Cache::doSnapshotItems() const
{
    // To avoid keeping mutex locked for too long, let's first gather all items
    // into a vector, unlock the mutex, and then iterate through them.
    std::vector<std::shared_ptr<CacheItem>> v;

    {
        std::unique_lock lk(Impl::mtx);
        v.reserve(pimpl->items.size());

        for (auto it = pimpl->items.begin(); it != pimpl->items.end(); ++it)
        {
            v.push_back(it->second);
        }
    }

    return v;
}

void Cache::Anchor::DestroyCapsule(PyObject *capsule)
{
    // Runs from PyThreadState_Clear: the GIL is held and the interpreter is
    // alive, which is exactly what the destructor path cannot assume.
    auto *owned = static_cast<std::shared_ptr<Anchor> *>(PyCapsule_GetPointer(capsule, kAnchorCapsuleName));
    if (owned == nullptr)
    {
        PyErr_Clear();
        return;
    }

    // This is a C callback: an exception unwinding into CPython's capsule
    // teardown is undefined behaviour, so failure leaks the handle rather than
    // propagating -- the same trade ~Cache() already makes.
    try
    {
        {
            std::lock_guard lk((*owned)->mtx);
            if (Cache *cache = (*owned)->cache)
            {
                cache->releaseItemsUnderGil();
            }
        }
        delete owned; // NOSONAR: the capsule destructor is where this handle dies
    }
    catch (...) // NOSONAR: anything escaping into CPython's C frames is undefined
    {
        // Swallowed on purpose, and `owned` is left to leak: there is no caller
        // to report to, and the process is better off with a stranded handle
        // than with an exception unwinding through the interpreter.
    }
}

void Cache::releaseItemsUnderGil()
{
    std::vector<std::shared_ptr<CacheItem>> readyItems;
    {
        std::lock_guard lk(Impl::mtx);
        readyItems.reserve(pimpl->items.size());
        sharedState().pendingItems.reserve(sharedState().pendingItems.size() + pimpl->items.size());
        for (auto it = pimpl->items.begin(); it != pimpl->items.end();)
        {
            if (HasPendingReleaseOwners(it->second))
            {
                sharedState().pendingItems.insert(pimpl->items.extract(it++));
            }
            else
            {
                unregisterItem(*it->second);
                readyItems.emplace_back(std::move(it->second));
                it = pimpl->items.erase(it);
            }
        }
    }
    // Destroyed outside the lock, like every other bulk release here.
}

bool Cache::anchorToPythonThreadState()
{
    if (!pythonRuntimeAvailable() || !util::HasPythonThreadState())
    {
        return false;
    }

    PyObject *dict = PyThreadState_GetDict(); // borrowed
    if (dict == nullptr)
    {
        return false;
    }

    m_anchor        = std::make_shared<Anchor>();
    m_anchor->cache = this;

    // The capsule owns a shared_ptr copy, so it stays valid even when it is
    // destroyed after this Cache is already gone.
    auto     *owned   = new std::shared_ptr<Anchor>(m_anchor); // NOSONAR: ownership passes to the capsule
    PyObject *capsule = PyCapsule_New(owned, kAnchorCapsuleName, &Anchor::DestroyCapsule);
    if (capsule == nullptr)
    {
        PyErr_Clear();
        delete owned; // NOSONAR: the capsule never took ownership
        m_anchor.reset();
        return false;
    }

    if (PyDict_SetItemString(dict, kAnchorDictKey, capsule) != 0)
    {
        PyErr_Clear();
        // The DECREF below is the capsule's last reference, and its destructor
        // would release this thread's items on the way out. Detach first, so a
        // failed install leaves the cache exactly as it found it and the next
        // call can try again.
        {
            std::lock_guard lk(m_anchor->mtx);
            m_anchor->cache = nullptr;
        }
        m_anchor.reset();
        Py_DECREF(capsule);
        return false;
    }

    Py_DECREF(capsule);
    anchorsInstalled().fetch_add(1);
    return true;
}

Cache &Cache::Instance()
{
    thread_local Cache cache;
    // Must outlive the if: it records, for the life of the thread, whether this
    // cache is already anchored. Narrowing it to an init-statement would reset
    // it on every call and reinstall the capsule each time.
    thread_local bool  anchored = false;
    if (!anchored) // NOSONAR: the flag is per-thread state, not a temporary
    {
        // Only latched on success: an install can fail on allocation, and the
        // fallback for an unanchored thread is leaking its items at exit, so a
        // later attempt is worth the two cheap C-API calls this costs.
        anchored = cache.anchorToPythonThreadState();
    }
    if (sharedState().deferredPrune.load(std::memory_order_relaxed)
        && sharedState().deferredPrune.exchange(false, std::memory_order_acq_rel))
        PruneToLimits();
    return cache;
}

void Cache::ClearAll()
{
    Items savedItems;
    {
        std::lock_guard lk(Cache::Impl::mtx);
        const size_t    totalItems = sharedState().pendingItems.size() + instanceItemCount();
        savedItems.reserve(totalItems);
        std::for_each(sharedState().instances.begin(), sharedState().instances.end(),
                      [&savedItems](const Cache *instance) { savedItems.merge(instance->pimpl->items); });
        savedItems.merge(sharedState().pendingItems);
        for (const auto &[key, item] : savedItems) unregisterItem(*item);
    }
    // savedItems is destroyed by leaving scope, outside the lock.
}

void Cache::PruneToLimits()
{
    if (!util::HasPythonThreadState())
    {
        sharedState().deferredPrune.store(true, std::memory_order_release);
        return;
    }

    std::vector<std::shared_ptr<CacheItem>> savedItems;
    {
        std::lock_guard lk(Cache::Impl::mtx);
        const bool      overLimit = std::any_of(
                 Cache::Impl::current_size_inbytes.begin(), Cache::Impl::current_size_inbytes.end(),
                 [](const auto &entry)
                 {
                auto limit = Cache::Impl::cache_limit_inbytes.find(entry.first);
                return entry.second > (limit != Cache::Impl::cache_limit_inbytes.end() ? limit->second : 0);
            });
        if (const bool hasCompletedPending
            = std::any_of(sharedState().pendingItems.begin(), sharedState().pendingItems.end(),
                          [](const auto &entry) { return !HasPendingReleaseOwners(entry.second); });
            !overLimit && !hasCompletedPending)
            return;

        size_t itemCount = sharedState().pendingItems.size();
        if (overLimit)
            itemCount += instanceItemCount();
        try
        {
            if (sharedState().failNextPruneAllocation.exchange(false, std::memory_order_relaxed))
                throw std::bad_alloc{};
            savedItems.reserve(itemCount);
        }
        catch (...)
        {
            sharedState().deferredPrune.store(true, std::memory_order_release);
            throw;
        }

        for (auto it = sharedState().pendingItems.begin(); it != sharedState().pendingItems.end();)
        {
            if (!HasPendingReleaseOwners(it->second))
            {
                unregisterItem(*it->second);
                savedItems.emplace_back(std::move(it->second));
                it = sharedState().pendingItems.erase(it);
            }
            else
            {
                ++it;
            }
        }

        if (!overLimit)
            return;

        for (const Cache *instance : sharedState().instances)
        {
            for (auto it = instance->pimpl->items.begin(); it != instance->pimpl->items.end();)
            {
                const int dev = it->first->deviceId();
                if (instance->doGetDeviceSize(dev) > instance->doGetDeviceLimit(dev)
                    && !HasPendingReleaseOwners(it->second))
                {
                    savedItems.emplace_back(std::move(it->second));
                    unregisterItem(*savedItems.back());
                    it = instance->pimpl->items.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
    }
}

size_t Cache::TotalSize()
{
    std::lock_guard lk(Cache::Impl::mtx);
    return sharedState().pendingItems.size()
         + std::accumulate(sharedState().instances.cbegin(), sharedState().instances.cend(), static_cast<size_t>(0),
                           [](size_t sum, const Cache *instance) { return sum + instance->pimpl->items.size(); });
}

void Cache::Export(py::module_ &m)
{
    using namespace nanobind::literals;

    // Initialize per-device cache limits to half each GPU's total memory.
    // Tolerate hosts with no CUDA device or only a stub libcuda available
    // (CPU-only build/CI nodes, CUDA_VISIBLE_DEVICES="", manylinux build
    // hosts that resolve libcuda.so.1 to a stub). cudaGetDeviceCount may
    // return cudaSuccess with deviceCount=0, cudaErrorNoDevice, or
    // cudaErrorStubLibrary. In any of those there is nothing to seed;
    // skipping leaves `import cvcuda` working. The cache cannot actually
    // be used until a real device is present, so deferring is safe.
    {
        int deviceCount = 0;
        if (cudaError_t err = cudaGetDeviceCount(&deviceCount); err == cudaErrorNoDevice || err == cudaErrorStubLibrary)
        {
            (void)cudaGetLastError(); // clear sticky error
            deviceCount = 0;
        }
        else
        {
            util::CheckThrow(err);
        }
        if (deviceCount > 0)
        {
            int savedDev = 0;
            util::CheckThrow(cudaGetDevice(&savedDev));
            for (int d = 0; d < deviceCount; ++d)
            {
                util::CheckThrow(cudaSetDevice(d));
                size_t free_mem;
                size_t total_mem;
                util::CheckThrow(cudaMemGetInfo(&free_mem, &total_mem));
                Impl::cache_limit_inbytes[d] = static_cast<int64_t>(total_mem / 2);
            }
            util::CheckThrow(cudaSetDevice(savedDev));
        }
    }

    // Make sure cache is cleared up when script ends.
    util::RegisterCleanup(m, Cache::ClearAll);

    m.def(
        "clear_cache",
        [](ThreadScope scope)
        {
            // A stream holds each op's resources until the GPU retires the
            // work that used them. Drain those holds first, so clearing the
            // cache also releases their resources -- and, more importantly, so
            // no buffer is freed here while a kernel can still be reading it.
            Stream::DrainAllPendingReleases();
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
        "Returns the cache limit [in bytes] for the current CUDA device.");
    m.def(
        "set_cache_limit_inbytes",
        [](int64_t new_cache_limit_inbytes) { Cache::Instance().setCacheLimit(new_cache_limit_inbytes); },
        "Sets the retained-cache budget [in bytes] for the current CUDA device, shared across threads. "
        "Pending GPU resources remain charged until retirement and may temporarily exceed this budget. "
        "Zero disables retention of completed allocations, not safe reuse of pending tensors.");

    m.def(
        "current_cache_size_inbytes", [] { return Cache::Instance().getCurrentSizeInBytes(); },
        "Returns indexed cache storage [in bytes] for the current CUDA device, including pending entries. "
        "This is not total device memory usage; user-owned uncached and wrapped external storage is excluded.");

    // Just to check if fetchAll compiles, it's harmless
    Cache::Instance().fetchAll<Cache>();
}

void Cache::ExportTestHooks(py::module_ &m)
{
    m.def("cache_filter_on_live_item",
          [](Container &container, bool scan)
          {
              // An owned descriptor may be mid-rebind on another thread. Reject
              // any filter call here without relying on a racy stress test.
              auto reject = [](const CacheItem &, const IKey &) -> bool
              {
                  throw CacheTestHookError("Cache filter inspected a live descriptor");
              };
              auto prefer = [](const CacheItem &a, const CacheItem &b)
              {
                  return a.GetSizeInBytes() < b.GetSizeInBytes();
              };
              if (auto item
                  = Cache::Instance().fetchOne(container.key(), reject, nullptr, {}, {}, scan ? +prefer : nullptr))
              {
                  item->releaseReuseClaim();
                  throw CacheTestHookError("Cache returned a live descriptor");
              }
          });
    m.def("fail_next_cache_prune",
          [](bool armed) { return sharedState().failNextPruneAllocation.exchange(armed, std::memory_order_relaxed); });
    m.def("add_to_unanchored_worker_cache",
          [](Container &container)
          {
              auto               item = std::make_shared<CachedObjectItem<Container>>(SharedContainerFrom(container));
              std::exception_ptr error;
              std::thread        worker(
                  [&item, &error]
                  {
                      try
                      {
                          Cache::Instance().add(*item);
                      }
                      catch (...) // NOSONAR: rethrow on the calling Python thread
                      {
                          error = std::current_exception();
                      }
                  });
              worker.join();
              if (error)
                  std::rethrow_exception(error);
          });
    m.def("arm_cache_tls_destructor", [] { cacheDestructionTestHook().arm(); });
    m.def(
        "wait_cache_tls_destructor", [] { cacheDestructionTestHook().waitUntilEntered(); },
        py::call_guard<py::gil_scoped_release>());
    m.def("release_and_wait_cache_tls_destructor", [] { cacheDestructionTestHook().releaseAndWaitUntilCompleted(); });
    m.def("cache_anchors_installed", [] { return anchorsInstalled().load(); });
}

} // namespace nvcvpy::priv
