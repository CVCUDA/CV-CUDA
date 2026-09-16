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

#include "Definitions.hpp"
#include "Stream.hpp"
#include "ThreadScope.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
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
    // Safe to destroy at exit: glibc runs thread_local destructors before
    // static ones, so every ~Cache that consults the hook has already run, and
    // the hook is inert anyway unless a test opted the process in.
    static CacheDestructionTestHook hook;
    return hook;
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

namespace {

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
    std::mutex mtx;
    Cache     *cache = nullptr;
};

std::unordered_set<Cache *> &Cache::instances()
{
    return sharedState().instances;
}

Cache::Cache()
{
    pimpl = std::make_unique<Impl>();
    std::lock_guard lk(Impl::mtx);
    instances().insert(this);
}

Cache::~Cache() noexcept
{
    std::unique_ptr<Impl> localPimpl;

    // Before anything else: a capsule destroyed after this point must not reach
    // a Cache that no longer exists. The interpreter can clear another thread's
    // state from the finalizing thread, so this races with destroyAnchorCapsule
    // and the lock is what orders them.
    if (m_anchor)
    {
        std::lock_guard lk(m_anchor->mtx);
        m_anchor->cache = nullptr;
    }

    try
    {
        {
            std::lock_guard lk(Impl::mtx);
            instances().erase(this);
            // It might not be safe to call destructors here, decrease the size manually
            for (const auto &[nodeKey, node] : pimpl->items)
            {
                int dev = nodeKey->deviceId();
                Impl::current_size_inbytes[dev] -= node->GetSizeInBytes();
            }
        }

        localPimpl          = std::move(this->pimpl);
        bool testHookActive = cacheDestructionTestHook().enter();

        // The anchor already released the Python-owned items, on a thread that
        // held the GIL with the interpreter alive, so there is nothing left here
        // that needs either. A thread that could not be anchored still reaches
        // this with items; destroying them would need the GIL, which this
        // context cannot get, so leaking is the only safe answer.
        if (localPimpl->items.empty() || (pythonRuntimeAvailable() && PyGILState_Check()))
        {
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
        // Leak intentionally if the Python runtime is not available anymore.
        // See https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
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
    Items savedItems;
    {
        std::unique_lock lk(Impl::mtx);
        int              dev = item.key().deviceId();

        if (item.GetSizeInBytes() > doGetDeviceLimit(dev))
        {
            return;
        }

        if (item.GetSizeInBytes() + doGetDeviceSize(dev) > doGetDeviceLimit(dev))
        {
            // Evict only items belonging to this device. The size is shared
            // with every other thread's cache, so credit back what this one
            // actually drops -- zeroing it would discard their bytes too and
            // let the device's total grow past the limit unnoticed.
            for (auto it = pimpl->items.begin(); it != pimpl->items.end();)
            {
                if (it->first->deviceId() == dev)
                {
                    Impl::current_size_inbytes[dev] -= it->second->GetSizeInBytes();
                    savedItems.insert(pimpl->items.extract(it++));
                }
                else
                {
                    ++it;
                }
            }
        }

        pimpl->items.emplace(&item.key(), item.shared_from_this());
        Impl::current_size_inbytes[dev] += item.GetSizeInBytes();
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
        std::unique_lock lk(Impl::mtx);

        auto [firstItem, lastItem] = pimpl->items.equal_range(&key);

        auto numItems = std::distance(firstItem, lastItem);

        auto it = firstItem;
        for (decltype(numItems) i = 0; i < numItems; ++i)
        {
            if (!it->second->isInUse())
            {
                holdItemsUntilMtxUnlocked.push_back(it->second);
                int dev = it->first->deviceId();
                Impl::current_size_inbytes[dev] -= it->second->GetSizeInBytes();
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

std::shared_ptr<CacheItem> Cache::fetchOne(const IKey &key) const
{
    std::unique_lock lk(Impl::mtx);

    auto [firstItem, lastItem] = pimpl->items.equal_range(&key);

    for (auto it = firstItem; it != lastItem; ++it)
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
    // The byte counter is shared by every thread's cache, so dropping items
    // without crediting them back leaves it reporting memory that is already
    // gone, and the next add() on any thread evicts against that stale total.
    Items savedItems;
    {
        std::lock_guard lk(Impl::mtx);
        for (const auto &[nodeKey, node] : pimpl->items)
        {
            Impl::current_size_inbytes[nodeKey->deviceId()] -= node->GetSizeInBytes();
        }
        savedItems.swap(pimpl->items);
    }
    // Destroyed after the lock is dropped, like every other bulk release here:
    // item destructors run arbitrary code, including paths back into the cache.
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

    Items savedItems;
    {
        std::unique_lock lk(Impl::mtx);
        if (doGetDeviceSize(dev) > new_cache_limit_inbytes)
        {
            // Evict only items belonging to this device, crediting back what
            // this cache actually drops: the size is shared with every other
            // thread's cache and theirs survive this call.
            for (auto it = pimpl->items.begin(); it != pimpl->items.end();)
            {
                if (it->first->deviceId() == dev)
                {
                    Impl::current_size_inbytes[dev] -= it->second->GetSizeInBytes();
                    savedItems.insert(pimpl->items.extract(it++));
                }
                else
                {
                    ++it;
                }
            }
        }
        Impl::cache_limit_inbytes[dev] = new_cache_limit_inbytes;
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

void Cache::destroyAnchorCapsule(PyObject *capsule)
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
    Items items;
    {
        std::lock_guard lk(Impl::mtx);
        for (const auto &[nodeKey, node] : pimpl->items)
        {
            Impl::current_size_inbytes[nodeKey->deviceId()] -= node->GetSizeInBytes();
        }
        items.swap(pimpl->items);
    }
    // Destroyed outside the lock, like every other bulk release here.
}

bool Cache::anchorToPythonThreadState()
{
    if (!pythonRuntimeAvailable() || !PyGILState_Check())
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
    PyObject *capsule = PyCapsule_New(owned, kAnchorCapsuleName, &Cache::destroyAnchorCapsule);
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
    return cache;
}

void Cache::ClearAll()
{
    Items savedItems;
    {
        std::lock_guard lk(Cache::Impl::mtx);
        std::for_each(instances().begin(), instances().end(),
                      [&savedItems](Cache *instance) { savedItems.merge(instance->pimpl->items); });
        Cache::Impl::current_size_inbytes.clear();
    }
    // savedItems is destroyed by leaving scope, outside the lock.
}

size_t Cache::TotalSize()
{
    std::lock_guard lk(Cache::Impl::mtx);
    return std::accumulate(instances().cbegin(), instances().cend(), static_cast<size_t>(0),
                           [](size_t sum, const Cache *instance) { return sum + instance->size(); });
}

void Cache::Export(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<CacheItem, std::shared_ptr<CacheItem>> cacheItem(nullptr, "CacheItem", py::module_local());

    py::class_<ExternalCacheItem, CacheItem, std::shared_ptr<ExternalCacheItem>> externalCacheItem(
        nullptr, "ExternalCacheItem", py::module_local());
    (void)cacheItem;
    (void)externalCacheItem;

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
            // ResourceGuard releases completed holds through auxiliary-stream callbacks.
            // Drain them so clearing the cache also releases their resources.
            Stream::SynchronizeAndClearGCBag();
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
        "Sets the cache limit [in bytes] for the current CUDA device.");

    m.def(
        "current_cache_size_inbytes", [] { return Cache::Instance().getCurrentSizeInBytes(); },
        "Returns the current cache size [in bytes] for the current CUDA device.");

    py::module_ internal = m.attr(INTERNAL_SUBMODULE_NAME);
    internal.def("nbytes_in_cache", [](const CacheItem &item) { return item.GetSizeInBytes(); });

    // Just to check if fetchAll compiles, it's harmless
    Cache::Instance().fetchAll<Cache>();
}

void Cache::ExportTestHooks(py::module &m)
{
    m.def("arm_cache_tls_destructor", [] { cacheDestructionTestHook().arm(); });
    m.def(
        "wait_cache_tls_destructor", [] { cacheDestructionTestHook().waitUntilEntered(); },
        py::call_guard<py::gil_scoped_release>());
    m.def("release_and_wait_cache_tls_destructor", [] { cacheDestructionTestHook().releaseAndWaitUntilCompleted(); });
    m.def("cache_anchors_installed", [] { return anchorsInstalled().load(); });
}

} // namespace nvcvpy::priv
