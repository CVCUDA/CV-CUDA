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

#include "Stream.hpp"

#include "../NvtxRange.hpp"
#include "Cache.hpp"
#include "Definitions.hpp"
#include "Resource.hpp"
#include "StreamStack.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#ifdef NB_FREE_THREADED
#    include <cuda.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <list>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace nvcvpy::priv {
namespace {

class StreamError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

uint64_t &ThreadIdentitySlot() noexcept
{
    static thread_local uint64_t identity = 0;
    return identity;
}

uint64_t AllocateThreadIdentity()
{
    static std::atomic_uint64_t next{1};
    uint64_t                    value = next.load(std::memory_order_relaxed);
    for (;;)
    {
        if (value == 0 || value == std::numeric_limits<uint64_t>::max())
            throw StreamError("CUDA stream thread identity space exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed, std::memory_order_relaxed))
            return value;
    }
}

uint64_t AssignCurrentThreadIdentity()
{
    uint64_t &identity = ThreadIdentitySlot();
    if (identity == 0)
    {
        identity = AllocateThreadIdentity();
    }
    return identity;
}

uint64_t CurrentThreadIdentity() noexcept
{
    return ThreadIdentitySlot();
}

} // namespace

StreamHandleOwner::StreamHandleOwner(cudaStream_t handle, int device, bool owns, py::object owner,
                                     bool externalOwnerRetainsLifetime)
    : m_handle(handle)
    , m_device(device)
    , m_owns(owns)
    , m_owner(std::move(owner))
    , m_externalOwnerRetainsLifetime(externalOwnerRetainsLifetime)
    , m_creatorThreadIdentity(handle == cudaStreamPerThread ? AssignCurrentThreadIdentity() : 0)
{
#ifdef NB_FREE_THREADED
    if (handle == cudaStreamPerThread)
    {
        util::CheckThrow(cudaEventCreateWithFlags(&m_perThreadCompletion, cudaEventDisableTiming));
        auto status = capturePerThreadCompletion();
        if (status != cudaSuccess)
        {
            util::CheckLog(cudaEventDestroy(m_perThreadCompletion));
            util::CheckThrow(status);
        }
    }
#endif
}

bool StreamHandleOwner::sharesQueueWith(const StreamHandleOwner &other) const noexcept
{
    return m_handle == other.m_handle
        && (m_handle != cudaStreamPerThread || m_creatorThreadIdentity == other.m_creatorThreadIdentity);
}

bool StreamHandleOwner::isCurrentThreadQueue() const noexcept
{
    return m_handle != cudaStreamPerThread || m_creatorThreadIdentity == CurrentThreadIdentity();
}

StreamHandleOwner::~StreamHandleOwner()
{
    if (const bool ownsHandle = m_owns && m_handle != nullptr; !ownsHandle
#ifdef NB_FREE_THREADED
                                                               && !m_perThreadCompletion
#endif
    )
    {
        return;
    }

    int         savedDevice = -1;
    cudaError_t status      = cudaGetDevice(&savedDevice);
    if (status == cudaSuccess)
    {
        status = cudaSetDevice(m_device);
    }
    if (status != cudaSuccess)
    {
        // Leaking the stream is the safe failure here: without its device
        // current, cudaStreamDestroy would not reach it. Report it and clear
        // the sticky error, so it cannot surface out of an unrelated CUDA call
        // made later by whoever happened to drop the last reference.
        util::CheckLog(status);
        (void)cudaGetLastError();
        return;
    }

    // CUDA keeps queued work alive after this asynchronous destroy.
#ifdef NB_FREE_THREADED
    if (m_perThreadCompletion)
        util::CheckLog(cudaEventDestroy(m_perThreadCompletion));
#endif
    if (m_owns && m_handle)
        util::CheckLog(cudaStreamDestroy(m_handle));
    util::CheckLog(cudaSetDevice(savedDevice));
}

namespace {

uint64_t AllocateStreamObjectId()
{
    static std::atomic_uint64_t next{0};
    return next.fetch_add(1, std::memory_order_relaxed);
}

class ReleaseEvent
{
public:
    explicit ReleaseEvent(int device)
        : m_device(device)
    {
        util::CheckThrow(cudaEventCreateWithFlags(&m_handle, cudaEventDisableTiming));
    }

    ~ReleaseEvent()
    {
        reset();
    }

    ReleaseEvent(const ReleaseEvent &)            = delete;
    ReleaseEvent &operator=(const ReleaseEvent &) = delete;

    cudaEvent_t handle() const
    {
        return m_handle;
    }

    int device() const
    {
        return m_device;
    }

private:
    void reset() noexcept
    {
        if (m_handle == nullptr)
        {
            return;
        }

        int         savedDevice = -1;
        cudaError_t status      = cudaGetDevice(&savedDevice);
        if (status == cudaSuccess)
        {
            status = cudaSetDevice(m_device);
        }
        if (status == cudaSuccess)
        {
            util::CheckLog(cudaEventDestroy(m_handle));
            util::CheckLog(cudaSetDevice(savedDevice));
        }
        else
        {
            // Same trade as ~StreamHandleOwner: leak the event rather than
            // leave a sticky error for an unrelated later call to trip on.
            util::CheckLog(status);
            (void)cudaGetLastError();
        }
        m_handle = nullptr;
    }

    cudaEvent_t m_handle = nullptr;
    int         m_device = -1;
};

std::atomic_bool &FailNextReleaseEventRecord()
{
    static std::atomic_bool fail{false};
    return fail;
}

void RecordReleaseEvent(cudaEvent_t event, cudaStream_t stream)
{
    // Exercise publication rollback without poisoning the CUDA context.
    if (std::atomic_bool &fail = FailNextReleaseEventRecord();
        fail.load(std::memory_order_relaxed) && fail.exchange(false, std::memory_order_relaxed))
    {
        throw std::invalid_argument("injected pending-release event record failure");
    }
    util::CheckThrow(cudaEventRecord(event, stream));
}

struct PendingRelease
{
    explicit PendingRelease(int device)
        : event(device)
    {
    }

    void clearNativeOwners()
    {
        for (const std::shared_ptr<Resource> &resource : resources)
        {
            resource->releasePendingReleaseOwner(ownersBlocked);
        }
        resources.clear();
        resourceSet.clear();
        stream.reset();
        lifetimeOwner.reset();
        ownersBlocked = false;
    }

    bool hasPythonTeardown() const noexcept
    {
        return !pythonOwners.empty() || (stream && stream->handleOwner()->hasPythonOwner())
            || (lifetimeOwner && lifetimeOwner->hasPythonOwner());
    }

    void blockNativeOwners()
    {
        if (ownersBlocked)
        {
            return;
        }
        for (const std::shared_ptr<Resource> &resource : resources)
        {
            resource->blockPendingReleaseOwner();
        }
        ownersBlocked = true;
    }

    bool sharesQueueWith(Stream &candidate, int device) const
    {
        return stream && event.device() == device && !ownersBlocked && stream->sharesQueueWith(candidate);
    }

    ReleaseEvent                             event;
    std::shared_ptr<const Stream>            stream;
    std::shared_ptr<const StreamHandleOwner> lifetimeOwner;
    ResourceBag                              resources;
    std::unordered_set<Resource *>           resourceSet;
    PythonOwnerBag                           pythonOwners;
    std::unordered_set<PyObject *>           pythonOwnerSet;
    bool                                     ownersBlocked = false;
};

using ReleaseList = std::list<PendingRelease>;

struct PendingReleaseState
{
    std::mutex           mutex;
    std::recursive_mutex drainMutex;
    ReleaseList          pending;
    ReleaseList          idle;
    ReleaseList          quarantine;
    ReleaseList          awaitingGilCleanup;
    std::atomic_bool     hasDeferredPythonOwners{false};
};

PendingReleaseState &PendingReleases()
{
    // CUDA and Python may already be shutting down when static destructors
    // run. Cleanup drains this state explicitly; the shell stays immortal.
    static auto *state = new PendingReleaseState; // NOSONAR: deliberately immortal
    return *state;
}

void DrainPendingReleases();
void ReleaseDeferredPythonOwners();

void RecycleCompleted(ReleaseList completed)
{
    if (completed.empty())
        return;
    ReleaseList awaitingGil;
    const bool  canReleasePython = util::HasPythonThreadState();
    for (auto it = completed.begin(); it != completed.end();)
    {
        auto current = it++;
        if (current->hasPythonTeardown() && !canReleasePython)
        {
            // Native submission may retire a legacy batch without the GIL.
            // Keep the whole ownership set in the same global release state
            // until a Python entry point can safely drop its wrappers.
            current->blockNativeOwners();
            awaitingGil.splice(awaitingGil.end(), completed, current);
            continue;
        }
        current->clearNativeOwners();
        current->pythonOwners.clear();
        current->pythonOwnerSet.clear();
    }

    constexpr size_t     kMaxIdleEvents = 64;
    PendingReleaseState &state          = PendingReleases();
    {
        std::lock_guard lock(state.mutex);
        if (!awaitingGil.empty())
        {
            state.awaitingGilCleanup.splice(state.awaitingGilCleanup.end(), awaitingGil);
            state.hasDeferredPythonOwners.store(true, std::memory_order_release);
        }
        while (!completed.empty() && state.idle.size() < kMaxIdleEvents)
        {
            state.idle.splice(state.idle.end(), completed, completed.begin());
        }
    }
    try
    {
        Cache::PruneToLimits();
    }
    catch (...) // NOSONAR: ownership may already be committed; pruning records its own retry request.
    {
        std::fputs("[cvcuda] Cache pruning failed after release retirement; deferring budget cleanup\n", stderr);
    }
}

void ReleaseDeferredPythonOwners()
{
    PendingReleaseState &state = PendingReleases();
    if (!state.hasDeferredPythonOwners.load(std::memory_order_acquire))
    {
        return;
    }
    if (!util::HasPythonThreadState())
    {
        return;
    }

    ReleaseList completed;
    {
        std::lock_guard lock(state.mutex);
        completed.splice(completed.end(), state.awaitingGilCleanup);
        state.hasDeferredPythonOwners.store(false, std::memory_order_release);
    }

    for (PendingRelease &batch : completed)
    {
        batch.pythonOwners.clear();
        batch.pythonOwnerSet.clear();
    }
    RecycleCompleted(std::move(completed));
}

// Moves every batch the GPU has provably finished with out of `pending`.
// There are at most two native batches per active underlying stream, so
// scanning past a busy stream is bounded by stream count and lets unrelated
// streams retire promptly. The older batch remains a stable completion
// checkpoint while the newer batch advances to the stream tail. When the
// checkpoint completes, the next submission rotates in a fresh generation.
//
// `device` selects which batches are eligible. A batch of another device does
// not stop the sweep, because per-device issue order is what the reasoning
// above rests on.
ReleaseList CollectReadyReleases(int device)
{
    ReleaseDeferredPythonOwners();
    PendingReleaseState &state = PendingReleases();
    ReleaseList          completed;
    cudaError_t          queryError = cudaSuccess;

    {
        std::lock_guard lock(state.mutex);
        for (auto it = state.pending.begin(); it != state.pending.end();)
        {
            auto current = it++;
            if (current->event.device() != device)
            {
                continue;
            }

            cudaError_t status = cudaEventQuery(current->event.handle());
            if (status == cudaErrorNotReady)
            {
                continue;
            }
            if (status == cudaSuccess)
            {
                completed.splice(completed.end(), state.pending, current);
            }
            else
            {
                queryError = status;
                current->blockNativeOwners();
                state.quarantine.splice(state.quarantine.end(), state.pending, current);
                (void)cudaGetLastError();
            }
        }
    }

    if (queryError != cudaSuccess)
    {
        std::fprintf(stderr, "[cvcuda] CUDA event query failed (%s); quarantining pending resources\n",
                     cudaGetErrorName(queryError));
    }
    return completed;
}

using ResourceSourceBag    = std::vector<std::shared_ptr<Resource> *>;
using PythonOwnerSourceBag = std::vector<py::object *>;

bool HasSameOwners(const PendingRelease &batch, const ResourceOwnerTransfer &resources,
                   const PythonOwnerBag &pythonOwners)
{
    if (batch.resources.size() != resources.inlineSize + resources.overflowSize
        || batch.pythonOwners.size() != pythonOwners.size())
        return false;
    for (size_t i = 0; i < resources.inlineSize; ++i)
        if (batch.resources[i] != resources.inlineData[i])
            return false;
    for (size_t i = 0; i < resources.overflowSize; ++i)
        if (batch.resources[resources.inlineSize + i] != resources.overflowData[i])
            return false;
    for (size_t i = 0; i < pythonOwners.size(); ++i)
        if (batch.pythonOwners[i].ptr() != pythonOwners[i].ptr())
            return false;
    return true;
}

bool HasNewOwner(const PendingRelease &batch, const ResourceOwnerTransfer &resources,
                 const PythonOwnerBag &pythonOwners)
{
    for (size_t i = 0; i < resources.inlineSize; ++i)
    {
        if (batch.resourceSet.find(resources.inlineData[i].get()) == batch.resourceSet.end())
            return true;
    }
    for (size_t i = 0; i < resources.overflowSize; ++i)
    {
        if (batch.resourceSet.find(resources.overflowData[i].get()) == batch.resourceSet.end())
            return true;
    }
    return std::any_of(pythonOwners.begin(), pythonOwners.end(),
                       [&batch](const py::object &owner)
                       { return batch.pythonOwnerSet.find(owner.ptr()) == batch.pythonOwnerSet.end(); });
}

template<class T>
void ReserveForAppend(std::vector<T> &items, size_t count)
{
    if (count > items.max_size() - items.size())
        throw std::length_error("pending release batch is too large");

    const size_t required = items.size() + count;
    if (required > items.capacity())
    {
        const size_t growth = std::min(items.capacity(), items.max_size() - items.capacity());
        items.reserve(std::max(required, items.capacity() + growth));
    }
}

void PublishStreamOwner(PendingRelease &batch, Stream &stream, std::shared_ptr<Stream> &streamOwner)
{
    // Any same-queue owner suffices. Replacing one could run Python teardown
    // off-GIL during publication; retirement already handles that boundary.
    if (!batch.lifetimeOwner && stream.handleOwner()->retainsHandleLifetime())
    {
        batch.lifetimeOwner = stream.handleOwner();
    }
    if (!batch.stream)
    {
        batch.stream = std::move(streamOwner);
    }
}

void StageResource(PendingRelease &batch, std::shared_ptr<Resource> &resource, ResourceSourceBag &newResources)
{
    if (batch.resourceSet.insert(resource.get()).second)
    {
        newResources.push_back(&resource);
    }
}

void RollbackStagedOwners(PendingRelease &batch, const ResourceSourceBag &newResources,
                          const PythonOwnerSourceBag &newPythonOwners, bool pendingOwnersAcquired)
{
    if (pendingOwnersAcquired)
    {
        for (const std::shared_ptr<Resource> *resource : newResources) (*resource)->releasePendingReleaseOwner(false);
    }
    for (const std::shared_ptr<Resource> *resource : newResources)
    {
        batch.resourceSet.erase(resource->get());
    }
    for (const py::object *owner : newPythonOwners)
    {
        batch.pythonOwnerSet.erase(owner->ptr());
    }
}

void AppendToBatch(PendingRelease &batch, Stream &stream, std::shared_ptr<Stream> &streamOwner,
                   const ResourceOwnerTransfer &resources, PythonOwnerBag &pythonOwners)
{
    if (!HasNewOwner(batch, resources, pythonOwners))
    {
        RecordReleaseEvent(batch.event.handle(), stream.handle());
        PublishStreamOwner(batch, stream, streamOwner);
        return;
    }

    const size_t resourceCount = resources.inlineSize + resources.overflowSize;
    ReserveForAppend(batch.resources, resourceCount);
    batch.resourceSet.reserve(batch.resourceSet.size() + resourceCount);
    ReserveForAppend(batch.pythonOwners, pythonOwners.size());
    batch.pythonOwnerSet.reserve(batch.pythonOwnerSet.size() + pythonOwners.size());

    ResourceSourceBag    newResources;
    PythonOwnerSourceBag newPythonOwners;
    newResources.reserve(resourceCount);
    newPythonOwners.reserve(pythonOwners.size());

    bool pendingOwnersAcquired = false;
    try
    {
        for (size_t i = 0; i < resources.inlineSize; ++i)
        {
            StageResource(batch, resources.inlineData[i], newResources);
        }
        for (size_t i = 0; i < resources.overflowSize; ++i)
        {
            StageResource(batch, resources.overflowData[i], newResources);
        }
        for (py::object &owner : pythonOwners)
        {
            if (batch.pythonOwnerSet.insert(owner.ptr()).second)
            {
                newPythonOwners.push_back(&owner);
            }
        }
        for (const std::shared_ptr<Resource> *resource : newResources) (*resource)->acquirePendingReleaseOwner();
        pendingOwnersAcquired = true;
        RecordReleaseEvent(batch.event.handle(), stream.handle());
    }
    catch (...) // NOSONAR: every staged owner must be rolled back before propagating any failure.
    {
        RollbackStagedOwners(batch, newResources, newPythonOwners, pendingOwnersAcquired);
        throw;
    }

    PublishStreamOwner(batch, stream, streamOwner);
    for (std::shared_ptr<Resource> *resource : newResources)
    {
        (*resource)->indexPendingResources();
        batch.resources.emplace_back(std::move(*resource));
    }
    for (py::object *owner : newPythonOwners)
    {
        batch.pythonOwners.emplace_back(std::move(*owner));
    }
}

PendingRelease *NewestBatchFor(Stream &stream, int device, ReleaseList &pending, size_t &count)
{
    PendingRelease *newest = nullptr;
    count                  = 0;
    for (PendingRelease &candidate : pending)
    {
        if (candidate.sharesQueueWith(stream, device))
        {
            newest = &candidate;
            ++count;
        }
    }
    return newest;
}

void InheritLifetimeOwner(PendingRelease &target, const PendingRelease *newest,
                          const std::shared_ptr<const StreamHandleOwner> &completedOwner)
{
    if (!target.lifetimeOwner)
        target.lifetimeOwner = newest && newest->lifetimeOwner ? newest->lifetimeOwner : completedOwner;
}

void AppendRelease(Stream &stream, std::shared_ptr<Stream> &streamOwner, const ResourceOwnerTransfer &resources,
                   PythonOwnerBag &pythonOwners, int device, ReleaseList &completed)
{
    PendingReleaseState                     &state = PendingReleases();
    std::shared_ptr<const StreamHandleOwner> completedOwner;
    if (!stream.handleOwner()->retainsHandleLifetime())
    {
        // A raw alias may have queued new work while a completed record still
        // carries the sole owner of its handle. Transfer that anchor too.
        for (auto it = completed.rbegin(); it != completed.rend(); ++it)
            if (it->lifetimeOwner && it->sharesQueueWith(stream, device))
            {
                completedOwner = it->lifetimeOwner;
                break;
            }
    }

    {
        std::lock_guard lock(state.mutex);
        size_t          generationCount;
        PendingRelease *newest = NewestBatchFor(stream, device, state.pending, generationCount);
        if (generationCount >= 2)
        {
            InheritLifetimeOwner(*newest, newest, completedOwner);
            AppendToBatch(*newest, stream, streamOwner, resources, pythonOwners);
            return;
        }
        auto reusable = std::find_if(completed.begin(), completed.end(),
                                     [&](const PendingRelease &batch)
                                     {
                                         return batch.sharesQueueWith(stream, device)
                                             && (!batch.hasPythonTeardown() || util::HasPythonThreadState())
                                             && HasSameOwners(batch, resources, pythonOwners);
                                     });
        if (reusable != completed.end())
        {
            // Exact equality, not containment: extending a completed superset
            // would keep unused allocations charged across later submissions.
            InheritLifetimeOwner(*reusable, newest, completedOwner);
            AppendToBatch(*reusable, stream, streamOwner, resources, pythonOwners);
            state.pending.splice(state.pending.end(), completed, reusable);
            return;
        }
    }

    ReleaseList incoming;
    {
        std::lock_guard lock(state.mutex);
        auto            idle = std::find_if(state.idle.begin(), state.idle.end(),
                                            [device](const PendingRelease &batch) { return batch.event.device() == device; });
        if (idle != state.idle.end())
        {
            incoming.splice(incoming.end(), state.idle, idle);
        }
    }
    if (incoming.empty())
    {
        incoming.emplace_back(device);
    }

    {
        std::lock_guard lock(state.mutex);
        size_t          generationCount;
        PendingRelease *newest = NewestBatchFor(stream, device, state.pending, generationCount);
        if (generationCount >= 2)
        {
            InheritLifetimeOwner(*newest, newest, completedOwner);
            AppendToBatch(*newest, stream, streamOwner, resources, pythonOwners);
            return;
        }

        InheritLifetimeOwner(incoming.front(), newest, completedOwner);
        AppendToBatch(incoming.front(), stream, streamOwner, resources, pythonOwners);
        state.pending.splice(state.pending.end(), incoming);
    }
}

void EnqueueRelease(Stream &stream, std::shared_ptr<Stream> &streamOwner, const ResourceOwnerTransfer &resources,
                    PythonOwnerBag &pythonOwners, int submissionDevice)
{
    if (resources.inlineSize + resources.overflowSize == 0 && pythonOwners.empty())
        return;

    // Keep completed owners through publication so an unchanged batch snapshot
    // does not deactivate/reactivate every time the GPU outruns the host.
    // Recycle unused generations outside queue locks; Python teardown may defer.
    auto completed = CollectReadyReleases(submissionDevice);
    try
    {
        AppendRelease(stream, streamOwner, resources, pythonOwners, submissionDevice, completed);
    }
    catch (...) // NOSONAR: completed ownership must retire on every publication failure.
    {
        try
        {
            RecycleCompleted(std::move(completed));
        }
        catch (...) // NOSONAR: pruning must not replace the original publication failure.
        {
            std::fputs("[cvcuda] Cache pruning failed while unwinding pending release publication\n", stderr);
        }
        throw;
    }
    RecycleCompleted(std::move(completed));
}

template<class AfterLock>
void DrainPendingReleasesImpl(AfterLock &&afterLock)
{
    ReleaseDeferredPythonOwners();
    PendingReleaseState &state = PendingReleases();
    std::unique_lock     drainLock(state.drainMutex, std::try_to_lock);
    if (!drainLock.owns_lock())
    {
        // Another drainer may be waiting on a callback that needs the GIL.
        py::gil_scoped_release release;
        drainLock.lock();
    }
    afterLock();

    ReleaseList waiting;
    ReleaseList completed;
    ReleaseList failed;
    cudaError_t firstError = cudaSuccess;

    for (;;)
    {
        {
            std::lock_guard lock(state.mutex);
            waiting.splice(waiting.end(), state.pending);
            for (PendingRelease &batch : waiting)
            {
                batch.blockNativeOwners();
            }
        }
        if (waiting.empty())
        {
            break;
        }

        {
            // A callback ahead of our event may need the GIL to complete.
            py::gil_scoped_release release;
            for (auto it = waiting.begin(); it != waiting.end();)
            {
                auto        current = it++;
                cudaError_t status  = cudaEventSynchronize(current->event.handle());
                if (status == cudaSuccess)
                {
                    completed.splice(completed.end(), waiting, current);
                    continue;
                }

                if (firstError == cudaSuccess)
                {
                    firstError = status;
                }
                failed.splice(failed.end(), waiting, current);
                (void)cudaGetLastError();
            }
        }
    }

    if (!failed.empty())
    {
        std::lock_guard lock(state.mutex);
        state.quarantine.splice(state.quarantine.end(), failed);
    }
    RecycleCompleted(std::move(completed));
    drainLock.unlock();
    if (firstError != cudaSuccess)
    {
        util::CheckThrow(firstError);
    }
}

void DrainPendingReleases()
{
    DrainPendingReleasesImpl([] { /* production drains need no lock-acquired hook */ });
}

void DestroyIdleReleaseEvents()
{
    ReleaseDeferredPythonOwners();
    PendingReleaseState &state = PendingReleases();
    ReleaseList          idle;
    {
        std::lock_guard lock(state.mutex);
        idle.splice(idle.end(), state.idle);
    }
}

} // namespace

// Static members initialization
std::weak_ptr<Stream> Stream::m_defaultStream;

// Here we define the representation of external CUDA streams and nanobind's
// type casters from Python objects.
// to the corresponding ExternalStream<E>.

// Defines each external stream represetation we support.
enum ExternalStreamType
{
    VOIDP,
    INT,
    TORCH,
    CUPY,
};

template<ExternalStreamType E>
class ExternalStream : public IExternalStream
{
public:
    ExternalStream() = default;

    explicit ExternalStream(cudaStream_t cudaStream)
        : m_cudaStream(cudaStream)
    {
    }

    void setCudaStream(cudaStream_t cudaStream, py::object obj, bool retainsHandleLifetime = false)
    {
        m_cudaStream            = cudaStream;
        m_wrappedObj            = std::move(obj);
        m_retainsHandleLifetime = retainsHandleLifetime;
    }

    cudaStream_t handle() const override
    {
        return m_cudaStream;
    }

    py::object wrappedObject() const override
    {
        return m_wrappedObj;
    }

    bool retainsHandleLifetime() const override
    {
        return m_retainsHandleLifetime;
    }

private:
    cudaStream_t m_cudaStream;
    py::object   m_wrappedObj;
    bool         m_retainsHandleLifetime = false;
};

} // namespace nvcvpy::priv

namespace nanobind::detail {

using namespace std::literals;
namespace util = nvcvpy::util;
namespace priv = nvcvpy::priv;

template<>
struct type_caster<priv::ExternalStream<priv::VOIDP>>
{
    NB_TYPE_CASTER(priv::ExternalStream<priv::VOIDP>, const_name("ctypes.c_void_p"));

    bool from_python(handle src, uint32_t, cleanup_list *) noexcept
    {
        if (std::string strType = util::GetFullyQualifiedName(src); strType != "ctypes.c_void_p")
        {
            return false;
        }

        nanobind::buffer_info info = nanobind::borrow<nanobind::buffer>(src).request();

        NVCV_ASSERT(info.itemsize == sizeof(void *));

        void *data = *reinterpret_cast<void **>(info.ptr);

        value.setCudaStream(reinterpret_cast<cudaStream_t>(data), nanobind::borrow<nanobind::object>(src));
        return true;
    }
};

template<>
struct type_caster<priv::ExternalStream<priv::INT>>
{
    NB_TYPE_CASTER(priv::ExternalStream<priv::INT>, const_name("int"));

    bool from_python(handle src, uint32_t, cleanup_list *) noexcept
    {
        try
        {
            // REVISIT: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = nanobind::cast<intptr_t>(src);
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), nanobind::borrow<nanobind::object>(src));
            return true;
        }
        catch (...) // NOSONAR: binding type casters probe conversions by catching failures.
        {
            return false;
        }
    }
};

template<>
struct type_caster<priv::ExternalStream<priv::TORCH>>
{
    NB_TYPE_CASTER(priv::ExternalStream<priv::TORCH>, const_name("torch.cuda.Stream"));

    bool from_python(handle src, uint32_t, cleanup_list *) noexcept
    {
        std::string strType = util::GetFullyQualifiedName(src);
        if (strType != "torch.cuda.streams.Stream" && strType != "torch.cuda.streams.ExternalStream")
        {
            return false;
        }

        try
        {
            // REVISIT: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = nanobind::cast<intptr_t>(src.attr("cuda_stream"));
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), nanobind::borrow<nanobind::object>(src),
                                strType == "torch.cuda.streams.Stream");
            return true;
        }
        catch (...) // NOSONAR: binding type casters probe conversions by catching failures.
        {
            return false;
        }
    }
};

template<>
struct type_caster<priv::ExternalStream<priv::CUPY>>
{
    NB_TYPE_CASTER(priv::ExternalStream<priv::CUPY>, const_name("cupy.cuda.Stream"));

    bool from_python(handle src, uint32_t, cleanup_list *) noexcept
    {
        std::string strType = util::GetFullyQualifiedName(src);
        if (strType != "cupy.cuda.stream.Stream" && strType != "cupy.cuda.stream.ExternalStream")
        {
            return false;
        }

        try
        {
            intptr_t data = nanobind::cast<intptr_t>(src.attr("ptr"));
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), nanobind::borrow<nanobind::object>(src),
                                strType == "cupy.cuda.stream.Stream");
            return true;
        }
        catch (...) // NOSONAR: binding type casters probe conversions by catching failures.
        {
            return false;
        }
    }
};

} // namespace nanobind::detail

namespace nvcvpy::priv {

// In terms of caching, all streams are the same.
// Any stream in the cache can be fetched and used.
size_t Stream::Key::doGetHash() const
{
    return 0;
}

bool Stream::Key::doIsCompatible(const IKey &that) const
{
    return true;
}

std::shared_ptr<Stream> Stream::Create()
{
    ::cvcudapy::NvtxRange                   nvtxRange("cvcuda.Stream.create");
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Stream::Key{});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Stream> stream(new Stream()); // NOSONAR: constructor is private.
        Cache::Instance().add(*stream);
        return stream;
    }
    else
    {
        // Get the first one
        return std::static_pointer_cast<Stream>(vcont[0]->cachedObject());
    }
}

Stream::Stream()
    : m_id(AllocateStreamObjectId())
    , m_owns(true)
    , m_size_inbytes(doComputeSizeInBytes())
{
    util::CheckThrow(cudaStreamCreateWithFlags(&m_handle, cudaStreamNonBlocking));
    try
    {
        m_handleOwner = std::make_shared<StreamHandleOwner>(m_handle, m_key.deviceId(), true);
    }
    catch (...)
    {
        util::CheckLog(cudaStreamDestroy(m_handle));
        throw;
    }
}

Stream::Stream(IExternalStream &extStream)
    : m_id(AllocateStreamObjectId())
    , m_handle(extStream.handle())
    , m_handleOwner(std::make_shared<StreamHandleOwner>(m_handle, m_key.deviceId(), false, extStream.wrappedObject(),
                                                        extStream.retainsHandleLifetime()))
{
    if (unsigned int flags; cudaStreamGetFlags(m_handle, &flags) != cudaSuccess)
    {
        throw StreamError("Invalid cuda stream");
    }
}

Stream::~Stream()
{
    destroy();
}

uint64_t Stream::id() const
{
    return m_id;
}

void Stream::destroy()
{
    {
        std::lock_guard lock(m_eventMutex);
        int             savedDev = 0;
        util::CheckLog(cudaGetDevice(&savedDev));
        for (const auto &[dev, evt] : m_events)
        {
            util::CheckLog(cudaSetDevice(dev));
            util::CheckLog(cudaEventDestroy(evt));
        }
        m_events.clear();
        util::CheckLog(cudaSetDevice(savedDev));
    }
}

int64_t Stream::doComputeSizeInBytes() const
{
    // We only cache the stream's handles, which are 8 byte on CPU memory, hence 0 bytes gpu memory.
    return 0;
}

int64_t Stream::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1
                && "Stream has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
}

std::shared_ptr<Stream> Stream::sharedStream()
{
    return std::static_pointer_cast<Stream>(Object::shared_from_this());
}

std::shared_ptr<const Stream> Stream::sharedStream() const
{
    return std::static_pointer_cast<const Stream>(Object::shared_from_this());
}

cudaStream_t Stream::handle() const
{
    return m_handle;
}

int Stream::deviceId() const
{
    return m_key.deviceId();
}

int Stream::submissionDeviceId() const
{
    validateCurrentThread();
    if (m_handle != nullptr && m_handle != cudaStreamLegacy && m_handle != cudaStreamPerThread)
    {
        return m_key.deviceId();
    }

    int currentDevice = 0;
    util::CheckThrow(cudaGetDevice(&currentDevice));
    return currentDevice;
}

bool Stream::sharesQueueWith(const Stream &other) const
{
    return m_handleOwner->sharesQueueWith(*other.m_handleOwner);
}

void Stream::validateCurrentThread() const
{
    if (!m_handleOwner->isCurrentThreadQueue())
    {
        throw StreamError("A cudaStreamPerThread wrapper may only be used on the host thread that created it");
    }
}

const std::shared_ptr<StreamHandleOwner> &Stream::handleOwner() const
{
    return m_handleOwner;
}

intptr_t Stream::pyhandle() const
{
    return reinterpret_cast<intptr_t>(m_handle);
}

void Stream::sync()
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.sync");
    int                   submissionDevice = submissionDeviceId();
    {
        py::gil_scoped_release release;
        util::CheckThrow(cudaStreamSynchronize(m_handle));
    }

    RecycleCompleted(CollectReadyReleases(submissionDevice));
}

void Stream::wait_stream(std::shared_ptr<Stream> other)
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.wait_stream");
    if (!other)
        throw std::invalid_argument("other is null");

    validateCurrentThread();
    other->validateCurrentThread();

    if (other->handle() == m_handle)
        return;

    py::gil_scoped_release release;
    waitFor(other->handle());
}

Stream &Stream::Current()
{
    auto currentStream = StreamStack::Instance().top();
    if (!currentStream)
    {
        currentStream = m_defaultStream.lock();
    }
    if (!currentStream)
    {
        // An unavailable default stream means Stream::Export skipped wrapping
        // the legacy default stream — only happens on no-device hosts. Throw
        // a Python-visible error rather than NVCV_ASSERT (which would abort()).
        throw StreamError("No default cvcuda.Stream available (no CUDA device visible).");
    }
    return *currentStream;
}

void Stream::activate()
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.__enter__");
    validateCurrentThread();
    if (m_owns)
    {
        int curDev = 0;
        util::CheckThrow(cudaGetDevice(&curDev));
        if (m_key.deviceId() != curDev)
        {
            throw StreamError("Cannot activate cvcuda.Stream (created on device " + std::to_string(m_key.deviceId())
                              + ") while current CUDA device is " + std::to_string(curDev));
        }
    }
    StreamStack::Instance().push(*this);
}

void Stream::deactivate(py::handle exc_type, py::handle, py::handle) const
{
    ::cvcudapy::NvtxRange nvtxRange("cvcuda.Stream.__exit__");
    validateCurrentThread();
    StreamStack &streamStack = StreamStack::Instance();
    if (streamStack.top().get() != this)
    {
        // Raising while the with-body is already unwinding would make this the
        // visible failure and demote the caller's to __context__, so the bug
        // they need to see gets buried behind a report of the mess it made.
        if (!exc_type.is_none())
        {
            return;
        }
        throw StreamError("Cannot deactivate cvcuda.Stream because it is not the current stream on this thread.");
    }
    streamStack.pop();
}

void Stream::waitFor(cudaStream_t producer)
{
    validateCurrentThread();
    int dev;
    util::CheckThrow(cudaGetDevice(&dev));

    // A wait captures the event's current record, independently of later
    // records. Serialize this pair so concurrent callers cannot replace it
    // before the wait is installed, then reuse the event without a host wait.
    std::lock_guard lock(m_eventMutex);
    auto [it, inserted] = m_events.try_emplace(dev, nullptr);
    if (inserted)
    {
        cudaEvent_t evt = nullptr;
        if (auto error = cudaEventCreateWithFlags(&evt, cudaEventDisableTiming); error != cudaSuccess)
        {
            m_events.erase(it);
            util::CheckThrow(error);
        }
        it->second = evt;
    }
    util::CheckThrow(cudaEventRecord(it->second, producer));
    util::CheckThrow(cudaStreamWaitEvent(m_handle, it->second, 0));
}

void Stream::holdResources(ResourceBag usedResources, int submissionDevice)
{
    validateCurrentThread();
    auto           streamOwner = sharedStream();
    PythonOwnerBag pythonOwners;
    EnqueueRelease(*this, streamOwner, {usedResources.data(), usedResources.size(), nullptr, 0}, pythonOwners,
                   submissionDevice);
}

void Stream::holdResources(ResourceBag usedResources, PythonOwnerBag pythonOwners, int submissionDevice)
{
    validateCurrentThread();
    auto streamOwner = sharedStream();
    EnqueueRelease(*this, streamOwner, {usedResources.data(), usedResources.size(), nullptr, 0}, pythonOwners,
                   submissionDevice);
}

void Stream::holdResources(const ResourceOwnerTransfer &usedResources, std::shared_ptr<Stream> &streamOwner,
                           int submissionDevice)
{
    validateCurrentThread();
    NVCV_ASSERT(streamOwner.get() == this);
    PythonOwnerBag pythonOwners;
    EnqueueRelease(*this, streamOwner, usedResources, pythonOwners, submissionDevice);
}

#ifdef NB_FREE_THREADED
void Stream::holdResources(const ResourceOwnerTransfer &resources, std::shared_ptr<Stream> &streamOwner,
                           PythonOwnerBag &pythonOwners, int submissionDevice)
{
    validateCurrentThread();
    EnqueueRelease(*this, streamOwner, resources, pythonOwners, submissionDevice);
}
#endif

void Stream::ensureNotCapturing(int submissionDevice) const
{
    validateCurrentThread();
    (void)submissionDevice;
#ifdef NB_FREE_THREADED
    CUstreamCaptureStatus driverStatus;
    if (cuStreamIsCapturing(reinterpret_cast<CUstream>(m_handle), &driverStatus) == CUDA_SUCCESS
        && driverStatus == CU_STREAM_CAPTURE_STATUS_NONE)
    {
        return;
    }
    // Preserve runtime diagnostics for captures and failed driver queries.
#endif
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    util::CheckThrow(cudaStreamIsCapturing(m_handle, &captureStatus));
    if (captureStatus != cudaStreamCaptureStatusNone)
    {
        throw StreamError("CV-CUDA Python operators are not supported during CUDA stream capture");
    }
}

void Stream::DrainAllPendingReleases()
{
    DrainPendingReleases();
}

bool Stream::IsPendingOnly(const std::shared_ptr<Object> &resourceOwner)
{
    auto *resource = dynamic_cast<Resource *>(resourceOwner.get());
    if (resource == nullptr || !util::HasPythonThreadState())
    {
        return false;
    }

    ReleaseDeferredPythonOwners();

    const std::optional<uint64_t> pendingOwners = resource->reusablePendingReleaseOwners();
    if (!pendingOwners.has_value())
    {
        return false;
    }

    // A registered wrapper is an observable user owner. Legacy queue entries
    // may themselves retain that wrapper, so conservatively exclude those as
    // well instead of depending on the wrapper's exact Python refcount.
    if (py::object pythonOwner = py::find(resource); pythonOwner.is_valid())
    {
        return false;
    }

    // The expected native owners are the cache's object, this function's
    // caller, and the explicitly registered pending queue entries. Anything
    // else is an owner that may still observe or mutate the allocation.
    const auto expectedOwners = static_cast<long>(2 + *pendingOwners);
    return resourceOwner.use_count() == expectedOwners && resource->pendingReleaseOwnershipMatches(*pendingOwners);
}

std::ostream &operator<<(std::ostream &out, const Stream &stream)
{
    return out << "<nvcv.cuda.Stream id=" << stream.id() << " handle=" << stream.handle() << '>';
}

template<ExternalStreamType E>
static void ExportExternalStream(py::module_ &m)
{
    m.def("as_stream",
          [](ExternalStream<E> extStream)
          {
              ::cvcudapy::NvtxRange nvtxRange("cvcuda.as_stream");
              return std::make_shared<Stream>(extStream);
          });
}

static void LogCleanupWarning(const std::exception &e)
{
    std::cerr << "Warning CVCUDA cleanup may be incomplete due to: " << e.what() << std::endl;
}

void Stream::CleanupAtExit(const std::shared_ptr<Stream> &globalStream)
{
    // No globalStream means no device at module-init time, no streams were ever
    // created, nothing to sync. Cleanup is a no-op.
    if (!globalStream)
    {
        return;
    }

    try
    {
        for (std::shared_ptr<Stream> stream : Cache::Instance().fetchAll<Stream>())
        {
            stream->sync();
        }
        globalStream->sync();

        // Nothing seeds the stack, so anything left is a context that was
        // entered and never exited. Draining it would release nothing -- the
        // entries are weak, and the thread-local storage goes away with the
        // thread -- so just report it. This runs on whichever thread finalizes
        // the interpreter, which need not be the one that imported the module.
        if (StreamStack::Instance().top())
        {
            std::cerr << "Stream stack leak detected on cleanup thread" << std::endl;
        }

        DrainPendingReleases();
        DestroyIdleReleaseEvents();
    }
    catch (const py::error_already_set &e)
    {
        LogCleanupWarning(e);
    }
    catch (const StreamError &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::invalid_argument &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::domain_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::length_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::out_of_range &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::range_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::overflow_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::underflow_error &e)
    {
        LogCleanupWarning(e);
    }
    catch (const std::bad_alloc &e)
    {
        LogCleanupWarning(e);
    }
}

void Stream::Export(py::module_ &m)
{
    py::class_<Stream> stream(m, "Stream",
                              R"pbdoc(
        A CUDA stream that cvcuda operators can be submitted to.

        Streams created via ``cvcuda.Stream()`` are non-blocking
        (``cudaStreamNonBlocking``) so they can run concurrently with CUDA's
        legacy default stream and with other non-blocking streams.

        Cross-library stream safety is handled automatically via the CUDA
        Array Interface (CAI) v3 ``stream`` field:

        * On **input**, ``cvcuda.as_tensor`` / ``cvcuda.as_image`` read
          ``__cuda_array_interface__["stream"]`` from the producer (cupy,
          torch, jax, etc.) and arrange for the first cvcuda operator to
          insert the proper ``cudaStreamWaitEvent`` before reading the data.
          No user-side ``synchronize()`` call is required.
        * On **output**, tensors returned by ``cvcuda.Tensor.cuda()`` /
          ``cvcuda.Image.cuda()`` populate ``__cuda_array_interface__
          ["stream"]`` with the cvcuda stream the data was last written on,
          so downstream consumers can sync themselves.

        Advanced usage:

        * **Opt out of the implicit wait**: export your buffer with
          ``stream: -1`` in its CAI dict to tell cvcuda you've already
          synchronized; the first op then runs without an extra event.
        * **Legacy default stream**: if your producer uses v2 CAI (no
          ``stream`` field) or advertises ``stream: 1``, cvcuda waits on the
          legacy default stream.  This is the safe conservative default but
          forfeits concurrency with stream-0 work.
        * **Use the default stream**: ``cvcuda.Stream.default`` wraps stream
          0 and has the usual legacy-default blocking semantics; use it when
          you want implicit sync with everything else on stream 0 and don't
          need the concurrency of a dedicated stream.
        )pbdoc",
                              py::is_weak_referenceable());

    stream.def(py::new_(&Stream::Create),
               "Create a new CUDA stream.  Streams created this way are non-blocking; "
               "cross-library synchronization is handled automatically via the CAI "
               "``stream`` field (see the class docstring).");

    py::module_ internal = m.attr(INTERNAL_SUBMODULE_NAME);
    internal.def("drain_pending_releases", &Stream::DrainAllPendingReleases);
    internal.def("syncAuxStream", &Stream::DrainAllPendingReleases);
    internal.def("_pending_release_batch_count",
                 []
                 {
                     PendingReleaseState &state = PendingReleases();
                     std::lock_guard      lock(state.mutex);
                     return state.pending.size();
                 });
    internal.def("_fail_next_release_event_record",
                 [](bool fail) { FailNextReleaseEventRecord().store(fail, std::memory_order_relaxed); });
    internal.def("_drain_pending_releases_with_lock_hook",
                 [](py::object afterLock) { DrainPendingReleasesImpl([&afterLock] { afterLock(); }); });

    // Wrap the CUDA legacy default stream as `cvcuda.Stream.default`. It is not
    // pushed onto a stream stack: that would seed only the importing thread's,
    // so `m_defaultStream` serves every thread uniformly instead.
    //
    // Skipped on hosts with no CUDA device (CPU-only build/CI nodes,
    // CUDA_VISIBLE_DEVICES=""): wrapping stream 0 unavoidably touches the
    // CUDA runtime (e.g. the `Stream` ctor's cudaStreamGetFlags), which fails
    // without a driver-bound device. We also skip
    // registering the `current` static property in that case — there is no
    // sensible value, and exposing a getter that throws breaks introspection
    // tools (nanobind_stubgen, sphinx-autodoc) that walk class members.
    int deviceCount = 0;
    if (cudaError_t devCntErr = cudaGetDeviceCount(&deviceCount);
        devCntErr == cudaErrorNoDevice || devCntErr == cudaErrorStubLibrary)
    {
        (void)cudaGetLastError();
        deviceCount = 0;
    }
    std::shared_ptr<Stream> globalStream;
    if (deviceCount > 0)
    {
        stream.def_prop_ro_static(
            "current", [](py::object) { return Current().sharedStream(); },
            "Get the current CUDA stream for this thread. Threads with no active stream context get "
            "cvcuda.Stream.default.");

        static priv::ExternalStream<priv::VOIDP> cudaDefaultStream(static_cast<cudaStream_t>(nullptr));
        globalStream    = std::make_shared<Stream>(cudaDefaultStream);
        m_defaultStream = globalStream;

        // Owns the wrapper. It cannot live under the public name, which is a
        // read-only property: assigning `default` used to succeed and change
        // nothing, since Current() resolves through m_defaultStream. The
        // atexit handler drops its own capture while the interpreter is still
        // alive, so something Python-owned has to outlast it.
        internal.attr("default_stream") = globalStream;

        stream.def_prop_ro_static(
            "default", [](py::object) { return m_defaultStream.lock(); },
            "The CUDA legacy default stream (stream 0). Read-only: to run work on another stream, enter "
            "its context or pass it as the operator's stream argument.");
    }

    // Order from most specific to less specific
    ExportExternalStream<TORCH>(m);
    ExportExternalStream<CUPY>(m);
    ExportExternalStream<VOIDP>(m);
    ExportExternalStream<INT>(m);

    fflush(stdout);

    stream.def("__enter__", &Stream::activate, "Activate the CUDA stream as the current stream for this thread.")
        .def("__exit__", &Stream::deactivate, py::arg().none(), py::arg().none(), py::arg().none(),
             "Deactivate the CUDA stream as the current stream for this thread. Raises RuntimeError if it is "
             "not the current stream on this thread, unless an exception is already propagating out of the "
             "context body, which is left to reach the caller unchanged.")
        .def("sync", &Stream::sync, "Wait for all preceding CUDA calls in the current stream to complete.")
        .def("wait_stream", &Stream::wait_stream, py::arg("other"),
             "Insert a dependency on 'other' into this stream. All subsequent work enqueued on this stream "
             "will wait until all work currently enqueued on 'other' has completed. "
             "Calling wait_stream(self) is a no-op. "
             "Cross-device usage is not supported and will raise a CUDA error.")
        .def("__int__", &Stream::pyhandle, "Cast the CUDA stream object to an integer handle.")
        .def("__repr__", &util::ToString<Stream>, "Return a string representation of the CUDA stream object.")
        .def_prop_ro("handle", &Stream::pyhandle, "Get the integer handle for the CUDA stream object.")
        .def_prop_ro("id", &Stream::id, "Get the unique ID for the CUDA stream object.");

    internal.def("nbytes_in_cache", [](const Stream &value) { return value.GetSizeInBytes(); });

    // Make sure all streams we've created are synced when script ends.
    // Also make cleanup hold the globalStream reference during script execution.
    util::RegisterCleanup(m, [globalStream]() { CleanupAtExit(globalStream); });
}

} // namespace nvcvpy::priv
