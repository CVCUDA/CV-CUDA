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

#include "Resource.hpp"

#include "Stream.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <cuda_runtime.h>

#include <atomic>

namespace nvcvpy::priv {

namespace {

constexpr uint64_t kPendingReleaseOwnerMask = (uint64_t{1} << 32) - 1;
constexpr uint64_t kBlockedReleaseOwnerUnit = uint64_t{1} << 32;

} // namespace

Resource::Resource([[maybe_unused]] bool createSubmissionState)
{
#ifdef NB_FREE_THREADED
    // Wrappers that share allocation state install it before publication.
    if (createSubmissionState)
        m_submissionStates.push_back(std::make_shared<SubmissionState>());
    static std::atomic<uint64_t> idnext{0};
    m_id = idnext.fetch_add(1, std::memory_order_relaxed);
#else
    static uint64_t idnext = 0;
    m_id                   = idnext++;
#endif
}

std::mutex *Resource::submissionMutex() const
{
    return nullptr;
}

void Resource::acquirePendingReleaseOwner() noexcept
{
    const uint64_t previous = m_pendingReleaseOwnership.fetch_add(1, std::memory_order_release);
    NVCV_ASSERT((previous & kPendingReleaseOwnerMask) != kPendingReleaseOwnerMask);
}

void Resource::blockPendingReleaseOwner() noexcept
{
    const uint64_t previous
        = m_pendingReleaseOwnership.fetch_add(kBlockedReleaseOwnerUnit - 1, std::memory_order_acq_rel);
    NVCV_ASSERT((previous & kPendingReleaseOwnerMask) != 0);
}

void Resource::unblockPendingReleaseOwner() noexcept
{
    const uint64_t previous
        = m_pendingReleaseOwnership.fetch_sub(kBlockedReleaseOwnerUnit - 1, std::memory_order_acq_rel);
    NVCV_ASSERT(previous >= kBlockedReleaseOwnerUnit);
}

void Resource::indexPendingResources() noexcept
{
    Cache::IndexPending(*this);
}

void Resource::releasePendingReleaseOwner(bool blocked) noexcept
{
    const uint64_t unit     = blocked ? kBlockedReleaseOwnerUnit : 1;
    const uint64_t previous = m_pendingReleaseOwnership.fetch_sub(unit, std::memory_order_acq_rel);
    NVCV_ASSERT(blocked ? previous >= kBlockedReleaseOwnerUnit : (previous & kPendingReleaseOwnerMask) != 0);
}

std::optional<uint64_t> Resource::reusablePendingReleaseOwners() const
{
    const uint64_t ownership = m_pendingReleaseOwnership.load(std::memory_order_acquire);
    if (ownership == 0 || ownership > kPendingReleaseOwnerMask)
    {
        return std::nullopt;
    }
    return ownership;
}

bool Resource::pendingReleaseOwnershipMatches(uint64_t snapshot) const
{
    return m_pendingReleaseOwnership.load(std::memory_order_acquire) == snapshot;
}

bool Resource::hasPendingReleaseOwners() const
{
    return m_pendingReleaseOwnership.load(std::memory_order_acquire) != 0;
}

Resource::~Resource() = default;

uint64_t Resource::id() const
{
    return m_id;
}

void Resource::submitSync(Stream &stream)
{
    submitSync(stream, stream.submissionDeviceId());
}

bool Resource::isOrderedOn(Stream &stream)
{
    const int device = stream.submissionDeviceId();
#ifdef NB_FREE_THREADED
    SubmissionLocks locks;
    locks.add(m_submissionStates, LOCK_MODE_READWRITE);
    if (!locks.try_lock())
        return false;
    return std::all_of(m_submissionStates.begin(), m_submissionStates.end(),
                       [&](const auto &state) { return state->isOrderedOn(stream, device); });
#else
    if (isSynchronizedWith(stream, device))
        return true;
    std::lock_guard lock(m_mtx);
    if (!m_lastStreamOwner && m_lastStreamHandle == nullptr)
        return true;
    const auto previous       = m_lastStreamOwner ? m_lastStreamOwner->handle() : m_lastStreamHandle;
    const auto next           = stream.handle();
    const bool explicitStream = next != nullptr && next != cudaStreamLegacy && next != cudaStreamPerThread;
    return previous == next
        && (explicitStream
            || (m_lastStreamOwner && m_lastDevice == device
                && m_lastStreamOwner->sharesQueueWith(*stream.handleOwner())));
#endif
}

bool Resource::isSynchronizedWith(Stream &stream, int submissionDevice) const
{
    return m_synchronizedWith.load() == stream.handleOwner().get() && submissionDevice == stream.deviceId();
}

void Resource::cacheSynchronization(Stream &stream)
{
    // Publish only completed transitions. The owner identifies the device for
    // default streams; per-thread handles additionally need their thread context.
    if (stream.handle() != cudaStreamPerThread && m_lastDevice == stream.deviceId())
        m_synchronizedWith.store(m_lastStreamOwner.get());
}

std::shared_ptr<const StreamHandleOwner> Resource::replaceLastStreamOwner(Stream &stream, StreamOwnerBag *retiredOwners)
{
    if (retiredOwners && m_lastStreamOwner)
        retiredOwners->push_back(m_lastStreamOwner);
    auto previous      = std::move(m_lastStreamOwner);
    m_lastStreamHandle = nullptr;
    m_lastStreamOwner  = stream.handleOwner();
    return previous;
}

void Resource::submitSync(Stream &stream, int submissionDevice, StreamOwnerBag *retiredOwners)
{
#ifdef NB_FREE_THREADED
    ResourceBag     retained;
    SubmissionLocks locks;
    visitSubmissionResources(LOCK_MODE_READWRITE,
                             [&](Resource &resource, auto mode, bool retain)
                             {
                                 if (retain)
                                     retained.push_back(resource.shared_from_this());
                                 locks.add(resource.submissionStates(), mode);
                             });
    locks.lock();
    locks.prepare(stream, submissionDevice);
    util::CheckThrow(stream.handleOwner()->capturePerThreadCompletion());
    return;
#endif
    if (isSynchronizedWith(stream, submissionDevice))
        return;
    // Declared before the lock so it is destroyed after it: dropping the
    // last reference to a handle owner runs cudaStreamDestroy, and this
    // mutex is taken on the operator hot path. Reads as a dead store, is
    // not one.
    std::shared_ptr<const StreamHandleOwner> previous;
    std::unique_lock                         lk(m_mtx);
    // A cache hit can precede this transition or follow its publication, but
    // must not observe partially updated history or an owner being released.
    m_synchronizedWith.store(nullptr);

    // Compute the "previous" stream handle. Either a cvcuda handle owner or
    // a raw handle seeded from an external producer via
    // seedLastStream() (m_lastStreamHandle).  Both being unset means this is
    // the first submission for this resource.
    bool         hasPrevious = m_lastStreamOwner != nullptr || m_lastStreamHandle != nullptr;
    cudaStream_t prevHandle  = m_lastStreamOwner != nullptr ? m_lastStreamOwner->handle() : m_lastStreamHandle;
    // Fast path: previously bound to the same stream means no sync work.
    // Explicit streams need only a pointer comparison. A null/default handle
    // also needs the current device, which a batch resolves once and shares
    // with all of its children.
    // cudaStream_t is an opaque handle compared by value, never dereferenced;
    // pointer-to-const would only spell CUDA's private struct name here.
    if (cudaStream_t nextHandle = stream.handle(); // NOSONAR: opaque handle, compared by value only.
        hasPrevious && prevHandle == nextHandle)
    {
        const bool isExplicitStream
            = nextHandle != nullptr && nextHandle != cudaStreamLegacy && nextHandle != cudaStreamPerThread;
        const bool isTrustedDefault = m_lastStreamOwner != nullptr && m_lastDevice == submissionDevice
                                   && m_lastStreamOwner->sharesQueueWith(*stream.handleOwner());
        if (isExplicitStream || isTrustedDefault)
        {
            // A trustworthy raw explicit handle adopts this wrapper. Raw
            // default sentinels do not return here because CAI producers may
            // advertise one while their writes run on another stream.
            if (m_lastStreamOwner == nullptr
                || (!m_lastStreamOwner->retainsHandleLifetime() && stream.handleOwner()->retainsHandleLifetime()))
            {
                previous = replaceLastStreamOwner(stream, retiredOwners);
            }
            cacheSynchronization(stream);
            return;
        }
    }

    // Slow path: need the current device for either first-binding state or
    // cross-stream/cross-device sync logic.
    int curDev;
    util::CheckThrow(cudaGetDevice(&curDev));

    // First submission for this resource — take ownership on the current
    // stream and return with no sync work.
    if (!hasPrevious)
    {
        m_lastStreamOwner = stream.handleOwner();
        m_lastDevice      = curDev;
        cacheSynchronization(stream);
        return;
    }

    // Defensive sync for CAI default-stream sentinels.  A `prevHandle` of
    // `cudaStreamLegacy` (1) or `cudaStreamPerThread` (2) only reaches us
    // through CAI parsing — either a producer that explicitly advertised a
    // default-stream sentinel (cupy reports the cupy-current stream at CAI-
    // query time, not the actual writer stream) or a producer that didn't
    // advertise a stream at all (CAI v2, e.g. PyTorch) and got the
    // cudaStreamLegacy fallback.  Both populations include producers that
    // lie.  An event-based barrier on a sentinel does NOT capture work on
    // non-blocking streams, so we'd race against unfinished producer
    // kernels.  Fall back to `cudaDeviceSynchronize` so the wait is correct
    // regardless of which stream the producer actually used.
    //
    // No real cvcuda stream handle is 1 or 2, so cvcuda → cvcuda chains
    // (which advertise their actual writer stream as a real pointer) stay
    // on the event-based fast path below.  This branch only fires once per
    // wrapped buffer's first cvcuda use; subsequent ops on the same buffer
    // take the event-based path because m_lastStreamOwner is set after this.
    if (prevHandle == cudaStreamLegacy || prevHandle == cudaStreamPerThread)
    {
        if (m_lastDevice != curDev)
        {
            int savedDev = curDev;
            util::CheckThrow(cudaSetDevice(m_lastDevice));
            util::CheckThrow(cudaDeviceSynchronize());
            util::CheckThrow(cudaSetDevice(savedDev));
        }
        else
        {
            util::CheckThrow(cudaDeviceSynchronize());
        }
    }
    // If the resource is moving between devices, CUDA events cannot synchronize
    // across device boundaries. Fall back to a full stream synchronize.
    else if (m_lastDevice != curDev)
    {
        // Sync the old stream on its device to ensure all work completes.
        int savedDev = curDev;
        util::CheckThrow(cudaSetDevice(m_lastDevice));
        util::CheckThrow(cudaStreamSynchronize(prevHandle));
        util::CheckThrow(cudaSetDevice(savedDev));
    }
    else
    {
        stream.waitFor(prevHandle);
    }

    // update the last stream since we changed streams
    previous     = replaceLastStreamOwner(stream, retiredOwners);
    m_lastDevice = curDev;
    cacheSynchronization(stream);
}

Resource::SyncState Resource::syncState() const
{
    SyncState        state;
    std::unique_lock lk(m_mtx);

    state.device = m_lastDevice;
    state.owner  = m_lastStreamOwner;
    if (state.owner != nullptr)
    {
        state.stream = state.owner->handle();
        state.bound  = true;
    }
    else
    {
        state.stream = m_lastStreamHandle;
        state.bound  = m_lastStreamHandle != nullptr;
    }
    return state;
}

bool Resource::submitSyncThrough(Stream &stream, const SyncState &synchronizedState, int submissionDevice,
                                 StreamOwnerBag *retiredOwners)
{
    if (isSynchronizedWith(stream, submissionDevice))
        return true;
    // Declared before the lock so it is destroyed after it: dropping the
    // last reference to a handle owner runs cudaStreamDestroy, and this
    // mutex is taken on the operator hot path. Reads as a dead store, is
    // not one.
    std::shared_ptr<const StreamHandleOwner> currentOwner;
    std::unique_lock                         lk(m_mtx);

    bool         currentBound = m_lastStreamOwner != nullptr || m_lastStreamHandle != nullptr;
    // Opaque handle, compared by value only -- see submitSync().
    cudaStream_t currentStream // NOSONAR: opaque handle, compared by value only.
        = m_lastStreamOwner != nullptr ? m_lastStreamOwner->handle() : m_lastStreamHandle;
    if (currentBound != synchronizedState.bound || currentStream != synchronizedState.stream
        || m_lastDevice != synchronizedState.device
        || (currentStream == cudaStreamPerThread
            && (!m_lastStreamOwner || !synchronizedState.owner
                || !m_lastStreamOwner->sharesQueueWith(*synchronizedState.owner))))
    {
        return false;
    }
    m_synchronizedWith.store(nullptr);

    if (cudaStream_t nextHandle = stream.handle(); // NOSONAR: opaque handle, compared by value only.
        currentBound && currentStream == nextHandle && m_lastDevice == submissionDevice
        && (nextHandle != cudaStreamPerThread
            || (m_lastStreamOwner && m_lastStreamOwner->sharesQueueWith(*stream.handleOwner()))))
    {
        if (m_lastStreamOwner == nullptr
            || (!m_lastStreamOwner->retainsHandleLifetime() && stream.handleOwner()->retainsHandleLifetime()))
        {
            currentOwner = replaceLastStreamOwner(stream, retiredOwners);
        }
        cacheSynchronization(stream);
        return true;
    }

    // The parent already established this stream dependency, so only transfer
    // ownership; recording another event per child would duplicate that wait.
    currentOwner = replaceLastStreamOwner(stream, retiredOwners);
    m_lastDevice = submissionDevice;
    cacheSynchronization(stream);
    return true;
}

void Resource::seedLastStream(cudaStream_t handle, int device)
{
#ifdef NB_FREE_THREADED
    for (const auto &state : m_submissionStates) state->seed(handle, device);
    return;
#endif
    std::unique_lock lk(m_mtx);

    // If some stream ownership is already recorded, respect it — either we've
    // been seeded before, or a cvcuda op has already claimed this resource.
    if (m_lastStreamOwner != nullptr || m_lastStreamHandle != nullptr)
    {
        return;
    }

    m_synchronizedWith.store(nullptr);
    m_lastStreamHandle = handle;
    m_lastDevice       = device;
}

void Resource::resetLastStreamForRebind()
{
#ifdef NB_FREE_THREADED
    m_submissionStates = {std::make_shared<SubmissionState>()};
#endif
    // Declared before the lock so it is destroyed after it: dropping the
    // last reference to a handle owner runs cudaStreamDestroy, and this
    // mutex is taken on the operator hot path. Reads as a dead store, is
    // not one.
    std::shared_ptr<const StreamHandleOwner> previous;
    std::unique_lock                         lk(m_mtx);

    m_synchronizedWith.store(nullptr);
    previous           = std::move(m_lastStreamOwner);
    m_lastStreamHandle = nullptr;
    m_lastDevice       = -1;
}

Resource::ExportStreamState Resource::exportStreamState() const
{
#ifdef NB_FREE_THREADED
    SubmissionLocks locks;
    locks.add(m_submissionStates, LOCK_MODE_READWRITE);
    locks.lock();
    auto access = ExportSubmissionAccess(m_submissionStates);
    return {access.stream, std::move(access.owner), m_submissionStates};
#endif
    ExportStreamState state;
    std::unique_lock  lk(m_mtx);

    state.owner = m_lastStreamOwner;
    if (state.owner != nullptr)
    {
        state.handle = state.owner->handle();
    }
    else if (m_lastStreamHandle != nullptr)
    {
        state.handle = m_lastStreamHandle;
    }
    return state;
}

RetainedMemory Resource::retainedMemory() const
{
    return {0, 1};
}

std::shared_ptr<Resource> Resource::shared_from_this()
{
    return std::static_pointer_cast<Resource>(Object::shared_from_this());
}

std::shared_ptr<const Resource> Resource::shared_from_this() const
{
    return std::static_pointer_cast<const Resource>(Object::shared_from_this());
}

void Resource::Export(py::module_ &m)
{
    py::class_<Resource>(m, "Resource", "Resource")
        .def_prop_ro("id", &Resource::id, "Unique resource instance identifier")
        .def("submitStreamSync", static_cast<void (Resource::*)(Stream &)>(&Resource::submitSync), py::arg("stream"),
             "Syncs object on new Stream");
}

} // namespace nvcvpy::priv
