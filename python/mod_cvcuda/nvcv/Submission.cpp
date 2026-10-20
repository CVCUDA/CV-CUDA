/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Submission.hpp"

#ifdef NB_FREE_THREADED
#    include <common/CheckError.hpp>
#    include <cuda.h>

#    include <condition_variable>
#    include <map>

namespace nvcvpy::priv {
namespace {
class DeviceGuard
{
public:
    explicit DeviceGuard(int device)
    {
        util::CheckThrow(cudaGetDevice(&m_previous));
        m_current = m_previous;
        setDevice(device);
    }

    void setDevice(int device)
    {
        if (device != m_current)
        {
            util::CheckThrow(cudaSetDevice(device));
            m_current = device;
        }
    }

    ~DeviceGuard()
    {
        if (m_current != m_previous)
            util::CheckLog(cudaSetDevice(m_previous));
    }

private:
    int m_previous = -1;
    int m_current  = -1;
};

bool SameStream(const SubmissionState::Access &a, const SubmissionState::Access &b)
{
    return a.device == b.device && a.stream == b.stream
        && (a.stream != cudaStreamPerThread || (a.owner && b.owner && a.owner->sharesQueueWith(*b.owner)));
}
} // namespace

std::shared_ptr<SubmissionState> AllocationSubmissionState(const void *pointer, int device)
{
    struct Registry
    {
        nanobind::ft_mutex                                                    mutex;
        std::map<std::pair<int, CUdeviceptr>, std::weak_ptr<SubmissionState>> allocations;
        unsigned int                                                          insertions = 0;
    };

    static auto *registry   = new Registry; // Registry can be used during Python teardown.
    CUdeviceptr  base       = reinterpret_cast<CUdeviceptr>(pointer);
    CUdeviceptr  allocation = 0;
    if (cuPointerGetAttribute(&allocation, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, base) == CUDA_SUCCESS)
        base = allocation;
    if (device < 0)
    {
        cudaPointerAttributes attributes{};
        util::CheckThrow(cudaPointerGetAttributes(&attributes, pointer));
        device = attributes.device;
    }
    std::lock_guard lock(registry->mutex);
    auto           &weak = registry->allocations[{device, base}];
    if (auto state = weak.lock())
        return state;
    auto state = std::make_shared<SubmissionState>();
    weak       = state;
    if (++registry->insertions % 64 == 0)
    {
        for (auto it = registry->allocations.begin(); it != registry->allocations.end();)
        {
            if (it->second.expired())
                it = registry->allocations.erase(it);
            else
                ++it;
        }
    }
    return state;
}

// Registration holds the mutex only while checking active access sets. CUDA
// enqueue and state preparation run outside it, including compatible readers.
struct SubmissionLocks::Registry
{
    nanobind::ft_mutex          mutex;
    std::condition_variable_any changed;
    SubmissionLocks            *head    = nullptr;
    bool                        waiting = false;
};

SubmissionLocks::Registry &SubmissionLocks::registry()
{
    static auto *value = new Registry;
    return *value;
}

void SubmissionLocks::addMultiple(const SubmissionStates &states, LockMode mode)
{
    // State lists are sorted and deduplicated when changed, and retained until unlock.
    auto              *previous = entriesBegin();
    const auto        *last     = entriesEnd();
    std::vector<Entry> merged((last - previous) + states.size());
    size_t             count = 0;
    for (const auto &state : states)
    {
        Entry entry{state.get(), mode};
        while (previous != last && *previous < entry) merged[count++] = *previous++;
        if (previous != last && previous->state == entry.state)
        {
            entry.mode = static_cast<LockMode>(entry.mode | previous->mode);
            ++previous;
        }
        merged[count++] = entry;
    }
    std::copy(previous, last, merged.begin() + count);
    count += last - previous;
    merged.resize(count);
    m_entries = std::move(merged);
}

bool SubmissionLocks::conflicts(const SubmissionLocks &other) const
{
    auto *a = entriesBegin(), *aEnd = entriesEnd();
    auto *b = other.entriesBegin(), *bEnd = other.entriesEnd();
    while (a != aEnd && b != bEnd)
    {
        if (a->state == b->state)
        {
            if ((a->mode | b->mode) & LOCK_MODE_WRITE)
                return true;
            ++a;
            ++b;
        }
        else if (std::less<SubmissionState *>{}(a->state, b->state))
            ++a;
        else
            ++b;
    }
    return false;
}

bool SubmissionLocks::tryRegister(Registry &registry)
{
    for (auto *active = registry.head; active; active = active->m_next)
        if (conflicts(*active))
            return false;
    m_next        = registry.head;
    registry.head = this;
    m_registered  = true;
    return true;
}

bool SubmissionLocks::try_lock()
{
    auto           &r = registry();
    std::lock_guard lock(r.mutex);
    return tryRegister(r);
}

void SubmissionLocks::lock()
{
    auto            &r = registry();
    std::unique_lock lock(r.mutex);
    while (!tryRegister(r))
    {
        r.waiting = true;
        util::ScopedThreadDetach release;
        r.changed.wait(lock);
    }
}

void SubmissionLocks::unlock()
{
    if (!m_registered)
        return;
    auto &r = registry();
    bool  notify;
    {
        std::lock_guard lock(r.mutex);
        auto          **entry = &r.head;
        while (*entry != this) entry = &(*entry)->m_next;
        *entry       = m_next;
        m_next       = nullptr;
        m_registered = false;
        notify       = std::exchange(r.waiting, false);
    }
    if (notify)
        r.changed.notify_all();
}

SubmissionState::~SubmissionState()
{
    if (m_events.empty())
        return;
    int previous;
    if (cudaGetDevice(&previous) != cudaSuccess)
        return;
    for (auto [device, event] : m_events)
    {
        if (cudaSetDevice(device) == cudaSuccess)
            util::CheckLog(cudaEventDestroy(event));
    }
    util::CheckLog(cudaSetDevice(previous));
}

void SubmissionState::wait(const Access &previous, const Access &next)
{
    if (SameStream(previous, next))
        return;
    if (previous.owner && previous.owner->perThreadCompletion())
    {
        DeviceGuard guard(next.device);
        util::CheckThrow(cudaStreamWaitEvent(next.stream, previous.owner->perThreadCompletion()));
        return;
    }
    DeviceGuard guard(previous.device);
    auto       &event = m_events[previous.device];
    if (!event)
        util::CheckThrow(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    util::CheckThrow(cudaEventRecord(event, previous.stream));
    guard.setDevice(next.device);
    util::CheckThrow(cudaStreamWaitEvent(next.stream, event));
}

bool SubmissionState::isOrderedOn(const Stream &stream, int device) const
{
    const Access next{stream.handle(), device, stream.handleOwner()};
    return (!m_writer || SameStream(*m_writer, next))
        && std::all_of(m_readers.begin(), m_readers.end(),
                       [&](const Access &reader) { return SameStream(reader, next); });
}

void SubmissionState::prepare(Stream &stream, int device, LockMode mode, StreamOwnerBag &retiredOwners)
{
    const bool  write     = mode & LOCK_MODE_WRITE;
    const auto  handle    = stream.handle();
    const auto &nextOwner = stream.handleOwner();
    const auto *owner     = nextOwner.get();
    // A default handle is device-relative. Cache it only on the wrapper's
    // recorded device so its owner identity also identifies the device.
    const bool  cacheReader = handle != cudaStreamPerThread && device == stream.deviceId();
    if (!write && cacheReader && m_lastReader.load(std::memory_order_relaxed) == owner)
        return;

    // A write reservation excludes every other access. Compatible readers
    // still need a mutex while extending their shared dependency history.
    std::unique_lock lock(m_mutex, std::defer_lock);
    if (!write)
        lock.lock();
    auto sameStream = [&](const Access &access)
    {
        return access.device == device && access.stream == handle
            && (handle != cudaStreamPerThread || (access.owner && access.owner->sharesQueueWith(*nextOwner)));
    };
    auto adoptOwner = [&](Access &access)
    {
        if (access.owner != nextOwner
            && (handle == cudaStreamPerThread || !access.owner
                || (!access.owner->retainsHandleLifetime() && nextOwner->retainsHandleLifetime())))
        {
            // The old owner may run Python during destruction. Keep it until
            // the caller releases its submission reservation.
            retiredOwners.push_back(access.owner);
            access.owner = nextOwner;
        }
    };
    auto sameReader = std::find_if(m_readers.begin(), m_readers.end(), sameStream);
    if (!write && sameReader != m_readers.end())
    {
        adoptOwner(*sameReader);
        if (cacheReader)
            m_lastReader.store(sameReader->owner.get(), std::memory_order_relaxed);
        return;
    }
    if (write && m_readers.empty() && m_writer && sameStream(*m_writer))
    {
        adoptOwner(*m_writer);
        return;
    }
    Access next{handle, device, nextOwner};
    if (m_writer)
        wait(*m_writer, next);
    if (mode & LOCK_MODE_WRITE)
    {
        for (const auto &reader : m_readers) wait(reader, next);
        retiredOwners.reserve(retiredOwners.size() + m_readers.size() + (m_writer ? 1 : 0));
        for (auto &reader : m_readers) retiredOwners.push_back(std::move(reader.owner));
        m_readers.clear();
        m_lastReader.store(nullptr, std::memory_order_relaxed);
        if (m_writer)
            retiredOwners.push_back(std::move(m_writer->owner));
        m_writer = std::move(next);
    }
    else
    {
        m_readers.push_back(std::move(next));
        if (cacheReader)
            m_lastReader.store(owner, std::memory_order_relaxed);
    }
}

void SubmissionState::seed(cudaStream_t stream, int device)
{
    DeviceGuard deviceGuard(device);
    Access      next{stream, device, {}};
    if (stream == cudaStreamPerThread)
        next.owner = std::make_shared<StreamHandleOwner>(stream, device, false);
    if (stream == cudaStreamLegacy)
    {
        util::ScopedThreadDetach release;
        util::CheckThrow(cudaDeviceSynchronize());
        // Completed external work adds no dependency. Keep any existing alias
        // history: another thread may have submitted work during the sync.
        return;
    }
    std::optional<Access> previous;
    std::vector<Access>   readers;
    SubmissionLocks       lease;
    lease.add(*this, LOCK_MODE_READWRITE);
    lease.lock();
    if (m_writer)
        wait(*m_writer, next);
    for (const auto &reader : m_readers) wait(reader, next);
    // The snapshot must include the waits just merged into this thread's stream.
    if (next.owner)
        util::CheckThrow(next.owner->capturePerThreadCompletion());
    previous = std::move(m_writer);
    readers.swap(m_readers);
    m_lastReader.store(nullptr, std::memory_order_relaxed);
    m_writer = std::move(next);
}

SubmissionState::Access SubmissionState::exportAccess()
{
    Access result;
    if (m_writer)
        result = *m_writer;
    else if (!m_readers.empty())
        result = m_readers.front();
    else
        util::CheckThrow(cudaGetDevice(&result.device));
    if (result.stream == cudaStreamPerThread)
    {
        if (!m_handoff)
        {
            DeviceGuard  guard(result.device);
            cudaStream_t stream;
            util::CheckThrow(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            try
            {
                m_handoff
                    = Access{stream, result.device, std::make_shared<StreamHandleOwner>(stream, result.device, true)};
            }
            catch (...)
            {
                util::CheckLog(cudaStreamDestroy(stream));
                throw;
            }
        }
        wait(result, *m_handoff);
        result = *m_handoff;
    }
    // A writable export must wait for outstanding readers as well as writers.
    for (const auto &reader : m_readers) wait(reader, result);
    return result;
}

void SubmissionState::exportTo(const Access &consumer)
{
    if (m_writer)
        wait(*m_writer, consumer);
    for (const auto &reader : m_readers) wait(reader, consumer);
}

SubmissionState::Access ExportSubmissionAccess(const SubmissionStates &states)
{
    auto result = states.front()->exportAccess();
    if (states.size() == 1)
        return result;
    for (size_t i = 1; i < states.size(); ++i)
    {
        auto access = states[i]->exportAccess();
        if ((result.stream == nullptr || result.stream == cudaStreamLegacy) && access.stream != nullptr
            && access.stream != cudaStreamLegacy)
            result = std::move(access);
    }
    for (const auto &state : states) state->exportTo(result);
    return result;
}
} // namespace nvcvpy::priv
#endif
