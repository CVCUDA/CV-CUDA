/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PYTHON_NATIVE_RESOURCE_GUARD_HPP
#define CVCUDA_PYTHON_NATIVE_RESOURCE_GUARD_HPP

#include "../nvcv/BatchLock.hpp"
#include "../nvcv/Image.hpp"
#include "../nvcv/ImageBatch.hpp"
#include "../nvcv/PythonObjectAdapters.hpp"
#include "../nvcv/Resource.hpp"
#include "../nvcv/Stream.hpp"

#include <common/PyUtil.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace cvcudapy {

namespace py = ::nanobind;

class NativeResourceGuard
{
public:
    explicit NativeResourceGuard(nvcvpy::priv::Stream &stream)
        : m_stream(stream.sharedStream())
        , m_submissionDevice(m_stream->submissionDeviceId())
    {
    }

    NativeResourceGuard(const NativeResourceGuard &)            = delete;
    NativeResourceGuard(NativeResourceGuard &&)                 = delete;
    NativeResourceGuard &operator=(const NativeResourceGuard &) = delete;
    NativeResourceGuard &operator=(NativeResourceGuard &&)      = delete;

    ~NativeResourceGuard()
    {
        finishNoThrow();
    }

    NativeResourceGuard &add(nvcvpy::LockMode                                                      mode,
                             std::initializer_list<std::reference_wrapper<nvcvpy::priv::Resource>> resources)
    {
        for (nvcvpy::priv::Resource &resource : resources)
        {
#ifdef NB_FREE_THREADED
            if (nvcvpy::util::HasPythonThreadState())
                if (auto owner = py::find(&resource))
                    m_pythonOwners.push_back(std::move(owner));
            resource.visitSubmissionResources(mode,
                                              [&](nvcvpy::priv::Resource &owner, auto access, bool retain)
                                              {
                                                  if ((access & nvcvpy::LOCK_MODE_WRITE) && owner.readOnly())
                                                      throw py::value_error("Cannot write to a read-only buffer");
                                                  if (access != nvcvpy::LockMode::LOCK_MODE_NONE)
                                                      m_submissionLocks.add(owner.submissionStates(), access);
                                                  if (retain)
                                                      addEntry(owner.shared_from_this(),
                                                               access != nvcvpy::LockMode::LOCK_MODE_NONE);
                                              });
#else
            addEntry(resource.shared_from_this(), mode != nvcvpy::LockMode::LOCK_MODE_NONE);
            if (const auto *batch = dynamic_cast<nvcvpy::priv::ImageBatchVarShape *>(&resource))
                addEntry(batch->pendingOwners(), false);
#endif
        }
        return *this;
    }

#ifdef NB_FREE_THREADED
    void begin()
#else
    template<class F>
    void run(F &&fn)
#endif
    {
        // Mutable native operator state may be cached and reused while an
        // earlier submission is still in flight. Hold its host lease from
        // synchronization through consumer enqueue so another thread cannot
        // record a cross-stream dependency before this call has enqueued its
        // operator work.
#ifdef NB_FREE_THREADED
        m_submissionLocks.lock();
        auto &submissionLease            = m_submissionLease;
        auto &additionalSubmissionLeases = m_additionalSubmissionLeases;
#else
        std::unique_lock<std::mutex> submissionLease;
        std::vector<std::unique_lock<std::mutex>> additionalSubmissionLeases;
#endif
        if (m_overflowSubmissionMutexes.empty())
        {
            std::sort(m_inlineSubmissionMutexes.begin(), m_inlineSubmissionMutexes.begin() + m_numSubmissionMutexes,
                      std::less<std::mutex *>{});
            if (m_numSubmissionMutexes != 0)
            {
                submissionLease = SubmissionLease{*m_inlineSubmissionMutexes[0]};
                additionalSubmissionLeases.reserve(m_numSubmissionMutexes - 1);
                for (size_t i = 1; i < m_numSubmissionMutexes; ++i)
                {
                    additionalSubmissionLeases.emplace_back(*m_inlineSubmissionMutexes[i]);
                }
            }
        }
        else
        {
            std::sort(m_overflowSubmissionMutexes.begin(), m_overflowSubmissionMutexes.end(),
                      std::less<std::mutex *>{});
            submissionLease = SubmissionLease{*m_overflowSubmissionMutexes[0]};
            additionalSubmissionLeases.reserve(m_overflowSubmissionMutexes.size() - 1);
            for (auto it = m_overflowSubmissionMutexes.begin() + 1; it != m_overflowSubmissionMutexes.end(); ++it)
            {
                additionalSubmissionLeases.emplace_back(**it);
            }
        }

        // Pending resource release uses CUDA events/callbacks that are illegal
        // during stream capture. Check after acquiring operator leases so a
        // wait cannot make an earlier capture result stale; still reject
        // before synchronization mutates resource state or enqueues work.
        try
        {
            m_stream->ensureNotCapturing(m_submissionDevice);
        }
        catch (const std::exception &e)
        {
            // The legacy C-API adapter maps internal stream failures to
            // ValueError. Preserve that Python contract on the native path.
            throw py::value_error(e.what());
        }

        try
        {
            const auto &injection = nvcvpy::priv::GetResourceGuardFailureInjection();
            nvcvpy::priv::ThrowIfResourceGuardFailureInjected(injection.resourcesSubmitSyncOnly,
                                                              "injected Resources_SubmitSyncOnly failure");

#ifdef NB_FREE_THREADED
            m_submissionLocks.prepare(*m_stream, m_submissionDevice);
#else
            forEachSyncResource([this](nvcvpy::priv::Resource *resource)
                                { resource->submitSync(*m_stream, m_submissionDevice, &m_retiredStreamOwners); });
#endif
        }
        catch (const std::exception &e)
        {
            // No consumer work was submitted, so no lifetime hold is needed.
            m_finished = true;
            // The compatibility C API reports every submit-sync failure as
            // ValueError. Keep the native fast path's public exception type
            // identical while avoiding a Python/C-API round trip.
            throw py::value_error(e.what());
        }
        catch (...) // NOSONAR: mark finished whatever escapes submitSync; rethrown unchanged.
        {
            m_finished = true;
            throw;
        }

        // Set before submission so unwind retains resources if the callable
        // throws after partially queueing work.
        m_submitted = true;
#ifndef NB_FREE_THREADED
        std::forward<F>(fn)();

        // The cross-stream ordering hazard ends once consumer work is in the
        // stream. Do not serialize event recording and release bookkeeping
        // for later submissions of the same cached operator.
        additionalSubmissionLeases.clear();
        if (submissionLease.owns_lock())
        {
            submissionLease.unlock();
        }
        commit();
#endif
    }

#ifdef NB_FREE_THREADED
    template<class F>
    void run(F &&fn)
    {
        begin();
        std::forward<F>(fn)();
        commit();
    }

    void finish()
    {
        commit();
    }
#endif

private:
    using ResourceOwner = std::shared_ptr<nvcvpy::priv::Resource>;
#ifdef NB_FREE_THREADED
    class SubmissionLease : public std::unique_lock<std::mutex>
    {
    public:
        SubmissionLease() = default;

        explicit SubmissionLease(std::mutex &mutex)
            : std::unique_lock<std::mutex>(mutex, std::defer_lock)
        {
            if (!try_lock())
            {
                nvcvpy::util::ScopedThreadDetach release;
                lock();
            }
        }
    };
#else
    using SubmissionLease = std::unique_lock<std::mutex>;
#endif

    static constexpr size_t kInlineCapacity = 8;

    class PyErrPreserver
    {
    public:
        PyErrPreserver() noexcept
        {
            if (!m_attached)
                return;
            PyErr_Fetch(&m_type, &m_value, &m_traceback);
        }

        ~PyErrPreserver() noexcept
        {
            if (!m_attached)
                return;
            PyErr_Clear();
            PyErr_Restore(m_type, m_value, m_traceback);
        }

        PyErrPreserver(const PyErrPreserver &)            = delete;
        PyErrPreserver &operator=(const PyErrPreserver &) = delete;

    private:
        bool      m_attached  = nvcvpy::util::HasPythonThreadState();
        PyObject *m_type      = nullptr;
        PyObject *m_value     = nullptr;
        PyObject *m_traceback = nullptr;
    };

    void releaseSubmissionLocks()
    {
#ifdef NB_FREE_THREADED
        m_additionalSubmissionLeases.clear();
        if (m_submissionLease.owns_lock())
            m_submissionLease.unlock();
        m_submissionLocks.unlock();
        nvcvpy::priv::BatchLockScope::Release();
#endif
    }

    void commit()
    {
#ifdef NB_FREE_THREADED
        auto captureStatus
            = (!m_finished && m_submitted) ? m_stream->handleOwner()->capturePerThreadCompletion() : cudaSuccess;
#endif
        releaseSubmissionLocks();
        if (m_finished)
        {
            return;
        }

        // Terminal before fallible code: the destructor must not retry a
        // failed hold during stack unwind.
        m_finished = true;
        if (!m_submitted)
        {
            return;
        }

        try
        {
#ifdef NB_FREE_THREADED
            nvcvpy::util::CheckThrow(captureStatus);
#endif
            const auto &injection = nvcvpy::priv::GetResourceGuardFailureInjection();
            nvcvpy::priv::ThrowIfResourceGuardFailureInjected(injection.streamHoldResources,
                                                              "injected Stream_HoldResources failure");

            // Stream moves these owners only after all fallible release setup
            // is complete. Until then they remain available for failure drain
            // or quarantine.
#ifdef NB_FREE_THREADED
            m_stream->holdResources(
                {m_inlineResources.data(), m_inlineSize, m_overflowResources.data(), m_overflowResources.size()},
                m_stream, m_pythonOwners, m_submissionDevice);
#else
            m_stream->holdResources(
                {m_inlineResources.data(), m_inlineSize, m_overflowResources.data(), m_overflowResources.size()},
                m_stream, m_submissionDevice);
#endif
            clearResources();
        }
        catch (const std::exception &e)
        {
            if (!drainStreamNoThrow())
            {
                quarantineNoThrow();
            }
            // Stream_HoldResources historically crosses the compatibility C
            // API, which maps all C++ failures to ValueError. Preserve that
            // observable contract on the direct native path.
            throw py::value_error(e.what());
        }
        catch (...)
        {
            if (!drainStreamNoThrow())
            {
                quarantineNoThrow();
            }
            throw;
        }
    }

    void finishNoThrow() noexcept
    {
        if (m_finished)
        {
            releaseSubmissionLocks();
            return;
        }

        PyErrPreserver preserver;
        try
        {
            commit();
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[cvcuda] ~ResourceGuard: commit() threw: %s\n", e.what());
        }
        catch (...)
        {
            std::fprintf(stderr, "[cvcuda] ~ResourceGuard: commit() threw unknown exception\n");
        }
    }

    bool drainStreamNoThrow() const noexcept
    {
        PyErrPreserver preserver;
        try
        {
            int         savedDevice = 0;
            cudaError_t status      = cudaGetDevice(&savedDevice);
            if (status != cudaSuccess)
            {
                std::fprintf(stderr, "[cvcuda] ResourceGuard: get device failed (%s); quarantining resources\n",
                             cudaGetErrorName(status));
                (void)cudaGetLastError();
                return false;
            }
            if (savedDevice != m_submissionDevice)
            {
                status = cudaSetDevice(m_submissionDevice);
            }
            if (status != cudaSuccess)
            {
                std::fprintf(stderr, "[cvcuda] ResourceGuard: set device failed (%s); quarantining resources\n",
                             cudaGetErrorName(status));
                (void)cudaGetLastError();
                return false;
            }
            {
                // A foreign callback queued on this stream may need the GIL.
#ifdef NB_FREE_THREADED
                nvcvpy::util::ScopedThreadDetach release;
#else
                py::gil_scoped_release release;
#endif
                status = cudaStreamSynchronize(m_stream->handle());
            }
            if (savedDevice != m_submissionDevice)
            {
                cudaError_t restoreStatus = cudaSetDevice(savedDevice);
                if (restoreStatus != cudaSuccess)
                {
                    std::fprintf(stderr, "[cvcuda] ResourceGuard: restore device failed (%s)\n",
                                 cudaGetErrorName(restoreStatus));
                    (void)cudaGetLastError();
                    return false;
                }
            }

            if (status == cudaSuccess)
            {
                return true;
            }

            std::fprintf(stderr, "[cvcuda] ResourceGuard: stream drain failed (%s); quarantining resources\n",
                         cudaGetErrorName(status));
            (void)cudaGetLastError();
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[cvcuda] ResourceGuard: stream drain threw: %s; quarantining resources\n", e.what());
        }
        catch (...)
        {
            std::fprintf(stderr, "[cvcuda] ResourceGuard: stream drain threw; quarantining resources\n");
        }
        return false;
    }

    void quarantineNoThrow() noexcept
    {
        struct QuarantinedResources
        {
            std::shared_ptr<nvcvpy::priv::Stream>      stream;
            std::array<ResourceOwner, kInlineCapacity> inlineResources;
            size_t                                     inlineSize;
            std::vector<ResourceOwner>                 overflowResources;
#ifdef NB_FREE_THREADED
            nvcvpy::priv::PythonOwnerBag pythonOwners;
#endif
        };

        // Completion could not be proven.  Leak the native owners rather than
        // free or reuse GPU resources that may still be in flight.
        auto *quarantine = new (std::nothrow) // NOSONAR: deliberately never freed, see above.
            QuarantinedResources{std::move(m_stream), std::move(m_inlineResources), m_inlineSize,
                                 std::move(m_overflowResources)
#ifdef NB_FREE_THREADED
                                     ,
                                 std::move(m_pythonOwners)
#endif
            };
        if (quarantine == nullptr)
        {
            // An out-of-memory failure leaves no safe way to retain shared
            // ownership; terminating is safer than a use-after-free on GPU.
            std::terminate();
        }
        (void)quarantine;
    }

    void addEntry(ResourceOwner resource, bool synchronize)
    {
        nvcvpy::priv::Resource *resourcePtr = resource.get();
        if (m_inlineSize < kInlineCapacity)
        {
            m_inlineResources[m_inlineSize] = std::move(resource);
            ++m_inlineSize;
        }
        else
        {
            m_overflowResources.push_back(std::move(resource));
        }

        if (!synchronize)
        {
            return;
        }

#ifndef NB_FREE_THREADED
        addSyncResource(resourcePtr);
#endif
        if (std::mutex *mutex = resourcePtr->submissionMutex())
        {
            addSubmissionMutex(mutex);
        }
    }

    void addSyncResource(nvcvpy::priv::Resource *resource)
    {
        if (m_numInlineSyncResources < kInlineCapacity)
        {
            m_inlineSyncResources[m_numInlineSyncResources] = resource;
            ++m_numInlineSyncResources;
        }
        else
        {
            m_overflowSyncResources.push_back(resource);
        }
    }

    void addSubmissionMutex(std::mutex *mutex)
    {
        auto inlineBegin = m_inlineSubmissionMutexes.begin();
        auto inlineEnd   = inlineBegin + m_numSubmissionMutexes;
        if (std::find(inlineBegin, inlineEnd, mutex) != inlineEnd
            || std::find(m_overflowSubmissionMutexes.begin(), m_overflowSubmissionMutexes.end(), mutex)
                   != m_overflowSubmissionMutexes.end())
        {
            return;
        }

        if (m_overflowSubmissionMutexes.empty() && m_numSubmissionMutexes < kInlineCapacity)
        {
            m_inlineSubmissionMutexes[m_numSubmissionMutexes] = mutex;
            ++m_numSubmissionMutexes;
            return;
        }

        if (m_overflowSubmissionMutexes.empty())
        {
            m_overflowSubmissionMutexes.reserve(kInlineCapacity * 2);
            m_overflowSubmissionMutexes.insert(m_overflowSubmissionMutexes.end(), inlineBegin, inlineEnd);
        }
        m_overflowSubmissionMutexes.push_back(mutex);
    }

    template<class F>
    void forEachSyncResource(F &&fn) const
    {
        for (size_t i = 0; i < m_numInlineSyncResources; ++i)
        {
            fn(m_inlineSyncResources[i]);
        }
        for (nvcvpy::priv::Resource *resource : m_overflowSyncResources)
        {
            fn(resource);
        }
    }

    void clearResources()
    {
        for (size_t i = 0; i < m_inlineSize; ++i)
        {
            m_inlineResources[i] = {};
        }
        m_inlineSize = 0;
        m_overflowResources.clear();
        m_numInlineSyncResources = 0;
        m_overflowSyncResources.clear();
        m_numSubmissionMutexes = 0;
        m_overflowSubmissionMutexes.clear();
    }

    std::shared_ptr<nvcvpy::priv::Stream> m_stream;
#ifdef NB_FREE_THREADED
    nvcvpy::priv::SubmissionLocks m_submissionLocks;
    nvcvpy::priv::PythonOwnerBag  m_pythonOwners;
#else
    // A displaced owner's finalizer may submit conflicting work. Retain it
    // until our consumer is enqueued and every submission lease is released.
    nvcvpy::priv::StreamOwnerBag m_retiredStreamOwners;
#endif
    int                                                   m_submissionDevice = -1;
    std::array<ResourceOwner, kInlineCapacity>            m_inlineResources;
    size_t                                                m_inlineSize = 0;
    std::vector<ResourceOwner>                            m_overflowResources;
    std::array<nvcvpy::priv::Resource *, kInlineCapacity> m_inlineSyncResources{};
    size_t                                                m_numInlineSyncResources = 0;
    std::vector<nvcvpy::priv::Resource *>                 m_overflowSyncResources;
    std::array<std::mutex *, kInlineCapacity>             m_inlineSubmissionMutexes{};
    size_t                                                m_numSubmissionMutexes = 0;
    std::vector<std::mutex *>                             m_overflowSubmissionMutexes;
#ifdef NB_FREE_THREADED
    SubmissionLease              m_submissionLease;
    std::vector<SubmissionLease> m_additionalSubmissionLeases;
#endif
    bool m_submitted = false;
    bool m_finished  = false;
};

} // namespace cvcudapy

#endif // CVCUDA_PYTHON_NATIVE_RESOURCE_GUARD_HPP
