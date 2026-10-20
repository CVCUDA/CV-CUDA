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

#ifndef NVCV_PYTHON_RESOURCE_GUARD_HPP
#define NVCV_PYTHON_RESOURCE_GUARD_HPP

#include "CAPI.hpp"
#include "LockMode.hpp"
#include "Resource.hpp"
#include "Stream.hpp"

#include <cstdio>
#include <vector>

namespace nvcvpy {

namespace py = nanobind;

class ResourceGuard
{
public:
    explicit ResourceGuard(Stream &stream)
        : m_pyStream(py::reinterpret_borrow<py::object>(stream.ptr()))
    {
    }

    ResourceGuard(const ResourceGuard &)            = delete;
    ResourceGuard(ResourceGuard &&)                 = delete;
    ResourceGuard &operator=(const ResourceGuard &) = delete;
    ResourceGuard &operator=(ResourceGuard &&)      = delete;

    ~ResourceGuard()
    {
        finishNoThrow();
        releaseRefs();
    }

    ResourceGuard &add([[maybe_unused]] LockMode                                     mode,
                       std::initializer_list<std::reference_wrapper<const Resource>> resources)
    {
#ifdef Py_GIL_DISABLED
        if (m_submission || m_finished)
            throw std::logic_error("Cannot add resources after submission has started");
        m_modes.reserve(m_resources.size() + resources.size());
        m_resources.reserve(m_resources.size() + resources.size());
#endif
        // Just append; sync is deferred to run() (or commit() in the legacy
        // path used by out-of-tree consumers that haven't migrated).
        for (const std::reference_wrapper<const Resource> &r : resources)
        {
            PyObject *obj = r.get().ptr();
            // Append before taking ownership: vector growth may throw, whereas
            // Py_INCREF cannot. This keeps the reference count consistent with
            // the vector's contents while letting vector keep its geometric
            // growth policy across add() calls.
            m_resources.push_back(obj);
#ifdef Py_GIL_DISABLED
            m_modes.push_back(mode);
#endif
            Py_INCREF(obj);
        }

#ifdef Py_GIL_DISABLED
        if (m_running && !m_resources.empty())
            beginSubmission();
#endif
        return *this;
    }

    // Run the consumer's kernel-submitting callable with sync barriers
    // already in place.  This inserts cudaStreamWaitEvent on the consumer
    // stream for every previously-add()'d resource BEFORE invoking the
    // callable, so the kernel(s) the callable queues are guaranteed to see
    // the producer-stream work complete.  The hold half (Stream_HoldResources)
    // runs immediately after the callable, or during exception cleanup if the
    // callable throws after partially submitting work.
    //
    // Use this in op shims:
    //
    //     ResourceGuard guard(*pstream);
    //     guard.add(LOCK_MODE_READ,  {input});
    //     guard.add(LOCK_MODE_WRITE, {output});
    //     guard.add(LOCK_MODE_NONE,  {*op});
    //     guard.run([&]() { op->submit(pstream->cudaHandle(), input, output); });
    //
    // Calling run() before op->submit() is what makes sync barriers effective
    // — `cudaStreamWaitEvent` only gates commands enqueued AFTER it on the
    // same stream, so it must precede the kernel.  The previous pattern
    // (sync-on-destruction) inserted barriers behind the kernel and did not
    // protect it.
    template<class F>
    void run(F &&fn)
    {
#ifdef Py_GIL_DISABLED
        if (m_running || m_finished)
            throw std::logic_error("ResourceGuard can only be submitted once");
        m_running = true;
        if (!m_resources.empty())
            beginSubmission();
#else
        if (!m_resources.empty())
        {
            // The C API callback reports failures only through a pending Python
            // error (its implementation traps all C++ exceptions), so an explicit
            // check needs no general catch clause here.
            capi().Resources_SubmitSyncOnlyRaw(m_pyStream.ptr(), m_resources.data(), m_resources.size());
            if (PyErr_Occurred())
            {
                // The callable never ran, so no consumer work needs a lifetime
                // hold.  Marking the guard finished stops the destructor from
                // retrying the failed synchronization through the legacy path.
                m_finished = true;
                throw py::error_already_set();
            }
        }
#endif

        // Set before invoking the callable: if it throws after (partially)
        // submitting work, unwind runs ~ResourceGuard, which installs the
        // hold while keeping the callable's exception primary.
        m_synced = true;
        std::forward<F>(fn)();

        // Finalize on the normal path so a hold failure reaches Python
        // instead of being swallowed by the noexcept destructor.
        commit();
    }

    void commit()
    {
        if (m_finished)
        {
            return;
        }
        if (m_resources.empty())
        {
            m_finished = true;
            return;
        }
        // Terminal before fallible code: makes commit() idempotent and
        // prevents a destructor retry during unwind.
        m_finished = true;
#ifdef Py_GIL_DISABLED
        if (m_submission)
        {
            void *submission = m_submission;
            m_submission     = nullptr;
            submissionCapi().End(submission);
            CheckCAPIError();
            return;
        }
#endif
        bool canDrain = m_synced;
        try
        {
            if (m_synced)
            {
                // Sync was performed by run() before the kernel; only the hold
                // remains.  Skips redundant per-resource submitSync work.
                capi().Stream_HoldResourcesRaw(m_pyStream.ptr(), m_resources.data(), m_resources.size());
                CheckCAPIError();
            }
            else
            {
                // Legacy path for guards that didn't use run() — typically
                // out-of-tree code compiled against an older header.  This
                // still syncs and holds in one call, but the syncs are queued
                // AFTER the consumer kernel and do not protect it from
                // external-producer races.  In-tree cvcuda ops should use
                // run().
                capi().Resources_SubmitSyncOnlyRaw(m_pyStream.ptr(), m_resources.data(), m_resources.size());
                CheckCAPIError();
                // Every producer dependency is now ordered on this stream, so
                // if the hold below fails, draining it proves both the
                // producers and the already-submitted consumer work complete.
                canDrain = true;
                capi().Stream_HoldResourcesRaw(m_pyStream.ptr(), m_resources.data(), m_resources.size());
                CheckCAPIError();
            }
        }
        catch (...)
        {
            // The hold is what keeps resources alive until the submitted
            // kernel completes; releasing the references now could free GPU
            // memory the kernel still uses.  Draining the stream proves every
            // resource idle.  The legacy combined call cannot distinguish a
            // producer-sync failure from a hold failure, so no drain of this
            // stream proves safety there — retain the references instead.
            if (!canDrain || !drainStreamNoThrow())
            {
                quarantineNoThrow();
            }
            throw;
        }
    }

private:
#ifdef Py_GIL_DISABLED
    void beginSubmission()
    {
        m_submission = submissionCapi().Begin(m_pyStream.ptr(), m_resources.data(), m_modes.data(), m_resources.size());
        if (!m_submission)
        {
            m_finished = true;
            CheckCAPIError();
            throw std::runtime_error("Unable to begin resource submission");
        }
    }
#endif
    // Saves any pending Python error at construction and re-instates it at
    // scope exit, discarding errors raised in between.  Keeps the primary
    // exception authoritative across cleanup that may set secondary errors.
    class PyErrPreserver
    {
    public:
        PyErrPreserver() noexcept
        {
            PyErr_Fetch(&m_type, &m_value, &m_traceback);
        }

        ~PyErrPreserver() noexcept
        {
            PyErr_Clear();
            PyErr_Restore(m_type, m_value, m_traceback);
        }

        PyErrPreserver(const PyErrPreserver &)            = delete;
        PyErrPreserver &operator=(const PyErrPreserver &) = delete;

    private:
        PyObject *m_type      = nullptr;
        PyObject *m_value     = nullptr;
        PyObject *m_traceback = nullptr;
    };

    void finishNoThrow() noexcept
    {
        if (m_finished)
        {
            return;
        }

        // Preserve an exception already being propagated by the submitted
        // callable; cleanup failures are secondary and must not replace it.
        PyErrPreserver preserver;
        try
        {
            this->commit();
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
        bool           drained = false;
        try
        {
            cudaStream_t stream = capi().Stream_GetCudaHandle(m_pyStream.ptr());
            if (!PyErr_Occurred())
            {
                cudaError_t status;
                {
                    // A foreign host callback queued on this stream may need
                    // the GIL in order to finish.
                    py::gil_scoped_release release;
                    status = cudaStreamSynchronize(stream);
                }

                if (status == cudaSuccess)
                {
                    drained = true;
                }
                else
                {
                    std::fprintf(stderr, "[cvcuda] ResourceGuard: stream drain failed (%s); quarantining resources\n",
                                 cudaGetErrorName(status));
                    cudaGetLastError();
                }
            }
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "[cvcuda] ResourceGuard: stream drain threw: %s; quarantining resources\n", e.what());
        }
        catch (...)
        {
            std::fprintf(stderr, "[cvcuda] ResourceGuard: stream drain threw; quarantining resources\n");
        }
        return drained;
    }

    void quarantineNoThrow() noexcept
    {
        // Intentionally leak one reference per resource and the stream. This path
        // is only reached when completion cannot be proven; releasing or
        // reusing a GPU resource that may still be active would be unsafe.
        // Clearing without decref'ing is what makes the leak permanent, and it
        // also stops the destructor from handing the references back.
        m_resources.clear();
        (void)m_pyStream.release();
    }

    // Drops the references add() took.  Safe to call once the resources are
    // either held by the stream (commit succeeded) or deliberately quarantined.
    void releaseRefs() noexcept
    {
        for (PyObject *obj : m_resources)
        {
            Py_DECREF(obj);
        }
        m_resources.clear();
    }

    py::object m_pyStream;
#ifdef Py_GIL_DISABLED
    std::vector<uint8_t> m_modes;
    void                *m_submission = nullptr;
    bool                 m_running    = false;
#endif
    std::vector<PyObject *> m_resources;
    bool                    m_synced   = false;
    bool                    m_finished = false;
};

} // namespace nvcvpy

#endif // NVCV_PYTHON_RESOURCE_GUARD_HPP
