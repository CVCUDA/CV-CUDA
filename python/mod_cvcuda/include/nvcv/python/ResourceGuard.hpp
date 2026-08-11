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

namespace nvcvpy {

namespace py = pybind11;

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
    }

    ResourceGuard &add(LockMode mode, std::initializer_list<std::reference_wrapper<const Resource>> resources)
    {
        py::object pyLockMode;
        switch (mode)
        {
        case LockMode::LOCK_MODE_NONE:
            pyLockMode = py::str("");
            break;
        case LockMode::LOCK_MODE_READ:
            pyLockMode = py::str("r");
            break;
        case LockMode::LOCK_MODE_WRITE:
            pyLockMode = py::str("w");
            break;
        case LockMode::LOCK_MODE_READWRITE:
            pyLockMode = py::str("rw");
            break;
        }

        // Just append; sync is deferred to run() (or commit() in the legacy
        // path used by out-of-tree consumers that haven't migrated).
        for (const std::reference_wrapper<const Resource> &r : resources)
        {
            py::object pyRes = py::reinterpret_borrow<py::object>(r.get().ptr());
            m_resourcesPerLockMode.append(std::make_pair(pyLockMode, std::move(pyRes)));
        }

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
        // The C API callback reports failures only through a pending Python
        // error (its implementation traps all C++ exceptions), so an explicit
        // check needs no general catch clause here.
        capi().Resources_SubmitSyncOnly(m_pyStream.ptr(), m_resourcesPerLockMode.ptr());
        if (PyErr_Occurred())
        {
            // The callable never ran, so no consumer work needs a lifetime
            // hold.  Marking the guard finished stops the destructor from
            // retrying the failed synchronization through the legacy path.
            m_finished = true;
            throw py::error_already_set();
        }

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
        // Terminal before fallible code: makes commit() idempotent and
        // prevents a destructor retry during unwind.
        m_finished = true;
        try
        {
            if (m_synced)
            {
                // Sync was performed by run() before the kernel; only the hold
                // remains.  Skips redundant per-resource submitSync work.
                capi().Stream_HoldResources(m_pyStream.ptr(), m_resourcesPerLockMode.ptr());
            }
            else
            {
                // Legacy path for guards that didn't use run() — typically
                // out-of-tree code compiled against an older header.  This
                // still syncs and holds in one call, but the syncs are queued
                // AFTER the consumer kernel and do not protect it from
                // external-producer races.  In-tree cvcuda ops should use
                // run().
                capi().Resources_SyncAndHold(m_pyStream.ptr(), m_resourcesPerLockMode.ptr());
            }
            CheckCAPIError();
        }
        catch (...)
        {
            // The hold is what keeps resources alive until the submitted
            // kernel completes; releasing the references now could free GPU
            // memory the kernel still uses.  Draining the stream proves every
            // resource idle.  The legacy combined call cannot distinguish a
            // producer-sync failure from a hold failure, so no drain of this
            // stream proves safety there — retain the references instead.
            if (!m_synced || !drainStreamNoThrow())
            {
                quarantineNoThrow();
            }
            throw;
        }
    }

private:
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

    bool drainStreamNoThrow() noexcept
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
        // Intentionally leak one reference to the list and stream.  This path
        // is only reached when completion cannot be proven; releasing or
        // reusing a GPU resource that may still be active would be unsafe.
        (void)m_resourcesPerLockMode.release();
        (void)m_pyStream.release();
    }

    py::object m_pyStream;
    py::object m_pyLockMode;
    py::list   m_resourcesPerLockMode;
    bool       m_synced   = false;
    bool       m_finished = false;
};

} // namespace nvcvpy

#endif // NVCV_PYTHON_RESOURCE_GUARD_HPP
