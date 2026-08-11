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

namespace nvcvpy::priv {

Resource::Resource()
{
    static uint64_t idnext = 0;

    m_id = idnext++;
}

Resource::~Resource()
{
    int savedDev = -1;
    util::CheckLog(cudaGetDevice(&savedDev));

    for (const auto &[dev, evt] : m_events)
    {
        util::CheckLog(cudaSetDevice(dev));
        util::CheckLog(cudaEventDestroy(evt));
    }

    if (savedDev >= 0)
    {
        util::CheckLog(cudaSetDevice(savedDev));
    }
}

uint64_t Resource::id() const
{
    return m_id;
}

cudaEvent_t Resource::event()
{
    int dev;
    util::CheckThrow(cudaGetDevice(&dev));

    auto it = m_events.find(dev);
    if (it == m_events.end())
    {
        cudaEvent_t evt = nullptr;
        util::CheckThrow(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        m_events.emplace(dev, evt);
        return evt;
    }
    return it->second;
}

void Resource::submitSync(Stream &stream)
{
    std::unique_lock lk(m_mtx);

    // Compute the "previous" stream handle.  Either a cvcuda-owned Stream
    // (m_lastStream) or a raw handle seeded from an external producer via
    // seedLastStream() (m_lastStreamHandle).  Both being unset means this is
    // the first submission for this resource.
    cudaStream_t prevHandle = nullptr;
    if (m_lastStream.has_value())
    {
        prevHandle = m_lastStream.value()->handle();
    }
    else if (m_lastStreamHandle != nullptr)
    {
        prevHandle = m_lastStreamHandle;
    }

    // Fast path: previously bound to the same stream → no sync work, and
    // no need to query the current device. Streams are sequential, so the
    // last operation on this stream is already ordered before whatever the
    // caller is about to enqueue. This path runs once per resource per Python
    // op call after the first, so keeping it cheap (mutex + pointer compare,
    // no CUDA driver round-trip) is the difference between an op wrapper
    // adding ~5µs vs ~30µs of constant overhead per call. cudaGetDevice
    // costs a few µs each invocation; with 5-7 resources locked per op,
    // skipping it on the fast path saves 25-35µs per gamma_contrast_into /
    // erase / brightness_contrast / etc.
    if (prevHandle != nullptr && prevHandle == stream.handle())
    {
        return;
    }

    // Slow path: need the current device for either first-binding state or
    // cross-stream/cross-device sync logic.
    int curDev;
    util::CheckThrow(cudaGetDevice(&curDev));

    // First submission for this resource — take ownership on the current
    // stream and return with no sync work.
    if (prevHandle == nullptr)
    {
        m_lastStream.emplace(stream.sharedStream());
        m_lastDevice = curDev;
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
    // take the event-based path because m_lastStream is set after this.
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
        // Same device — use the efficient event-based synchronization.
        // Write event on the old stream, the new stream will wait for it.
        util::CheckThrow(cudaEventRecord(event(), prevHandle));
        util::CheckThrow(cudaStreamWaitEvent(stream.handle(), event()));
    }

    // update the last stream since we changed streams
    m_lastStream.reset();
    m_lastStreamHandle = nullptr;
    m_lastStream.emplace(stream.sharedStream());
    m_lastDevice = curDev;
}

Resource::SyncState Resource::syncState() const
{
    std::unique_lock lk(m_mtx);

    if (m_lastStream.has_value())
    {
        return {m_lastStream.value()->handle(), m_lastDevice};
    }
    return {m_lastStreamHandle, m_lastDevice};
}

bool Resource::submitSyncThrough(Stream &stream, SyncState synchronizedState)
{
    std::unique_lock lk(m_mtx);

    SyncState currentState{m_lastStreamHandle, m_lastDevice};
    if (m_lastStream.has_value())
    {
        currentState.stream = m_lastStream.value()->handle();
    }
    if (currentState.stream != synchronizedState.stream || currentState.device != synchronizedState.device)
    {
        return false;
    }
    if (currentState.stream == stream.handle())
    {
        return true;
    }

    // The parent already established this stream dependency, so only transfer
    // ownership; recording another event per child would duplicate that wait.
    m_lastStream.reset();
    m_lastStreamHandle = nullptr;
    m_lastStream.emplace(stream.sharedStream());
    m_lastDevice = stream.deviceId();
    return true;
}

void Resource::seedLastStream(cudaStream_t handle, int device)
{
    std::unique_lock lk(m_mtx);

    // If some stream ownership is already recorded, respect it — either we've
    // been seeded before, or a cvcuda op has already claimed this resource.
    if (m_lastStream.has_value() || m_lastStreamHandle != nullptr)
    {
        return;
    }

    m_lastStreamHandle = handle;
    m_lastDevice       = device;
}

void Resource::resetLastStreamForRebind()
{
    std::unique_lock lk(m_mtx);

    m_lastStream.reset();
    m_lastStreamHandle = nullptr;
    m_lastDevice       = -1;
}

cudaStream_t Resource::getLastStreamHandle() const
{
    std::unique_lock lk(m_mtx);

    if (m_lastStream.has_value())
    {
        return m_lastStream.value()->handle();
    }
    return m_lastStreamHandle;
}

std::shared_ptr<Resource> Resource::shared_from_this()
{
    return std::dynamic_pointer_cast<Resource>(Object::shared_from_this());
}

std::shared_ptr<const Resource> Resource::shared_from_this() const
{
    return std::dynamic_pointer_cast<const Resource>(Object::shared_from_this());
}

void Resource::Export(py::module &m)
{
    py::class_<Resource, std::shared_ptr<Resource>>(m, "Resource", "Resource")
        .def_property_readonly("id", &Resource::id, "Unique resource instance identifier");
}

void Resource::ExportStreamMethods(py::module &m)
{
    // Re-open the already-registered Resource class and add the stream-using
    // method now that Stream is a known pybind11 type.  Without this
    // deferral, pybind11 would emit the raw C++ typename
    // `nvcvpy::priv::Stream` in the signature.
    py::object cls = m.attr("Resource");
    cls.attr("submitStreamSync")
        = py::cpp_function(&Resource::submitSync, py::name("submitStreamSync"), py::is_method(cls), py::arg("stream"),
                           "Syncs object on new Stream");
}

} // namespace nvcvpy::priv
