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

#ifndef NVCV_PYTHON_PRIV_RESOURCE_HPP
#define NVCV_PYTHON_PRIV_RESOURCE_HPP

#include "Object.hpp"
#include "Stream.hpp"

#include <nvcv/detail/CudaFwd.h>
#include <nvcv/python/LockMode.hpp>
#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>
#include <unordered_map>

// fwd declaration from driver_types.h (cudaStream_t comes via Stream.hpp → cuda_runtime.h)
struct CUevent_st;
using cudaEvent_t = CUevent_st *;

namespace nvcvpy::priv {
namespace py = pybind11;

class ImageBatchVarShape;

/**
 * @brief A class representing a CUDA resource.
 *
 * This class encapsulates a CUDA resource and provides methods for synchronization
 * with CUDA streams.
 */
class PYBIND11_EXPORT Resource : public virtual Object // NOSONAR: CUDA resources share one Object base.
{
public:
    /**
     * @brief Destructor.
     */
    ~Resource() override;

    /**
     * @brief Export the Resource class to Python.
     *
     * Must be called before any class that inherits from Resource (e.g.
     * Container).  It does NOT bind methods whose signatures reference
     * Stream — those are added later by ExportStreamMethods so pybind11
     * emits the proper Python typename in the stubs.
     *
     * @param m The Python module to export the class to.
     */
    static void Export(py::module &m);

    /**
     * @brief Bind Resource methods that reference Stream.
     *
     * Must be called AFTER Stream::Export so pybind11 can resolve Stream
     * to its Python type in the method signature (otherwise the stubs
     * leak the raw C++ typename `nvcvpy::priv::Stream`).
     *
     * @param m The Python module.
     */
    static void ExportStreamMethods(py::module &m);

    /**
     * @brief Get the unique identifier of the resource.
     *
     * @return uint64_t The unique identifier of the resource.
     */
    uint64_t id() const;

    /**
     * @brief Submit the resource for synchronization with a CUDA stream.
     *
     * This method synchronizes the resource with the specified CUDA stream.
     *
     * @param stream The CUDA stream to synchronize with.
     */
    virtual void submitSync(Stream &stream);

    /**
     * @brief Seed the "last stream" tracking with a raw CUDA stream handle.
     *
     * Used when wrapping an external buffer (e.g., a cupy/torch tensor) that
     * advertises its producer stream via `__cuda_array_interface__["stream"]`.
     * The next call to `submitSync` will insert the appropriate event-wait
     * (or cross-device `cudaStreamSynchronize`) before the cvcuda op runs.
     *
     * No-op if the resource already has a last-stream set (subsequent ops
     * own the sync chain).
     *
     * @param handle The producer CUDA stream handle.
     * @param device The CUDA device the producer stream lives on.
     */
    void seedLastStream(cudaStream_t handle, int device);

    /**
     * @brief Get the handle of the stream this resource is currently pending on.
     *
     * Returns 0 if no prior stream has been recorded.  Used by the CAI /
     * DLPack exporters to populate the outgoing "stream" field so downstream
     * consumers know which stream they must synchronize with.
     */
    cudaStream_t getLastStreamHandle() const;

    /**
     * @brief Get a shared pointer to this resource.
     *
     * @return std::shared_ptr<Resource> A shared pointer to this resource.
     */
    std::shared_ptr<Resource> shared_from_this();

    /**
     * @brief Get a shared pointer to this const resource.
     *
     * @return std::shared_ptr<const Resource> A shared pointer to this const resource.
     */
    std::shared_ptr<const Resource> shared_from_this() const;

protected:
    Resource();

    /**
     * @brief Clear stream ownership before a cached wrapper is rebound to new storage.
     *
     * The caller must guarantee that the wrapper is no longer in flight on its
     * recorded stream, as the cache does before returning a reusable wrapper.
     */
    void resetLastStreamForRebind();

private:
    friend class ImageBatchVarShape;

    struct SyncState
    {
        cudaStream_t stream = nullptr;
        int          device = -1;
    };

    SyncState syncState() const;
    bool      submitSyncThrough(Stream &stream, SyncState synchronizedState);

    uint64_t                                     m_id;         /**< The unique identifier of the resource. */
    std::unordered_map<int, cudaEvent_t>         m_events;     /**< Per-device CUDA events for synchronization. */
    std::optional<std::shared_ptr<const Stream>> m_lastStream; /**< Cache the last stream used for this resource. */
    /**
     * Raw handle of the last stream when it is an external (non-cvcuda) stream
     * seeded via `seedLastStream` (e.g., from CAI `stream` field).  Mutually
     * exclusive with `m_lastStream`: at most one is set.
     */
    cudaStream_t                                 m_lastStreamHandle = nullptr;
    int                                          m_lastDevice       = -1; /**< Device ID of the last stream. */
    mutable std::mutex                           m_mtx; /**< Lock reads and writes to the resource.  */

    cudaEvent_t event(); /**< Returns the CUDA event for the current device. */
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_RESOURCE_HPP
