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
#include "Submission.hpp"

#include <common/Binding.hpp>
#include <nvcv/detail/CudaFwd.h>
#include <nvcv/python/LockMode.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace nvcvpy::priv {
namespace py = nanobind;

class ImageBatchVarShape;
class BatchMutex;
class Cache;
class CacheItem;

struct RetainedMemory
{
    uint64_t knownBytes       = 0;
    size_t   unknownResources = 0;
};

/**
 * @brief A class representing a CUDA resource.
 *
 * This class encapsulates a CUDA resource and provides methods for synchronization
 * with CUDA streams.
 */
class NB_EXPORT Resource : public Object
{
public:
    struct ExportStreamState
    {
        std::optional<cudaStream_t>              handle;
        std::shared_ptr<const StreamHandleOwner> owner;
#ifdef NB_FREE_THREADED
        SubmissionStates states;
#endif
    };

    /**
     * @brief Destructor.
     */
    ~Resource() override;

    /**
     * @brief Export the Resource class to Python.
     *
     * Must be called after Stream::Export and before any class that inherits
     * from Resource (e.g. Container).
     *
     * @param m The Python module to export the class to.
     */
    static void Export(py::module_ &m);

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
    virtual void submitSync(Stream &stream, int submissionDevice, StreamOwnerBag *retiredOwners = nullptr);

    /** Describes storage whose lifetime is extended by retaining this owner. */
    virtual RetainedMemory retainedMemory() const;

    /** Mutable cached operator state opts into enqueue-time host serialization. */
    virtual std::mutex *submissionMutex() const;

    // Ordinary cache reuse is safe for a Resource with no wrapper owner.
    // Reusing one still owned only by the GPU release ledger requires an
    // explicit opt-in because CUDA ordering cannot protect mutable host state.
    virtual bool allowsCacheReuse(bool pendingRelease) const
    {
        return !pendingRelease;
    }

    // Requires an exclusive cache claim and pending-only ownership. Predicts
    // whether the next write can submit without adding a cross-stream wait.
    bool isOrderedOn(Stream &stream);

    // The global release queue records its ownership explicitly. Cache
    // pressure can then distinguish safe pending owners from quarantined or
    // deferred-cleanup owners without reconstructing the queue from Python
    // reference counts.
    virtual void            acquirePendingReleaseOwner() noexcept;
    virtual void            blockPendingReleaseOwner() noexcept;
    virtual void            releasePendingReleaseOwner(bool blocked) noexcept;
    void                    unblockPendingReleaseOwner() noexcept;
    virtual void            indexPendingResources() noexcept;
    std::optional<uint64_t> reusablePendingReleaseOwners() const;
    bool                    pendingReleaseOwnershipMatches(uint64_t snapshot) const;
    bool                    hasPendingReleaseOwners() const;

#ifdef NB_FREE_THREADED
    virtual BatchMutex *submissionBatchMutex() const
    {
        return nullptr;
    }

    // Batch descriptors retain their children; those visits only contribute access modes.
    using SubmissionVisitor = std::function<void(Resource &, LockMode, bool retain)>;

    virtual void visitSubmissionResources(LockMode mode, const SubmissionVisitor &visit)
    {
        visit(*this, mode, true);
    }

    const SubmissionStates &submissionStates() const
    {
        return m_submissionStates;
    }

    void shareSubmissionStates(const Resource &other)
    {
        m_submissionStates = other.m_submissionStates;
        m_readOnly         = other.m_readOnly;
    }

    void setSubmissionStates(SubmissionStates states)
    {
        std::sort(states.begin(), states.end(),
                  [](const auto &a, const auto &b) { return std::less<SubmissionState *>{}(a.get(), b.get()); });
        states.erase(std::unique(states.begin(), states.end()), states.end());
        m_submissionStates = std::move(states);
    }

    bool readOnly() const
    {
        return m_readOnly;
    }

    void setReadOnly(bool value)
    {
        m_readOnly = value;
    }
#endif

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

    /** @brief Snapshot the last stream handle and the owner that keeps it valid. */
    ExportStreamState exportStreamState() const;

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
    explicit Resource(bool createSubmissionState = true);

#ifndef NB_FREE_THREADED
    bool isExportStreamCurrent(const StreamHandleOwner *owner) const
    {
        return owner && m_synchronizedWith.load() == owner;
    }
#endif

    /**
     * @brief Clear stream ownership before a cached wrapper is rebound to new storage.
     *
     * The caller must guarantee that the wrapper is no longer in flight on its
     * recorded stream, as the cache does before returning a reusable wrapper.
     */
    void resetLastStreamForRebind();

private:
#ifdef NB_FREE_THREADED
    SubmissionStates m_submissionStates;
    bool             m_readOnly = false;
#endif
    friend class ImageBatchVarShape;

    struct SyncState
    {
        cudaStream_t                             stream = nullptr;
        int                                      device = -1;
        std::shared_ptr<const StreamHandleOwner> owner;
        bool                                     bound = false;
    };

    SyncState syncState() const;
    bool      submitSyncThrough(Stream &stream, const SyncState &synchronizedState, int submissionDevice,
                                StreamOwnerBag *retiredOwners = nullptr);
    bool      isSynchronizedWith(Stream &stream, int submissionDevice) const;
    void      cacheSynchronization(Stream &stream);

    // Caller holds m_mtx and keeps the returned owner alive until after unlocking.
    [[nodiscard]] std::shared_ptr<const StreamHandleOwner> replaceLastStreamOwner(Stream         &stream,
                                                                                  StreamOwnerBag *retiredOwners);

    // A null/default handle names the current device, so batch callers resolve
    // it once before walking their children.
    uint64_t                                 m_id; /**< The unique identifier of the resource. */
    std::shared_ptr<const StreamHandleOwner> m_lastStreamOwner;
    std::atomic<const StreamHandleOwner *>   m_synchronizedWith{nullptr};
    // Low/high halves count reusable/blocked release-queue owners. Keeping the
    // transition in one atomic prevents cache lookup from observing neither.
    std::atomic_uint64_t                     m_pendingReleaseOwnership{0};
    friend class Cache;
    // The owning cache index keeps this registration alive. Cache reads and
    // clears it under its shared mutex, before an index owner can be released.
    CacheItem         *m_cacheItem = nullptr;
    /**
     * Raw handle of the last stream when it is an external (non-cvcuda) stream
     * seeded via `seedLastStream` (e.g., from CAI `stream` field).  Mutually
     * exclusive with `m_lastStreamOwner`: at most one is set.
     */
    cudaStream_t       m_lastStreamHandle = nullptr;
    int                m_lastDevice       = -1; /**< Device ID of the last stream. */
    mutable std::mutex m_mtx;                   /**< Lock reads and writes to the resource.  */
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_RESOURCE_HPP
