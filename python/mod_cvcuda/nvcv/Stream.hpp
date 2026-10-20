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

#ifndef NVCV_PYTHON_PRIV_STREAM_HPP
#define NVCV_PYTHON_PRIV_STREAM_HPP

#include "Cache.hpp"
#include "Object.hpp"

#include <common/Binding.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nvcvpy::priv {

class Resource;

// Owns the validity of a CUDA stream handle independently of the Python Stream
// wrapper it came from, so a resource or an exported CAI/DLPack view can keep
// the handle alive after that wrapper is gone.
class StreamHandleOwner final
{
public:
    StreamHandleOwner(cudaStream_t handle, int device, bool owns, py::object owner = {},
                      bool externalOwnerRetainsLifetime = false);
    ~StreamHandleOwner();

    // Sole ownership of the handle: it is always held through a shared_ptr, and
    // copying one would destroy the stream more than once.
    StreamHandleOwner(const StreamHandleOwner &)            = delete;
    StreamHandleOwner(StreamHandleOwner &&)                 = delete;
    StreamHandleOwner &operator=(const StreamHandleOwner &) = delete;
    StreamHandleOwner &operator=(StreamHandleOwner &&)      = delete;

    cudaStream_t handle() const
    {
        return m_handle;
    }

    bool retainsHandleLifetime() const
    {
        return m_owns || m_externalOwnerRetainsLifetime;
    }

    bool hasPythonOwner() const noexcept
    {
        return m_owner.is_valid();
    }

    bool sharesQueueWith(const StreamHandleOwner &other) const noexcept;
    bool isCurrentThreadQueue() const noexcept;
#ifdef NB_FREE_THREADED
    cudaEvent_t perThreadCompletion() const
    {
        return m_perThreadCompletion;
    }

    cudaError_t capturePerThreadCompletion() const
    {
        return m_perThreadCompletion ? cudaEventRecord(m_perThreadCompletion, m_handle) : cudaSuccess;
    }
#endif

private:
    cudaStream_t m_handle;
    int          m_device;
    bool         m_owns;
    py::object   m_owner;
    bool         m_externalOwnerRetainsLifetime;
    uint64_t     m_creatorThreadIdentity;
#ifdef NB_FREE_THREADED
    cudaEvent_t m_perThreadCompletion = nullptr;
#endif
};

class IExternalStream
{
public:
    virtual ~IExternalStream() = default;

    virtual cudaStream_t handle() const                = 0;
    virtual py::object   wrappedObject() const         = 0;
    virtual bool         retainsHandleLifetime() const = 0;
};

using ResourceBag    = std::vector<std::shared_ptr<Resource>>;
using StreamOwnerBag = std::vector<std::shared_ptr<const StreamHandleOwner>>;
using PythonOwnerBag = std::vector<py::object>;

// Mutable owner segments transferred by the native operator guard. Sources
// are moved from only after release bookkeeping can no longer fail, so the
// guard retains every owner if holdResources throws.
struct ResourceOwnerTransfer
{
    std::shared_ptr<Resource> *inlineData   = nullptr;
    size_t                     inlineSize   = 0;
    std::shared_ptr<Resource> *overflowData = nullptr;
    size_t                     overflowSize = 0;
};

class NB_EXPORT Stream : public Object
{
public:
    static void Export(py::module_ &m);

    static Stream &Current();

    static std::shared_ptr<Stream> Create();

    ~Stream() override;

    std::shared_ptr<Stream>       sharedStream();
    std::shared_ptr<const Stream> sharedStream() const;

    void activate();
    void deactivate(py::handle exc_type, py::handle exc_value, py::handle exc_tb) const;

    void holdResources(ResourceBag usedResources, int submissionDevice);
    void holdResources(ResourceBag usedResources, PythonOwnerBag pythonOwners, int submissionDevice);
    void holdResources(const ResourceOwnerTransfer &usedResources, std::shared_ptr<Stream> &streamOwner,
                       int submissionDevice);
#ifdef NB_FREE_THREADED
    void holdResources(const ResourceOwnerTransfer &usedResources, std::shared_ptr<Stream> &streamOwner,
                       PythonOwnerBag &pythonOwners, int submissionDevice);
#endif

    void ensureNotCapturing(int submissionDevice) const;

    // Blocks until every resource batch queued by any stream has been
    // released. Used by cache clear and teardown, which must not free a
    // buffer a queued kernel can still read.
    static void DrainAllPendingReleases();

    // Returns true when every live owner other than `resource` itself and its
    // cache entry belongs to the pending-release queue. Cache lookup uses this
    // to reuse a Tensor without a host wait. ResourceGuard preserves ordering
    // on the next submission; below the limit, lookup additionally checks the
    // resource's synchronization history to avoid new cross-stream waits.
    static bool IsPendingOnly(const std::shared_ptr<Object> &resourceOwner);

    int64_t GetSizeInBytes() const;

    void                                      sync();
    void                                      wait_stream(std::shared_ptr<Stream> other);
    cudaStream_t                              handle() const;
    int                                       deviceId() const;
    int                                       submissionDeviceId() const;
    bool                                      sharesQueueWith(const Stream &other) const;
    const std::shared_ptr<StreamHandleOwner> &handleOwner() const;
    uint64_t                                  id() const;

    const IKey &key() const
    {
        return m_key;
    }

    // Returns the cuda handle in python
    intptr_t pyhandle() const;

    explicit Stream(IExternalStream &extStream);

    friend std::ostream &operator<<(std::ostream &out, const Stream &stream);

private:
    Stream(Stream &&) = delete;
    Stream();

    int64_t doComputeSizeInBytes() const;

    class Key final : public IKey
    {
    private:
        size_t doGetHash() const override;
        bool   doIsCompatible(const IKey &that) const override;
    };

    friend class Resource;
    void destroy();
    void waitFor(cudaStream_t producer);
    void validateCurrentThread() const;

    Key                                  m_key;
    uint64_t                             m_id     = 0;
    bool                                 m_owns   = false;
    cudaStream_t                         m_handle = nullptr;
    std::shared_ptr<StreamHandleOwner>   m_handleOwner;
    std::unordered_map<int, cudaEvent_t> m_events;
    std::mutex                           m_eventMutex;
    int64_t                              m_size_inbytes = -1;

    static std::weak_ptr<Stream> m_defaultStream;

    static void CleanupAtExit(const std::shared_ptr<Stream> &globalStream);
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_STREAM_HPP
