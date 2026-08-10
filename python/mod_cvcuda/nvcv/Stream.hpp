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

#include <cuda_runtime.h>
#include <nvcv/python/LockMode.hpp>

#include <atomic>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace nvcvpy::priv {

class Resource;

class IExternalStream
{
public:
    virtual ~IExternalStream() = default;

    virtual cudaStream_t handle() const        = 0;
    virtual py::object   wrappedObject() const = 0;
};

using LockResources = std::unordered_multimap<LockMode, std::shared_ptr<const Resource>>;

class PYBIND11_EXPORT Stream : public CacheItem
{
public:
    static void Export(py::module &m);

    static Stream &Current();

    static std::shared_ptr<Stream> Create();

    ~Stream() override;

    std::shared_ptr<Stream>       sharedStream();
    std::shared_ptr<const Stream> sharedStream() const;

    void activate();
    void deactivate(py::object exc_type, py::object exc_value, py::object exc_tb) const;

    void holdResources(LockResources usedResources);

    static void SynchronizeAndClearGCBag();

    int64_t GetSizeInBytes() const override;

    void         sync();
    void         wait_stream(std::shared_ptr<Stream> other);
    cudaStream_t handle() const;
    int          deviceId() const;

    // Returns the cuda handle in python
    intptr_t pyhandle() const;

    explicit Stream(IExternalStream &extStream);

    friend std::ostream &operator<<(std::ostream &out, const Stream &stream);

private:
    Stream(Stream &&) = delete;
    Stream();

    int64_t doComputeSizeInBytes() const;

    // Singleton access to the auxiliary CUDA stream

    class Key final : public IKey
    {
    private:
        size_t doGetHash() const override;
        bool   doIsCompatible(const IKey &that) const override;
    };

    const Key &key() const override
    {
        return m_key;
    }

    void        destroy();
    cudaEvent_t getEvent();

    Key                                  m_key;
    bool                                 m_owns   = false;
    cudaStream_t                         m_handle = nullptr;
    std::unordered_map<int, cudaEvent_t> m_events;
    std::shared_mutex                    m_eventMutex;
    py::object                           m_wrappedObj;
    int64_t                              m_size_inbytes = -1;

    // REVISIT: these don't have to be static members, but simply defined
    // as local entities in Stream.cpp, thereby minimizing code coupling and
    // unnecessary rebuilds.

    //per-device aux streams and protection. this is a bit overkill
    //for now as python is single threaded, but it is a good practice
    static std::shared_mutex                     m_auxStreamMutex;
    static std::atomic<int>                      m_instanceCount;
    static std::unordered_map<int, cudaStream_t> m_auxStreams;

    static void         incrementInstanceCount();
    static int          decrementInstanceCount();
    static cudaStream_t GetAuxStream();
    static void         SyncAuxStream();
    static void         CleanupAtExit(const std::shared_ptr<Stream> &globalStream);

    // Adds the object to the garbage-collector's bag to delay its destruction
    // until it's safe to destroy it.
    // Safe here means: not from a thread that is processing tasks in a cuda stream,
    // i.e., not inside the callback given to cudaStreamAddCallback. If this happens,
    // cuda calls will be made from within the callback, and CUDA docs prohibit it.
    struct HostFunctionClosure;
    static void AddToGCBag(std::unique_ptr<HostFunctionClosure> obj);

    // Clear the garbage-collector's bag. It's supposed to be called by
    // functions that
    static void ClearGCBag();

    using GCBag = std::vector<std::unique_ptr<HostFunctionClosure>>;
    static std::mutex m_gcMutex;

    static GCBag &GetGCBag();
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_STREAM_HPP
