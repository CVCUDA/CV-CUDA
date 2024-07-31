/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_map>
#include <vector>

namespace nvcvpy::priv {

class Resource;

class IExternalStream
{
public:
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

    virtual ~Stream();

    std::shared_ptr<Stream>       shared_from_this();
    std::shared_ptr<const Stream> shared_from_this() const;

    void activate();
    void deactivate(py::object exc_type, py::object exc_value, py::object exc_tb);

    void holdResources(LockResources usedResources);

    int64_t GetSizeInBytes() const override;

    void         sync();
    cudaStream_t handle() const;

    // Returns the cuda handle in python
    intptr_t pyhandle() const;

    Stream(IExternalStream &extStream);

    friend std::ostream &operator<<(std::ostream &out, const Stream &stream);

private:
    Stream(Stream &&) = delete;
    Stream();

    int64_t doComputeSizeInBytes();

    // Singleton access to the auxiliary CUDA stream

    class Key final : public IKey
    {
    private:
        virtual size_t doGetHash() const override;
        virtual bool   doIsCompatible(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        static Key key;
        return key;
    }

    void destroy();

    bool         m_owns   = false;
    cudaStream_t m_handle = nullptr;
    cudaEvent_t  m_event  = nullptr;
    py::object   m_wrappedObj;
    int64_t      m_size_inbytes = -1;

    // TODO: these don't have to be static members, but simply defined
    // as local entities in Stream.cpp, thereby minimizing code coupling and
    // unnecessary rebuilds.

    //singleton aux stream and protection. this a a bit overkill
    //for now as python is single threaded, but it is a good practice
    static std::mutex       m_auxStreamMutex;
    static std::atomic<int> m_instanceCount;
    static cudaStream_t     m_auxStream;

    static void          incrementInstanceCount();
    static int           decrementInstanceCount();
    static cudaStream_t &GetAuxStream();
    static void          SyncAuxStream();

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
