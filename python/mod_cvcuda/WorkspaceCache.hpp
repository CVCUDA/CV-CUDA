/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PYTHON_WORKSPACE_CACHE_HPP
#define CVCUDA_PYTHON_WORKSPACE_CACHE_HPP

#include <common/CheckError.hpp>
#include <cvcuda/Workspace.hpp>
#include <cvcuda/util/Event.hpp>
#include <cvcuda/util/PerStreamCache.hpp>
#include <cvcuda/util/SimpleCache.hpp>
#include <cvcuda/util/StreamId.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <atomic>
#include <cassert>
#include <map>
#include <mutex>

namespace cvcudapy {

using WorkspaceMemDestructor_t = std::function<void(cvcuda::WorkspaceMem &)>;

enum class MemoryKind
{
    Host,
    Pinned,
    Cuda
};

template<MemoryKind kind>
class CachedWorkspaceMem : public cvcuda::WorkspaceMem
{
public:
    CachedWorkspaceMem()
        : cvcuda::WorkspaceMem({})
    {
        assert(data == nullptr);
        assert(ready == nullptr);
    }

    CachedWorkspaceMem(const cvcuda::WorkspaceMem &mem, WorkspaceMemDestructor_t destructor)
        : cvcuda::WorkspaceMem(mem)
        , m_destructor(std::move(destructor))
    {
    }

    CachedWorkspaceMem(CachedWorkspaceMem &&mem)
    {
        *this = std::move(mem);
    }

    CachedWorkspaceMem &operator=(CachedWorkspaceMem &&mem)
    {
        std::swap(wsMem(), mem.wsMem());
        std::swap(m_destructor, mem.m_destructor);
        mem.reset();
        return *this;
    }

    ~CachedWorkspaceMem()
    {
        reset();
    }

    void reset()
    {
        if (m_destructor)
        {
            m_destructor(*this);
            m_destructor = {};
        }
        wsMem() = {};
    }

    explicit operator bool() const noexcept
    {
        return data != nullptr;
    }

private:
    cvcuda::WorkspaceMem &wsMem() &
    {
        return static_cast<cvcuda::WorkspaceMem &>(*this);
    }

    const cvcuda::WorkspaceMem &wsMem() const &
    {
        return static_cast<const cvcuda::WorkspaceMem &>(*this);
    }

    WorkspaceMemDestructor_t m_destructor;
};

template<MemoryKind kind>
inline size_t StreamCachePayloadSize(const CachedWorkspaceMem<kind> &mem)
{
    return mem.req.size;
}

template<MemoryKind kind>
inline size_t StreamCachePayloadAlignment(const CachedWorkspaceMem<kind> &mem)
{
    return mem.req.alignment;
}

template<MemoryKind kind>
class WorkspaceMemCache
{
public:
    using Mem  = CachedWorkspaceMem<kind>;
    using Base = nvcv::util::PerStreamCache<Mem>;

    WorkspaceMemCache(nvcv::Allocator alloc, std::shared_ptr<nvcv::util::EventCache> eventCache)
        : m_alloc(std::move(alloc))
        , m_eventCache(std::move(eventCache))
    {
    }

    ~WorkspaceMemCache()
    {
        assert(m_outstandingAllocs == 0);
    }

    Mem get(cvcuda::WorkspaceMemRequirements req, std::optional<cudaStream_t> stream)
    {
        if (req.size == 0)
            return {};

        ++m_outstandingAllocs;
        auto opt = m_memCache.get(req.size, req.alignment, stream);
        if (opt)
            return std::move(opt).value();

        return create(req);
    }

    void put(Mem &&mem, std::optional<cudaStream_t> stream)
    {
        m_memCache.put(std::move(mem), stream);
        --m_outstandingAllocs;
    }

    void clear()
    {
        assert(m_outstandingAllocs == 0);
        m_memCache.purge();
    }

private:
    void *allocateMem(size_t size, size_t alignment) const
    {
        if constexpr (kind == MemoryKind::Host)
            return m_alloc.hostMem().alloc(size, alignment);
        else if constexpr (kind == MemoryKind::Pinned)
            return m_alloc.hostPinnedMem().alloc(size, alignment);
        else if constexpr (kind == MemoryKind::Cuda)
            return m_alloc.cudaMem().alloc(size, alignment);
        else
            return nullptr; // should never happen
    }

    void freeMem(void *mem, size_t size, size_t alignment) const
    {
        if constexpr (kind == MemoryKind::Host)
            return m_alloc.hostMem().free(mem, size, alignment);
        else if constexpr (kind == MemoryKind::Pinned)
            return m_alloc.hostPinnedMem().free(mem, size, alignment);
        else if constexpr (kind == MemoryKind::Cuda)
            return m_alloc.cudaMem().free(mem, size, alignment);
    }

    auto getMemDeleter() const
    {
        return [this](cvcuda::WorkspaceMem &mem)
        {
            // free the memory
            freeMem(mem.data, mem.req.size, mem.req.alignment);
            // return the event to the event cache
            if (mem.ready)
            {
                m_eventCache->put(nvcv::util::CudaEvent(mem.ready));
                mem.ready = nullptr;
            }
        };
    }

    Mem create(cvcuda::WorkspaceMemRequirements req)
    {
        WorkspaceMemDestructor_t del = getMemDeleter();

        auto  evt  = nvcv::util::CudaEvent::Create();
        void *data = allocateMem(req.size, req.alignment);

        cvcuda::WorkspaceMem wsmem = {req, data, evt.get()};

        Mem mem(wsmem, std::move(del));
        evt.release(); // from now on, the event handle is managed by `mem`.
        return mem;
    }

    nvcv::Allocator m_alloc;

    std::shared_ptr<nvcv::util::EventCache> m_eventCache;

    nvcv::util::PerStreamCache<CachedWorkspaceMem<kind>> m_memCache;

    std::atomic_int m_outstandingAllocs;
};

class WorkspaceCache;

class WorkspaceLease
{
public:
    cvcuda::Workspace get() const
    {
        return {m_host, m_pinned, m_cuda};
    }

    ~WorkspaceLease();

private:
    friend class WorkspaceCache;
    WorkspaceLease(WorkspaceCache *owner, CachedWorkspaceMem<MemoryKind::Host> &&host,
                   CachedWorkspaceMem<MemoryKind::Pinned> &&pinned, CachedWorkspaceMem<MemoryKind::Cuda> &&cuda,
                   std::optional<cudaStream_t> hostReleaseStream, std::optional<cudaStream_t> pinnedReleaseStream,
                   std::optional<cudaStream_t> cudaReleaseStream);

    WorkspaceCache                        *m_owner;
    CachedWorkspaceMem<MemoryKind::Host>   m_host;
    CachedWorkspaceMem<MemoryKind::Pinned> m_pinned;
    CachedWorkspaceMem<MemoryKind::Cuda>   m_cuda;

    std::optional<cudaStream_t> m_hostReleaseStream, m_pinnedReleaseStream, m_cudaReleaseStream;
};

class WorkspaceCache
{
public:
    WorkspaceCache();

    WorkspaceCache(nvcv::Allocator allocator);

    /** Gets a workspace with custom stream semantics
     *
     * @param req                 The workspace memory sizes and alignments
     * @param hostAcquireStream   The stream on which regular host memory will be initialky used; typically nullopt
     * @param hostReleaseStream   The stream on which regular host memory usage will be completed; typically nullopt
     * @param pinnedAcquireStream The stream on which pinned memory will be initialky used; typically nullopt
     * @param pinnedReleaseStream The stream on which pinned memory usage will be completed; typically the main stream
     *                            on which the operator is executed
     * @param cudaAcquireStream   The stream on which device memory will be initialky used
     * @param cudaReleaseStream   The stream on which device memory usage will be completed
     */
    WorkspaceLease get(cvcuda::WorkspaceRequirements req, std::optional<cudaStream_t> hostAcquireStream,
                       std::optional<cudaStream_t> hostReleaseStream, std::optional<cudaStream_t> pinnedAcquireStream,
                       std::optional<cudaStream_t> pinnedReleaseStream, std::optional<cudaStream_t> cudaAcquireStream,
                       std::optional<cudaStream_t> cudaReleaseStream);

    /** Gets a workspace with default stream semantics
     *
     * The default stream semantics are:
     * - host memory doesn't use any streams
     * - pinned memory is used for h2d copy (released in stream order)
     * - device memory is acquired and released on the same stream
     *
     * NOTE: If these semantics are not honored by the user, the code should still be correct, just less efficient.
     */
    WorkspaceLease get(cvcuda::WorkspaceRequirements req, cudaStream_t stream)
    {
        return get(req, std::nullopt, std::nullopt, std::nullopt, stream, stream, stream);
    }

    auto &host()
    {
        return m_host;
    }

    auto &pinned()
    {
        return m_pinned;
    }

    auto &cuda()
    {
        return m_cuda;
    }

    static WorkspaceCache &instance();

    void clear();

private:
    std::shared_ptr<nvcv::util::EventCache> m_eventCache;
    WorkspaceMemCache<MemoryKind::Host>     m_host;
    WorkspaceMemCache<MemoryKind::Pinned>   m_pinned;
    WorkspaceMemCache<MemoryKind::Cuda>     m_cuda;

    friend class WorkspaceLease;
};

} // namespace cvcudapy

#endif // CVCUDA_PYTHON_WORKSPACE_CACHE_HPP
