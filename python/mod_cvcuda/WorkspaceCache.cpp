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

#include "WorkspaceCache.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace cvcudapy {

namespace {

template<MemoryKind kind>
void ReleaseWorkspaceMem(WorkspaceMemCache<kind> &cache, CachedWorkspaceMem<kind> &mem,
                         std::optional<cudaStream_t> releaseStream) noexcept
{
    if (!mem)
        return;

    try
    {
        cache.put(std::move(mem), releaseStream);
    }
    catch (...)
    {
        mem.reset();
    }
}

} // namespace

WorkspaceLease::WorkspaceLease(WorkspaceCache *owner, CachedWorkspaceMem<MemoryKind::Host> &&host,
                               CachedWorkspaceMem<MemoryKind::Pinned> &&pinned,
                               CachedWorkspaceMem<MemoryKind::Cuda>   &&cuda,
                               std::optional<cudaStream_t>              hostReleaseStream,
                               std::optional<cudaStream_t>              pinnedReleaseStream,
                               std::optional<cudaStream_t>              cudaReleaseStream)
    : m_owner(owner)
    , m_host(std::move(host))
    , m_pinned(std::move(pinned))
    , m_cuda(std::move(cuda))
    , m_hostReleaseStream(std::move(hostReleaseStream))
    , m_pinnedReleaseStream(std::move(pinnedReleaseStream))
    , m_cudaReleaseStream(std::move(cudaReleaseStream))
{
}

WorkspaceLease::~WorkspaceLease() noexcept
{
    if (m_owner == nullptr)
        return;

    ReleaseWorkspaceMem(m_owner->m_host, m_host, m_hostReleaseStream);
    ReleaseWorkspaceMem(m_owner->m_pinned, m_pinned, m_pinnedReleaseStream);
    ReleaseWorkspaceMem(m_owner->m_cuda, m_cuda, m_cudaReleaseStream);
}

WorkspaceCache::WorkspaceCache(nvcv::Allocator allocator)
    : m_eventCache(std::make_shared<nvcv::util::EventCache>())
    , m_host(allocator, m_eventCache)
    , m_pinned(allocator, m_eventCache)
    , m_cuda(allocator, m_eventCache)
{
}

WorkspaceCache::WorkspaceCache()
    : WorkspaceCache(nvcv::CustomAllocator<>{})
{
}

WorkspaceLease WorkspaceCache::get(const cvcuda::WorkspaceRequirements &req,
                                   std::optional<cudaStream_t>          hostAcquireStream,
                                   std::optional<cudaStream_t>          hostReleaseStream,
                                   std::optional<cudaStream_t>          pinnedAcquireStream,
                                   std::optional<cudaStream_t>          pinnedReleaseStream,
                                   std::optional<cudaStream_t>          cudaAcquireStream,
                                   std::optional<cudaStream_t>          cudaReleaseStream)
{
    return WorkspaceLease(this, m_host.get(req.hostMem, hostAcquireStream),
                          m_pinned.get(req.pinnedMem, pinnedAcquireStream), m_cuda.get(req.cudaMem, cudaAcquireStream),
                          hostReleaseStream, pinnedReleaseStream, cudaReleaseStream);
}

WorkspaceCache &WorkspaceCache::instance()
{
    // Per-device singleton: each CUDA device gets its own WorkspaceCache
    // so that device memory allocations are always on the correct GPU.
    // Uses shared_ptr for heap stability — unordered_map rehashing won't
    // invalidate the objects that outstanding references point to.
    static std::unordered_map<int, std::shared_ptr<WorkspaceCache>> instances;
    static std::shared_mutex                                        instances_mutex;

    int dev = 0;
    nvcvpy::util::CheckThrow(cudaGetDevice(&dev));

    // Shared lock: concurrent readers when the entry already exists.
    {
        std::shared_lock lock(instances_mutex);
        auto             it = instances.find(dev);
        if (it != instances.end())
            return *it->second;
    }

    // Exclusive lock: serializes the one-time insertion of a new entry.
    std::unique_lock lock(instances_mutex);
    auto [it, _] = instances.try_emplace(dev, nullptr);
    if (!it->second)
        it->second = std::make_shared<WorkspaceCache>();
    return *it->second;
}

void WorkspaceCache::clear()
{
    m_cuda.clear();
    m_pinned.clear();
    m_host.clear();
    m_eventCache->purge();
}

} // namespace cvcudapy
