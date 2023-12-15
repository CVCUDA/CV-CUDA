/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace cvcudapy {

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

WorkspaceLease::~WorkspaceLease()
{
    if (m_host)
        m_owner->m_host.put(std::move(m_host), m_hostReleaseStream);
    if (m_pinned)
        m_owner->m_pinned.put(std::move(m_pinned), m_pinnedReleaseStream);
    if (m_cuda)
        m_owner->m_cuda.put(std::move(m_cuda), m_hostReleaseStream);
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

WorkspaceLease WorkspaceCache::get(cvcuda::WorkspaceRequirements req, std::optional<cudaStream_t> hostAcquireStream,
                                   std::optional<cudaStream_t> hostReleaseStream,
                                   std::optional<cudaStream_t> pinnedAcquireStream,
                                   std::optional<cudaStream_t> pinnedReleaseStream,
                                   std::optional<cudaStream_t> cudaAcquireStream,
                                   std::optional<cudaStream_t> cudaReleaseStream)
{
    return WorkspaceLease(this, m_host.get(req.hostMem, hostAcquireStream),
                          m_pinned.get(req.pinnedMem, pinnedAcquireStream), m_cuda.get(req.cudaMem, cudaAcquireStream),
                          hostReleaseStream, pinnedReleaseStream, cudaReleaseStream);
}

WorkspaceCache &WorkspaceCache::instance()
{
    static WorkspaceCache instance;
    return instance;
}

void WorkspaceCache::clear()
{
    m_cuda.clear();
    m_pinned.clear();
    m_host.clear();
    m_eventCache->purge();
}

} // namespace cvcudapy
