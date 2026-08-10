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

#ifndef CVCUDAERATORS_WORKSPACE_HPP
#define CVCUDAERATORS_WORKSPACE_HPP

#include "Workspace.h"

#include <nvcv/alloc/Allocator.hpp>
#include <nvcv/detail/Align.hpp>

#include <cassert>
#include <functional>
#include <stdexcept>
#include <utility>

namespace cvcuda {

using Workspace                = NVCVWorkspace;
using WorkspaceMem             = NVCVWorkspaceMem;
using WorkspaceRequirements    = NVCVWorkspaceRequirements;
using WorkspaceMemRequirements = NVCVWorkspaceMemRequirements;

class WorkspaceError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/** Computes memory requirements that can cover both input requirements.
 *
 * The resulting memory requriements will have alignment and size that is not smaller than that of either
 * of the arguments.
 *
 * alignment = max(a.alignment, b.alignment)
 * size = align_up(max(a.size, b.size), alignment)
 */
inline WorkspaceMemRequirements MaxWorkspaceReq(WorkspaceMemRequirements a, WorkspaceMemRequirements b)
{
    WorkspaceMemRequirements ret;
    assert(!a.size || a.alignment > 0);
    assert(!b.size || b.alignment > 0);
    ret.alignment = b.alignment > a.alignment ? b.alignment : a.alignment;
    ret.size      = b.size > a.size ? b.size : a.size;
    assert((ret.alignment & (ret.alignment - 1)) == 0 && "Alignment must be a power of 2");
    ret.size = nvcv::detail::AlignUp(ret.size, ret.alignment);
    return ret;
}

/** Computes workspace requirements that can cover both input requirments. */
inline NVCVWorkspaceRequirements MaxWorkspaceReq(const WorkspaceRequirements &a, const WorkspaceRequirements &b)
{
    WorkspaceRequirements ret;
    ret.hostMem   = MaxWorkspaceReq(a.hostMem, b.hostMem);
    ret.pinnedMem = MaxWorkspaceReq(a.pinnedMem, b.pinnedMem);
    ret.cudaMem   = MaxWorkspaceReq(a.cudaMem, b.cudaMem);
    return ret;
}

inline void AlignUp(WorkspaceRequirements &ws)
{
    ws.hostMem.size   = nvcv::detail::AlignUp(ws.hostMem.size, ws.hostMem.alignment);
    ws.pinnedMem.size = nvcv::detail::AlignUp(ws.pinnedMem.size, ws.pinnedMem.alignment);
    ws.cudaMem.size   = nvcv::detail::AlignUp(ws.cudaMem.size, ws.cudaMem.alignment);
}

inline void SynchronizeWorkspaceMem(const WorkspaceMem &mem)
{
    if (mem.ready && cudaEventSynchronize(mem.ready) != cudaSuccess)
    {
        throw WorkspaceError("cudaEventSynchronize failed");
    }
}

template<class Allocator>
inline void FreeWorkspaceMem(WorkspaceMem &mem, Allocator alloc)
{
    if (!mem.data)
    {
        return;
    }

    SynchronizeWorkspaceMem(mem);
    alloc.free(mem.data, static_cast<int64_t>(mem.req.size), static_cast<int32_t>(mem.req.alignment));
    mem.data = nullptr;
}

/** A helper class that manages the lifetime of resources stored in a Workspace structure.
 *
 * This class works in a way similar to unique_ptr with a custom deleter.
 */
class UniqueWorkspace
{
public:
    using DeleterFunc = void(NVCVWorkspace &);
    using Deleter     = std::function<DeleterFunc>;

    UniqueWorkspace() = default;

    UniqueWorkspace(const UniqueWorkspace &) = delete;

    UniqueWorkspace(UniqueWorkspace &&ws) noexcept
    {
        swap(ws);
    }

    UniqueWorkspace &operator=(const UniqueWorkspace &) = delete;

    UniqueWorkspace &operator=(UniqueWorkspace &&ws) noexcept
    {
        swap(ws);
        ws.resetNoThrow();
        return *this;
    }

    explicit UniqueWorkspace(const Workspace &workspace, Deleter del = {})
        : m_impl(workspace)
        , m_del(std::move(del))
    {
    }

    UniqueWorkspace(const WorkspaceMem &host, const WorkspaceMem &pinned, const WorkspaceMem &cuda, Deleter del = {})
        : m_impl{host, pinned, cuda}
        , m_del(std::move(del))
    {
    }

    ~UniqueWorkspace() noexcept
    {
        resetNoThrow();
    }

    void reset() noexcept
    {
        resetNoThrow();
    }

    const Workspace &get() const
    {
        return m_impl;
    }

private:
    void resetImpl()
    {
        if (m_del)
        {
            m_del(m_impl);
            m_del  = {};
            m_impl = {};
        }
    }

    void resetNoThrow() noexcept
    {
        try
        {
            resetImpl();
        }
        catch (...)
        {
            m_del  = {};
            m_impl = {};
        }
    }

    void swap(UniqueWorkspace &ws) noexcept
    {
        std::swap(m_impl, ws.m_impl);
        std::swap(m_del, ws.m_del);
    }

    Workspace m_impl{};
    Deleter   m_del{};
};

/** Allocates a workspace with an allocator specified in `alloc` (or a default one).
 *
 * This function is meant as a simple helper to simplify the usage operators requiring a workspace, but its intense use
 * may degrade performance due to excessive allocations and deallocations.
 * For code used in tight loops, some workspace reuse scheme and/or resource pools are recommended.
 */
inline UniqueWorkspace AllocateWorkspace(const WorkspaceRequirements &req, nvcv::Allocator alloc = {})
{
    if (!alloc)
    {
        nvcv::CustomAllocator<> cust{};
        alloc = std::move(cust);
    }
    auto del = [alloc](NVCVWorkspace &ws)
    {
        // REVISIT(michalz): Add proper CUDA error handling in public API
        FreeWorkspaceMem(ws.hostMem, alloc.hostMem());
        FreeWorkspaceMem(ws.pinnedMem, alloc.hostPinnedMem());
        FreeWorkspaceMem(ws.cudaMem, alloc.cudaMem());
    };
    NVCVWorkspace ws = {};
    try
    {
        ws.hostMem.req   = req.hostMem;
        ws.pinnedMem.req = req.pinnedMem;
        ws.cudaMem.req   = req.cudaMem;

        if (req.hostMem.size)
            ws.hostMem.data = alloc.hostMem().alloc(static_cast<int64_t>(req.hostMem.size),
                                                    static_cast<int32_t>(req.hostMem.alignment));
        if (req.pinnedMem.size)
            ws.pinnedMem.data = alloc.hostPinnedMem().alloc(static_cast<int64_t>(req.pinnedMem.size),
                                                            static_cast<int32_t>(req.pinnedMem.alignment));
        if (req.cudaMem.size)
            ws.cudaMem.data = alloc.cudaMem().alloc(static_cast<int64_t>(req.cudaMem.size),
                                                    static_cast<int32_t>(req.cudaMem.alignment));
        return UniqueWorkspace(ws, std::move(del));
    }
    catch (...)
    {
        del(ws);
        throw;
    }
}

} // namespace cvcuda

#endif // CVCUDAERATORS_WORKSPACE_HPP
