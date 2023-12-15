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

#ifndef CVCUDA_PRIV_WORKSPACE_ESTIMATOR_HPP
#define CVCUDA_PRIV_WORKSPACE_ESTIMATOR_HPP

#include <cvcuda/Workspace.hpp>

namespace cvcuda {

struct WorkspaceMemEstimator
{
    explicit WorkspaceMemEstimator(size_t initial_size = 0, size_t base_alignment = alignof(std::max_align_t))
        : req{initial_size, base_alignment}
    {
    }

    WorkspaceMemRequirements req;

    template<typename T = char>
    WorkspaceMemEstimator &add(size_t count = 1, size_t alignment = alignof(T))
    {
        if (alignment > req.alignment)
            req.alignment = alignment;
        req.size = nvcv::detail::AlignUp(req.size, alignment);
        req.size += nvcv::detail::AlignUp(count * sizeof(T), alignment);
        return *this;
    }
};

struct WorkspaceEstimator
{
    static constexpr size_t kDefaultPinnedAlignment = 256;
    static constexpr size_t kDefaultDeviceAlignment = 256;

    WorkspaceMemEstimator hostMem;
    WorkspaceMemEstimator pinnedMem{0, kDefaultPinnedAlignment};
    WorkspaceMemEstimator cudaMem{0, kDefaultDeviceAlignment};

    template<typename T = char>
    WorkspaceEstimator &add(bool host, bool pinned, bool cuda, size_t count = 1, size_t alignment = alignof(T))
    {
        if (host)
            addHost<T>(count, alignment);
        if (pinned)
            addPinned<T>(count, alignment);
        if (cuda)
            addCuda<T>(count, alignment);
        return *this;
    }

    template<typename T = char>
    WorkspaceEstimator &addHost(size_t count = 1, size_t alignment = alignof(T))
    {
        hostMem.add<T>(count, alignment);
        return *this;
    }

    template<typename T = char>
    WorkspaceEstimator &addPinned(size_t count = 1, size_t alignment = alignof(T))
    {
        pinnedMem.add<T>(count, alignment);
        return *this;
    }

    template<typename T = char>
    WorkspaceEstimator &addCuda(size_t count = 1, size_t alignment = alignof(T))
    {
        cudaMem.add<T>(count, alignment);
        return *this;
    }
};

} // namespace cvcuda

#endif // CVCUDA_PRIV_WORKSPACE_ESTIMATOR_HPP
