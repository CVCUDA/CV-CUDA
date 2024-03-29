/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file OpPillowResize.hpp
 *
 * @brief Defines the public C++ Class for the pillow resize operation.
 * @defgroup NVCV_CPP_ALGORITHM_PILLOW_RESIZE Pillow Resize
 * @{
 */

#ifndef CVCUDA_PILLOW_RESIZE_HPP
#define CVCUDA_PILLOW_RESIZE_HPP

#include "IOperator.hpp"
#include "OpPillowResize.h"
#include "Workspace.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class PillowResize final : public IOperator
{
public:
    PillowResize();

    ~PillowResize();

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, const nvcv::Size2D *in_sizes,
                                                   const nvcv::Size2D *out_sizes, nvcv::ImageFormat fmt);

    WorkspaceRequirements getWorkspaceRequirements(int maxBatchSize, nvcv::Size2D maxInSize, nvcv::Size2D maxOutSize,
                                                   nvcv::ImageFormat fmt);

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVInterpolationType interpolation);

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &in,
                    const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline PillowResize::PillowResize()
{
    nvcv::detail::CheckThrow(cvcudaPillowResizeCreate(&m_handle));
    assert(m_handle);
}

inline PillowResize::~PillowResize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline WorkspaceRequirements PillowResize::getWorkspaceRequirements(int batchSize, const nvcv::Size2D *in_sizes,
                                                                    const nvcv::Size2D *out_sizes,
                                                                    nvcv::ImageFormat   fmt)
{
    WorkspaceRequirements req{};
    nvcv::detail::CheckThrow(cvcudaPillowResizeVarShapeGetWorkspaceRequirements(m_handle, batchSize, in_sizes,
                                                                                out_sizes, fmt.cvalue(), &req));
    return req;
}

inline WorkspaceRequirements PillowResize::getWorkspaceRequirements(int maxBatchSize, nvcv::Size2D maxInSize,
                                                                    nvcv::Size2D maxOutSize, nvcv::ImageFormat fmt)
{
    WorkspaceRequirements req{};
    nvcv::detail::CheckThrow(cvcudaPillowResizeGetWorkspaceRequirements(
        m_handle, maxBatchSize, maxInSize.w, maxInSize.h, maxOutSize.w, maxOutSize.h, fmt.cvalue(), &req));
    return req;
}

inline void PillowResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in,
                                     const nvcv::Tensor &out, const NVCVInterpolationType interpolation)
{
    nvcv::detail::CheckThrow(cvcudaPillowResizeSubmit(m_handle, stream, &ws, in.handle(), out.handle(), interpolation));
}

inline void PillowResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &in,
                                     const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation)
{
    nvcv::detail::CheckThrow(
        cvcudaPillowResizeVarShapeSubmit(m_handle, stream, &ws, in.handle(), out.handle(), interpolation));
}

inline NVCVOperatorHandle PillowResize::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_PILLOW_RESIZE_HPP
