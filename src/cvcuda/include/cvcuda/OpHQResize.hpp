/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpHQResize.hpp
 *
 * @brief Defines the public C++ Class for the HQResize operation.
 * @defgroup NVCV_CPP_ALGORITHM_HQ_RESIZE HQ Resize
 * @{
 */

#ifndef CVCUDA_HQ_RESIZE_HPP
#define CVCUDA_HQ_RESIZE_HPP

#include "IOperator.hpp"
#include "OpHQResize.h"
#include "Workspace.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Rect.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

#include <cassert>

namespace cvcuda {

class HQResize final : public IOperator
{
public:
    explicit HQResize();

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, const HQResizeTensorShapeI inputShape,
                                                   const HQResizeTensorShapeI  outputShape,
                                                   const NVCVInterpolationType minInterpolation,
                                                   const NVCVInterpolationType magInterpolation, bool antialias,
                                                   const HQResizeRoiF *roi = nullptr) const;

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, HQResizeTensorShapesI inputShapes,
                                                   const HQResizeTensorShapesI outputShapes,
                                                   const NVCVInterpolationType minInterpolation,
                                                   const NVCVInterpolationType magInterpolation, bool antialias,
                                                   const HQResizeRoisF roi = {}) const;

    WorkspaceRequirements getWorkspaceRequirements(int maxBatchSize, const HQResizeTensorShapeI maxShape) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias = false, const HQResizeRoiF *roi = nullptr) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias = false, const HQResizeRoisF roi = {}) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::TensorBatch &in, const nvcv::TensorBatch &out,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias = false, const HQResizeRoisF roi = {}) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline HQResize::HQResize()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaHQResizeCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline WorkspaceRequirements HQResize::getWorkspaceRequirements(int batchSize, const HQResizeTensorShapeI inputShape,
                                                                const HQResizeTensorShapeI  outputShape,
                                                                const NVCVInterpolationType minInterpolation,
                                                                const NVCVInterpolationType magInterpolation,
                                                                bool antialias, const HQResizeRoiF *roi) const
{
    WorkspaceRequirements req{};
    nvcv::detail::CheckThrow(cvcudaHQResizeTensorGetWorkspaceRequirements(
        m_handle.get(), batchSize, inputShape, outputShape, minInterpolation, magInterpolation, antialias, roi, &req));
    return req;
}

inline WorkspaceRequirements HQResize::getWorkspaceRequirements(int batchSize, const HQResizeTensorShapesI inputShapes,
                                                                const HQResizeTensorShapesI outputShapes,
                                                                const NVCVInterpolationType minInterpolation,
                                                                const NVCVInterpolationType magInterpolation,
                                                                bool antialias, const HQResizeRoisF roi) const
{
    WorkspaceRequirements req{};
    nvcv::detail::CheckThrow(cvcudaHQResizeTensorBatchGetWorkspaceRequirements(m_handle.get(), batchSize, inputShapes,
                                                                               outputShapes, minInterpolation,
                                                                               magInterpolation, antialias, roi, &req));
    return req;
}

inline WorkspaceRequirements HQResize::getWorkspaceRequirements(int                        maxBatchSize,
                                                                const HQResizeTensorShapeI maxShape) const
{
    WorkspaceRequirements req{};
    nvcv::detail::CheckThrow(cvcudaHQResizeGetMaxWorkspaceRequirements(m_handle.get(), maxBatchSize, maxShape, &req));
    return req;
}

inline void HQResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in,
                                 const nvcv::Tensor &out, const NVCVInterpolationType minInterpolation,
                                 const NVCVInterpolationType magInterpolation, bool antialias,
                                 const HQResizeRoiF *roi) const
{
    nvcv::detail::CheckThrow(cvcudaHQResizeSubmit(m_handle.get(), stream, &ws, in.handle(), out.handle(),
                                                  minInterpolation, magInterpolation, antialias, roi));
}

inline void HQResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatch &in,
                                 const nvcv::ImageBatch &out, const NVCVInterpolationType minInterpolation,
                                 const NVCVInterpolationType magInterpolation, bool antialias,
                                 const HQResizeRoisF roi) const
{
    nvcv::detail::CheckThrow(cvcudaHQResizeImageBatchSubmit(m_handle.get(), stream, &ws, in.handle(), out.handle(),
                                                            minInterpolation, magInterpolation, antialias, roi));
}

inline void HQResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::TensorBatch &in,
                                 const nvcv::TensorBatch &out, const NVCVInterpolationType minInterpolation,
                                 const NVCVInterpolationType magInterpolation, bool antialias,
                                 const HQResizeRoisF roi) const
{
    nvcv::detail::CheckThrow(cvcudaHQResizeTensorBatchSubmit(m_handle.get(), stream, &ws, in.handle(), out.handle(),
                                                             minInterpolation, magInterpolation, antialias, roi));
}

inline NVCVOperatorHandle HQResize::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_HQ_RESIZE_HPP
