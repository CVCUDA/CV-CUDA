/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file HQResize.hpp
 *
 * @brief Defines the private C++ Class for the HQResize operation.
 */

#ifndef CVCUDA_PRIV_HQ_RESIZE_HPP
#define CVCUDA_PRIV_HQ_RESIZE_HPP
#include "IOperator.hpp"
#include "cvcuda/Workspace.hpp"

#include <cvcuda/OpHQResize.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

#include <memory>

namespace cvcuda::priv {

namespace hq_resize {

class IHQResizeImpl
{
public:
    virtual WorkspaceRequirements getWorkspaceRequirements(int batchSize, const HQResizeTensorShapeI inputShape,
                                                           const HQResizeTensorShapeI  outputShape,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoiF *roi) const = 0;

    virtual WorkspaceRequirements getWorkspaceRequirements(int batchSize, const HQResizeTensorShapesI inputShapes,
                                                           const HQResizeTensorShapesI outputShapes,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoisF rois) const = 0;

    virtual WorkspaceRequirements getWorkspaceRequirements(int                        maxBatchSize,
                                                           const HQResizeTensorShapeI maxShape) const = 0;

    virtual void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                            const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                            bool antialias, const HQResizeRoiF *roi)
        = 0;

    virtual void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &src,
                            const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                            const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi)
        = 0;

    virtual void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::TensorBatch &src,
                            const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                            const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi)
        = 0;

    virtual ~IHQResizeImpl() = default;
};

} // namespace hq_resize

class HQResize final : public IOperator
{
public:
    explicit HQResize();

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, const HQResizeTensorShapeI inputShape,
                                                   const HQResizeTensorShapeI  outputShape,
                                                   const NVCVInterpolationType minInterpolation,
                                                   const NVCVInterpolationType magInterpolation, bool antialias,
                                                   const HQResizeRoiF *roi) const;

    WorkspaceRequirements getWorkspaceRequirements(int batchSize, const HQResizeTensorShapesI inputShapes,
                                                   const HQResizeTensorShapesI outputShapes,
                                                   const NVCVInterpolationType minInterpolation,
                                                   const NVCVInterpolationType magInterpolation, bool antialias,
                                                   const HQResizeRoisF rois) const;

    WorkspaceRequirements getWorkspaceRequirements(int maxBatchSize, const HQResizeTensorShapeI maxShape) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias, const HQResizeRoiF *roi) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &src,
                    const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi) const;

    void operator()(cudaStream_t stream, const Workspace &ws, const nvcv::TensorBatch &src,
                    const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi) const;

private:
    std::unique_ptr<hq_resize::IHQResizeImpl> m_impl;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_HQ_RESIZE_HPP
