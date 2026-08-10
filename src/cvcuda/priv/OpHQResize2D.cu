/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpHQResizeDispatch.hpp"

#include "OpHQResizeKernel.cuh"

namespace cvcuda::priv::hq_resize {

using kernel::HQResizeRun;

class HQResizeImpl2D final : public IHQResizeImpl
{
public:
    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapeI inputShape,
                                                           const HQResizeTensorShapeI  outputShape,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoiF *roi) const override
    {
        HQResizeRun<2> resize(m_filtersFactory);
        return resize.getWorkspaceRequirements(numSamples, inputShape, outputShape, minInterpolation, magInterpolation,
                                               antialias, roi);
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapesI inputShapes,
                                                           const HQResizeTensorShapesI outputShapes,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoisF rois) const override
    {
        HQResizeRun<2> resize(m_filtersFactory);
        return resize.getWorkspaceRequirements(numSamples, inputShapes, outputShapes, minInterpolation,
                                               magInterpolation, antialias, rois);
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int                        maxBatchSize,
                                                           const HQResizeTensorShapeI maxShape) const override
    {
        HQResizeRun<2> resize(m_filtersFactory);
        return resize.getWorkspaceRequirements(maxBatchSize, maxShape);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias, const HQResizeRoiF *roi) override
    {
        CVCUDA_NVTX_RANGE("cvcuda::hq_resize::HQResizeImpl2D::operator()[Tensor]");
        HQResizeRun<2> resize(m_filtersFactory);
        resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, roi);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::ImageBatchVarShape &src,
                    const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) override
    {
        CVCUDA_NVTX_RANGE("cvcuda::hq_resize::HQResizeImpl2D::operator()[ImageBatchVarShape]");
        HQResizeRun<2> resize(m_filtersFactory);
        resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::TensorBatch &src,
                    const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) override
    {
        CVCUDA_NVTX_RANGE("cvcuda::hq_resize::HQResizeImpl2D::operator()[TensorBatch]");
        HQResizeRun<2> resize(m_filtersFactory);
        resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
    }

private:
    filter::ResamplingFiltersFactory m_filtersFactory;
};

std::unique_ptr<IHQResizeImpl> makeImpl2D()
{
    return std::make_unique<HQResizeImpl2D>();
}

} // namespace cvcuda::priv::hq_resize
