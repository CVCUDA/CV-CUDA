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

#include "Nvtx.hpp"
#include "OpHQResize.hpp"
#include "OpHQResizeDispatch.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cassert>
#include <vector>

namespace cvcuda::priv {
namespace hq_resize {

namespace {

// Expand a planar (NCHW/CHW) tensor batch into a batch of single-channel (H,W,1) HWC views, one per
// (sample, channel) plane. HQResize resizes channels independently, so resizing each plane as an
// independent single-channel image reproduces the interleaved result bit-for-bit. The views alias
// the original device memory (no copy); the returned batch must outlive the operator call. The
// caller sizes the workspace from the matching expanded (sum-of-channels single-channel) shapes.
static nvcv::TensorBatch ExpandPlanarTensorBatch(const nvcv::TensorBatch &batch)
{
    std::vector<nvcv::Tensor> views;
    for (int i = 0; i < batch.numTensors(); ++i)
    {
        nvcv::Tensor tensor = batch[i];
        auto         data   = tensor.exportData<nvcv::TensorDataStridedCuda>();
        if (!data)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar HQResize requires cuda-accessible "
                                  "tensors");
        }
        auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
        if (!access)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar HQResize requires planar-accessible "
                                  "tensors");
        }

        const int     numSamples  = access->numSamples();
        const int     numChannels = access->numChannels();
        const int64_t H = access->numRows(), W = access->numCols();
        for (int s = 0; s < numSamples; ++s)
        {
            for (int c = 0; c < numChannels; ++c)
            {
                nvcv::TensorDataStridedCuda::Buffer buf;
                buf.basePtr    = reinterpret_cast<NVCVByte *>(access->sampleData(s) + c * access->chStride());
                buf.strides[0] = access->rowStride(); // H
                buf.strides[1] = access->colStride(); // W
                buf.strides[2] = access->colStride(); // C == 1
                nvcv::TensorDataStridedCuda viewData{
                    nvcv::TensorShape{{H, W, 1}, "HWC"},
                    data->dtype(), buf
                };
                views.push_back(nvcv::TensorWrapData(viewData));
            }
        }
    }
    nvcv::TensorBatch out(static_cast<int32_t>(views.size()));
    out.pushBack(views.begin(), views.end());
    return out;
}

} // namespace

class HQResizeImpl final : public IHQResizeImpl
{
public:
    HQResizeImpl()
        : m_impl2d(makeImpl2D())
        , m_impl3d(makeImpl3D())
    {
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapeI inputShape,
                                                           const HQResizeTensorShapeI  outputShape,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoiF *roi) const override
    {
        return implForNDim(inputShape.ndim)
            .getWorkspaceRequirements(numSamples, inputShape, outputShape, minInterpolation, magInterpolation,
                                      antialias, roi);
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapesI inputShapes,
                                                           const HQResizeTensorShapesI outputShapes,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoisF rois) const override
    {
        return implForNDim(inputShapes.ndim)
            .getWorkspaceRequirements(numSamples, inputShapes, outputShapes, minInterpolation, magInterpolation,
                                      antialias, rois);
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int                        maxBatchSize,
                                                           const HQResizeTensorShapeI maxShape) const override
    {
        return implForNDim(maxShape.ndim).getWorkspaceRequirements(maxBatchSize, maxShape);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias, const HQResizeRoiF *roi) override
    {
        CVCUDA_NVTX_RANGE("cvcuda::hq_resize::HQResizeImpl::operator()[Tensor]");
        bool is3d = src.layout().find('D') >= 0;
        implForNDim(is3d ? 3 : 2)(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, roi);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::ImageBatchVarShape &src,
                    const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) override
    {
        CVCUDA_NVTX_RANGE("cvcuda::hq_resize::HQResizeImpl::operator()[ImageBatchVarShape]");
        (*m_impl2d)(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::TensorBatch &src,
                    const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) override
    {
        CVCUDA_NVTX_RANGE("cvcuda::hq_resize::HQResizeImpl::operator()[TensorBatch]");
        // Planar (channel-first 2D, NCHW/CHW) batches are expanded into single-channel (H,W,1) plane
        // views and run through the regular 2D path; the views alias the originals so the result is
        // bit-identical to the interleaved path. Views are kept alive for the duration of the call.
        const nvcv::TensorLayout layout = src.layout();
        if (layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW)
        {
            nvcv::TensorBatch planarSrc = ExpandPlanarTensorBatch(src);
            nvcv::TensorBatch planarDst = ExpandPlanarTensorBatch(dst);
            (*m_impl2d)(stream, ws, planarSrc, planarDst, minInterpolation, magInterpolation, antialias, rois);
            return;
        }
        bool is3d = layout.find('D') >= 0;
        implForNDim(is3d ? 3 : 2)(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
    }

private:
    IHQResizeImpl &implForNDim(int ndim) const
    {
        if (ndim == 2)
            return *m_impl2d;
        if (ndim == 3)
            return *m_impl3d;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only 2D or 3D resize is supported. Got unexpected number of extents to resize.");
    }

    std::unique_ptr<IHQResizeImpl> m_impl2d;
    std::unique_ptr<IHQResizeImpl> m_impl3d;
};

} // namespace hq_resize

// Constructor -----------------------------------------------------------------

HQResize::HQResize()
{
    m_impl = std::make_unique<hq_resize::HQResizeImpl>();
}

// Operator --------------------------------------------------------------------

cvcuda::WorkspaceRequirements HQResize::getWorkspaceRequirements(int batchSize, const HQResizeTensorShapeI inputShape,
                                                                 const HQResizeTensorShapeI  outputShape,
                                                                 const NVCVInterpolationType minInterpolation,
                                                                 const NVCVInterpolationType magInterpolation,
                                                                 bool antialias, const HQResizeRoiF *roi) const
{
    return m_impl->getWorkspaceRequirements(batchSize, inputShape, outputShape, minInterpolation, magInterpolation,
                                            antialias, roi);
}

cvcuda::WorkspaceRequirements HQResize::getWorkspaceRequirements(int batchSize, const HQResizeTensorShapesI inputShapes,
                                                                 const HQResizeTensorShapesI outputShapes,
                                                                 const NVCVInterpolationType minInterpolation,
                                                                 const NVCVInterpolationType magInterpolation,
                                                                 bool antialias, const HQResizeRoisF rois) const
{
    return m_impl->getWorkspaceRequirements(batchSize, inputShapes, outputShapes, minInterpolation, magInterpolation,
                                            antialias, rois);
}

cvcuda::WorkspaceRequirements HQResize::getWorkspaceRequirements(int                        maxBatchSize,
                                                                 const HQResizeTensorShapeI maxShape) const
{
    return m_impl->getWorkspaceRequirements(maxBatchSize, maxShape);
}

void HQResize::operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::Tensor &src,
                          const nvcv::Tensor &dst, const NVCVInterpolationType minInterpolation,
                          const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HQResize::operator()[Tensor]");
    assert(m_impl);
    m_impl->operator()(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, roi);
}

void HQResize::operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::ImageBatchVarShape &src,
                          const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                          const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HQResize::operator()[ImageBatchVarShape]");
    assert(m_impl);
    m_impl->operator()(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
}

void HQResize::operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::TensorBatch &src,
                          const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                          const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HQResize::operator()[TensorBatch]");
    assert(m_impl);
    m_impl->operator()(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
}

} // namespace cvcuda::priv
