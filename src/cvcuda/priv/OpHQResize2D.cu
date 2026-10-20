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

#include <cub/block/block_scan.cuh>

#include <cassert>

namespace cvcuda::priv::hq_resize::batch_wrapper::tensor {
namespace {

constexpr int kSetupBlockSize = 256;

inline __device__ void SetupPlanarTensorBatchElement(PlanarTensorBatchElement            &out,
                                                     const NVCVTensorBatchElementStrided &in, int sample, int channel,
                                                     bool hasSampleDim)
{
    const int channelDim = hasSampleDim ? 1 : 0;
    const int heightDim  = channelDim + 1;

    out.data      = in.data + (hasSampleDim ? sample * in.stride[0] : 0) + channel * in.stride[channelDim];
    out.rowStride = in.stride[heightDim];
}

template<typename StrideT>
__global__ void SetupPlanarTensorBatchElementsKernel(
    PlanarTensorBatchElement *expandedSrc, PlanarTensorBatchElement *expandedDst,
    dynamic::TensorAccessDesc<StrideT> *intermediateMeta, float *intermediate,
    const kernel::resampling::SampleDesc<2> *sampleDescs, const NVCVTensorBatchElementStrided *src,
    const NVCVTensorBatchElementStrided *dst, int numTensors, bool hasSampleDim)
{
    using PlaneScan = cub::BlockScan<int, kSetupBlockSize>;
    using ByteScan  = cub::BlockScan<int64_t, kSetupBlockSize>;
    __shared__ typename PlaneScan::TempStorage planeScanStorage;
    __shared__ typename ByteScan::TempStorage  byteScanStorage;
    __shared__ int                             planeOffsets[kSetupBlockSize + 1];
    __shared__ int                             tensorChannels[kSetupBlockSize];

    int     expandedBase           = 0;
    int64_t intermediateByteOffset = 0;
    for (int tensorBase = 0; tensorBase < numTensors; tensorBase += kSetupBlockSize)
    {
        const int tileTensors = min(kSetupBlockSize, numTensors - tensorBase);
        const int tensor      = tensorBase + threadIdx.x;
        int       numSamples{};
        int       numChannels{};
        if (tensor < numTensors)
        {
            const int channelDim = hasSampleDim ? 1 : 0;
            numSamples           = hasSampleDim ? static_cast<int>(src[tensor].shape[0]) : 1;
            numChannels          = static_cast<int>(src[tensor].shape[channelDim]);
        }

        const int numPlanes = numSamples * numChannels;
        int       tensorOffset;
        int       tilePlanes;
        PlaneScan(planeScanStorage).ExclusiveSum(numPlanes, tensorOffset, tilePlanes);

        if (threadIdx.x < tileTensors)
        {
            planeOffsets[threadIdx.x]   = tensorOffset;
            tensorChannels[threadIdx.x] = numChannels;
        }
        if (threadIdx.x == 0)
        {
            planeOffsets[tileTensors] = tilePlanes;
        }
        __syncthreads();

        for (int planeBase = 0; planeBase < tilePlanes; planeBase += kSetupBlockSize)
        {
            const int plane = planeBase + threadIdx.x;
            int       owner{};
            if (plane < tilePlanes)
            {
                int hi = tileTensors;
                while (owner + 1 < hi)
                {
                    const int mid = (owner + hi) / 2;
                    if (planeOffsets[mid] <= plane)
                    {
                        owner = mid;
                    }
                    else
                    {
                        hi = mid;
                    }
                }
            }

            const int expandedIdx = expandedBase + plane;
            VecI<2>   shape{};
            int64_t   rowStride{};
            int64_t   planeBytes{};
            if (plane < tilePlanes)
            {
                shape      = sampleDescs[expandedIdx].shapes[1];
                rowStride  = static_cast<int64_t>(shape.x) * sizeof(float);
                planeBytes = rowStride * shape.y;
            }

            int64_t planeByteOffset;
            int64_t planeTileBytes;
            ByteScan(byteScanStorage).ExclusiveSum(planeBytes, planeByteOffset, planeTileBytes);

            if (plane < tilePlanes)
            {
                const int channelCount = tensorChannels[owner];
                const int tensorPlane  = plane - planeOffsets[owner];
                const int sample       = tensorPlane / channelCount;
                const int channel      = tensorPlane % channelCount;
                const int ownerTensor  = tensorBase + owner;
                SetupPlanarTensorBatchElement(expandedSrc[expandedIdx], src[ownerTensor], sample, channel,
                                              hasSampleDim);
                SetupPlanarTensorBatchElement(expandedDst[expandedIdx], dst[ownerTensor], sample, channel,
                                              hasSampleDim);

                assert(rowStride <= cuda::TypeTraits<StrideT>::max);
                intermediateMeta[expandedIdx].basePtr
                    = reinterpret_cast<unsigned char *>(intermediate) + intermediateByteOffset + planeByteOffset;
                intermediateMeta[expandedIdx].strides[0] = static_cast<StrideT>(rowStride);
            }

            intermediateByteOffset += planeTileBytes;
            __syncthreads();
        }

        expandedBase += tilePlanes;
        __syncthreads();
    }
}

} // namespace

template<typename StrideT>
void SetupPlanarTensorBatchElements(PlanarTensorBatchElement *expandedSrc, PlanarTensorBatchElement *expandedDst,
                                    dynamic::TensorAccessDesc<StrideT> *intermediateMeta, float *intermediate,
                                    const kernel::resampling::SampleDesc<2> *sampleDescs,
                                    const nvcv::TensorBatchDataStridedCuda  &src,
                                    const nvcv::TensorBatchDataStridedCuda &dst, bool hasSampleDim, cudaStream_t stream)
{
    SetupPlanarTensorBatchElementsKernel<StrideT>
        <<<1, kSetupBlockSize, 0, stream>>>(expandedSrc, expandedDst, intermediateMeta, intermediate, sampleDescs,
                                            src.buffer().tensors, dst.buffer().tensors, src.numTensors(), hasSampleDim);
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace cvcuda::priv::hq_resize::batch_wrapper::tensor

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
