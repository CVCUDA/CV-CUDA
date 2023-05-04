/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/SaturateCast.hpp>

using namespace nvcv::legacy::helpers;
using namespace nvcv::legacy::cuda_op;

namespace nvcv::legacy::cuda_op {

__global__ void UpdateMasksAnchors(cuda::Tensor1DWrap<int2> masks, cuda::Tensor1DWrap<int2> anchors, int numImages,
                                   int iteration)
{
    int1 coord;
    coord.x = cuda::StaticCast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (coord.x >= numImages)
        return;

    int2 mask_size = masks[coord];
    int2 anchor    = anchors[coord];

    if (mask_size.x == -1 || mask_size.y == -1)
        mask_size.x = mask_size.y = 3;
    if (anchor.x < 0)
        anchor.x = mask_size.x / 2;
    if (anchor.y < 0)
        anchor.y = mask_size.y / 2;

    mask_size = mask_size + (iteration - 1) * (mask_size - 1);
    anchor    = anchor * iteration;

    masks[coord]   = mask_size;
    anchors[coord] = anchor;
}

template<class SrcWrapper, class DstWrapper, typename D = typename DstWrapper::ValueType,
         typename BT = typename cuda::BaseType<D>>
__global__ void dilate(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int2> kernelSizeArr,
                       cuda::Tensor1DWrap<int2> kernelAnchorArr, BT maxmin)
{
    D         res       = cuda::SetAll<D>(maxmin);
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    int3 srcCoord = {0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = cuda::max(res, src[srcCoord]);
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<D>(res);
}

template<class SrcWrapper, class DstWrapper, typename D = typename DstWrapper::ValueType,
         typename BT = typename cuda::BaseType<D>>
__global__ void erode(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int2> kernelSizeArr,
                      cuda::Tensor1DWrap<int2> kernelAnchorArr, BT maxmin)
{
    D         res       = cuda::SetAll<D>(maxmin);
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    int2 anchor     = kernelAnchorArr[batch_idx];

    int3 srcCoord = {0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = cuda::min(res, src[srcCoord]);
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<D>(res);
}

template<typename D, NVCVBorderType B>
void MorphFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                         const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                         const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type, cudaStream_t stream)
{
    cuda::Tensor1DWrap<int2> kernelSizeTensor(kMasks);
    cuda::Tensor1DWrap<int2> kernelAnchorTensor(kAnchors);

    Size2D outMaxSize = outData.maxSize();
    int    maxWidth   = outMaxSize.w;
    int    maxHeight  = outMaxSize.h;

    dim3 block(16, 16);
    dim3 grid(divUp(maxWidth, block.x), divUp(maxHeight, block.y), outData.numImages());

    using BT = nvcv::cuda::BaseType<D>;
    BT val   = (morph_type == NVCVMorphologyType::NVCV_DILATE) ? std::numeric_limits<BT>::min()
                                                               : std::numeric_limits<BT>::max();

    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(val));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    if (morph_type == NVCVMorphologyType::NVCV_ERODE)
    {
        erode<<<grid, block, 0, stream>>>(src, dst, kernelSizeTensor, kernelAnchorTensor, val);
        checkKernelErrors();
    }
    else if (morph_type == NVCVMorphologyType::NVCV_DILATE)
    {
        dilate<<<grid, block, 0, stream>>>(src, dst, kernelSizeTensor, kernelAnchorTensor, val);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void MorphFilter2D(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                   const TensorDataStridedCuda &kMasks, const TensorDataStridedCuda &kAnchors,
                   NVCVMorphologyType morph_type, NVCVBorderType borderMode, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                           const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type, cudaStream_t stream);

    static const func_t funcs[]
        = {MorphFilter2DCaller<D, NVCV_BORDER_CONSTANT>, MorphFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           MorphFilter2DCaller<D, NVCV_BORDER_REFLECT>, MorphFilter2DCaller<D, NVCV_BORDER_WRAP>,
           MorphFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kMasks, kAnchors, morph_type, stream);
}

MorphologyVarShape::MorphologyVarShape(const int maxBatchSize)
    : CudaBaseOp()
    , m_maxBatchSize(maxBatchSize)
    , m_kernelMaskSizes(maxBatchSize)
    , m_kernelAnchors(maxBatchSize)
{
    if (m_maxBatchSize > 0)
    {
        // {Width, Height} per image for mask and anchor
        size_t totalNumElements = m_maxBatchSize * 2;

        m_kernelMaskSizes.resize(totalNumElements);
        if (m_kernelMaskSizes.size() != totalNumElements)
        {
            throw std::runtime_error("Host memory allocation error!");
        }

        m_kernelAnchors.resize(totalNumElements);
        if (m_kernelAnchors.size() != totalNumElements)
        {
            throw std::runtime_error("Host memory allocation error!");
        }
    }
}

MorphologyVarShape::~MorphologyVarShape() {}

ErrorCode MorphologyVarShape::infer(const nvcv::IImageBatchVarShape &inBatch, const nvcv::IImageBatchVarShape &outBatch,
                                    NVCVMorphologyType morph_type, const TensorDataStridedCuda &masks,
                                    const TensorDataStridedCuda &anchors, int iteration, NVCVBorderType borderMode,
                                    cudaStream_t stream)
{
    auto inData = inBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        LOG_ERROR("Input must be varshape image batch");
    }
    auto outData = outBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        LOG_ERROR("Output must be varshape image batch");
    }

    DataFormat input_format  = GetLegacyDataFormat(*inData);
    DataFormat output_format = GetLegacyDataFormat(*outData);
    DataType   data_type     = GetLegacyDataType(inData->uniqueFormat());

    if (inData->numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Number of VarShape Images exceeds configured max size");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    const int channels = inData->uniqueFormat().numChannels();

    if (!(channels == 1 || channels == 3 || channels == 4))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (iteration == 0)
    {
        for (auto init = inBatch.begin(), outit = outBatch.begin(); init != inBatch.end(), outit != outBatch.end();
             ++init, ++outit)
        {
            const IImage            &inimg      = *init;
            const IImage            &outimg     = *outit;
            auto                     inimgdata  = inimg.exportData<ImageDataStridedCuda>();
            auto                     outimgdata = outimg.exportData<ImageDataStridedCuda>();
            const ImagePlaneStrided &inplane    = inimgdata->plane(0);
            const ImagePlaneStrided &outplane   = outimgdata->plane(0);
            checkCudaErrors(cudaMemcpy2DAsync(outplane.basePtr, outplane.rowStride, inplane.basePtr, inplane.rowStride,
                                              inplane.rowStride, inplane.height, cudaMemcpyDeviceToDevice, stream));
        }
        return ErrorCode::SUCCESS;
    }

    dim3                     block(32), grid(divUp(inData->numImages(), 32));
    cuda::Tensor1DWrap<int2> kmasks(masks), kanchors(anchors);
    UpdateMasksAnchors<<<grid, block, 0, stream>>>(kmasks, kanchors, inData->numImages(), iteration);

    typedef void (*filter2D_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &kMasks,
                               const TensorDataStridedCuda &kAnchors, NVCVMorphologyType morph_type,
                               NVCVBorderType borderMode, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { MorphFilter2D<uchar>, 0,  MorphFilter2D<uchar3>,  MorphFilter2D<uchar4>},
        {                    0, 0,                      0,                      0},
        {MorphFilter2D<ushort>, 0, MorphFilter2D<ushort3>, MorphFilter2D<ushort4>},
        {                    0, 0,                      0,                      0},
        {                    0, 0,                      0,                      0},
        { MorphFilter2D<float>, 0,  MorphFilter2D<float3>,  MorphFilter2D<float4>},
    };

    funcs[data_type][channels - 1](*inData, *outData, masks, anchors, morph_type, borderMode, stream);

    return ErrorCode::SUCCESS;
}
} // namespace nvcv::legacy::cuda_op
