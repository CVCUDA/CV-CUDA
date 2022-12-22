/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template<typename T, class SrcWrapper, class DstWrapper>
__global__ void dilate(SrcWrapper src, DstWrapper dst, Size2D dstSize, Size2D kernelSize, int2 kernelAnchor, T maxmin)
{
    using PT = typename DstWrapper::ValueType;
    PT res   = cuda::SetAll<PT>(maxmin);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int3 coord{x, y, batch_idx};

    for (int i = 0; i < kernelSize.h; ++i)
    {
        coord.y = y - kernelAnchor.y + i;
        for (int j = 0; j < kernelSize.w; ++j)
        {
            coord.x = x - kernelAnchor.x + j;
            res     = cuda::max(res, src[coord]);
        }
    }
    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<typename T, class SrcWrapper, class DstWrapper>
__global__ void erode(SrcWrapper src, DstWrapper dst, Size2D dstSize, Size2D kernelSize, int2 kernelAnchor, T maxmin)
{
    using PT = typename DstWrapper::ValueType;
    PT res   = cuda::SetAll<PT>(maxmin);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int3 coord{x, y, batch_idx};

    for (int i = 0; i < kernelSize.h; ++i)
    {
        coord.y = y - kernelAnchor.y + i;
        for (int j = 0; j < kernelSize.w; ++j)
        {
            coord.x = x - kernelAnchor.x + j;
            res     = cuda::min(res, src[coord]);
        }
    }
    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<typename D, NVCVBorderType B>
void MorphFilter2DCaller(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                         NVCVMorphologyType morph_type, Size2D kernelSize, int2 kernelAnchor, cudaStream_t stream)
{
    using BT = cuda::BaseType<D>;
    BT val   = (morph_type == NVCVMorphologyType::NVCV_DILATE) ? std::numeric_limits<BT>::min()
                                                               : std::numeric_limits<BT>::max();

    cuda::BorderWrapNHW<const D, B> src(inData, cuda::SetAll<D>(val));
    cuda::Tensor3DWrap<D>           dst(outData);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);
    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};
    dim3   block(16, 16);
    dim3   grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    if (morph_type == NVCVMorphologyType::NVCV_ERODE)
    {
        erode<BT><<<grid, block, 0, stream>>>(src, dst, dstSize, kernelSize, kernelAnchor, val);
        checkKernelErrors();
    }
    else if (morph_type == NVCVMorphologyType::NVCV_DILATE)
    {
        dilate<BT><<<grid, block, 0, stream>>>(src, dst, dstSize, kernelSize, kernelAnchor, val);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void MorphFilter2D(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                   NVCVMorphologyType morph_type, Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode,
                   cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_MORPH_CASE(BORDERTYPE)                                                                        \
    case BORDERTYPE:                                                                                       \
        MorphFilter2DCaller<D, BORDERTYPE>(inData, outData, morph_type, kernelSize, kernelAnchor, stream); \
        break

        NVCV_MORPH_CASE(NVCV_BORDER_CONSTANT);
        NVCV_MORPH_CASE(NVCV_BORDER_REPLICATE);
        NVCV_MORPH_CASE(NVCV_BORDER_REFLECT);
        NVCV_MORPH_CASE(NVCV_BORDER_WRAP);
        NVCV_MORPH_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_MORPH_CASE
    default:
        NVCV_ASSERT("Unknown bortertype");
        break;
    }
}

ErrorCode Morphology::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                            NVCVMorphologyType morph_type, Size2D mask_size, int2 anchor, int iteration,
                            const NVCVBorderType borderMode, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());
    DataType   data_type     = GetLegacyDataType(inData.dtype());

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());
    int       channels    = input_shape.C;

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

    if (!(channels == 1 || channels == 3 || channels == 4))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }
    Size2D mask_size_ = mask_size;
    if (mask_size.w == -1 || mask_size.h == -1)
    {
        mask_size_.w = 3;
        mask_size_.h = 3;
    }

    int2 anchor_ = anchor;
    normalizeAnchor(anchor_, mask_size_);

    if (iteration == 0 || mask_size_.w * mask_size_.h == 1)
    {
        // just a unity copy here
        for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
        {
            nvcv::Byte *inSampData  = inAccess->sampleData(i);
            nvcv::Byte *outSampData = outAccess->sampleData(i);

            for (int32_t p = 0; p < inAccess->numPlanes(); ++p)
            {
                checkCudaErrors(cudaMemcpy2DAsync(outAccess->planeData(p, outSampData), outAccess->rowStride(),
                                                  inAccess->planeData(p, inSampData), inAccess->rowStride(),
                                                  inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                                  cudaMemcpyDeviceToDevice, stream));
            }
        }
        return SUCCESS;
    }

    mask_size_.w = mask_size_.w + (iteration - 1) * (mask_size_.w - 1);
    mask_size_.h = mask_size_.h + (iteration - 1) * (mask_size_.h - 1);
    anchor_      = anchor_ * iteration;

    typedef void (*filter2D_t)(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                               NVCVMorphologyType morph_type, Size2D kernelSize, int2 kernelAnchor,
                               NVCVBorderType borderMode, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { MorphFilter2D<uchar>, 0,  MorphFilter2D<uchar3>,  MorphFilter2D<uchar4>},
        {                    0, 0,                      0,                      0},
        {MorphFilter2D<ushort>, 0, MorphFilter2D<ushort3>, MorphFilter2D<ushort4>},
        {                    0, 0,                      0,                      0},
        {                    0, 0,                      0,                      0},
        { MorphFilter2D<float>, 0,  MorphFilter2D<float3>,  MorphFilter2D<float4>},
    };

    funcs[data_type][channels - 1](inData, outData, morph_type, mask_size_, anchor_, borderMode, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
