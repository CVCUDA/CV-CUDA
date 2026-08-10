/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <cvcuda/cuda_tools/Compat.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>

#include <cstddef>
#include <cstdio>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

template<int NRY, class SrcWrapper, class DstWrapper>
__global__ void custom_crop_kernel(const SrcWrapper src, DstWrapper dst, int start_x, int start_y, int width,
                                   int height)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = (blockIdx.y * blockDim.y + threadIdx.y) * NRY;
    const int batch_idx = get_batch_idx();
    if (x >= width)
        return;

#pragma unroll
    for (int i = 0; i < NRY; ++i)
    {
        if (y + i < height)
        {
            *dst.ptr(batch_idx, y + i, x) = *src.ptr(batch_idx, y + i + start_y, x + start_x);
        }
    }
}

template<typename T>
ErrorCode customCrop(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                     NVCVRectI roi, cudaStream_t stream)
{
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    constexpr int blockWidth    = sizeof(T) <= 3 ? 32 : 16;
    constexpr int rowsPerThread = sizeof(T) == 3 ? 2 : 1;
    dim3          block(blockWidth, 256 / blockWidth);
    dim3          grid(divUp(roi.width, block.x), divUp(roi.height, block.y * rowsPerThread), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= nvcv::cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        custom_crop_kernel<rowsPerThread><<<grid, block, 0, stream>>>(src, dst, roi.x, roi.y, roi.width, roi.height);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

static bool tryCopyDenseU8NHWC(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                               NVCVRectI roi, cudaStream_t stream)
{
    if (inData.rank() != 4 || outData.rank() != 4 || inData.layout() != nvcv::TENSOR_NHWC
        || outData.layout() != nvcv::TENSOR_NHWC || inData.dtype() != outData.dtype())
    {
        return false;
    }

    const auto bitsPerChannel = inData.dtype().bitsPerChannel();
    if (bitsPerChannel[0] != 8)
    {
        return false;
    }
    const int64_t bytesPerChannel = bitsPerChannel[0] / 8;

    const int64_t inBatch = inData.shape(0);
    const int64_t inRows  = inData.shape(1);
    const int64_t inCols  = inData.shape(2);
    const int64_t inCh    = inData.shape(3);

    const int64_t outBatch = outData.shape(0);
    const int64_t outRows  = outData.shape(1);
    const int64_t outCols  = outData.shape(2);
    const int64_t outCh    = outData.shape(3);

    const int64_t pixelBytes = inCh * bytesPerChannel;
    if (inBatch != outBatch || inCh != 1 || outCh != 1 || inData.stride(3) != bytesPerChannel
        || outData.stride(3) != bytesPerChannel || inData.stride(2) != pixelBytes || outData.stride(2) != pixelBytes
        || inData.stride(1) < inCols * pixelBytes || outData.stride(1) < outCols * pixelBytes
        || inData.stride(0) != inData.stride(1) * inRows || outData.stride(0) != outData.stride(1) * outRows)
    {
        return false;
    }

    cudaMemcpy3DParms params{};
    params.srcPtr
        = make_cudaPitchedPtr(inData.basePtr(), static_cast<std::size_t>(inData.stride(1)),
                              static_cast<std::size_t>(inCols * pixelBytes), static_cast<std::size_t>(inRows));
    params.dstPtr
        = make_cudaPitchedPtr(outData.basePtr(), static_cast<std::size_t>(outData.stride(1)),
                              static_cast<std::size_t>(outCols * pixelBytes), static_cast<std::size_t>(outRows));
    params.srcPos = make_cudaPos(static_cast<std::size_t>(roi.x * pixelBytes), static_cast<std::size_t>(roi.y), 0);
    params.dstPos = make_cudaPos(0, 0, 0);
    params.extent = make_cudaExtent(static_cast<std::size_t>(roi.width * pixelBytes),
                                    static_cast<std::size_t>(roi.height), static_cast<std::size_t>(inBatch));
    params.kind   = cudaMemcpyDeviceToDevice;

    checkCudaErrors(cudaMemcpy3DAsync(&params, stream));
    return true;
}

namespace nvcv::legacy::cuda_op {

ErrorCode CustomCrop::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, NVCVRectI roi,
                            cudaStream_t stream)
{
    if (roi.width <= 0 || roi.height <= 0)
    {
        LOG_ERROR("ROI width and height must be positive");
        return ErrorCode::INVALID_PARAMETER;
    }

    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invliad DataFormat both Input and Output must be kHWC or kHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const cuda_op::DataType data_type = GetLegacyDataType(inData.dtype());

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_16F || data_type == kCV_32S || data_type == kCV_32F || data_type == kCV_64F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (batch != outAccess->numSamples() || channels != outAccess->numChannels())
    {
        LOG_ERROR("Input and output must have the same sample and channel counts");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (roi.height > outAccess->size().h || roi.width > outAccess->size().w)
    {
        LOG_ERROR("ROI larger than dst buffer");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int data_size = DataSize(GetLegacyDataType(inData.dtype()));
    int start_x   = roi.x;
    int start_y   = roi.y;
    int end_x     = start_x + roi.width - 1;
    int end_y     = start_y + roi.height - 1;
#ifdef CUDA_DEBUG_LOG
    LOG_ERROR("x " << roi.x << " y " << roi.y << " width " << roi.width << " height " << roi.height);
#endif

    if (start_x < 0 || start_y < 0 || end_x >= cols || end_y >= rows)
    {
        LOG_ERROR("Invalid Roi range x " << roi.x << " y " << roi.y << " width " << roi.width << " height "
                                         << roi.height);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (tryCopyDenseU8NHWC(inData, outData, roi, stream))
    {
        return ErrorCode::SUCCESS;
    }

    typedef ErrorCode (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                NVCVRectI roi, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {customCrop<uchar1>,  customCrop<uchar2>,  customCrop<uchar3>,      customCrop<uchar4>},
        {customCrop<ushort>, customCrop<ushort2>, customCrop<ushort3>,     customCrop<ushort4>},
        {   customCrop<int>,    customCrop<int2>,    customCrop<int3>,        customCrop<int4>},
        {                 0,                   0,                   0,                       0},
        {customCrop<double>, customCrop<double2>, customCrop<double3>, customCrop<double4_16a>}
    };

    return funcs[data_size / 2][channels - 1](inData, outData, roi, stream);
}

} // namespace nvcv::legacy::cuda_op
