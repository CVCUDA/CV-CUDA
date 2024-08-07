/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

namespace nvcv::legacy::cuda_op {

using namespace nvcv::legacy::helpers;

template<typename SrcWrapper, typename DstWrapper>
__global__ void flipHorizontal(SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    const int32_t dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();

    int32_t out_height = dstSize.h;
    int32_t out_width  = dstSize.w;

    if (dst_x < out_width && dst_y < out_height)
    {
        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, dst_y, (out_width - 1 - dst_x));
    }
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void flipVertical(SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    const int32_t dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();

    int32_t out_height = dstSize.h;
    int32_t out_width  = dstSize.w;

    if (dst_x < out_width && dst_y < out_height)
    {
        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, (out_height - 1 - dst_y), dst_x);
    }
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void flipHorizontalVertical(SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    const int32_t dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();

    int32_t out_height = dstSize.h;
    int32_t out_width  = dstSize.w;

    if (dst_x < out_width && dst_y < out_height)
    {
        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, (out_height - 1 - dst_y), (out_width - 1 - dst_x));
    }
}

template<typename SrcWrap, typename DstWrap>
void runFlipKernel(SrcWrap src, DstWrap dst, Size2D dstSize, int numSamples, int32_t flipCode, cudaStream_t stream)
{
    constexpr uint32_t BLOCK = 32;

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(dstSize.w, blockSize.x), divUp(dstSize.h, blockSize.y), numSamples);

    if (flipCode > 0)
    {
        flipHorizontal<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
        checkKernelErrors();
    }
    else if (flipCode == 0)
    {
        flipVertical<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
        checkKernelErrors();
    }
    else
    {
        flipHorizontalVertical<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
        checkKernelErrors();
    }
}

template<typename T>
ErrorCode flip(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output, const int32_t flipCode,
               cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(output);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(input);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(input);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(output);

        runFlipKernel(src, dst, dstSize, outAccess->numSamples(), flipCode, stream);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif // CUDA_DEBUG_LOG

    return ErrorCode::SUCCESS;
}

ErrorCode Flip::infer(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output, const int32_t flipCode,
                      cudaStream_t stream)
{
    if (input.dtype() != output.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << input.dtype() << ") and output (" << output.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat inputFormat  = GetLegacyDataFormat(input.layout());
    DataFormat outputFormat = GetLegacyDataFormat(output.layout());
    if (inputFormat != outputFormat)
    {
        LOG_ERROR("Invalid DataFormat between input (" << inputFormat << ") and output (" << outputFormat << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = inputFormat;
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    cuda_op::DataType dataType = GetLegacyDataType(input.dtype());
    if (!(dataType == kCV_8U || dataType == kCV_16U || dataType == kCV_32S || dataType == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << dataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inputWrapper = TensorDataAccessStridedImagePlanar::Create(input);
    NVCV_ASSERT(inputWrapper);

    cuda_op::DataShape inputShape = GetLegacyDataShape(inputWrapper->infoShape());
    if (inputShape.C > 4)
    {
        LOG_ERROR("Invalid channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // using flip_t = void(const TensorDataStridedCuda & input,
    //                     const TensorDataStridedCuda & output,
    //                     const int32_t flipCode, cudaStream_t stream);
    typedef ErrorCode (*flip_t)(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output,
                                const int32_t flipCode, cudaStream_t stream);

    static const flip_t funcs[6][4] = {
        {  flip<uchar>, 0,  flip<uchar3>,  flip<uchar4>},
        {            0, 0,             0,             0},
        { flip<ushort>, 0, flip<ushort3>, flip<ushort4>},
        {            0, 0,             0,             0},
        {flip<int32_t>, 0,    flip<int3>,    flip<int4>},
        {  flip<float>, 0,  flip<float3>,  flip<float4>}
    };

    const int32_t channels = inputShape.C;
    return funcs[dataType][channels - 1](input, output, flipCode, stream);
}

} // namespace nvcv::legacy::cuda_op
