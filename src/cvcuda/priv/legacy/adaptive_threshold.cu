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
#include "filter_utils.cuh"

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

template<typename T>
struct MyGreater
{
    __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const
    {
        return lhs > rhs;
    }
};

template<typename T>
struct MyLessEqual
{
    __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const
    {
        return lhs <= rhs;
    }
};

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define X_STEPS     4

template<typename CMP, typename SrcWrapper, typename DstWrapper, typename KernelWrapper>
__global__ void adaptive_threshold(SrcWrapper src, DstWrapper dst, Size2D dstSize, const uchar maxValue,
                                   KernelWrapper kernel, const int blockSize, const int idelta)
{
    const int         batch_idx = get_batch_idx();
    const int         r         = blockSize >> 1;
    // (2 * r + BLOCK_DIM_X * X_STEPS) * (2 * r + BLOCK_DIM_Y) + blockSize * blockSize * sizeof(float)
    extern __shared__ __align__(sizeof(float)) uchar s[];
    const int                                        s_width  = 2 * r + BLOCK_DIM_X * X_STEPS;
    const int                                        s_height = 2 * r + BLOCK_DIM_Y;
    float                                           *s_k      = (float *)(s + s_width * s_height); // for kernel
    // load image data into shared memory
    const int                                        shift_x = blockIdx.x * BLOCK_DIM_X * X_STEPS - r;
    const int                                        shift_y = blockIdx.y * BLOCK_DIM_Y - r;
    int3                                             srcCoord{0, 0, batch_idx};
    for (int start_y = 0; start_y < s_height; start_y += BLOCK_DIM_Y)
    {
        int local_y = start_y + threadIdx.y;
        srcCoord.y  = shift_y + local_y;
        for (int start_x = 0; start_x < s_width; start_x += BLOCK_DIM_X)
        {
            int local_x = start_x + threadIdx.x;
            srcCoord.x  = shift_x + local_x;

            if (local_y < s_height && local_x < s_width)
            {
                s[local_y * s_width + local_x] = src[srcCoord];
            }
        }
    }
    // load kernel data into shared memory
    const int kernel_size = blockSize * blockSize;
    int       local_idx   = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    while (local_idx < kernel_size)
    {
        s_k[local_idx] = kernel[local_idx];
        local_idx += BLOCK_DIM_X * BLOCK_DIM_Y;
    }
    __syncthreads();

    // calculate convolution
    int out_x = blockIdx.x * BLOCK_DIM_X * X_STEPS + threadIdx.x;
    int out_y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    if (out_x >= dstSize.w || out_y >= dstSize.h)
        return;
    CMP cmp;
#pragma unroll
    for (int k = 0; k < X_STEPS; ++k)
    {
        float  res     = 0.f;
        int    kInd    = 0;
        int    start_x = out_x - shift_x - r;
        int    start_y = out_y - shift_y - r;
        uchar *p       = s + start_y * s_width + start_x;
        for (int i = 0; i < blockSize; ++i)
        {
            for (int j = 0; j < blockSize; ++j)
            {
                res += p[j] * s_k[kInd++];
            }
            // next row in shared memory
            p += s_width;
        }
        uchar t = cmp(s[(out_y - shift_y) * s_width + out_x - shift_x] + idelta, cuda::SaturateCast<uchar>(res))
                    ? maxValue
                    : 0;
        *dst.ptr(batch_idx, out_y, out_x) = t;
        out_x += BLOCK_DIM_X;
        if (out_x >= dstSize.w)
            return;
    }
}

template<typename T, NVCVBorderType B, typename CMP, class KernelWrapper>
void adaptive_threshold_caller(const TensorDataStridedCuda &in, const TensorDataStridedCuda &out, const uchar maxValue,
                               KernelWrapper kernel, const int blockSize, const int idelta, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    auto src = cuda::CreateBorderWrapNHW<const T, B>(in, cuda::SetAll<T>(0.f));
    auto dst = cuda::CreateTensorWrapNHW<T>(out);

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(divUp(dstSize.w, BLOCK_DIM_X * X_STEPS), divUp(dstSize.h, block.y), outAccess->numSamples());

    int s_mem_size = (blockSize - 1 + BLOCK_DIM_X * X_STEPS) * (blockSize - 1 + BLOCK_DIM_Y)
                   + blockSize * blockSize * sizeof(float);
    adaptive_threshold<CMP>
        <<<grid, block, s_mem_size, stream>>>(src, dst, dstSize, maxValue, kernel, blockSize, idelta);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

AdaptiveThreshold::AdaptiveThreshold(DataShape maxInputShape, DataShape maxOutputShape, int32_t maxBlockSize)
    : CudaBaseOp(maxInputShape, maxOutputShape)
{
    if (maxBlockSize <= 0)
    {
        LOG_ERROR("Invalid num of max block size " << maxBlockSize);
        throw std::runtime_error("Parameter error!");
    }
    size_t bufferSize = maxBlockSize * maxBlockSize * sizeof(float);
    NVCV_CHECK_THROW(cudaMalloc(&m_kernel, bufferSize));
}

AdaptiveThreshold::~AdaptiveThreshold()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

size_t AdaptiveThreshold::calBufferSize(DataShape maxInputShape, DataShape maxOutputShape, int maxBlockSize)
{
    return maxBlockSize * maxBlockSize * sizeof(float);
}

ErrorCode AdaptiveThreshold::infer(const TensorDataStridedCuda &in, const TensorDataStridedCuda &out,
                                   const double maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const int32_t blockSize, const double c,
                                   cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(in.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(out.layout());

    if (in.dtype() != out.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << in.dtype()
                                                                         << " output format = " << out.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (input_format != output_format)
    {
        LOG_ERROR("Input data format (" << input_format << ") and output data format (" << output_format
                                        << ") must be the same.");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if ((input_format != kNHWC) && (input_format != kHWC))
    {
        LOG_ERROR("Invalid DataFormat both Input and Output must be kHWC or kNHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(in);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(in.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());
    int                channels    = input_shape.C;

    if (data_type != kCV_8U)
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (channels != 1)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (adaptiveMethod != NVCV_ADAPTIVE_THRESH_MEAN_C && adaptiveMethod != NVCV_ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        LOG_ERROR("Invalid AdaptiveMethod " << adaptiveMethod);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (thresholdType != NVCV_THRESH_BINARY && thresholdType != NVCV_THRESH_BINARY_INV)
    {
        LOG_ERROR("Invalid ThresholdType " << thresholdType);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(blockSize % 2 == 1 && blockSize > 1))
    {
        LOG_ERROR("Invalid BlockSize " << blockSize);
        return ErrorCode::INVALID_PARAMETER;
    }

    float *kernelPtr = (float *)m_kernel;
    if (m_adaptiveMethod != adaptiveMethod || m_blockSize != blockSize)
    {
        if (adaptiveMethod == NVCV_ADAPTIVE_THRESH_MEAN_C)
        {
            int kSize = blockSize * blockSize;
            computeMeanKernel<<<1, kSize, 0, stream>>>(kernelPtr, kSize);
        }
        else
        {
            dim3    block(32, 4);
            dim3    grid(divUp(blockSize, block.x), divUp(blockSize, block.y));
            Size2D  kernelSize{blockSize, blockSize};
            double2 sigma;
            sigma.x = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8;
            sigma.y = sigma.x;

            computeGaussianKernel<<<grid, block, 0, stream>>>(kernelPtr, kernelSize, sigma);
        }
        m_adaptiveMethod = adaptiveMethod;
        m_blockSize      = blockSize;
    }
    checkKernelErrors();

    uchar imaxval = cuda::SaturateCast<uchar>(maxValue);
    int   idelta  = thresholdType == NVCV_THRESH_BINARY ? (int)std::ceil(c) : (int)std::floor(c);
    if (thresholdType == NVCV_THRESH_BINARY)
    {
        adaptive_threshold_caller<uchar, NVCV_BORDER_REPLICATE, MyGreater<int>>(in, out, imaxval, kernelPtr, blockSize,
                                                                                idelta, stream);
    }
    else
    {
        adaptive_threshold_caller<uchar, NVCV_BORDER_REPLICATE, MyLessEqual<int>>(in, out, imaxval, kernelPtr,
                                                                                  blockSize, idelta, stream);
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
