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
#include "filter_utils.cuh"

#include <nvcv/TensorDataAccess.hpp>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

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

static ErrorCode validateParameterTensor(const TensorDataStridedCuda &tensor, nvcv::DataType expectedType,
                                         int expectedLength, const char *name)
{
    if (tensor.dtype() != expectedType)
    {
        LOG_ERROR("Invalid " << name << " DataType " << tensor.dtype());
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int rank = tensor.layout().rank();
    if (rank != 1)
    {
        LOG_ERROR("Invalid " << name << " Dim " << rank);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (tensor.shape()[0] != expectedLength)
    {
        LOG_ERROR("Invalid " << name << " length " << tensor.shape()[0] << ", expected " << expectedLength);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (tensor.stride(0) != expectedType.strideBytes())
    {
        LOG_ERROR("Invalid " << name << " stride " << tensor.stride(0));
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    return ErrorCode::SUCCESS;
}

template<int XSteps, bool ReuseCoefficients, typename CMP, cuda::RoundMode RM, typename SrcWrapper, typename DstWrapper>
__global__ void adaptive_threshold(const SrcWrapper src, DstWrapper dst,
                                   cuda::Tensor1DWrap<double, int32_t> maxValueArr,
                                   cuda::Tensor1DWrap<int, int32_t>    blockSizeArr,
                                   cuda::Tensor1DWrap<double, int32_t> cArr, cuda::Tensor3DWrap<float, int32_t> kernel)
{
    const int batch_idx  = get_batch_idx();
    int       out_x      = blockIdx.x * BLOCK_DIM_X * XSteps;
    int       out_y      = blockIdx.y * BLOCK_DIM_Y;
    int       out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    // var shape version, the upper-left corner may be invalid
    if (out_x >= out_width || out_y >= out_height)
        return;
    const int         kbs = blockSizeArr[batch_idx];
    const int         r   = kbs >> 1;
    // (2 * r + BLOCK_DIM_X * XSteps) * (2 * r + BLOCK_DIM_Y) + kbs * kbs * sizeof(float)
    extern __shared__ __align__(sizeof(float)) uchar s[];
    const int                                        s_width  = 2 * r + BLOCK_DIM_X * XSteps;
    const int                                        s_height = 2 * r + BLOCK_DIM_Y;
    float                                           *s_k      = (float *)(s + s_width * s_height); // for kernel
    // load image data into shared memory
    const int                                        shift_x = blockIdx.x * BLOCK_DIM_X * XSteps - r;
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
    const int kernel_size = kbs * kbs;
    int       local_idx   = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    while (local_idx < kernel_size)
    {
        int i          = local_idx / kbs;
        int j          = local_idx - i * kbs;
        s_k[local_idx] = *kernel.ptr(batch_idx, i, j);
        local_idx += BLOCK_DIM_X * BLOCK_DIM_Y;
    }
    __syncthreads();

    // calculate convolution
    out_x += threadIdx.x;
    out_y += threadIdx.y;
    if (out_x >= out_width || out_y >= out_height)
        return;
    CMP         cmp;
    const uchar maxv  = cuda::SaturateCast<uchar>(maxValueArr[batch_idx]);
    const int   delta = cuda::round<RM>(cArr[batch_idx]);
    if constexpr (ReuseCoefficients)
    {
        // Accumulate the XSteps outputs together so each shared kernel coefficient is loaded once.
        // Each output still visits coefficients in the original row-major order, preserving rounding.
        float  res[XSteps]{};
        int    kInd    = 0;
        int    start_x = out_x - shift_x - r;
        int    start_y = out_y - shift_y - r;
        uchar *p       = s + start_y * s_width + start_x;
        for (int i = 0; i < kbs; ++i)
        {
            for (int j = 0; j < kbs; ++j)
            {
                float coeff = s_k[kInd++];
#pragma unroll
                for (int k = 0; k < XSteps; ++k)
                {
                    res[k] += p[j + k * BLOCK_DIM_X] * coeff;
                }
            }
            p += s_width;
        }

#pragma unroll
        for (int k = 0; k < XSteps; ++k)
        {
            uchar t = cmp(s[(out_y - shift_y) * s_width + out_x - shift_x] + delta, cuda::SaturateCast<uchar>(res[k]))
                        ? maxv
                        : 0;
            *dst.ptr(batch_idx, out_y, out_x) = t;
            out_x += BLOCK_DIM_X;
            if (out_x >= out_width)
                return;
        }
    }
    else
    {
#pragma unroll
        for (int k = 0; k < XSteps; ++k)
        {
            float  res     = 0.f;
            int    kInd    = 0;
            int    start_x = out_x - shift_x - r;
            int    start_y = out_y - shift_y - r;
            uchar *p       = s + start_y * s_width + start_x;
            for (int i = 0; i < kbs; ++i)
            {
                for (int j = 0; j < kbs; ++j)
                {
                    res += p[j] * s_k[kInd++];
                }
                // next row in shared memory
                p += s_width;
            }
            uchar t = cmp(s[(out_y - shift_y) * s_width + out_x - shift_x] + delta, cuda::SaturateCast<uchar>(res))
                        ? maxv
                        : 0;
            *dst.ptr(batch_idx, out_y, out_x) = t;
            out_x += BLOCK_DIM_X;
            if (out_x >= out_width)
                return;
        }
    }
}

template<int XSteps, bool ReuseCoefficients, typename D, NVCVBorderType B, typename CMP>
void adaptive_threshold_caller_impl(const ImageBatchVarShapeDataStridedCuda &in,
                                    const ImageBatchVarShapeDataStridedCuda &out,
                                    cuda::Tensor1DWrap<double, int32_t>      maxValueArr,
                                    NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                                    cuda::Tensor1DWrap<int, int32_t>    blockSizeArr,
                                    cuda::Tensor1DWrap<double, int32_t> cArr, cuda::Tensor3DWrap<float, int32_t> kernel,
                                    int maxBlockSize, cudaStream_t stream)
{
    (void)thresholdType;
    float                                borderValue = .0f;
    cuda::BorderVarShapeWrap<const D, B> src(in, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(out);

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    int  maxHeight = in.maxSize().h;
    int  maxWidth  = in.maxSize().w;
    dim3 grid(divUp(maxWidth, block.x), divUp(maxHeight, block.y), out.numImages());
    int  s_mem_size = (maxBlockSize - 1 + BLOCK_DIM_X * XSteps) * (maxBlockSize - 1 + BLOCK_DIM_Y)
                   + maxBlockSize * maxBlockSize * sizeof(float);

    if constexpr (std::is_same_v<CMP, MyGreater<int>>)
    {
        adaptive_threshold<XSteps, ReuseCoefficients, CMP, cuda::RoundMode::UP>
            <<<grid, block, s_mem_size, stream>>>(src, dst, maxValueArr, blockSizeArr, cArr, kernel);
    }
    else
    {
        adaptive_threshold<XSteps, ReuseCoefficients, CMP, cuda::RoundMode::DOWN>
            <<<grid, block, s_mem_size, stream>>>(src, dst, maxValueArr, blockSizeArr, cArr, kernel);
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B, typename CMP>
void adaptive_threshold_caller(const ImageBatchVarShapeDataStridedCuda &in,
                               const ImageBatchVarShapeDataStridedCuda &out,
                               cuda::Tensor1DWrap<double, int32_t>      maxValueArr,
                               NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                               cuda::Tensor1DWrap<int, int32_t> blockSizeArr, cuda::Tensor1DWrap<double, int32_t> cArr,
                               cuda::Tensor3DWrap<float, int32_t> kernel, int maxBlockSize,
                               AdaptiveThresholdKernelPolicy kernelPolicy, cudaStream_t stream)
{
    if (kernelPolicy == AdaptiveThresholdKernelPolicy::kLegacyX4)
    {
        adaptive_threshold_caller_impl<4, false, D, B, CMP>(in, out, maxValueArr, adaptiveMethod, thresholdType,
                                                            blockSizeArr, cArr, kernel, maxBlockSize, stream);
    }
    else
    {
        adaptive_threshold_caller_impl<8, true, D, B, CMP>(in, out, maxValueArr, adaptiveMethod, thresholdType,
                                                           blockSizeArr, cArr, kernel, maxBlockSize, stream);
    }
}

AdaptiveThresholdVarShape::AdaptiveThresholdVarShape(DataShape maxInputShape, DataShape maxOutputShape,
                                                     int32_t maxBlockSize, int32_t maxVarShapeBatchSize,
                                                     AdaptiveThresholdKernelPolicy kernelPolicy)
    : CudaBaseOp(maxInputShape, maxOutputShape)
    , m_maxBatchSize(maxVarShapeBatchSize)
    , m_maxBlockSize(maxBlockSize)
    , m_kernelPolicy(kernelPolicy)
{
    if (maxBlockSize <= 0)
    {
        LOG_ERROR("Invalid num of max block size " << maxBlockSize);
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBlockSize must be >= 0");
    }
    if (maxVarShapeBatchSize > 0)
    {
        size_t bufferSize = sizeof(float) * maxVarShapeBatchSize * maxBlockSize * maxBlockSize;
        NVCV_CHECK_LOG(cudaMalloc(&m_kernel, bufferSize));
    }
}

AdaptiveThresholdVarShape::~AdaptiveThresholdVarShape()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

ErrorCode AdaptiveThresholdVarShape::infer(const ImageBatchVarShapeDataStridedCuda &in,
                                           const ImageBatchVarShapeDataStridedCuda &out,
                                           const TensorDataStridedCuda             &maxValue,
                                           const NVCVAdaptiveThresholdType          adaptiveMethod,
                                           const NVCVThresholdType                  thresholdType,
                                           const TensorDataStridedCuda &blockSize, const TensorDataStridedCuda &c,
                                           cudaStream_t stream)
{
    if (m_maxBatchSize <= 0 || in.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (!in.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!out.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(in);
    DataFormat output_format = helpers::GetLegacyDataFormat(out);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    const bool isPlanar = format == kNCHW || format == kCHW;
    if (!(format == kNHWC || format == kHWC || isPlanar))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(in.uniqueFormat());

    if (data_type != kCV_8U)
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = in.uniqueFormat().numChannels();

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

    const int numImages = in.numImages();
    ErrorCode status    = validateParameterTensor(maxValue, nvcv::TYPE_F64, numImages, "maxValue");
    if (status != ErrorCode::SUCCESS)
    {
        return status;
    }

    status = validateParameterTensor(blockSize, nvcv::TYPE_S32, numImages, "blockSize");
    if (status != ErrorCode::SUCCESS)
    {
        return status;
    }

    status = validateParameterTensor(c, nvcv::TYPE_F64, numImages, "c");
    if (status != ErrorCode::SUCCESS)
    {
        return status;
    }

    auto blockSizeAccess = nvcv::TensorDataAccessStrided::Create(blockSize);
    NVCV_ASSERT(blockSizeAccess);

    std::vector<int> blockSizes(numImages);
    checkCudaErrors(cudaMemcpy2DAsync(blockSizes.data(), sizeof(int), blockSizeAccess->sampleData(0),
                                      blockSizeAccess->sampleStride(), sizeof(int), numImages, cudaMemcpyDeviceToHost,
                                      stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        const int bsize = blockSizes[i];
        if (!(bsize % 2 == 1 && bsize > 1 && bsize <= m_maxBlockSize))
        {
            LOG_ERROR("Invalid BlockSize " << bsize);
            return ErrorCode::INVALID_PARAMETER;
        }
    }

    cuda::Tensor1DWrap<double, int32_t> maxValueArr(maxValue);
    cuda::Tensor1DWrap<int, int32_t>    blockSizeArr(blockSize);
    cuda::Tensor1DWrap<double, int32_t> cArr(c);

    int kernelPitch2 = static_cast<int>(m_maxBlockSize * sizeof(float));
    int kernelPitch1 = m_maxBlockSize * kernelPitch2;

    float                             *kernelPtr = m_kernel;
    cuda::Tensor3DWrap<float, int32_t> kernelTensor(kernelPtr, kernelPitch1, kernelPitch2);

    dim3 block(32, 4);
    dim3 grid(divUp(m_maxBlockSize, block.x), divUp(m_maxBlockSize, block.y), out.numImages());

    if (adaptiveMethod == NVCV_ADAPTIVE_THRESH_MEAN_C)
    {
        computeMeanKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, blockSizeArr);
    }
    else
    {
        computeGaussianKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, blockSizeArr);
    }
    checkKernelErrors();

    if (thresholdType == NVCV_THRESH_BINARY)
    {
        adaptive_threshold_caller<uchar, NVCV_BORDER_REPLICATE, MyGreater<int>>(
            in, out, maxValueArr, adaptiveMethod, thresholdType, blockSizeArr, cArr, kernelTensor, m_maxBlockSize,
            m_kernelPolicy, stream);
    }
    else
    {
        adaptive_threshold_caller<uchar, NVCV_BORDER_REPLICATE, MyLessEqual<int>>(
            in, out, maxValueArr, adaptiveMethod, thresholdType, blockSizeArr, cArr, kernelTensor, m_maxBlockSize,
            m_kernelPolicy, stream);
    }
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
