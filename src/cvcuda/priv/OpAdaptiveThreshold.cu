/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "CudaDeviceUtils.hpp"
#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpAdaptiveThreshold.hpp"
#include "TensorAccessValidation.hpp"

#include "FilterCommon.cuh"

#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuda   = nvcv::cuda;
namespace filter = cvcuda::priv::filter;
namespace util   = nvcv::util;

namespace {

using uchar = unsigned char;

using cvcuda::priv::AdaptiveThresholdKernelPolicy;

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

// Block geometry of the coefficient-generation launches, which are sized by the neighborhood
// extent rather than by the image.
constexpr int kCoeffBlockDimX = 32;
constexpr int kCoeffBlockDimY = 4;

inline AdaptiveThresholdKernelPolicy AdaptiveThresholdKernelPolicyForCurrentDevice()
{
    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));

    return cvcuda::priv::AdaptiveThresholdKernelPolicyForSM(sm);
}

template<typename T>
struct Greater
{
    __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const
    {
        return lhs > rhs;
    }
};

template<typename T>
struct LessEqual
{
    __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const
    {
        return lhs <= rhs;
    }
};

// The coefficient generators below are the AdaptiveThreshold-specific entry points of what used to
// be the legacy filter_utils translation unit. They take nvcv::Size2D rather than the legacy int2,
// and the per-position gaussian weight they share with the filter cluster is
// filter::ComputeSingleGaussianValue in FilterCommon.cuh.

// The tensor path evaluates this on the host and the var-shape path per image, so both must read
// the same expression.
inline __host__ __device__ double2 GaussianSigmaForBlockSize(int blockSize)
{
    double sigma = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8;

    return make_double2(sigma, sigma);
}

__global__ void ComputeMeanKernel(float *kernel_ptr, int k_size)
{
    float kernelVal = 1.0 / k_size;
    int   tid       = threadIdx.x;
    if (tid < k_size)
    {
        kernel_ptr[tid] = kernelVal;
    }
}

__global__ void ComputeGaussianKernel(float *kernel, nvcv::Size2D kernelSize, double2 sigma)
{
    int2 coord = cuda::StaticCast<int>(cuda::DropCast<2>(blockIdx * blockDim + threadIdx));

    if (coord.x >= kernelSize.w || coord.y >= kernelSize.h)
    {
        return;
    }

    int2 half{kernelSize.w / 2, kernelSize.h / 2};

    kernel[coord.y * kernelSize.w + coord.x] = filter::ComputeSingleGaussianValue(coord, half, sigma);
}

__global__ void ComputeMeanKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel,
                                          cuda::Tensor1DWrap<int, int32_t>   blockSizeArr)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    int blockSize = blockSizeArr[coord.z];

    if (coord.x >= blockSize || coord.y >= blockSize)
    {
        return;
    }

    kernel[coord] = 1.f / (blockSize * blockSize);
}

__global__ void ComputeGaussianKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel,
                                              cuda::Tensor1DWrap<int, int32_t>   blockSizeArr)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    int blockSize = blockSizeArr[coord.z];

    if (coord.x >= blockSize || coord.y >= blockSize)
    {
        return;
    }

    int2 half{blockSize / 2, blockSize / 2};

    kernel[coord] = filter::ComputeSingleGaussianValue(coord, half, GaussianSigmaForBlockSize(blockSize));
}

template<int KSize, int XSteps, bool ReuseCoefficients, typename CMP, typename SrcWrapper, typename DstWrapper,
         typename KernelWrapper>
__global__ void AdaptiveThresholdKernel(SrcWrapper src, DstWrapper dst, nvcv::Size2D dstSize, const uchar maxValue,
                                        KernelWrapper kernel, const int blockSize, const int idelta)
{
    const int         batch_idx          = blockIdx.z;
    const int         effectiveBlockSize = KSize > 0 ? KSize : blockSize;
    const int         r                  = effectiveBlockSize >> 1;
    // (2 * r + kBlockDimX * XSteps) * (2 * r + kBlockDimY) + blockSize * blockSize * sizeof(float)
    extern __shared__ __align__(sizeof(float)) uchar s[];
    const int                                        s_width  = 2 * r + kBlockDimX * XSteps;
    const int                                        s_height = 2 * r + kBlockDimY;
    float                                           *s_k      = (float *)(s + s_width * s_height);
    const int                                        shift_x  = blockIdx.x * kBlockDimX * XSteps - r;
    const int                                        shift_y  = blockIdx.y * kBlockDimY - r;
    int3                                             srcCoord{0, 0, batch_idx};
    for (int start_y = 0; start_y < s_height; start_y += kBlockDimY)
    {
        int local_y = start_y + threadIdx.y;
        srcCoord.y  = shift_y + local_y;
        for (int start_x = 0; start_x < s_width; start_x += kBlockDimX)
        {
            int local_x = start_x + threadIdx.x;
            srcCoord.x  = shift_x + local_x;

            if (local_y < s_height && local_x < s_width)
            {
                s[local_y * s_width + local_x] = src[srcCoord];
            }
        }
    }
    const int kernel_size = effectiveBlockSize * effectiveBlockSize;
    int       local_idx   = threadIdx.y * kBlockDimX + threadIdx.x;
    while (local_idx < kernel_size)
    {
        s_k[local_idx] = kernel[local_idx];
        local_idx += kBlockDimX * kBlockDimY;
    }
    __syncthreads();

    int out_x = blockIdx.x * kBlockDimX * XSteps + threadIdx.x;
    int out_y = blockIdx.y * kBlockDimY + threadIdx.y;
    if (out_x >= dstSize.w || out_y >= dstSize.h)
        return;
    CMP cmp;
    if constexpr (ReuseCoefficients)
    {
        // Accumulate the XSteps outputs together so each shared kernel coefficient is loaded once.
        // Each output still visits coefficients in the original row-major order, preserving rounding.
        float  res[XSteps]{};
        int    kInd    = 0;
        int    start_x = out_x - shift_x - r;
        int    start_y = out_y - shift_y - r;
        uchar *p       = s + start_y * s_width + start_x;
        if constexpr (KSize > 0)
        {
#pragma unroll
            for (int i = 0; i < KSize; ++i)
            {
#pragma unroll
                for (int j = 0; j < KSize; ++j)
                {
                    float coeff = s_k[kInd++];
#pragma unroll
                    for (int k = 0; k < XSteps; ++k)
                    {
                        res[k] += p[j + k * kBlockDimX] * coeff;
                    }
                }
                p += s_width;
            }
        }
        else
        {
            for (int i = 0; i < blockSize; ++i)
            {
                for (int j = 0; j < blockSize; ++j)
                {
                    float coeff = s_k[kInd++];
#pragma unroll
                    for (int k = 0; k < XSteps; ++k)
                    {
                        res[k] += p[j + k * kBlockDimX] * coeff;
                    }
                }
                p += s_width;
            }
        }

#pragma unroll
        for (int k = 0; k < XSteps; ++k)
        {
            uchar t = cmp(s[(out_y - shift_y) * s_width + out_x - shift_x] + idelta, cuda::SaturateCast<uchar>(res[k]))
                        ? maxValue
                        : 0;
            *dst.ptr(batch_idx, out_y, out_x) = t;
            out_x += kBlockDimX;
            if (out_x >= dstSize.w)
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
            if constexpr (KSize > 0)
            {
#pragma unroll
                for (int i = 0; i < KSize; ++i)
                {
#pragma unroll
                    for (int j = 0; j < KSize; ++j)
                    {
                        res += p[j] * s_k[kInd++];
                    }
                    p += s_width;
                }
            }
            else
            {
                for (int i = 0; i < blockSize; ++i)
                {
                    for (int j = 0; j < blockSize; ++j)
                    {
                        res += p[j] * s_k[kInd++];
                    }
                    p += s_width;
                }
            }
            uchar t = cmp(s[(out_y - shift_y) * s_width + out_x - shift_x] + idelta, cuda::SaturateCast<uchar>(res))
                        ? maxValue
                        : 0;
            *dst.ptr(batch_idx, out_y, out_x) = t;
            out_x += kBlockDimX;
            if (out_x >= dstSize.w)
                return;
        }
    }
}

template<int XSteps, bool ReuseCoefficients, typename CMP, cuda::RoundMode RM, typename SrcWrapper, typename DstWrapper>
__global__ void AdaptiveThresholdVarShapeKernel(const SrcWrapper src, DstWrapper dst,
                                                cuda::Tensor1DWrap<double, int32_t> maxValueArr,
                                                cuda::Tensor1DWrap<int, int32_t>    blockSizeArr,
                                                cuda::Tensor1DWrap<double, int32_t> cArr,
                                                cuda::Tensor3DWrap<float, int32_t>  kernel)
{
    const int batch_idx  = blockIdx.z;
    int       out_x      = blockIdx.x * kBlockDimX * XSteps;
    int       out_y      = blockIdx.y * kBlockDimY;
    int       out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    // var shape version, the upper-left corner may be invalid
    if (out_x >= out_width || out_y >= out_height)
        return;
    const int         kbs = blockSizeArr[batch_idx];
    const int         r   = kbs >> 1;
    // (2 * r + kBlockDimX * XSteps) * (2 * r + kBlockDimY) + kbs * kbs * sizeof(float)
    extern __shared__ __align__(sizeof(float)) uchar s[];
    const int                                        s_width  = 2 * r + kBlockDimX * XSteps;
    const int                                        s_height = 2 * r + kBlockDimY;
    float                                           *s_k      = (float *)(s + s_width * s_height);
    const int                                        shift_x  = blockIdx.x * kBlockDimX * XSteps - r;
    const int                                        shift_y  = blockIdx.y * kBlockDimY - r;
    int3                                             srcCoord{0, 0, batch_idx};
    for (int start_y = 0; start_y < s_height; start_y += kBlockDimY)
    {
        int local_y = start_y + threadIdx.y;
        srcCoord.y  = shift_y + local_y;
        for (int start_x = 0; start_x < s_width; start_x += kBlockDimX)
        {
            int local_x = start_x + threadIdx.x;
            srcCoord.x  = shift_x + local_x;

            if (local_y < s_height && local_x < s_width)
            {
                s[local_y * s_width + local_x] = src[srcCoord];
            }
        }
    }
    const int kernel_size = kbs * kbs;
    int       local_idx   = threadIdx.y * kBlockDimX + threadIdx.x;
    while (local_idx < kernel_size)
    {
        int i          = local_idx / kbs;
        int j          = local_idx - i * kbs;
        s_k[local_idx] = *kernel.ptr(batch_idx, i, j);
        local_idx += kBlockDimX * kBlockDimY;
    }
    __syncthreads();

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
                    res[k] += p[j + k * kBlockDimX] * coeff;
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
            out_x += kBlockDimX;
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
                p += s_width;
            }
            uchar t = cmp(s[(out_y - shift_y) * s_width + out_x - shift_x] + delta, cuda::SaturateCast<uchar>(res))
                        ? maxv
                        : 0;
            *dst.ptr(batch_idx, out_y, out_x) = t;
            out_x += kBlockDimX;
            if (out_x >= out_width)
                return;
        }
    }
}

template<int XSteps, bool ReuseCoefficients, typename T, NVCVBorderType B, typename CMP, class KernelWrapper>
void RunAdaptiveThresholdImpl(const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                              const uchar maxValue, KernelWrapper kernel, const int blockSize, const int idelta,
                              cudaStream_t stream)
{
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in);
    NVCV_ASSERT(inAccess);

    nvcv::Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(kBlockDimX, kBlockDimY);
    // Deliberate divergence from the legacy divUp(), which was ceil((float)a/b) and so returned one
    // block too few once the extent reached 2^24+1 -- dropping the trailing column or row. The int32
    // stride guard below does not bite first: a 1x1x16777217x1 U8 tensor is ~0.8% of INT32_MAX.
    dim3 grid(util::DivUp(dstSize.w, kBlockDimX * XSteps), util::DivUp(dstSize.h, kBlockDimY), outAccess->numSamples());

    int s_mem_size
        = (blockSize - 1 + kBlockDimX * XSteps) * (blockSize - 1 + kBlockDimY) + blockSize * blockSize * sizeof(float);

    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }

    auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(in, cuda::SetAll<T>(0.f));
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(out);

    if (blockSize == 7)
    {
        AdaptiveThresholdKernel<7, XSteps, ReuseCoefficients, CMP>
            <<<grid, block, s_mem_size, stream>>>(src, dst, dstSize, maxValue, kernel, blockSize, idelta);
    }
    else if (blockSize == 3)
    {
        AdaptiveThresholdKernel<3, XSteps, ReuseCoefficients, CMP>
            <<<grid, block, s_mem_size, stream>>>(src, dst, dstSize, maxValue, kernel, blockSize, idelta);
    }
    else
    {
        AdaptiveThresholdKernel<0, XSteps, ReuseCoefficients, CMP>
            <<<grid, block, s_mem_size, stream>>>(src, dst, dstSize, maxValue, kernel, blockSize, idelta);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B, typename CMP, class KernelWrapper>
void RunAdaptiveThreshold(const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                          const uchar maxValue, KernelWrapper kernel, const int blockSize, const int idelta,
                          AdaptiveThresholdKernelPolicy kernelPolicy, cudaStream_t stream)
{
    if (kernelPolicy == AdaptiveThresholdKernelPolicy::kLegacyX4)
    {
        RunAdaptiveThresholdImpl<4, false, T, B, CMP>(in, out, maxValue, kernel, blockSize, idelta, stream);
        return;
    }
    RunAdaptiveThresholdImpl<8, true, T, B, CMP>(in, out, maxValue, kernel, blockSize, idelta, stream);
}

template<int XSteps, bool ReuseCoefficients, typename D, NVCVBorderType B, typename CMP, cuda::RoundMode RM>
void RunAdaptiveThresholdVarShapeImpl(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                      const nvcv::ImageBatchVarShapeDataStridedCuda &out,
                                      cuda::Tensor1DWrap<double, int32_t>            maxValueArr,
                                      cuda::Tensor1DWrap<int, int32_t>               blockSizeArr,
                                      cuda::Tensor1DWrap<double, int32_t>            cArr,
                                      cuda::Tensor3DWrap<float, int32_t> kernel, int maxBlockSize, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(in, cuda::SetAll<D>(0.f));
    cuda::ImageBatchVarShapeWrap<D>      dst(out);

    dim3 block(kBlockDimX, kBlockDimY);
    int  maxHeight = in.maxSize().h;
    int  maxWidth  = in.maxSize().w;
    dim3 grid(util::DivUp(maxWidth, kBlockDimX), util::DivUp(maxHeight, kBlockDimY), out.numImages());
    int  s_mem_size = (maxBlockSize - 1 + kBlockDimX * XSteps) * (maxBlockSize - 1 + kBlockDimY)
                   + maxBlockSize * maxBlockSize * sizeof(float);

    AdaptiveThresholdVarShapeKernel<XSteps, ReuseCoefficients, CMP, RM>
        <<<grid, block, s_mem_size, stream>>>(src, dst, maxValueArr, blockSizeArr, cArr, kernel);

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename D, NVCVBorderType B, typename CMP, cuda::RoundMode RM>
void RunAdaptiveThresholdVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                  const nvcv::ImageBatchVarShapeDataStridedCuda &out,
                                  cuda::Tensor1DWrap<double, int32_t>            maxValueArr,
                                  cuda::Tensor1DWrap<int, int32_t>               blockSizeArr,
                                  cuda::Tensor1DWrap<double, int32_t> cArr, cuda::Tensor3DWrap<float, int32_t> kernel,
                                  int maxBlockSize, AdaptiveThresholdKernelPolicy kernelPolicy, cudaStream_t stream)
{
    if (kernelPolicy == AdaptiveThresholdKernelPolicy::kLegacyX4)
    {
        RunAdaptiveThresholdVarShapeImpl<4, false, D, B, CMP, RM>(in, out, maxValueArr, blockSizeArr, cArr, kernel,
                                                                  maxBlockSize, stream);
        return;
    }
    RunAdaptiveThresholdVarShapeImpl<8, true, D, B, CMP, RM>(in, out, maxValueArr, blockSizeArr, cArr, kernel,
                                                             maxBlockSize, stream);
}

// The four layouts the legacy GetLegacyDataFormat helper could produce for this operator. It is
// reproduced here, rather than replaced by a direct layout comparison, so that the rejections keep
// their original order and status.
enum class ImageDataFormat
{
    kNCHW,
    kCHW,
    kNHWC,
    kHWC
};

inline ImageDataFormat DataFormatForLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return ImageDataFormat::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return ImageDataFormat::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return ImageDataFormat::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return ImageDataFormat::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline ImageDataFormat DataFormatForVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &batch)
{
    nvcv::ImageFormat fmt = cvcuda::priv::legacy_format::RequireUniformVarShapeFormat(batch);

    if (fmt.numPlanes() >= 2)
    {
        return batch.numImages() >= 2 ? ImageDataFormat::kNCHW : ImageDataFormat::kCHW;
    }

    return batch.numImages() >= 2 ? ImageDataFormat::kNHWC : ImageDataFormat::kHWC;
}

inline void ValidateParameterTensor(const nvcv::TensorDataStridedCuda &tensor, nvcv::DataType expectedType,
                                    int expectedLength, const char *name)
{
    if (tensor.dtype() != expectedType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s DataType", name);
    }

    const int rank = tensor.layout().rank();
    if (rank != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s Dim %d", name, rank);
    }

    if (tensor.shape()[0] != expectedLength)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s length %ld, expected %d", name,
                              static_cast<long>(tensor.shape()[0]), expectedLength);
    }

    if (tensor.stride(0) != expectedType.strideBytes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s stride %ld", name,
                              static_cast<long>(tensor.stride(0)));
    }
}

} // namespace

namespace cvcuda::priv {

AdaptiveThreshold::TensorState::TensorState(int32_t maxBlockSize, AdaptiveThresholdKernelPolicy policy)
    : kernelPolicy(policy)
{
    size_t bufferSize = maxBlockSize * maxBlockSize * sizeof(float);
    NVCV_CHECK_THROW(cudaMalloc(&kernel, bufferSize));
}

AdaptiveThreshold::TensorState::~TensorState()
{
    NVCV_CHECK_LOG(cudaFree(kernel));
}

AdaptiveThreshold::VarShapeState::VarShapeState(int32_t maxBlockSize, int32_t maxVarShapeBatchSize,
                                                AdaptiveThresholdKernelPolicy policy)
    : kernelPolicy(policy)
{
    if (maxVarShapeBatchSize > 0)
    {
        size_t bufferSize = sizeof(float) * maxVarShapeBatchSize * maxBlockSize * maxBlockSize;
        NVCV_CHECK_THROW(cudaMalloc(&kernel, bufferSize));
    }
}

AdaptiveThreshold::VarShapeState::~VarShapeState()
{
    NVCV_CHECK_LOG(cudaFree(kernel));
}

AdaptiveThreshold::AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize)
    : m_maxBlockSize(maxBlockSize)
    , m_maxVarShapeBatchSize(maxVarShapeBatchSize)
    // PerDeviceResource runs the factory with the target device already current.
    , m_tensorState(
          [maxBlockSize](int)
          { return std::make_unique<TensorState>(maxBlockSize, AdaptiveThresholdKernelPolicyForCurrentDevice()); })
    , m_varShapeState(
          [maxBlockSize, maxVarShapeBatchSize](int)
          {
              return std::make_unique<VarShapeState>(maxBlockSize, maxVarShapeBatchSize,
                                                     AdaptiveThresholdKernelPolicyForCurrentDevice());
          })
{
    if (maxBlockSize <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBlockSize must be > 0");
    }
}

void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const double maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const int32_t blockSize, const double c) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AdaptiveThreshold::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    // The admitted set is exactly {TYPE_U8}: the packed TYPE_2U8/3U8/4U8 types are rejected here,
    // which is what the frozen negative tests observe.
    if (inData->dtype() != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data type must be U8. Unsupported data type.");
    }

    if (outData->dtype() != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output data type must be U8. Unsupported data type.");
    }

    TensorState &state = m_tensorState.get();

    ImageDataFormat inputFormat  = DataFormatForLayout(inData->layout());
    ImageDataFormat outputFormat = DataFormatForLayout(outData->layout());

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data format and output data format must be the same.");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    NVCV_ASSERT(outAccess);

    // The kernels index the input with the destination's extents, so a larger output would read
    // past the input's samples or rows.
    if (inAccess->numSamples() != outAccess->numSamples() || inAccess->numChannels() != outAccess->numChannels()
        || inAccess->numRows() != outAccess->numRows() || inAccess->numCols() != outAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same shape");
    }

    const int channels = inAccess->numChannels();
    if (channels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (adaptiveMethod != NVCV_ADAPTIVE_THRESH_MEAN_C && adaptiveMethod != NVCV_ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid AdaptiveMethod %d",
                              static_cast<int>(adaptiveMethod));
    }

    if (thresholdType != NVCV_THRESH_BINARY && thresholdType != NVCV_THRESH_BINARY_INV)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid ThresholdType %d",
                              static_cast<int>(thresholdType));
    }

    if (!(blockSize % 2 == 1 && blockSize > 1 && blockSize <= m_maxBlockSize))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid BlockSize %d", blockSize);
    }

    float *kernelPtr = state.kernel;
    if (state.adaptiveMethod != adaptiveMethod || state.blockSize != blockSize)
    {
        if (adaptiveMethod == NVCV_ADAPTIVE_THRESH_MEAN_C)
        {
            int kSize = blockSize * blockSize;
            ComputeMeanKernel<<<1, kSize, 0, stream>>>(kernelPtr, kSize);
        }
        else
        {
            dim3         block(kCoeffBlockDimX, kCoeffBlockDimY);
            dim3         grid(util::DivUp(blockSize, kCoeffBlockDimX), util::DivUp(blockSize, kCoeffBlockDimY));
            nvcv::Size2D kernelSize{blockSize, blockSize};

            ComputeGaussianKernel<<<grid, block, 0, stream>>>(kernelPtr, kernelSize,
                                                              GaussianSigmaForBlockSize(blockSize));
        }
        state.adaptiveMethod = adaptiveMethod;
        state.blockSize      = blockSize;
    }
    NVCV_CHECK_THROW(cudaGetLastError());

    uchar imaxval = cuda::SaturateCast<uchar>(maxValue);
    int   idelta  = thresholdType == NVCV_THRESH_BINARY ? (int)std::ceil(c) : (int)std::floor(c);
    if (thresholdType == NVCV_THRESH_BINARY)
    {
        RunAdaptiveThreshold<uchar, NVCV_BORDER_REPLICATE, Greater<int>>(*inData, *outData, imaxval, kernelPtr,
                                                                         blockSize, idelta, state.kernelPolicy, stream);
    }
    else
    {
        RunAdaptiveThreshold<uchar, NVCV_BORDER_REPLICATE, LessEqual<int>>(
            *inData, *outData, imaxval, kernelPtr, blockSize, idelta, state.kernelPolicy, stream);
    }
}

void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &maxValue,
                                   const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const nvcv::Tensor &blockSize,
                                   const nvcv::Tensor &c) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AdaptiveThreshold::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    // As in the tensor overload, the admitted set is exactly {TYPE_U8}.
    if (!inData->uniqueFormat() || inData->uniqueFormat().planeDataType(0) != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data type must be U8. Unsupported data type.");
    }

    if (!outData->uniqueFormat() || outData->uniqueFormat().planeDataType(0) != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output data type must be U8. Unsupported data type.");
    }

    auto maxvalueData  = RequireCudaTensor(maxValue, "maxValue");
    auto blocksizeData = RequireCudaTensor(blockSize, "blockSize");
    auto cData         = RequireCudaTensor(c, "C");

    VarShapeState &state = m_varShapeState.get();

    if (m_maxVarShapeBatchSize <= 0 || inData->numImages() > m_maxVarShapeBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum batch size");
    }

    ImageDataFormat inputFormat  = DataFormatForVarShape(*inData);
    ImageDataFormat outputFormat = DataFormatForVarShape(*outData);
    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }

    const int channels = inData->uniqueFormat().numChannels();
    if (channels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (adaptiveMethod != NVCV_ADAPTIVE_THRESH_MEAN_C && adaptiveMethod != NVCV_ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid AdaptiveMethod %d",
                              static_cast<int>(adaptiveMethod));
    }

    if (thresholdType != NVCV_THRESH_BINARY && thresholdType != NVCV_THRESH_BINARY_INV)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid ThresholdType %d",
                              static_cast<int>(thresholdType));
    }

    // The parameter tensors are sized against the input batch while the launch paths walk the
    // output batch, so a mismatch would read past both.
    RequireSameNumImages(*inData, *outData);

    const int numImages = inData->numImages();
    ValidateParameterTensor(*maxvalueData, nvcv::TYPE_F64, numImages, "maxValue");
    ValidateParameterTensor(*blocksizeData, nvcv::TYPE_S32, numImages, "blockSize");
    ValidateParameterTensor(*cData, nvcv::TYPE_F64, numImages, "c");

    auto blockSizeAccess = nvcv::TensorDataAccessStrided::Create(*blocksizeData);
    NVCV_ASSERT(blockSizeAccess);

    // The per-image block sizes gate the shared-memory footprint, so they are read back and
    // validated on the host before any kernel is launched.
    std::vector<int> blockSizes(numImages);
    NVCV_CHECK_THROW(cudaMemcpy2DAsync(blockSizes.data(), sizeof(int), blockSizeAccess->sampleData(0),
                                       blockSizeAccess->sampleStride(), sizeof(int), numImages, cudaMemcpyDeviceToHost,
                                       stream));
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        const int bsize = blockSizes[i];
        if (!(bsize % 2 == 1 && bsize > 1 && bsize <= m_maxBlockSize))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid BlockSize %d", bsize);
        }
    }

    cuda::Tensor1DWrap<double, int32_t> maxValueArr(*maxvalueData);
    cuda::Tensor1DWrap<int, int32_t>    blockSizeArr(*blocksizeData);
    cuda::Tensor1DWrap<double, int32_t> cArr(*cData);

    int kernelPitch2 = static_cast<int>(m_maxBlockSize * sizeof(float));
    int kernelPitch1 = m_maxBlockSize * kernelPitch2;

    cuda::Tensor3DWrap<float, int32_t> kernelTensor(state.kernel, kernelPitch1, kernelPitch2);

    dim3 block(kCoeffBlockDimX, kCoeffBlockDimY);
    dim3 grid(util::DivUp(m_maxBlockSize, kCoeffBlockDimX), util::DivUp(m_maxBlockSize, kCoeffBlockDimY),
              outData->numImages());

    if (adaptiveMethod == NVCV_ADAPTIVE_THRESH_MEAN_C)
    {
        ComputeMeanKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, blockSizeArr);
    }
    else
    {
        ComputeGaussianKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, blockSizeArr);
    }
    NVCV_CHECK_THROW(cudaGetLastError());

    if (thresholdType == NVCV_THRESH_BINARY)
    {
        RunAdaptiveThresholdVarShape<uchar, NVCV_BORDER_REPLICATE, Greater<int>, cuda::RoundMode::UP>(
            *inData, *outData, maxValueArr, blockSizeArr, cArr, kernelTensor, m_maxBlockSize, state.kernelPolicy,
            stream);
    }
    else
    {
        RunAdaptiveThresholdVarShape<uchar, NVCV_BORDER_REPLICATE, LessEqual<int>, cuda::RoundMode::DOWN>(
            *inData, *outData, maxValueArr, blockSizeArr, cArr, kernelTensor, m_maxBlockSize, state.kernelPolicy,
            stream);
    }
}

} // namespace cvcuda::priv
