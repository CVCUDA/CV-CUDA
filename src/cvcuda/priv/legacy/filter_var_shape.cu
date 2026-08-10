/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "../Assert.h"
#include "../SafeSize.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "filter_utils.cuh"

#include <type_traits>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

__device__ __forceinline__ int2 ResolveGaussianKernelSize(cuda::Tensor1DWrap<int2, int32_t>    kernelSizeArr,
                                                          cuda::Tensor1DWrap<double2, int32_t> sigmaArr,
                                                          Size2D maxKernelSize, int dataKernelSize, int batch_idx)
{
    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    return kernelSize;
}

} // namespace

template<class SrcWrapper, class DstWrapper>
__global__ void filter2D(const SrcWrapper src, DstWrapper dst, cuda::ImageBatchVarShapeWrap<float> kernel,
                         cuda::Tensor1DWrap<int2, int32_t> kernelAnchor)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 anchor = kernelAnchor[batch_idx];

    int2 kernelSize{kernel.width(batch_idx), kernel.height(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    int3 srcCoord{0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<int Outputs, int KernelW, int KernelH, class SrcWrapper, class DstWrapper>
__global__ void filter2DTiledFixed(const SrcWrapper src, DstWrapper dst, cuda::ImageBatchVarShapeWrap<float> kernel,
                                   cuda::Tensor1DWrap<int2, int32_t> kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * Outputs;
    const int x0        = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int2 anchor = kernelAnchor[batch_idx];

    int2 kernelSize{kernel.width(batch_idx), kernel.height(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    const int outBaseX = blockIdx.x * outputW;
    const int outBaseY = blockIdx.y * blockDim.y;

    if (outBaseX >= dst.width(batch_idx) || outBaseY >= dst.height(batch_idx))
        return;

    const int tileW       = outputW + KernelW - 1;
    const int tileH       = blockDim.y + KernelH - 1;
    const int baseX       = outBaseX - anchor.x;
    const int baseY       = outBaseY - anchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 srcCoord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[srcCoord];
    }

    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    int kInd = 0;
    if (kernelSize.x == KernelW && kernelSize.y == KernelH)
    {
#pragma unroll
        for (int i = 0; i < KernelH; ++i)
        {
            const int tileY = threadIdx.y + i;

#pragma unroll
            for (int j = 0; j < KernelW; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize.y; ++i)
        {
            const int tileY = threadIdx.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }

#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dst.width(batch_idx))
        {
            *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void filter2DPlanar(const SrcWrapper src, DstWrapper dst, cuda::ImageBatchVarShapeWrap<float> kernel,
                               cuda::Tensor1DWrap<int2, int32_t> kernelAnchor, int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx, 0) || y >= dst.height(batch_idx, 0))
        return;

    int2 anchor = kernelAnchor[batch_idx];

    int2 kernelSize{kernel.width(batch_idx), kernel.height(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    for (int plane = 0; plane < channels; ++plane)
    {
        work_type res      = cuda::SetAll<work_type>(0);
        int4      srcCoord = {0, 0, plane, batch_idx};

        for (int i = 0; i < kernelSize.y; ++i)
        {
            srcCoord.y = y - anchor.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                srcCoord.x = x - anchor.x + j;

                res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
            }
        }

        *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
    }
}

template<int Outputs, int KernelW, int KernelH, class SrcWrapper, class DstWrapper>
__global__ void filter2DPlanarTiledFixed(const SrcWrapper src, DstWrapper dst,
                                         cuda::ImageBatchVarShapeWrap<float> kernel,
                                         cuda::Tensor1DWrap<int2, int32_t> kernelAnchor, int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * Outputs;
    const int x0        = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z / channels;
    const int plane     = blockIdx.z - batch_idx * channels;

    int2 anchor = kernelAnchor[batch_idx];

    int2 kernelSize{kernel.width(batch_idx), kernel.height(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    const int outBaseX = blockIdx.x * outputW;
    const int outBaseY = blockIdx.y * blockDim.y;

    if (outBaseX >= dst.width(batch_idx, plane) || outBaseY >= dst.height(batch_idx, plane))
        return;

    const int tileW       = outputW + KernelW - 1;
    const int tileH       = blockDim.y + KernelH - 1;
    const int baseX       = outBaseX - anchor.x;
    const int baseY       = outBaseY - anchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int4 srcCoord{baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx};
        tile[idx] = src[srcCoord];
    }

    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx, plane) || y >= dst.height(batch_idx, plane))
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    int kInd = 0;
    if (kernelSize.x == KernelW && kernelSize.y == KernelH)
    {
#pragma unroll
        for (int i = 0; i < KernelH; ++i)
        {
            const int tileY = threadIdx.y + i;

#pragma unroll
            for (int j = 0; j < KernelW; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize.y; ++i)
        {
            const int tileY = threadIdx.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }

#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dst.width(batch_idx, plane))
        {
            *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DPlanar(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float, int32_t> kernel,
                                       cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                       cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize, int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx, 0) || y >= dst.height(batch_idx, 0))
        return;

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    int2 anchor{kernelSize.x / 2, kernelSize.y / 2};

    for (int plane = 0; plane < channels; ++plane)
    {
        work_type res      = cuda::SetAll<work_type>(0);
        int4      srcCoord = {0, 0, plane, batch_idx};

        for (int i = 0; i < kernelSize.y; ++i)
        {
            srcCoord.y = y - anchor.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                srcCoord.x = x - anchor.x + j;

                res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
            }
        }

        *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DPlanarTiled(const SrcWrapper src, DstWrapper dst,
                                            cuda::Tensor3DWrap<float, int32_t> kernel,
                                            cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                            cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize,
                                            int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z / channels;
    const int plane     = blockIdx.z - batch_idx * channels;

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = blockDim.x + kernelSize.x - 1;
    const int  tileH       = blockDim.y + kernelSize.y - 1;
    const int  baseX       = blockIdx.x * blockDim.x - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int4 srcCoord{baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x >= dst.width(batch_idx, plane) || y >= dst.height(batch_idx, plane))
        return;

    work_type res  = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            const int tileX = threadIdx.x + j;

            res = res + tile[tileY * tileW + tileX] * kernelTile[kInd++];
        }
    }

    *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DPlanarTiledX2(const SrcWrapper src, DstWrapper dst,
                                              cuda::Tensor3DWrap<float, int32_t> kernel,
                                              cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                              cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize,
                                              int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * 2;
    const int x0        = blockIdx.x * outputW + threadIdx.x * 2;
    const int x1        = x0 + 1;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z / channels;
    const int plane     = blockIdx.z - batch_idx * channels;

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = outputW + kernelSize.x - 1;
    const int  tileH       = blockDim.y + kernelSize.y - 1;
    const int  baseX       = blockIdx.x * outputW - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int4 srcCoord{baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx, plane) || y >= dst.height(batch_idx, plane))
        return;

    work_type res0 = cuda::SetAll<work_type>(0);
    work_type res1 = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            const int   tileX = threadIdx.x * 2 + j;
            const float k     = kernelTile[kInd++];

            res0 = res0 + tile[tileY * tileW + tileX] * k;
            res1 = res1 + tile[tileY * tileW + tileX + 1] * k;
        }
    }

    *dst.ptr(batch_idx, plane, y, x0) = cuda::SaturateCast<T>(res0);
    if (x1 < dst.width(batch_idx, plane))
    {
        *dst.ptr(batch_idx, plane, y, x1) = cuda::SaturateCast<T>(res1);
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DPlanarTiledX4(const SrcWrapper src, DstWrapper dst,
                                              cuda::Tensor3DWrap<float, int32_t> kernel,
                                              cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                              cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize,
                                              int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * 4;
    const int x0        = blockIdx.x * outputW + threadIdx.x * 4;
    const int x1        = x0 + 1;
    const int x2        = x0 + 2;
    const int x3        = x0 + 3;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z / channels;
    const int plane     = blockIdx.z - batch_idx * channels;

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = outputW + kernelSize.x - 1;
    const int  tileH       = blockDim.y + kernelSize.y - 1;
    const int  baseX       = blockIdx.x * outputW - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int4 srcCoord{baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx, plane) || y >= dst.height(batch_idx, plane))
        return;

    work_type res0 = cuda::SetAll<work_type>(0);
    work_type res1 = cuda::SetAll<work_type>(0);
    work_type res2 = cuda::SetAll<work_type>(0);
    work_type res3 = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            const int   tileX = threadIdx.x * 4 + j;
            const float k     = kernelTile[kInd++];

            res0 = res0 + tile[tileY * tileW + tileX] * k;
            res1 = res1 + tile[tileY * tileW + tileX + 1] * k;
            res2 = res2 + tile[tileY * tileW + tileX + 2] * k;
            res3 = res3 + tile[tileY * tileW + tileX + 3] * k;
        }
    }

    *dst.ptr(batch_idx, plane, y, x0) = cuda::SaturateCast<T>(res0);
    if (x1 < dst.width(batch_idx, plane))
    {
        *dst.ptr(batch_idx, plane, y, x1) = cuda::SaturateCast<T>(res1);
    }
    if (x2 < dst.width(batch_idx, plane))
    {
        *dst.ptr(batch_idx, plane, y, x2) = cuda::SaturateCast<T>(res2);
    }
    if (x3 < dst.width(batch_idx, plane))
    {
        *dst.ptr(batch_idx, plane, y, x3) = cuda::SaturateCast<T>(res3);
    }
}

template<int Outputs, int KernelW, int KernelH, class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DPlanarTiledFixed(const SrcWrapper src, DstWrapper dst,
                                                 cuda::Tensor3DWrap<float, int32_t> kernel,
                                                 cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                                 cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize,
                                                 int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * Outputs;
    const int x0        = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z / channels;
    const int plane     = blockIdx.z - batch_idx * channels;

    const int2 kernelSize
        = ResolveGaussianKernelSize(kernelSizeArr, sigmaArr, maxKernelSize, dataKernelSize, batch_idx);
    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = outputW + KernelW - 1;
    const int  tileH       = blockDim.y + KernelH - 1;
    const int  baseX       = blockIdx.x * outputW - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int4 srcCoord{baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx, plane) || y >= dst.height(batch_idx, plane))
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    int kInd = 0;
    if (kernelSize.x == KernelW && kernelSize.y == KernelH)
    {
#pragma unroll
        for (int i = 0; i < KernelH; ++i)
        {
            const int tileY = threadIdx.y + i;

#pragma unroll
            for (int j = 0; j < KernelW; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize.y; ++i)
        {
            const int tileY = threadIdx.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }

#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dst.width(batch_idx, plane))
        {
            *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

template<int Outputs, typename T>
size_t Filter2DSharedBytes(dim3 block, Size2D maxKernelSize)
{
    const size_t kernelBytes = static_cast<size_t>(maxKernelSize.w) * maxKernelSize.h * sizeof(float);
    const size_t tileElems
        = static_cast<size_t>(block.x * Outputs + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t tileBytes    = tileElems * sizeof(T);
    const size_t kernelOffset = (tileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);

    return kernelOffset + kernelBytes;
}

template<typename D, NVCVBorderType B>
void Filter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
                    float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::ImageBatchVarShapeWrap<float>  kernel(kernelData);
    cuda::Tensor1DWrap<int2, int32_t>    kernelAnchor(kernelAnchorData);

    // Output count is dtype-gated from full-surface benchmarks: x8 wins for byte scalar/uchar4,
    // while float3/float4 need x2 to avoid register pressure from the wider vector work type.
    constexpr int kOutputs = std::is_same_v<D, uchar> || std::is_same_v<D, uchar4>
                               ? 8
                               : (std::is_same_v<D, float3> || std::is_same_v<D, float4> ? 2 : 4);

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());
    dim3 tiledGrid(divUp(inData.maxSize().w, block.x * kOutputs), divUp(inData.maxSize().h, block.y),
                   outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    const Size2D maxKernelSize = kernelData.maxSize();
    const size_t sharedBytes   = Filter2DSharedBytes<kOutputs, D>(block, maxKernelSize);

#define NVCV_CONV2D_VARSHAPE_FILTER_TILED_FIXED_CASE(KSIZE)                                  \
    if (maxKernelSize.w == KSIZE && maxKernelSize.h == KSIZE)                                \
    {                                                                                        \
        if (sharedBytes <= 48 * 1024)                                                        \
        {                                                                                    \
            filter2DTiledFixed<kOutputs, KSIZE, KSIZE>                                       \
                <<<tiledGrid, block, sharedBytes, stream>>>(src, dst, kernel, kernelAnchor); \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            filter2D<<<grid, block, 0, stream>>>(src, dst, kernel, kernelAnchor);            \
        }                                                                                    \
    }                                                                                        \
    else

    NVCV_CONV2D_VARSHAPE_FILTER_TILED_FIXED_CASE(3)
    NVCV_CONV2D_VARSHAPE_FILTER_TILED_FIXED_CASE(7)
    {
        filter2D<<<grid, block, 0, stream>>>(src, dst, kernel, kernelAnchor);
    }

#undef NVCV_CONV2D_VARSHAPE_FILTER_TILED_FIXED_CASE
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B>
void Filter2DPlanarCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                          const ImageBatchVarShapeDataStridedCuda &outData,
                          const ImageBatchVarShapeDataStridedCuda &kernelData,
                          const TensorDataStridedCuda &kernelAnchorData, int channels, float borderValue,
                          cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::ImageBatchVarShapeWrap<float>  kernel(kernelData);
    cuda::Tensor1DWrap<int2, int32_t>    kernelAnchor(kernelAnchorData);

    constexpr int kOutputs = std::is_same_v<D, uchar> ? 8 : 4;

    dim3          block(16, 16);
    const int     tiledGridX = divUp(outData.maxSize().w, block.x * kOutputs);
    const int     tiledGridY = divUp(outData.maxSize().h, block.y);
    const int64_t tiledGridZ = static_cast<int64_t>(outData.numImages()) * channels;

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    const Size2D maxKernelSize = kernelData.maxSize();
    const size_t sharedBytes   = Filter2DSharedBytes<kOutputs, D>(block, maxKernelSize);

#define NVCV_CONV2D_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE(KSIZE)                                        \
    if (maxKernelSize.w == KSIZE && maxKernelSize.h == KSIZE)                                             \
    {                                                                                                     \
        if (sharedBytes <= 48 * 1024 && tiledGridZ <= 65535)                                              \
        {                                                                                                 \
            dim3 grid(tiledGridX, tiledGridY, static_cast<unsigned int>(tiledGridZ));                     \
            filter2DPlanarTiledFixed<kOutputs, KSIZE, KSIZE>                                              \
                <<<grid, block, sharedBytes, stream>>>(src, dst, kernel, kernelAnchor, channels);         \
        }                                                                                                 \
        else                                                                                              \
        {                                                                                                 \
            dim3 fallbackGrid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),   \
                              outData.numImages());                                                       \
            filter2DPlanar<<<fallbackGrid, block, 0, stream>>>(src, dst, kernel, kernelAnchor, channels); \
        }                                                                                                 \
    }                                                                                                     \
    else

    NVCV_CONV2D_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE(3)
    NVCV_CONV2D_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE(7)
    {
        dim3 fallbackGrid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),
                          outData.numImages());
        filter2DPlanar<<<fallbackGrid, block, 0, stream>>>(src, dst, kernel, kernelAnchor, channels);
    }

#undef NVCV_CONV2D_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void Filter2D(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
              const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
              NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           const ImageBatchVarShapeDataStridedCuda &kernelData,
                           const TensorDataStridedCuda &kernelAnchorData, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {Filter2DCaller<D, NVCV_BORDER_CONSTANT>, Filter2DCaller<D, NVCV_BORDER_REPLICATE>,
                                   Filter2DCaller<D, NVCV_BORDER_REFLECT>, Filter2DCaller<D, NVCV_BORDER_WRAP>,
                                   Filter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelData, kernelAnchorData, borderValue, stream);
}

template<typename D>
void Filter2DPlanar(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                    const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
                    NVCVBorderType borderMode, float borderValue, int channels, cudaStream_t stream)
{
    typedef void (*func_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
        const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
        int channels, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {Filter2DPlanarCaller<D, NVCV_BORDER_CONSTANT>, Filter2DPlanarCaller<D, NVCV_BORDER_REPLICATE>,
           Filter2DPlanarCaller<D, NVCV_BORDER_REFLECT>, Filter2DPlanarCaller<D, NVCV_BORDER_WRAP>,
           Filter2DPlanarCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelData, kernelAnchorData, channels, borderValue, stream);
}

// Conv2DVarShape --------------------------------------------------------------

ErrorCode Conv2DVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                const ImageBatchVarShapeDataStridedCuda &outData,
                                const ImageBatchVarShapeDataStridedCuda &kernelData,
                                const TensorDataStridedCuda &kernelAnchorData, NVCVBorderType borderMode,
                                cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    const bool isPlanar = IsPlanar(format);

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef void (*filter2D_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
        const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
        NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        { Filter2D<short>, 0,  Filter2D<short3>,  Filter2D<short4>},
        {   Filter2D<int>, 0,    Filter2D<int3>,    Filter2D<int4>},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    if (isPlanar)
    {
        if (outData.numImages() > 65535)
        {
            LOG_ERROR("Planar Conv2D requires numImages <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        typedef void (*filter2D_planar_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            const ImageBatchVarShapeDataStridedCuda &kernelData, const TensorDataStridedCuda &kernelAnchorData,
            NVCVBorderType borderMode, float borderValue, int channels, cudaStream_t stream);

        static const filter2D_planar_t planarFuncs[6] = {
            Filter2DPlanar<uchar>, 0, Filter2DPlanar<ushort>, Filter2DPlanar<short>, Filter2DPlanar<int>,
            Filter2DPlanar<float>,
        };

        const filter2D_planar_t func = planarFuncs[data_type];

        NVCV_ASSERT(func != 0);

        func(inData, outData, kernelData, kernelAnchorData, borderMode, borderValue, channels, stream);
        return ErrorCode::SUCCESS;
    }

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    func(inData, outData, kernelData, kernelAnchorData, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

// LaplacianVarShape -----------------------------------------------------------

// @brief Laplacian 3x3 kernels for ksize == 1 and ksize == 3

constexpr int kLaplacianPlanarU8NIX      = 4;
constexpr int kLaplacianPlanarFloatNIX   = 4;
constexpr int kLaplacianPlanarBlockWidth = 32;

// clang-format off

__device__ cuda::math::Vector<float, 9> kLaplacianKernel1{
    {0.0f,  1.0f, 0.0f,
     1.0f, -4.0f, 1.0f,
     0.0f,  1.0f, 0.0f}
};
__device__ cuda::math::Vector<float, 9> kLaplacianKernel3{
    {2.0f,  0.0f, 2.0f,
     0.0f, -8.0f, 0.0f,
     2.0f,  0.0f, 2.0f}
};

// clang-format on

// Laplacian kernels are either one or the other (above)
template<class SrcWrapper, class DstWrapper>
__global__ void laplacianFilter2D(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int, int32_t> ksize,
                                  cuda::Tensor1DWrap<float, int32_t> scale)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    constexpr int2 kernelSize = int2{3, 3};
    constexpr int2 anchor     = int2{1, 1};

    const int ksizeVal = ksize[batch_idx];

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);
    cuda::math::Vector<float, 9> kernel = ksizeVal == 1 ? kLaplacianKernel1 : kLaplacianKernel3;

    kernel *= scale[batch_idx];

    int  kidx = 0;
    int3 srcCoord{0, 0, batch_idx};

#pragma unroll
    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

#pragma unroll
        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * kernel[kidx++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<class SrcWrapper, class DstWrapper>
__global__ void laplacianFilter2DPlanar(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int, int32_t> ksize,
                                        cuda::Tensor1DWrap<float, int32_t> scale, int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx, 0) || y >= dst.height(batch_idx, 0))
        return;

    constexpr int2 kernelSize = int2{3, 3};
    constexpr int2 anchor     = int2{1, 1};

    const int ksizeVal = ksize[batch_idx];

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);
    cuda::math::Vector<float, 9> kernel = ksizeVal == 1 ? kLaplacianKernel1 : kLaplacianKernel3;

    kernel *= scale[batch_idx];

    for (int plane = 0; plane < channels; ++plane)
    {
        work_type res      = cuda::SetAll<work_type>(0);
        int       kidx     = 0;
        int4      srcCoord = {0, 0, plane, batch_idx};

        for (int i = 0; i < kernelSize.y; ++i)
        {
            srcCoord.y = y - anchor.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                srcCoord.x = x - anchor.x + j;

                res = res + src[srcCoord] * kernel[kidx++];
            }
        }

        *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void laplacianFilter2DU8(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int, int32_t> ksize,
                                    cuda::Tensor1DWrap<float, int32_t> scale)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    const int   ksizeVal = ksize[batch_idx];
    const float scaleVal = scale[batch_idx];
    int3        srcCoord{x, y - 1, batch_idx};

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);

    work_type res;
    if (ksizeVal == 1)
    {
        const float axisK   = scaleVal;
        const float centerK = -4.0f * scaleVal;

        res        = src[srcCoord] * axisK;
        srcCoord.x = x - 1;
        srcCoord.y = y;
        res        = res + src[srcCoord] * axisK;
        srcCoord.x = x;
        res        = res + src[srcCoord] * centerK;
        srcCoord.x = x + 1;
        res        = res + src[srcCoord] * axisK;
        srcCoord.x = x;
        srcCoord.y = y + 1;
        res        = res + src[srcCoord] * axisK;
    }
    else
    {
        const float cornerK = 2.0f * scaleVal;
        const float centerK = -8.0f * scaleVal;

        srcCoord.x = x - 1;
        res        = src[srcCoord] * cornerK;
        srcCoord.x = x + 1;
        res        = res + src[srcCoord] * cornerK;
        srcCoord.x = x;
        srcCoord.y = y;
        res        = res + src[srcCoord] * centerK;
        srcCoord.x = x - 1;
        srcCoord.y = y + 1;
        res        = res + src[srcCoord] * cornerK;
        srcCoord.x = x + 1;
        res        = res + src[srcCoord] * cornerK;
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<int NIX, class SrcWrapper, class DstWrapper>
__global__ void laplacianFilter2DPlanarTiled(const SrcWrapper src, DstWrapper dst,
                                             cuda::Tensor1DWrap<int, int32_t>   ksize,
                                             cuda::Tensor1DWrap<float, int32_t> scale, int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    constexpr int kRadius = 1;

    constexpr int outputW   = kLaplacianPlanarBlockWidth * NIX;
    const int     outBaseX  = blockIdx.x * outputW;
    const int     outBaseY  = blockIdx.y * blockDim.y;
    const int     batch_idx = get_batch_idx();
    const int     dstWidth  = dst.width(batch_idx, 0);
    const int     dstHeight = dst.height(batch_idx, 0);

    if (outBaseX >= dstWidth || outBaseY >= dstHeight)
        return;

    const int ksizeVal = ksize[batch_idx];

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);

    const float   scaleVal  = scale[batch_idx];
    constexpr int tileW     = outputW + 2 * kRadius;
    const int     tileH     = blockDim.y + 2 * kRadius;
    const int     tileElems = tileW * tileH;
    const int     tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const int     y         = outBaseY + threadIdx.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);

    for (int plane = 0; plane < channels; ++plane)
    {
        for (int idx = tid; idx < tileElems; idx += blockDim.x * blockDim.y)
        {
            int4 coord{outBaseX - kRadius + idx % tileW, outBaseY - kRadius + idx / tileW, plane, batch_idx};
            tile[idx] = src[coord];
        }

        __syncthreads();

        if (y < dstHeight)
        {
            const int tileY         = threadIdx.y + kRadius;
            auto      computeOutput = [&](int tileX)
            {
                work_type res;
                if (ksizeVal == 1)
                {
                    const float axisK   = scaleVal;
                    const float centerK = -4.0f * scaleVal;

                    if constexpr (std::is_same_v<T, float>)
                    {
                        res = LaplacianFloatMulRN(tile[(tileY - 1) * tileW + tileX], axisK);
                        res = LaplacianFloatFmaRN(tile[tileY * tileW + tileX - 1], axisK, res);
                        res = LaplacianFloatFmaRN(tile[tileY * tileW + tileX], centerK, res);
                        res = LaplacianFloatFmaRN(tile[tileY * tileW + tileX + 1], axisK, res);
                        res = LaplacianFloatFmaRN(tile[(tileY + 1) * tileW + tileX], axisK, res);
                    }
                    else
                    {
                        res = tile[(tileY - 1) * tileW + tileX] * axisK;
                        res = res + tile[tileY * tileW + tileX - 1] * axisK;
                        res = res + tile[tileY * tileW + tileX] * centerK;
                        res = res + tile[tileY * tileW + tileX + 1] * axisK;
                        res = res + tile[(tileY + 1) * tileW + tileX] * axisK;
                    }
                }
                else
                {
                    const float cornerK = 2.0f * scaleVal;
                    const float centerK = -8.0f * scaleVal;

                    if constexpr (std::is_same_v<T, float>)
                    {
                        res = LaplacianFloatMulRN(tile[(tileY - 1) * tileW + tileX - 1], cornerK);
                        res = LaplacianFloatFmaRN(tile[(tileY - 1) * tileW + tileX + 1], cornerK, res);
                        res = LaplacianFloatFmaRN(tile[tileY * tileW + tileX], centerK, res);
                        res = LaplacianFloatFmaRN(tile[(tileY + 1) * tileW + tileX - 1], cornerK, res);
                        res = LaplacianFloatFmaRN(tile[(tileY + 1) * tileW + tileX + 1], cornerK, res);
                    }
                    else
                    {
                        res = tile[(tileY - 1) * tileW + tileX - 1] * cornerK;
                        res = res + tile[(tileY - 1) * tileW + tileX + 1] * cornerK;
                        res = res + tile[tileY * tileW + tileX] * centerK;
                        res = res + tile[(tileY + 1) * tileW + tileX - 1] * cornerK;
                        res = res + tile[(tileY + 1) * tileW + tileX + 1] * cornerK;
                    }
                }

                return cuda::SaturateCast<T>(res);
            };

            if constexpr (NIX == 4 && (std::is_same_v<T, uchar> || std::is_same_v<T, float>))
            {
                const int x0 = outBaseX + threadIdx.x * NIX;
                if (x0 < dstWidth)
                {
                    const int valid = dstWidth - x0 < NIX ? dstWidth - x0 : NIX;
                    T         outputs[NIX];

#pragma unroll
                    for (int i = 0; i < NIX; ++i)
                    {
                        if (i < valid)
                            outputs[i] = computeOutput(threadIdx.x * NIX + i + kRadius);
                    }

                    T               *dstPtr           = dst.ptr(batch_idx, plane, y, x0);
                    constexpr size_t kVectorAlignment = sizeof(T) * NIX;
                    if (valid == NIX && (reinterpret_cast<uintptr_t>(dstPtr) & (kVectorAlignment - 1)) == 0)
                    {
                        if constexpr (std::is_same_v<T, uchar>)
                            *reinterpret_cast<uchar4 *>(dstPtr)
                                = make_uchar4(outputs[0], outputs[1], outputs[2], outputs[3]);
                        else
                            *reinterpret_cast<float4 *>(dstPtr)
                                = make_float4(outputs[0], outputs[1], outputs[2], outputs[3]);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < NIX; ++i)
                        {
                            if (i < valid)
                                dstPtr[i] = outputs[i];
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < NIX; ++i)
                {
                    const int x = outBaseX + threadIdx.x + i * blockDim.x;
                    if (x < dstWidth)
                        *dst.ptr(batch_idx, plane, y, x) = computeOutput(threadIdx.x + i * blockDim.x + kRadius);
                }
            }
        }

        __syncthreads();
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void laplacianFilter2DFloat(const SrcWrapper src, DstWrapper dst, cuda::Tensor1DWrap<int, int32_t> ksize,
                                       cuda::Tensor1DWrap<float, int32_t> scale)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    const int   ksizeVal = ksize[batch_idx];
    const float scaleVal = scale[batch_idx];

    NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);

    int3 srcCoord{x, y - 1, batch_idx};

    if (ksizeVal == 1)
    {
        const float axisK   = scaleVal;
        const float centerK = -4.0f * scaleVal;

        work_type res = LaplacianFloatMulRN(src[srcCoord], axisK);
        srcCoord.x    = x - 1;
        srcCoord.y    = y;
        res           = LaplacianFloatFmaRN(src[srcCoord], axisK, res);
        srcCoord.x    = x;
        res           = LaplacianFloatFmaRN(src[srcCoord], centerK, res);
        srcCoord.x    = x + 1;
        res           = LaplacianFloatFmaRN(src[srcCoord], axisK, res);

        srcCoord.y = y + 1;
        srcCoord.x = x;
        res        = LaplacianFloatFmaRN(src[srcCoord], axisK, res);

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
    }
    else
    {
        const float cornerK = 2.0f * scaleVal;
        const float centerK = -8.0f * scaleVal;

        srcCoord.x    = x - 1;
        work_type res = LaplacianFloatMulRN(src[srcCoord], cornerK);
        srcCoord.x    = x + 1;
        res           = LaplacianFloatFmaRN(src[srcCoord], cornerK, res);

        srcCoord.x = x;
        srcCoord.y = y;
        res        = LaplacianFloatFmaRN(src[srcCoord], centerK, res);

        srcCoord.x = x - 1;
        srcCoord.y = y + 1;
        res        = LaplacianFloatFmaRN(src[srcCoord], cornerK, res);
        srcCoord.x = x + 1;
        res        = LaplacianFloatFmaRN(src[srcCoord], cornerK, res);

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
    }
}

template<typename D, NVCVBorderType B>
void LaplacianFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                             const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                             const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::Tensor1DWrap<int, int32_t>     kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float, int32_t>   kernelScale(scale);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    laplacianFilter2D<<<grid, block, 0, stream>>>(src, dst, kernelApertureSize, kernelScale);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B>
void LaplacianFilter2DU8Caller(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                               const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::Tensor1DWrap<int, int32_t>     kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float, int32_t>   kernelScale(scale);

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    laplacianFilter2DU8<<<grid, block, 0, stream>>>(src, dst, kernelApertureSize, kernelScale);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B>
void LaplacianFilter2DFloatCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                                  const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                                  const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::Tensor1DWrap<int, int32_t>     kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float, int32_t>   kernelScale(scale);

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    laplacianFilter2DFloat<<<grid, block, 0, stream>>>(src, dst, kernelApertureSize, kernelScale);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void LaplacianFilter2D(const ImageBatchVarShapeDataStridedCuda &inData,
                       const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                       const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                       cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                           const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {LaplacianFilter2DCaller<D, NVCV_BORDER_CONSTANT>, LaplacianFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           LaplacianFilter2DCaller<D, NVCV_BORDER_REFLECT>, LaplacianFilter2DCaller<D, NVCV_BORDER_WRAP>,
           LaplacianFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, ksize, scale, borderValue, stream);
}

template<typename D>
void LaplacianFilter2DU8(const ImageBatchVarShapeDataStridedCuda &inData,
                         const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                         const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                         cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                           const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {LaplacianFilter2DU8Caller<D, NVCV_BORDER_CONSTANT>, LaplacianFilter2DU8Caller<D, NVCV_BORDER_REPLICATE>,
           LaplacianFilter2DU8Caller<D, NVCV_BORDER_REFLECT>, LaplacianFilter2DU8Caller<D, NVCV_BORDER_WRAP>,
           LaplacianFilter2DU8Caller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, ksize, scale, borderValue, stream);
}

template<typename D>
void LaplacianFilter2DFloat(const ImageBatchVarShapeDataStridedCuda &inData,
                            const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                            const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                            cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                           const TensorDataStridedCuda &scale, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {LaplacianFilter2DFloatCaller<D, NVCV_BORDER_CONSTANT>,
           LaplacianFilter2DFloatCaller<D, NVCV_BORDER_REPLICATE>, LaplacianFilter2DFloatCaller<D, NVCV_BORDER_REFLECT>,
           LaplacianFilter2DFloatCaller<D, NVCV_BORDER_WRAP>, LaplacianFilter2DFloatCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, ksize, scale, borderValue, stream);
}

template<typename D, NVCVBorderType B>
void LaplacianFilter2DPlanarCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                                   const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                                   const TensorDataStridedCuda &scale, int channels, float borderValue,
                                   cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::Tensor1DWrap<int, int32_t>     kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float, int32_t>   kernelScale(scale);

    dim3 block(16, 16);
    dim3 grid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    laplacianFilter2DPlanar<<<grid, block, 0, stream>>>(src, dst, kernelApertureSize, kernelScale, channels);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<int NIX, typename D, NVCVBorderType B>
void LaplacianFilter2DPlanarTiledCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                                        const ImageBatchVarShapeDataStridedCuda &outData,
                                        const TensorDataStridedCuda &ksize, const TensorDataStridedCuda &scale,
                                        int channels, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);
    cuda::Tensor1DWrap<int, int32_t>     kernelApertureSize(ksize);
    cuda::Tensor1DWrap<float, int32_t>   kernelScale(scale);

    dim3 block(kLaplacianPlanarBlockWidth, std::is_same_v<D, float> ? 8 : 16);
    dim3 grid(divUp(outData.maxSize().w, block.x * NIX), divUp(outData.maxSize().h, block.y), outData.numImages());

    const size_t sharedBytes = static_cast<size_t>(block.x * NIX + 2) * (block.y + 2) * sizeof(D);
    laplacianFilter2DPlanarTiled<NIX>
        <<<grid, block, sharedBytes, stream>>>(src, dst, kernelApertureSize, kernelScale, channels);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void LaplacianFilter2DPlanar(const ImageBatchVarShapeDataStridedCuda &inData,
                             const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                             const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                             int channels, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                           const TensorDataStridedCuda &scale, int channels, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {
        LaplacianFilter2DPlanarCaller<D, NVCV_BORDER_CONSTANT>, LaplacianFilter2DPlanarCaller<D, NVCV_BORDER_REPLICATE>,
        LaplacianFilter2DPlanarCaller<D, NVCV_BORDER_REFLECT>, LaplacianFilter2DPlanarCaller<D, NVCV_BORDER_WRAP>,
        LaplacianFilter2DPlanarCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, ksize, scale, channels, borderValue, stream);
}

template<int NIX, typename D>
void LaplacianFilter2DPlanarTiled(const ImageBatchVarShapeDataStridedCuda &inData,
                                  const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                                  const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                                  int channels, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                           const TensorDataStridedCuda &scale, int channels, float borderValue, cudaStream_t stream);

    static const func_t funcs[] = {LaplacianFilter2DPlanarTiledCaller<NIX, D, NVCV_BORDER_CONSTANT>,
                                   LaplacianFilter2DPlanarTiledCaller<NIX, D, NVCV_BORDER_REPLICATE>,
                                   LaplacianFilter2DPlanarTiledCaller<NIX, D, NVCV_BORDER_REFLECT>,
                                   LaplacianFilter2DPlanarTiledCaller<NIX, D, NVCV_BORDER_WRAP>,
                                   LaplacianFilter2DPlanarTiledCaller<NIX, D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, ksize, scale, channels, borderValue, stream);
}

ErrorCode LaplacianVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                   const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                                   const TensorDataStridedCuda &scale, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData);
    DataFormat output_format = GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = IsPlanar(format);

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef void (*filter2D_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                               const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                               cudaStream_t stream);
    typedef void (*laplacian_u8_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                                   const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &ksize,
                                   const TensorDataStridedCuda &scale, NVCVBorderType borderMode, float borderValue,
                                   cudaStream_t stream);
    typedef void (*laplacian_float_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                                      const ImageBatchVarShapeDataStridedCuda &outData,
                                      const TensorDataStridedCuda &ksize, const TensorDataStridedCuda &scale,
                                      NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        {                        0, 0,                          0,                          0},
        {                        0, 0,                          0,                          0},
        {LaplacianFilter2D<ushort>, 0, LaplacianFilter2D<ushort3>, LaplacianFilter2D<ushort4>},
        {                        0, 0,                          0,                          0},
        {                        0, 0,                          0,                          0},
        {                        0, 0,                          0,                          0},
    };

    static const laplacian_u8_t u8Funcs[4]
        = {LaplacianFilter2DU8<uchar>, 0, LaplacianFilter2DU8<uchar3>, LaplacianFilter2DU8<uchar4>};
    static const laplacian_float_t floatFuncs[4]
        = {LaplacianFilter2DFloat<float>, 0, LaplacianFilter2DFloat<float3>, LaplacianFilter2DFloat<float4>};

    if (isPlanar)
    {
        if (outData.numImages() > 65535)
        {
            LOG_ERROR("Planar Laplacian requires numImages <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        typedef void (*planar_filter2D_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            const TensorDataStridedCuda &ksize, const TensorDataStridedCuda &scale, NVCVBorderType borderMode,
            float borderValue, int channels, cudaStream_t stream);
        static const planar_filter2D_t planarFuncs[6] = {
            0, 0, LaplacianFilter2DPlanar<ushort>, 0, 0, 0,
        };

        if (data_type == kCV_8U)
        {
            LaplacianFilter2DPlanarTiled<kLaplacianPlanarU8NIX, uchar>(inData, outData, ksize, scale, borderMode,
                                                                       borderValue, channels, stream);
            return ErrorCode::SUCCESS;
        }
        if (data_type == kCV_32F)
        {
            LaplacianFilter2DPlanarTiled<kLaplacianPlanarFloatNIX, float>(inData, outData, ksize, scale, borderMode,
                                                                          borderValue, channels, stream);
            return ErrorCode::SUCCESS;
        }

        const planar_filter2D_t planarFunc = planarFuncs[data_type];

        NVCV_ASSERT(planarFunc != 0);

        planarFunc(inData, outData, ksize, scale, borderMode, borderValue, channels, stream);

        return ErrorCode::SUCCESS;
    }

    if (data_type == kCV_8U)
    {
        const laplacian_u8_t u8Func = u8Funcs[channels - 1];
        NVCV_ASSERT(u8Func != 0);
        u8Func(inData, outData, ksize, scale, borderMode, borderValue, stream);
        return ErrorCode::SUCCESS;
    }
    if (data_type == kCV_32F)
    {
        const laplacian_float_t floatFunc = floatFuncs[channels - 1];
        NVCV_ASSERT(floatFunc != 0);
        floatFunc(inData, outData, ksize, scale, borderMode, borderValue, stream);
        return ErrorCode::SUCCESS;
    }

    const filter2D_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);
    func(inData, outData, ksize, scale, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

// GaussianVarShape ------------------------------------------------------------

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2D(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float, int32_t> kernel,
                                 cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                 cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    int2 anchor{kernelSize.x / 2, kernelSize.y / 2};

    int3 srcCoord{0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DTiled(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float, int32_t> kernel,
                                      cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                      cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = blockDim.x + kernelSize.x - 1;
    const int  tileH       = blockDim.y + kernelSize.y - 1;
    const int  baseX       = blockIdx.x * blockDim.x - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 srcCoord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    work_type res  = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            const int tileX = threadIdx.x + j;

            res = res + tile[tileY * tileW + tileX] * kernelTile[kInd++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DTiledX2(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float, int32_t> kernel,
                                        cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                        cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * 2;
    const int x0        = blockIdx.x * outputW + threadIdx.x * 2;
    const int x1        = x0 + 1;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = outputW + kernelSize.x - 1;
    const int  tileH       = blockDim.y + kernelSize.y - 1;
    const int  baseX       = blockIdx.x * outputW - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 srcCoord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    work_type res0 = cuda::SetAll<work_type>(0);
    work_type res1 = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            const int   tileX = threadIdx.x * 2 + j;
            const float k     = kernelTile[kInd++];

            res0 = res0 + tile[tileY * tileW + tileX] * k;
            res1 = res1 + tile[tileY * tileW + tileX + 1] * k;
        }
    }

    *dst.ptr(batch_idx, y, x0) = cuda::SaturateCast<T>(res0);
    if (x1 < dst.width(batch_idx))
    {
        *dst.ptr(batch_idx, y, x1) = cuda::SaturateCast<T>(res1);
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DTiledX4(const SrcWrapper src, DstWrapper dst, cuda::Tensor3DWrap<float, int32_t> kernel,
                                        cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                        cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * 4;
    const int x0        = blockIdx.x * outputW + threadIdx.x * 4;
    const int x1        = x0 + 1;
    const int x2        = x0 + 2;
    const int x3        = x0 + 3;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int2    kernelSize = kernelSizeArr[batch_idx];
    double2 sigma      = sigmaArr[batch_idx];

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = outputW + kernelSize.x - 1;
    const int  tileH       = blockDim.y + kernelSize.y - 1;
    const int  baseX       = blockIdx.x * outputW - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 srcCoord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    work_type res0 = cuda::SetAll<work_type>(0);
    work_type res1 = cuda::SetAll<work_type>(0);
    work_type res2 = cuda::SetAll<work_type>(0);
    work_type res3 = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            const int   tileX = threadIdx.x * 4 + j;
            const float k     = kernelTile[kInd++];

            res0 = res0 + tile[tileY * tileW + tileX] * k;
            res1 = res1 + tile[tileY * tileW + tileX + 1] * k;
            res2 = res2 + tile[tileY * tileW + tileX + 2] * k;
            res3 = res3 + tile[tileY * tileW + tileX + 3] * k;
        }
    }

    *dst.ptr(batch_idx, y, x0) = cuda::SaturateCast<T>(res0);
    if (x1 < dst.width(batch_idx))
    {
        *dst.ptr(batch_idx, y, x1) = cuda::SaturateCast<T>(res1);
    }
    if (x2 < dst.width(batch_idx))
    {
        *dst.ptr(batch_idx, y, x2) = cuda::SaturateCast<T>(res2);
    }
    if (x3 < dst.width(batch_idx))
    {
        *dst.ptr(batch_idx, y, x3) = cuda::SaturateCast<T>(res3);
    }
}

template<int Outputs, int KernelW, int KernelH, class SrcWrapper, class DstWrapper>
__global__ void gaussianFilter2DTiledFixed(const SrcWrapper src, DstWrapper dst,
                                           cuda::Tensor3DWrap<float, int32_t> kernel,
                                           cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr, Size2D maxKernelSize,
                                           cuda::Tensor1DWrap<double2, int32_t> sigmaArr, int dataKernelSize)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW   = blockDim.x * Outputs;
    const int x0        = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int2 kernelSize
        = ResolveGaussianKernelSize(kernelSizeArr, sigmaArr, maxKernelSize, dataKernelSize, batch_idx);
    const int2 anchor{kernelSize.x / 2, kernelSize.y / 2};
    const int  tileW       = outputW + KernelW - 1;
    const int  tileH       = blockDim.y + KernelH - 1;
    const int  baseX       = blockIdx.x * outputW - anchor.x;
    const int  baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int  tileElems   = tileW * tileH;
    const int  kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 srcCoord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[srcCoord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = *kernel.ptr(batch_idx, idx / kernelSize.x, idx % kernelSize.x);
    }

    __syncthreads();

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    int kInd = 0;
    if (kernelSize.x == KernelW && kernelSize.y == KernelH)
    {
#pragma unroll
        for (int i = 0; i < KernelH; ++i)
        {
            const int tileY = threadIdx.y + i;

#pragma unroll
            for (int j = 0; j < KernelW; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize.y; ++i)
        {
            const int tileY = threadIdx.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }

#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dst.width(batch_idx))
        {
            *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

constexpr int kGaussianBlockWidth  = 16;
constexpr int kGaussianBlockHeight = 16;

template<int Outputs, int KSize, typename T>
constexpr size_t GaussianFixedSharedBytes()
{
    constexpr size_t tileBytes = static_cast<size_t>(kGaussianBlockWidth * Outputs + KSize - 1)
                               * (kGaussianBlockHeight + KSize - 1) * sizeof(T);
    constexpr size_t kernelOffset = (tileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    return kernelOffset + static_cast<size_t>(KSize) * KSize * sizeof(float);
}

template<typename D, NVCVBorderType B>
void GaussianFilter2DCaller(const ImageBatchVarShapeDataStridedCuda  &inData,
                            const ImageBatchVarShapeDataStridedCuda  &outData,
                            const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                            const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                            const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize, bool enableX4,
                            float borderValue, cudaStream_t stream)
{
    using BaseT = cuda::BaseType<D>;

    constexpr bool kX4Always          = std::is_same_v<BaseT, uchar> || cuda::NumElements<D> == 1;
    constexpr bool kDynamicX4Possible = kX4Always || std::is_same_v<D, float3>;

    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(kGaussianBlockWidth, kGaussianBlockHeight);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());
    dim3 x2Grid(divUp(inData.maxSize().w, block.x * 2), divUp(inData.maxSize().h, block.y), outData.numImages());
    dim3 x4Grid(divUp(inData.maxSize().w, block.x * 4), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    const size_t kernelBytes  = static_cast<size_t>(maxKernelSize.w) * maxKernelSize.h * sizeof(float);
    const size_t tileElems    = static_cast<size_t>(block.x + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t tileBytes    = tileElems * sizeof(D);
    const size_t kernelOffset = (tileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    const size_t sharedBytes  = kernelOffset + kernelBytes;
    const size_t x2TileElems = static_cast<size_t>(block.x * 2 + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t x2TileBytes = x2TileElems * sizeof(D);
    const size_t x2KernelOffset = (x2TileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    const size_t x2SharedBytes  = x2KernelOffset + kernelBytes;
    const size_t x4TileElems = static_cast<size_t>(block.x * 4 + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t x4TileBytes = x4TileElems * sizeof(D);
    const size_t x4KernelOffset = (x4TileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    const size_t x4SharedBytes  = x4KernelOffset + kernelBytes;

#define NVCV_GAUSSIAN_VARSHAPE_FILTER_TILED_FIXED_CASE(KSIZE)                                          \
    if (maxKernelSize.w == KSIZE && maxKernelSize.h == KSIZE)                                          \
    {                                                                                                  \
        if constexpr (kX4Always)                                                                       \
        {                                                                                              \
            static_assert(GaussianFixedSharedBytes<4, KSIZE, D>() <= 48 * 1024);                       \
            gaussianFilter2DTiledFixed<4, KSIZE, KSIZE><<<x4Grid, block, x4SharedBytes, stream>>>(     \
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize); \
        }                                                                                              \
        else                                                                                           \
        {                                                                                              \
            static_assert(GaussianFixedSharedBytes<2, KSIZE, D>() <= 48 * 1024);                       \
            gaussianFilter2DTiledFixed<2, KSIZE, KSIZE><<<x2Grid, block, x2SharedBytes, stream>>>(     \
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize); \
        }                                                                                              \
    }                                                                                                  \
    else

    NVCV_GAUSSIAN_VARSHAPE_FILTER_TILED_FIXED_CASE(3)
    NVCV_GAUSSIAN_VARSHAPE_FILTER_TILED_FIXED_CASE(5)
    if constexpr (kDynamicX4Possible)
    {
        if (enableX4 && x4SharedBytes <= 48 * 1024)
        {
            gaussianFilter2DTiledX4<<<x4Grid, block, x4SharedBytes, stream>>>(
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize);
        }
        else if (x2SharedBytes <= 48 * 1024)
        {
            gaussianFilter2DTiledX2<<<x2Grid, block, x2SharedBytes, stream>>>(
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize);
        }
        else if (sharedBytes <= 48 * 1024)
        {
            gaussianFilter2DTiled<<<grid, block, sharedBytes, stream>>>(src, dst, kernelTensor, kernelSizeTensor,
                                                                        maxKernelSize, sigmaTensor, dataKernelSize);
        }
        else
        {
            gaussianFilter2D<<<grid, block, 0, stream>>>(src, dst, kernelTensor, kernelSizeTensor, maxKernelSize,
                                                         sigmaTensor, dataKernelSize);
        }
    }
    else if (x2SharedBytes <= 48 * 1024)
    {
        gaussianFilter2DTiledX2<<<x2Grid, block, x2SharedBytes, stream>>>(src, dst, kernelTensor, kernelSizeTensor,
                                                                          maxKernelSize, sigmaTensor, dataKernelSize);
    }
    else if (sharedBytes <= 48 * 1024)
    {
        gaussianFilter2DTiled<<<grid, block, sharedBytes, stream>>>(src, dst, kernelTensor, kernelSizeTensor,
                                                                    maxKernelSize, sigmaTensor, dataKernelSize);
    }
    else
    {
        gaussianFilter2D<<<grid, block, 0, stream>>>(src, dst, kernelTensor, kernelSizeTensor, maxKernelSize,
                                                     sigmaTensor, dataKernelSize);
    }

#undef NVCV_GAUSSIAN_VARSHAPE_FILTER_TILED_FIXED_CASE
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B>
void GaussianFilter2DPlanarCaller(const ImageBatchVarShapeDataStridedCuda  &inData,
                                  const ImageBatchVarShapeDataStridedCuda  &outData,
                                  const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                                  const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                                  const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize,
                                  int channels, bool enableX4, float borderValue, cudaStream_t stream)
{
    constexpr bool kX4Always = std::is_same_v<D, uchar> || std::is_same_v<D, float>;

    cuda::BorderVarShapeWrap<const D, B> src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<D>      dst(outData);

    dim3          block(kGaussianBlockWidth, kGaussianBlockHeight);
    const int64_t tiledGridZ = static_cast<int64_t>(outData.numImages()) * channels;

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    const size_t kernelBytes  = static_cast<size_t>(maxKernelSize.w) * maxKernelSize.h * sizeof(float);
    const size_t tileElems    = static_cast<size_t>(block.x + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t tileBytes    = tileElems * sizeof(D);
    const size_t kernelOffset = (tileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    const size_t sharedBytes  = kernelOffset + kernelBytes;
    const size_t x2TileElems = static_cast<size_t>(block.x * 2 + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t x2TileBytes = x2TileElems * sizeof(D);
    const size_t x2KernelOffset = (x2TileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    const size_t x2SharedBytes  = x2KernelOffset + kernelBytes;
    const size_t x4TileElems = static_cast<size_t>(block.x * 4 + maxKernelSize.w - 1) * (block.y + maxKernelSize.h - 1);
    const size_t x4TileBytes = x4TileElems * sizeof(D);
    const size_t x4KernelOffset = (x4TileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
    const size_t x4SharedBytes  = x4KernelOffset + kernelBytes;

#define NVCV_GAUSSIAN_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE(KSIZE)                                                 \
    if (maxKernelSize.w == KSIZE && maxKernelSize.h == KSIZE)                                                        \
    {                                                                                                                \
        if (tiledGridZ <= 65535)                                                                                     \
        {                                                                                                            \
            if constexpr (kX4Always)                                                                                 \
            {                                                                                                        \
                static_assert(GaussianFixedSharedBytes<4, KSIZE, D>() <= 48 * 1024);                                 \
                dim3 grid(divUp(outData.maxSize().w, block.x * 4), divUp(outData.maxSize().h, block.y),              \
                          static_cast<unsigned int>(tiledGridZ));                                                    \
                gaussianFilter2DPlanarTiledFixed<4, KSIZE, KSIZE><<<grid, block, x4SharedBytes, stream>>>(           \
                    src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels); \
            }                                                                                                        \
            else                                                                                                     \
            {                                                                                                        \
                static_assert(GaussianFixedSharedBytes<2, KSIZE, D>() <= 48 * 1024);                                 \
                dim3 grid(divUp(outData.maxSize().w, block.x * 2), divUp(outData.maxSize().h, block.y),              \
                          static_cast<unsigned int>(tiledGridZ));                                                    \
                gaussianFilter2DPlanarTiledFixed<2, KSIZE, KSIZE><<<grid, block, x2SharedBytes, stream>>>(           \
                    src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels); \
            }                                                                                                        \
        }                                                                                                            \
        else                                                                                                         \
        {                                                                                                            \
            dim3 fallbackGrid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),              \
                              outData.numImages());                                                                  \
            gaussianFilter2DPlanar<<<fallbackGrid, block, 0, stream>>>(                                              \
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);     \
        }                                                                                                            \
    }                                                                                                                \
    else

    NVCV_GAUSSIAN_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE(3)
    NVCV_GAUSSIAN_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE(5)
    if constexpr (kX4Always)
    {
        if (enableX4 && x4SharedBytes <= 48 * 1024 && tiledGridZ <= 65535)
        {
            dim3 grid(divUp(outData.maxSize().w, block.x * 4), divUp(outData.maxSize().h, block.y),
                      static_cast<unsigned int>(tiledGridZ));
            gaussianFilter2DPlanarTiledX4<<<grid, block, x4SharedBytes, stream>>>(
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
        }
        else if (x2SharedBytes <= 48 * 1024 && tiledGridZ <= 65535)
        {
            dim3 grid(divUp(outData.maxSize().w, block.x * 2), divUp(outData.maxSize().h, block.y),
                      static_cast<unsigned int>(tiledGridZ));
            gaussianFilter2DPlanarTiledX2<<<grid, block, x2SharedBytes, stream>>>(
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
        }
        else if (sharedBytes <= 48 * 1024 && tiledGridZ <= 65535)
        {
            dim3 grid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),
                      static_cast<unsigned int>(tiledGridZ));
            gaussianFilter2DPlanarTiled<<<grid, block, sharedBytes, stream>>>(
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
        }
        else
        {
            dim3 fallbackGrid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),
                              outData.numImages());
            gaussianFilter2DPlanar<<<fallbackGrid, block, 0, stream>>>(
                src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
        }
    }
    else if (x2SharedBytes <= 48 * 1024 && tiledGridZ <= 65535)
    {
        dim3 grid(divUp(outData.maxSize().w, block.x * 2), divUp(outData.maxSize().h, block.y),
                  static_cast<unsigned int>(tiledGridZ));
        gaussianFilter2DPlanarTiledX2<<<grid, block, x2SharedBytes, stream>>>(
            src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
    }
    else if (sharedBytes <= 48 * 1024 && tiledGridZ <= 65535)
    {
        dim3 grid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),
                  static_cast<unsigned int>(tiledGridZ));
        gaussianFilter2DPlanarTiled<<<grid, block, sharedBytes, stream>>>(
            src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
    }
    else
    {
        dim3 fallbackGrid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y),
                          outData.numImages());
        gaussianFilter2DPlanar<<<fallbackGrid, block, 0, stream>>>(
            src, dst, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize, channels);
    }

#undef NVCV_GAUSSIAN_VARSHAPE_PLANAR_FILTER_TILED_FIXED_CASE
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void GaussianFilter2D(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                      const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                      const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                      const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize,
                      NVCVBorderType borderMode, float borderValue, bool enableX4, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda  &inData,
                           const ImageBatchVarShapeDataStridedCuda  &outData,
                           const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                           const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                           const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize, bool enableX4,
                           float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {GaussianFilter2DCaller<D, NVCV_BORDER_CONSTANT>, GaussianFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           GaussianFilter2DCaller<D, NVCV_BORDER_REFLECT>, GaussianFilter2DCaller<D, NVCV_BORDER_WRAP>,
           GaussianFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize,
                      enableX4, borderValue, stream);
}

template<typename D>
void GaussianFilter2DPlanar(const ImageBatchVarShapeDataStridedCuda  &inData,
                            const ImageBatchVarShapeDataStridedCuda  &outData,
                            const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                            const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                            const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize,
                            NVCVBorderType borderMode, float borderValue, int channels, bool enableX4,
                            cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda  &inData,
                           const ImageBatchVarShapeDataStridedCuda  &outData,
                           const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                           const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                           const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize, int channels,
                           bool enableX4, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {GaussianFilter2DPlanarCaller<D, NVCV_BORDER_CONSTANT>,
           GaussianFilter2DPlanarCaller<D, NVCV_BORDER_REPLICATE>, GaussianFilter2DPlanarCaller<D, NVCV_BORDER_REFLECT>,
           GaussianFilter2DPlanarCaller<D, NVCV_BORDER_WRAP>, GaussianFilter2DPlanarCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelTensor, kernelSizeTensor, maxKernelSize, sigmaTensor, dataKernelSize,
                      channels, enableX4, borderValue, stream);
}

GaussianVarShape::GaussianVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize,
                                   int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
    , m_maxBatchSize(maxBatchSize)
{
    if (maxBatchSize > 0)
    {
        const size_t kernelBytes = cvcuda::priv::CheckedMulMany(
            {sizeof(float), cvcuda::priv::CheckedPositiveToSize(maxKernelSize.w, "maxKernelSize.w"),
             cvcuda::priv::CheckedPositiveToSize(maxKernelSize.h, "maxKernelSize.h"),
             cvcuda::priv::CheckedPositiveToSize(maxBatchSize, "maxBatchSize")},
            "GaussianVarShape kernel allocation size overflow");
        NVCV_CHECK_THROW(cudaMalloc(&m_kernel, kernelBytes));
    }
}

GaussianVarShape::~GaussianVarShape()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

ErrorCode GaussianVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                  const ImageBatchVarShapeDataStridedCuda &outData,
                                  const TensorDataStridedCuda &kernelSize, const TensorDataStridedCuda &sigma,
                                  NVCVBorderType borderMode, cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (m_maxBatchSize <= 0 || inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = IsPlanar(format);

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    int dataKernelSize = (data_type == kCV_8U ? 3 : 4);

    dim3 block(32, 4);
    dim3 grid(divUp(m_maxKernelSize.w, block.x), divUp(m_maxKernelSize.h, block.y), outData.numImages());

    cuda::Tensor1DWrap<int2, int32_t>    kernelSizeTensor(kernelSize);
    cuda::Tensor1DWrap<double2, int32_t> sigmaTensor(sigma);

    int kernelPitch2 = static_cast<int>(m_maxKernelSize.w * sizeof(float));
    int kernelPitch1 = m_maxKernelSize.h * kernelPitch2;

    cuda::Tensor3DWrap<float, int32_t> kernelTensor(m_kernel, kernelPitch1, kernelPitch2);

    computeGaussianKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, dataKernelSize, m_maxKernelSize,
                                                              kernelSizeTensor, sigmaTensor);

    checkKernelErrors();

    typedef void (*filter2D_t)(const ImageBatchVarShapeDataStridedCuda  &inData,
                               const ImageBatchVarShapeDataStridedCuda  &outData,
                               const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
                               const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
                               const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize,
                               NVCVBorderType borderMode, float borderValue, bool enableX4, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { GaussianFilter2D<uchar>, 0,  GaussianFilter2D<uchar3>,  GaussianFilter2D<uchar4>},
        {                       0, 0,                         0,                         0},
        {GaussianFilter2D<ushort>, 0, GaussianFilter2D<ushort3>, GaussianFilter2D<ushort4>},
        { GaussianFilter2D<short>, 0,  GaussianFilter2D<short3>,  GaussianFilter2D<short4>},
        {   GaussianFilter2D<int>, 0,    GaussianFilter2D<int3>,    GaussianFilter2D<int4>},
        { GaussianFilter2D<float>, 0,  GaussianFilter2D<float3>,  GaussianFilter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    if (isPlanar)
    {
        if (outData.numImages() > 65535)
        {
            LOG_ERROR("Planar Gaussian requires numImages <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        typedef void (*planar_filter2D_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            const cuda::Tensor3DWrap<float, int32_t> &kernelTensor,
            const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor, Size2D maxKernelSize,
            const cuda::Tensor1DWrap<double2, int32_t> &sigmaTensor, int dataKernelSize, NVCVBorderType borderMode,
            float borderValue, int channels, bool enableX4, cudaStream_t stream);

        static const planar_filter2D_t planarFuncs[6] = {
            GaussianFilter2DPlanar<uchar>,  0,
            GaussianFilter2DPlanar<ushort>, GaussianFilter2DPlanar<short>,
            GaussianFilter2DPlanar<int>,    GaussianFilter2DPlanar<float>,
        };

        const planar_filter2D_t planarFunc = planarFuncs[data_type];
        NVCV_ASSERT(planarFunc != 0);

        const bool enableX4 = data_type == kCV_8U || data_type == kCV_32F;
        planarFunc(inData, outData, kernelTensor, kernelSizeTensor, m_maxKernelSize, sigmaTensor, dataKernelSize,
                   borderMode, borderValue, channels, enableX4, stream);

        return ErrorCode::SUCCESS;
    }

    const bool enableX4 = data_type == kCV_8U || channels == 1
                       || (data_type == kCV_32F && channels == 3 && m_maxKernelSize.w > 5 && m_maxKernelSize.h > 5);
    func(inData, outData, kernelTensor, kernelSizeTensor, m_maxKernelSize, sigmaTensor, dataKernelSize, borderMode,
         borderValue, enableX4, stream);

    return ErrorCode::SUCCESS;
}

// AverageBlurVarShape ---------------------------------------------------------

template<class BorderWrapper, class SrcWrapper, class DstWrapper>
__global__ void avgBlurFilter2D(const BorderWrapper src, const SrcWrapper srcRaw, DstWrapper dst,
                                cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr,
                                cuda::Tensor1DWrap<int2, int32_t> kernelAnchorArr)
{
    using work_type = cuda::ConvertBaseTypeTo<float, typename DstWrapper::ValueType>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    NVCV_CUDA_ASSERT(kernelSize.x > 0 && kernelSize.y > 0, "E Wrong kernelSize=(%d,%d), expected both > 0",
                     kernelSize.x, kernelSize.y);

    int2 anchor = kernelAnchorArr[batch_idx];
    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;
    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    if (kernelSize.x == 5 && kernelSize.y == 5 && anchor.x == 2 && anchor.y == 2)
    {
        constexpr float kernelValue = static_cast<float>(1.0 / 25);
        int3            srcCoord{0, 0, batch_idx};

        if (x >= 2 && x < dst.width(batch_idx) - 2 && y >= 2 && y < dst.height(batch_idx) - 2)
        {
#pragma unroll
            for (int i = 0; i < 5; ++i)
            {
                srcCoord.y = y - 2 + i;

#pragma unroll
                for (int j = 0; j < 5; ++j)
                {
                    srcCoord.x = x - 2 + j;

                    res = res + srcRaw[srcCoord] * kernelValue;
                }
            }

            *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
            return;
        }

#pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            srcCoord.y = y - 2 + i;

#pragma unroll
            for (int j = 0; j < 5; ++j)
            {
                srcCoord.x = x - 2 + j;

                res = res + src[srcCoord] * kernelValue;
            }
        }

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
        return;
    }

    const float kernelValue = static_cast<float>(1.0 / (kernelSize.x * kernelSize.y));
    int3        srcCoord{0, 0, batch_idx};

    for (int i = 0; i < kernelSize.y; ++i)
    {
        srcCoord.y = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            srcCoord.x = x - anchor.x + j;

            res = res + src[srcCoord] * kernelValue;
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<typename DstWrapper::ValueType>(res);
}

template<class BorderWrapper, class SrcWrapper, class DstWrapper>
__global__ void avgBlurFilter2DPlanar(const BorderWrapper src, const SrcWrapper srcRaw, DstWrapper dst,
                                      cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr,
                                      cuda::Tensor1DWrap<int2, int32_t> kernelAnchorArr, int channels)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.width(batch_idx, 0) || y >= dst.height(batch_idx, 0))
        return;

    int2 kernelSize = kernelSizeArr[batch_idx];
    NVCV_CUDA_ASSERT(kernelSize.x > 0 && kernelSize.y > 0, "E Wrong kernelSize=(%d,%d), expected both > 0",
                     kernelSize.x, kernelSize.y);

    int2 anchor = kernelAnchorArr[batch_idx];
    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;
    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    if (kernelSize.x == 5 && kernelSize.y == 5 && anchor.x == 2 && anchor.y == 2)
    {
        constexpr float kernelValue = static_cast<float>(1.0 / 25);
        const bool isInterior = x >= 2 && x < dst.width(batch_idx, 0) - 2 && y >= 2 && y < dst.height(batch_idx, 0) - 2;

        for (int plane = 0; plane < channels; ++plane)
        {
            work_type res      = cuda::SetAll<work_type>(0);
            int4      srcCoord = {0, 0, plane, batch_idx};

            if (isInterior)
            {
#pragma unroll
                for (int i = 0; i < 5; ++i)
                {
                    srcCoord.y = y - 2 + i;

#pragma unroll
                    for (int j = 0; j < 5; ++j)
                    {
                        srcCoord.x = x - 2 + j;

                        res = res + srcRaw[srcCoord] * kernelValue;
                    }
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 5; ++i)
                {
                    srcCoord.y = y - 2 + i;

#pragma unroll
                    for (int j = 0; j < 5; ++j)
                    {
                        srcCoord.x = x - 2 + j;

                        res = res + src[srcCoord] * kernelValue;
                    }
                }
            }

            *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
        }

        return;
    }

    const float kernelValue = static_cast<float>(1.0 / (kernelSize.x * kernelSize.y));

    for (int plane = 0; plane < channels; ++plane)
    {
        work_type res      = cuda::SetAll<work_type>(0);
        int4      srcCoord = {0, 0, plane, batch_idx};

        for (int i = 0; i < kernelSize.y; ++i)
        {
            srcCoord.y = y - anchor.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                srcCoord.x = x - anchor.x + j;

                res = res + src[srcCoord] * kernelValue;
            }
        }

        *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
    }
}

template<typename D, NVCVBorderType B>
void AverageBlurFilter2DCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData,
                               const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
                               const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, float borderValue,
                               cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B>  src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<const D> srcRaw(inData);
    cuda::ImageBatchVarShapeWrap<D>       dst(outData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(32, 8);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    avgBlurFilter2D<<<grid, block, 0, stream>>>(src, srcRaw, dst, kernelSizeTensor, kernelAnchorTensor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D, NVCVBorderType B>
void AverageBlurFilter2DPlanarCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                                     const ImageBatchVarShapeDataStridedCuda &outData,
                                     const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
                                     const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, int channels,
                                     float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const D, B>  src(inData, cuda::SetAll<D>(borderValue));
    cuda::ImageBatchVarShapeWrap<const D> srcRaw(inData);
    cuda::ImageBatchVarShapeWrap<D>       dst(outData);

    dim3 block(32, 8);
    dim3 grid(divUp(outData.maxSize().w, block.x), divUp(outData.maxSize().h, block.y), outData.numImages());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    avgBlurFilter2DPlanar<<<grid, block, 0, stream>>>(src, srcRaw, dst, kernelSizeTensor, kernelAnchorTensor, channels);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void AverageBlurFilter2D(const ImageBatchVarShapeDataStridedCuda &inData,
                         const ImageBatchVarShapeDataStridedCuda &outData,
                         const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
                         const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, NVCVBorderType borderMode,
                         float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
        const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
        const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {AverageBlurFilter2DCaller<D, NVCV_BORDER_CONSTANT>, AverageBlurFilter2DCaller<D, NVCV_BORDER_REPLICATE>,
           AverageBlurFilter2DCaller<D, NVCV_BORDER_REFLECT>, AverageBlurFilter2DCaller<D, NVCV_BORDER_WRAP>,
           AverageBlurFilter2DCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelSizeTensor, kernelAnchorTensor, borderValue, stream);
}

template<typename D>
void AverageBlurFilter2DPlanar(const ImageBatchVarShapeDataStridedCuda &inData,
                               const ImageBatchVarShapeDataStridedCuda &outData,
                               const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
                               const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, NVCVBorderType borderMode,
                               float borderValue, int channels, cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
                           const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, int channels, float borderValue,
                           cudaStream_t stream);

    static const func_t funcs[] = {AverageBlurFilter2DPlanarCaller<D, NVCV_BORDER_CONSTANT>,
                                   AverageBlurFilter2DPlanarCaller<D, NVCV_BORDER_REPLICATE>,
                                   AverageBlurFilter2DPlanarCaller<D, NVCV_BORDER_REFLECT>,
                                   AverageBlurFilter2DPlanarCaller<D, NVCV_BORDER_WRAP>,
                                   AverageBlurFilter2DPlanarCaller<D, NVCV_BORDER_REFLECT101>};

    funcs[borderMode](inData, outData, kernelSizeTensor, kernelAnchorTensor, channels, borderValue, stream);
}

AverageBlurVarShape::AverageBlurVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize,
                                         int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
    , m_maxBatchSize(maxBatchSize)
{
}

AverageBlurVarShape::~AverageBlurVarShape() {}

ErrorCode AverageBlurVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                     const ImageBatchVarShapeDataStridedCuda &outData,
                                     const TensorDataStridedCuda &kernelSize, const TensorDataStridedCuda &kernelAnchor,
                                     NVCVBorderType borderMode, cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (m_maxBatchSize <= 0 || inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = IsPlanar(format);

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    cuda::Tensor1DWrap<int2, int32_t> kernelSizeTensor(kernelSize);
    cuda::Tensor1DWrap<int2, int32_t> kernelAnchorTensor(kernelAnchor);

    // clang-format off
    typedef void (*filter2D_t)(const ImageBatchVarShapeDataStridedCuda  &inData,
                               const ImageBatchVarShapeDataStridedCuda  &outData,
                               const cuda::Tensor1DWrap<int2,  int32_t> &kernelSizeTensor,
                               const cuda::Tensor1DWrap<int2,  int32_t> &kernelAnchorTensor,
                               NVCVBorderType borderMode,
                               float borderValue,
                               cudaStream_t stream);
    // clang-format on

    static const filter2D_t funcs[6][4] = {
        { AverageBlurFilter2D<uchar>, 0,  AverageBlurFilter2D<uchar3>,  AverageBlurFilter2D<uchar4>},
        {                          0, 0,                            0,                            0},
        {AverageBlurFilter2D<ushort>, 0, AverageBlurFilter2D<ushort3>, AverageBlurFilter2D<ushort4>},
        { AverageBlurFilter2D<short>, 0,  AverageBlurFilter2D<short3>,  AverageBlurFilter2D<short4>},
        {   AverageBlurFilter2D<int>, 0,    AverageBlurFilter2D<int3>,    AverageBlurFilter2D<int4>},
        { AverageBlurFilter2D<float>, 0,  AverageBlurFilter2D<float3>,  AverageBlurFilter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    if (isPlanar)
    {
        if (outData.numImages() > 65535)
        {
            LOG_ERROR("Planar AverageBlur requires numImages <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        typedef void (*planar_filter2D_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            const cuda::Tensor1DWrap<int2, int32_t> &kernelSizeTensor,
            const cuda::Tensor1DWrap<int2, int32_t> &kernelAnchorTensor, NVCVBorderType borderMode, float borderValue,
            int channels, cudaStream_t stream);

        static const planar_filter2D_t planarFuncs[6] = {
            AverageBlurFilter2DPlanar<uchar>,  0,
            AverageBlurFilter2DPlanar<ushort>, AverageBlurFilter2DPlanar<short>,
            AverageBlurFilter2DPlanar<int>,    AverageBlurFilter2DPlanar<float>,
        };

        const planar_filter2D_t planarFunc = planarFuncs[data_type];

        NVCV_ASSERT(planarFunc != 0);

        planarFunc(inData, outData, kernelSizeTensor, kernelAnchorTensor, borderMode, borderValue, channels, stream);

        return ErrorCode::SUCCESS;
    }

    func(inData, outData, kernelSizeTensor, kernelAnchorTensor, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
