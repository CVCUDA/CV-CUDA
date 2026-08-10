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

#include "../PlanarTensorView.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "filter_utils.cuh"

#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <type_traits>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

} // namespace

template<class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2D(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel, Size2D kernelSize,
                         int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int  kInd = 0;
    int3 coord{x, y, batch_idx};

    for (int i = 0; i < kernelSize.h; ++i)
    {
        coord.y = y - kernelAnchor.y + i;

        for (int j = 0; j < kernelSize.w; ++j)
        {
            coord.x = x - kernelAnchor.x + j;

            res = res + src[coord] * kernel[kInd++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2DTiled(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel, Size2D kernelSize,
                              int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x           = blockIdx.x * blockDim.x + threadIdx.x;
    const int y           = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx   = get_batch_idx();
    const int tileW       = blockDim.x + kernelSize.w - 1;
    const int tileH       = blockDim.y + kernelSize.h - 1;
    const int baseX       = blockIdx.x * blockDim.x - kernelAnchor.x;
    const int baseY       = blockIdx.y * blockDim.y - kernelAnchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = kernelSize.w * kernelSize.h;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 coord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[coord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = kernel[idx];
    }

    __syncthreads();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    work_type res  = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.h; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.w; ++j)
        {
            const int tileX = threadIdx.x + j;

            res = res + tile[tileY * tileW + tileX] * kernelTile[kInd++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2DTiledX2(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel, Size2D kernelSize,
                                int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW     = blockDim.x * 2;
    const int x0          = blockIdx.x * outputW + threadIdx.x * 2;
    const int x1          = x0 + 1;
    const int y           = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx   = get_batch_idx();
    const int tileW       = outputW + kernelSize.w - 1;
    const int tileH       = blockDim.y + kernelSize.h - 1;
    const int baseX       = blockIdx.x * outputW - kernelAnchor.x;
    const int baseY       = blockIdx.y * blockDim.y - kernelAnchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = kernelSize.w * kernelSize.h;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 coord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[coord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = kernel[idx];
    }

    __syncthreads();

    if (x0 >= dstSize.w || y >= dstSize.h)
        return;

    work_type res0 = cuda::SetAll<work_type>(0);
    work_type res1 = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.h; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.w; ++j)
        {
            const int   tileX = threadIdx.x * 2 + j;
            const float k     = kernelTile[kInd++];

            res0 = res0 + tile[tileY * tileW + tileX] * k;
            res1 = res1 + tile[tileY * tileW + tileX + 1] * k;
        }
    }

    *dst.ptr(batch_idx, y, x0) = cuda::SaturateCast<T>(res0);
    if (x1 < dstSize.w)
    {
        *dst.ptr(batch_idx, y, x1) = cuda::SaturateCast<T>(res1);
    }
}

template<class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2DTiledX4(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel, Size2D kernelSize,
                                int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW     = blockDim.x * 4;
    const int x0          = blockIdx.x * outputW + threadIdx.x * 4;
    const int x1          = x0 + 1;
    const int x2          = x0 + 2;
    const int x3          = x0 + 3;
    const int y           = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx   = get_batch_idx();
    const int tileW       = outputW + kernelSize.w - 1;
    const int tileH       = blockDim.y + kernelSize.h - 1;
    const int baseX       = blockIdx.x * outputW - kernelAnchor.x;
    const int baseY       = blockIdx.y * blockDim.y - kernelAnchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = kernelSize.w * kernelSize.h;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 coord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[coord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = kernel[idx];
    }

    __syncthreads();

    if (x0 >= dstSize.w || y >= dstSize.h)
        return;

    work_type res0 = cuda::SetAll<work_type>(0);
    work_type res1 = cuda::SetAll<work_type>(0);
    work_type res2 = cuda::SetAll<work_type>(0);
    work_type res3 = cuda::SetAll<work_type>(0);
    int       kInd = 0;

    for (int i = 0; i < kernelSize.h; ++i)
    {
        const int tileY = threadIdx.y + i;

        for (int j = 0; j < kernelSize.w; ++j)
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
    if (x1 < dstSize.w)
    {
        *dst.ptr(batch_idx, y, x1) = cuda::SaturateCast<T>(res1);
    }
    if (x2 < dstSize.w)
    {
        *dst.ptr(batch_idx, y, x2) = cuda::SaturateCast<T>(res2);
    }
    if (x3 < dstSize.w)
    {
        *dst.ptr(batch_idx, y, x3) = cuda::SaturateCast<T>(res3);
    }
}

template<int Outputs, int KernelW, int KernelH, class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2DTiledFixed(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel,
                                   int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW     = blockDim.x * Outputs;
    const int x0          = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y           = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx   = get_batch_idx();
    const int tileW       = outputW + KernelW - 1;
    const int tileH       = blockDim.y + KernelH - 1;
    const int baseX       = blockIdx.x * outputW - kernelAnchor.x;
    const int baseY       = blockIdx.y * blockDim.y - kernelAnchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = KernelW * KernelH;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);
    const size_t kernelOffset = (tileElems * sizeof(T) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float       *kernelTile   = reinterpret_cast<float *>(tileRaw + kernelOffset);

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        int3 coord{baseX + idx % tileW, baseY + idx / tileW, batch_idx};
        tile[idx] = src[coord];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = kernel[idx];
    }

    __syncthreads();

    if (x0 >= dstSize.w || y >= dstSize.h)
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    int kInd = 0;
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

#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dstSize.w)
        {
            *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

template<int KSize, class SrcWrapper, class DstWrapper>
__global__ void laplacian2DU8(SrcWrapper src, DstWrapper dst, Size2D dstSize, float scale)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int3 coord{x, y - 1, batch_idx};

    work_type res;
    if constexpr (KSize == 1)
    {
        const float axisK   = scale;
        const float centerK = -4.0f * scale;

        res     = src[coord] * axisK;
        coord.x = x - 1;
        coord.y = y;
        res     = res + src[coord] * axisK;
        coord.x = x;
        res     = res + src[coord] * centerK;
        coord.x = x + 1;
        res     = res + src[coord] * axisK;
        coord.x = x;
        coord.y = y + 1;
        res     = res + src[coord] * axisK;
    }
    else
    {
        static_assert(KSize == 3);
        const float cornerK = 2.0f * scale;
        const float centerK = -8.0f * scale;

        coord.x = x - 1;
        res     = src[coord] * cornerK;
        coord.x = x + 1;
        res     = res + src[coord] * cornerK;
        coord.x = x;
        coord.y = y;
        res     = res + src[coord] * centerK;
        coord.x = x - 1;
        coord.y = y + 1;
        res     = res + src[coord] * cornerK;
        coord.x = x + 1;
        res     = res + src[coord] * cornerK;
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<int KSize, class SrcWrapper, class DstWrapper>
__global__ void laplacian2DFloat(SrcWrapper src, DstWrapper dst, Size2D dstSize, float scale)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int3 coord{x, y - 1, batch_idx};

    if constexpr (KSize == 1)
    {
        const float axisK   = scale;
        const float centerK = -4.0f * scale;

        work_type res = LaplacianFloatMulRN(src[coord], axisK);
        coord.x       = x - 1;
        coord.y       = y;
        res           = LaplacianFloatFmaRN(src[coord], axisK, res);
        coord.x       = x;
        res           = LaplacianFloatFmaRN(src[coord], centerK, res);
        coord.x       = x + 1;
        res           = LaplacianFloatFmaRN(src[coord], axisK, res);

        coord.y = y + 1;
        coord.x = x;
        res     = LaplacianFloatFmaRN(src[coord], axisK, res);

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
    }
    else
    {
        static_assert(KSize == 3);
        const float cornerK = 2.0f * scale;
        const float centerK = -8.0f * scale;

        coord.x       = x - 1;
        work_type res = LaplacianFloatMulRN(src[coord], cornerK);
        coord.x       = x + 1;
        res           = LaplacianFloatFmaRN(src[coord], cornerK, res);

        coord.x = x;
        coord.y = y;
        res     = LaplacianFloatFmaRN(src[coord], centerK, res);

        coord.x = x - 1;
        coord.y = y + 1;
        res     = LaplacianFloatFmaRN(src[coord], cornerK, res);
        coord.x = x + 1;
        res     = LaplacianFloatFmaRN(src[coord], cornerK, res);

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
    }
}

constexpr int kLaplacianPlanarBlockWidth = 32;

template<int KSize, int NIX, class SrcWrapper, class DstWrapper>
__global__ void laplacian2DPlanarTiled(SrcWrapper src, DstWrapper dst, Size2D dstSize, float scale)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    constexpr int kRadius = 1;

    constexpr int outputW   = kLaplacianPlanarBlockWidth * NIX;
    const int     outBaseX  = blockIdx.x * outputW;
    const int     outBaseY  = blockIdx.y * blockDim.y;
    const int     batch_idx = get_batch_idx();

    if (outBaseX >= dstSize.w || outBaseY >= dstSize.h)
        return;

    constexpr int tileW     = outputW + 2 * kRadius;
    const int     tileH     = blockDim.y + 2 * kRadius;
    const int     tileElems = tileW * tileH;
    const int     tid       = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile = reinterpret_cast<T *>(tileRaw);

    for (int idx = tid; idx < tileElems; idx += blockDim.x * blockDim.y)
    {
        int3 coord{outBaseX - kRadius + idx % tileW, outBaseY - kRadius + idx / tileW, batch_idx};
        tile[idx] = src[coord];
    }

    __syncthreads();

    const int y = outBaseY + threadIdx.y;
    if (y >= dstSize.h)
        return;

    const int tileY         = threadIdx.y + kRadius;
    auto      computeOutput = [&](int tileX)
    {
        work_type res;
        if constexpr (KSize == 1)
        {
            const float axisK   = scale;
            const float centerK = -4.0f * scale;

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
            static_assert(KSize == 3);
            const float cornerK = 2.0f * scale;
            const float centerK = -8.0f * scale;

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

    if constexpr (sizeof(T) == 1 && NIX == 4)
    {
        const int x0 = outBaseX + threadIdx.x * NIX;
        if (x0 >= dstSize.w)
            return;

        const int valid = dstSize.w - x0 < NIX ? dstSize.w - x0 : NIX;
        T         outputs[NIX];

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            if (i < valid)
                outputs[i] = computeOutput(threadIdx.x * NIX + i + kRadius);
        }

        T *dstPtr = dst.ptr(batch_idx, y, x0);
        if (valid == NIX && (reinterpret_cast<uintptr_t>(dstPtr) & (alignof(uchar4) - 1)) == 0)
        {
            *reinterpret_cast<uchar4 *>(dstPtr) = make_uchar4(outputs[0], outputs[1], outputs[2], outputs[3]);
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
    else
    {
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int x = outBaseX + threadIdx.x + i * blockDim.x;
            if (x < dstSize.w)
                *dst.ptr(batch_idx, y, x) = computeOutput(threadIdx.x + i * blockDim.x + kRadius);
        }
    }
}

template<typename T, NVCVBorderType B, class KernelWrapper>
ErrorCode Filter2DCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                         KernelWrapper kernel, Size2D kernelSize, int2 kernelAnchor, float borderValue,
                         cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(16, 16);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        filter2D<<<grid, block, 0, stream>>>(src, dst, dstSize, kernel, kernelSize, kernelAnchor);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<int KSize, typename T, NVCVBorderType B>
ErrorCode LaplacianFloatCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float scale,
                               float borderValue, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(16, 16);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        laplacian2DFloat<KSize><<<grid, block, 0, stream>>>(src, dst, dstSize, scale);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<int KSize, typename T>
ErrorCode LaplacianFloat(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float scale,
                         NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_LAPLACIAN_FLOAT_CASE(BORDERTYPE) \
    case BORDERTYPE:                          \
        return LaplacianFloatCaller<KSize, T, BORDERTYPE>(inData, outData, scale, borderValue, stream)

        NVCV_LAPLACIAN_FLOAT_CASE(NVCV_BORDER_CONSTANT);
        NVCV_LAPLACIAN_FLOAT_CASE(NVCV_BORDER_REPLICATE);
        NVCV_LAPLACIAN_FLOAT_CASE(NVCV_BORDER_REFLECT);
        NVCV_LAPLACIAN_FLOAT_CASE(NVCV_BORDER_WRAP);
        NVCV_LAPLACIAN_FLOAT_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_LAPLACIAN_FLOAT_CASE
    default:
        return ErrorCode::INVALID_PARAMETER;
    }
}

template<typename T>
ErrorCode LaplacianFloat(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int ksize,
                         float scale, NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    if (ksize == 1)
    {
        return LaplacianFloat<1, T>(inData, outData, scale, borderMode, borderValue, stream);
    }
    return LaplacianFloat<3, T>(inData, outData, scale, borderMode, borderValue, stream);
}

template<int KSize, typename T, NVCVBorderType B>
ErrorCode LaplacianU8Caller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float scale,
                            float borderValue, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(16, 16);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        laplacian2DU8<KSize><<<grid, block, 0, stream>>>(src, dst, dstSize, scale);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<int KSize, typename T>
ErrorCode LaplacianU8(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float scale,
                      NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_LAPLACIAN_U8_CASE(BORDERTYPE) \
    case BORDERTYPE:                       \
        return LaplacianU8Caller<KSize, T, BORDERTYPE>(inData, outData, scale, borderValue, stream)

        NVCV_LAPLACIAN_U8_CASE(NVCV_BORDER_CONSTANT);
        NVCV_LAPLACIAN_U8_CASE(NVCV_BORDER_REPLICATE);
        NVCV_LAPLACIAN_U8_CASE(NVCV_BORDER_REFLECT);
        NVCV_LAPLACIAN_U8_CASE(NVCV_BORDER_WRAP);
        NVCV_LAPLACIAN_U8_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_LAPLACIAN_U8_CASE
    default:
        return ErrorCode::INVALID_PARAMETER;
    }
}

template<typename T>
ErrorCode LaplacianU8(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int ksize, float scale,
                      NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    if (ksize == 1)
    {
        return LaplacianU8<1, T>(inData, outData, scale, borderMode, borderValue, stream);
    }
    return LaplacianU8<3, T>(inData, outData, scale, borderMode, borderValue, stream);
}

template<int NIX, int KSize, typename T, NVCVBorderType B>
ErrorCode LaplacianPlanarTiledCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                     float scale, float borderValue, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(kLaplacianPlanarBlockWidth, 16);
    dim3 grid(divUp(dstSize.w, block.x * NIX), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        const size_t sharedBytes = static_cast<size_t>(block.x * NIX + 2) * (block.y + 2) * sizeof(T);
        laplacian2DPlanarTiled<KSize, NIX><<<grid, block, sharedBytes, stream>>>(src, dst, dstSize, scale);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<int NIX, int KSize, typename T>
ErrorCode LaplacianPlanarTiled(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float scale,
                               NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_LAPLACIAN_PLANAR_TILED_CASE(BORDERTYPE) \
    case BORDERTYPE:                                 \
        return LaplacianPlanarTiledCaller<NIX, KSize, T, BORDERTYPE>(inData, outData, scale, borderValue, stream)

        NVCV_LAPLACIAN_PLANAR_TILED_CASE(NVCV_BORDER_CONSTANT);
        NVCV_LAPLACIAN_PLANAR_TILED_CASE(NVCV_BORDER_REPLICATE);
        NVCV_LAPLACIAN_PLANAR_TILED_CASE(NVCV_BORDER_REFLECT);
        NVCV_LAPLACIAN_PLANAR_TILED_CASE(NVCV_BORDER_WRAP);
        NVCV_LAPLACIAN_PLANAR_TILED_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_LAPLACIAN_PLANAR_TILED_CASE
    default:
        return ErrorCode::INVALID_PARAMETER;
    }
}

template<int NIX, typename T>
ErrorCode LaplacianPlanarTiled(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int ksize,
                               float scale, NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    if (ksize == 1)
    {
        return LaplacianPlanarTiled<NIX, 1, T>(inData, outData, scale, borderMode, borderValue, stream);
    }
    return LaplacianPlanarTiled<NIX, 3, T>(inData, outData, scale, borderMode, borderValue, stream);
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

template<typename T, NVCVBorderType B, class KernelWrapper>
ErrorCode GaussianFilter2DTiledCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                      KernelWrapper kernel, Size2D kernelSize, int2 kernelAnchor, float borderValue,
                                      bool enableX4, cudaStream_t stream)
{
    using BaseT = cuda::BaseType<T>;

    constexpr bool kX4Always = std::is_same_v<BaseT, uchar> || std::is_same_v<T, float>;
    // Scalar U16/S16/S32 share this instantiation: packed C1 selects x4, while planar selects x2.
    constexpr bool kX4LayoutDependent = cuda::NumElements<T> == 1 && !kX4Always;
    constexpr bool kDynamicX4Possible = kX4Always || kX4LayoutDependent || std::is_same_v<T, float3>;

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(kGaussianBlockWidth, kGaussianBlockHeight);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());
    dim3 x2Grid(divUp(dstSize.w, block.x * 2), divUp(dstSize.h, block.y), outAccess->numSamples());
    dim3 x4Grid(divUp(dstSize.w, block.x * 4), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        const size_t kernelBytes  = static_cast<size_t>(kernelSize.w) * kernelSize.h * sizeof(float);
        const size_t tileElems    = static_cast<size_t>(block.x + kernelSize.w - 1) * (block.y + kernelSize.h - 1);
        const size_t tileBytes    = tileElems * sizeof(T);
        const size_t kernelOffset = (tileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
        const size_t sharedBytes  = kernelOffset + kernelBytes;
        const size_t x2TileElems  = static_cast<size_t>(block.x * 2 + kernelSize.w - 1) * (block.y + kernelSize.h - 1);
        const size_t x2TileBytes  = x2TileElems * sizeof(T);
        const size_t x2KernelOffset = (x2TileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
        const size_t x2SharedBytes  = x2KernelOffset + kernelBytes;
        const size_t x4TileElems = static_cast<size_t>(block.x * 4 + kernelSize.w - 1) * (block.y + kernelSize.h - 1);
        const size_t x4TileBytes = x4TileElems * sizeof(T);
        const size_t x4KernelOffset = (x4TileBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
        const size_t x4SharedBytes  = x4KernelOffset + kernelBytes;

#define NVCV_GAUSSIAN_FILTER_TILED_FIXED_CASE(KSIZE)                                                     \
    if (kernelSize.w == KSIZE && kernelSize.h == KSIZE)                                                  \
    {                                                                                                    \
        if constexpr (kX4Always)                                                                         \
        {                                                                                                \
            static_assert(GaussianFixedSharedBytes<4, KSIZE, T>() <= 48 * 1024);                         \
            filter2DTiledFixed<4, KSIZE, KSIZE>                                                          \
                <<<x4Grid, block, x4SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelAnchor);     \
        }                                                                                                \
        else if constexpr (kX4LayoutDependent)                                                           \
        {                                                                                                \
            static_assert(GaussianFixedSharedBytes<4, KSIZE, T>() <= 48 * 1024);                         \
            static_assert(GaussianFixedSharedBytes<2, KSIZE, T>() <= 48 * 1024);                         \
            if (enableX4)                                                                                \
            {                                                                                            \
                filter2DTiledFixed<4, KSIZE, KSIZE>                                                      \
                    <<<x4Grid, block, x4SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelAnchor); \
            }                                                                                            \
            else                                                                                         \
            {                                                                                            \
                filter2DTiledFixed<2, KSIZE, KSIZE>                                                      \
                    <<<x2Grid, block, x2SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelAnchor); \
            }                                                                                            \
        }                                                                                                \
        else                                                                                             \
        {                                                                                                \
            static_assert(GaussianFixedSharedBytes<2, KSIZE, T>() <= 48 * 1024);                         \
            filter2DTiledFixed<2, KSIZE, KSIZE>                                                          \
                <<<x2Grid, block, x2SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelAnchor);     \
        }                                                                                                \
    }                                                                                                    \
    else

        NVCV_GAUSSIAN_FILTER_TILED_FIXED_CASE(3)
        NVCV_GAUSSIAN_FILTER_TILED_FIXED_CASE(5)
        if constexpr (kDynamicX4Possible)
        {
            if (enableX4 && x4SharedBytes <= 48 * 1024)
            {
                filter2DTiledX4<<<x4Grid, block, x4SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelSize,
                                                                          kernelAnchor);
            }
            else if (x2SharedBytes <= 48 * 1024)
            {
                filter2DTiledX2<<<x2Grid, block, x2SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelSize,
                                                                          kernelAnchor);
            }
            else if (sharedBytes <= 48 * 1024)
            {
                filter2DTiled<<<grid, block, sharedBytes, stream>>>(src, dst, dstSize, kernel, kernelSize,
                                                                    kernelAnchor);
            }
            else
            {
                filter2D<<<grid, block, 0, stream>>>(src, dst, dstSize, kernel, kernelSize, kernelAnchor);
            }
        }
        else if (x2SharedBytes <= 48 * 1024)
        {
            filter2DTiledX2<<<x2Grid, block, x2SharedBytes, stream>>>(src, dst, dstSize, kernel, kernelSize,
                                                                      kernelAnchor);
        }
        else if (sharedBytes <= 48 * 1024)
        {
            filter2DTiled<<<grid, block, sharedBytes, stream>>>(src, dst, dstSize, kernel, kernelSize, kernelAnchor);
        }
        else
        {
            filter2D<<<grid, block, 0, stream>>>(src, dst, dstSize, kernel, kernelSize, kernelAnchor);
        }

#undef NVCV_GAUSSIAN_FILTER_TILED_FIXED_CASE
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<typename T, class KernelWrapper>
ErrorCode GaussianFilter2DTiled(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                KernelWrapper kernel, Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode,
                                float borderValue, bool enableX4, cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_GAUSSIAN_FILTER_TILED_CASE(BORDERTYPE)                                                          \
    case BORDERTYPE:                                                                                         \
        return GaussianFilter2DTiledCaller<T, BORDERTYPE>(inData, outData, kernel, kernelSize, kernelAnchor, \
                                                          borderValue, enableX4, stream)

        NVCV_GAUSSIAN_FILTER_TILED_CASE(NVCV_BORDER_CONSTANT);
        NVCV_GAUSSIAN_FILTER_TILED_CASE(NVCV_BORDER_REPLICATE);
        NVCV_GAUSSIAN_FILTER_TILED_CASE(NVCV_BORDER_REFLECT);
        NVCV_GAUSSIAN_FILTER_TILED_CASE(NVCV_BORDER_WRAP);
        NVCV_GAUSSIAN_FILTER_TILED_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_GAUSSIAN_FILTER_TILED_CASE
    default:
        break;
    }
    return ErrorCode::SUCCESS;
}

template<typename T, class KernelWrapper>
ErrorCode Filter2D(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, KernelWrapper kernel,
                   Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode, float borderValue,
                   cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_FILTER_CASE(BORDERTYPE) \
    case BORDERTYPE:                 \
        return Filter2DCaller<T, BORDERTYPE>(inData, outData, kernel, kernelSize, kernelAnchor, borderValue, stream)

        NVCV_FILTER_CASE(NVCV_BORDER_CONSTANT);
        NVCV_FILTER_CASE(NVCV_BORDER_REPLICATE);
        NVCV_FILTER_CASE(NVCV_BORDER_REFLECT);
        NVCV_FILTER_CASE(NVCV_BORDER_WRAP);
        NVCV_FILTER_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_FILTER_CASE
    default:
        break;
    }
    return ErrorCode::SUCCESS;
}

// Laplacian -------------------------------------------------------------------

// @brief Laplacian 3x3 kernels for ksize == 1 and ksize == 3

constexpr int kLaplacianPlanarU8NIX    = 4;
constexpr int kLaplacianPlanarFloatNIX = 4;

// clang-format off
constexpr Size2D kLaplacianKernelSize{3, 3};

constexpr cuda::math::Vector<float, 9> kLaplacianKernel1{
    {0.0f,  1.0f, 0.0f,
     1.0f, -4.0f, 1.0f,
     0.0f,  1.0f, 0.0f}
};
constexpr cuda::math::Vector<float, 9> kLaplacianKernel3{
    {2.0f,  0.0f, 2.0f,
     0.0f, -8.0f, 0.0f,
     2.0f,  0.0f, 2.0f}
};

// clang-format on

ErrorCode Laplacian::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int ksize,
                           float scale, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (!(ksize == 1 || ksize == 3))
    {
        LOG_ERROR("Invalid ksize " << ksize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int2 kernelAnchor{-1, -1};
    normalizeAnchor(kernelAnchor, kLaplacianKernelSize);
    float borderValue = .0f;

    typedef ErrorCode (*filter2D_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    cuda::math::Vector<float, 9> kernel, Size2D kernelSize, int2 kernelAnchor,
                                    NVCVBorderType borderMode, float borderValue, cudaStream_t stream);
    typedef ErrorCode (*laplacian_u8_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                        int ksize, float scale, NVCVBorderType borderMode, float borderValue,
                                        cudaStream_t stream);
    typedef ErrorCode (*laplacian_float_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                           int ksize, float scale, NVCVBorderType borderMode, float borderValue,
                                           cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        {               0, 0,                 0,                 0},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        {               0, 0,                 0,                 0},
        {               0, 0,                 0,                 0},
        {               0, 0,                 0,                 0},
    };

    static const laplacian_u8_t    u8Funcs[4] = {LaplacianU8<uchar>, 0, LaplacianU8<uchar3>, LaplacianU8<uchar4>};
    static const laplacian_float_t floatFuncs[4]
        = {LaplacianFloat<float>, 0, LaplacianFloat<float3>, LaplacianFloat<float4>};

    cuda::math::Vector<float, 9> kernel;

    if (ksize == 1)
    {
        kernel = kLaplacianKernel1;
    }
    else if (ksize == 3)
    {
        kernel = kLaplacianKernel3;
    }

    if (scale != 1)
    {
        kernel *= scale;
    }

    if (isPlanar)
    {
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int64_t numSamples = inAccess->numSamples();
        if (outAccess->numSamples() != numSamples || outAccess->numChannels() != channels)
        {
            LOG_ERROR("Planar Laplacian input and output must have matching sample and channel counts");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples > 1
            && (inAccess->sampleStride() != static_cast<int64_t>(channels) * inAccess->chStride()
                || outAccess->sampleStride() != static_cast<int64_t>(channels) * outAccess->chStride()))
        {
            LOG_ERROR("Planar Laplacian of a batched tensor requires tightly packed channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples * channels > 65535)
        {
            LOG_ERROR("Planar Laplacian requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto inView  = cvcuda::priv::PlanarAsSingleChannelView(inData, *inAccess);
        auto outView = cvcuda::priv::PlanarAsSingleChannelView(outData, *outAccess);
        if (data_type == kCV_8U)
        {
            return LaplacianPlanarTiled<kLaplacianPlanarU8NIX, uchar>(inView, outView, ksize, scale, borderMode,
                                                                      borderValue, stream);
        }
        if (data_type == kCV_32F)
        {
            return LaplacianPlanarTiled<kLaplacianPlanarFloatNIX, float>(inView, outView, ksize, scale, borderMode,
                                                                         borderValue, stream);
        }
        return funcs[data_type][0](inView, outView, kernel, kLaplacianKernelSize, kernelAnchor, borderMode, borderValue,
                                   stream);
    }

    if (data_type == kCV_8U)
    {
        const laplacian_u8_t u8Func = u8Funcs[channels - 1];
        NVCV_ASSERT(u8Func != 0);
        return u8Func(inData, outData, ksize, scale, borderMode, borderValue, stream);
    }
    if (data_type == kCV_32F)
    {
        const laplacian_float_t floatFunc = floatFuncs[channels - 1];
        NVCV_ASSERT(floatFunc != 0);
        return floatFunc(inData, outData, ksize, scale, borderMode, borderValue, stream);
    }

    return funcs[data_type][channels - 1](inData, outData, kernel, kLaplacianKernelSize, kernelAnchor, borderMode,
                                          borderValue, stream);
}

// Gaussian --------------------------------------------------------------------

Gaussian::Gaussian(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
{
    NVCV_CHECK_THROW(cudaMalloc(&m_kernel, maxKernelSize.w * maxKernelSize.h * sizeof(float)));
}

Gaussian::~Gaussian()
{
    NVCV_CHECK_LOG(cudaFree(m_kernel));
}

ErrorCode Gaussian::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Size2D kernelSize,
                          double2 sigma, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (sigma.y <= 0)
        sigma.y = sigma.x;

    // automatic detection of kernel size from sigma
    if (kernelSize.w <= 0 && sigma.x > 0)
        kernelSize.w = nvcv::cuda::round<int>(sigma.x * (data_type == kCV_8U ? 3 : 4) * 2 + 1) | 1;
    if (kernelSize.h <= 0 && sigma.y > 0)
        kernelSize.h = nvcv::cuda::round<int>(sigma.y * (data_type == kCV_8U ? 3 : 4) * 2 + 1) | 1;

    if (!(kernelSize.w > 0 && kernelSize.w % 2 == 1 && kernelSize.w <= m_maxKernelSize.w && kernelSize.h > 0
          && kernelSize.h % 2 == 1 && kernelSize.h <= m_maxKernelSize.h))
    {
        LOG_ERROR("Invalid kernel size = " << kernelSize.w << " " << kernelSize.h);
        return ErrorCode::INVALID_PARAMETER;
    }

    sigma.x = std::max(sigma.x, 0.0);
    sigma.y = std::max(sigma.y, 0.0);

    if (m_curSigma != sigma || m_curKernelSize != kernelSize)
    {
        dim3 block(32, 4);
        dim3 grid(divUp(kernelSize.w, block.x), divUp(kernelSize.h, block.y));

        computeGaussianKernel<<<grid, block, 0, stream>>>(m_kernel, kernelSize, sigma);

        checkKernelErrors();

        m_curKernelSize = kernelSize;
        m_curSigma      = sigma;
    }

    const int channels = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int2 kernelAnchor{-1, -1};
    normalizeAnchor(kernelAnchor, kernelSize);
    float borderValue = .0f;

    typedef ErrorCode (*filter2D_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    float *kernel, Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode,
                                    float borderValue, bool enableX4, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { GaussianFilter2DTiled<uchar>, 0,  GaussianFilter2DTiled<uchar3>,  GaussianFilter2DTiled<uchar4>},
        {                            0, 0,                              0,                              0},
        {GaussianFilter2DTiled<ushort>, 0, GaussianFilter2DTiled<ushort3>, GaussianFilter2DTiled<ushort4>},
        { GaussianFilter2DTiled<short>, 0,  GaussianFilter2DTiled<short3>,  GaussianFilter2DTiled<short4>},
        {   GaussianFilter2DTiled<int>, 0,    GaussianFilter2DTiled<int3>,    GaussianFilter2DTiled<int4>},
        { GaussianFilter2DTiled<float>, 0,  GaussianFilter2DTiled<float3>,  GaussianFilter2DTiled<float4>},
    };

    if (isPlanar)
    {
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int64_t numSamples = inAccess->numSamples();
        if (numSamples > 1
            && (inAccess->sampleStride() != static_cast<int64_t>(channels) * inAccess->chStride()
                || outAccess->sampleStride() != static_cast<int64_t>(channels) * outAccess->chStride()))
        {
            LOG_ERROR("Planar Gaussian of a batched tensor requires tightly packed channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples * channels > 65535)
        {
            LOG_ERROR("Planar Gaussian requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto       inView   = cvcuda::priv::PlanarAsSingleChannelView(inData, *inAccess);
        auto       outView  = cvcuda::priv::PlanarAsSingleChannelView(outData, *outAccess);
        const bool enableX4 = data_type == kCV_8U || data_type == kCV_32F;
        return funcs[data_type][0](inView, outView, m_kernel, kernelSize, kernelAnchor, borderMode, borderValue,
                                   enableX4, stream);
    }

    const bool enableX4 = data_type == kCV_8U || channels == 1
                       || (data_type == kCV_32F && channels == 3 && kernelSize.w > 5 && kernelSize.h > 5);
    return funcs[data_type][channels - 1](inData, outData, m_kernel, kernelSize, kernelAnchor, borderMode, borderValue,
                                          enableX4, stream);
}

// Average Blur ----------------------------------------------------------------

AverageBlur::AverageBlur(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_maxKernelSize(maxKernelSize)
{
}

AverageBlur::~AverageBlur() {}

template<class BorderWrapper, class SrcWrapper, class DstWrapper>
__global__ void averageBlur2D5x5(BorderWrapper borderSrc, SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    constexpr float kernelValue = static_cast<float>(1.0 / 25);
    int3            coord{x, y, batch_idx};

    if (x >= 2 && x < dstSize.w - 2 && y >= 2 && y < dstSize.h - 2)
    {
#pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            coord.y = y - 2 + i;

#pragma unroll
            for (int j = 0; j < 5; ++j)
            {
                coord.x = x - 2 + j;

                res = res + src[coord] * kernelValue;
            }
        }

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
        return;
    }

#pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        coord.y = y - 2 + i;

#pragma unroll
        for (int j = 0; j < 5; ++j)
        {
            coord.x = x - 2 + j;

            res = res + borderSrc[coord] * kernelValue;
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<class BorderWrapper, class SrcWrapper, class DstWrapper>
__global__ void averageBlur2D(BorderWrapper borderSrc, SrcWrapper src, DstWrapper dst, Size2D dstSize,
                              Size2D kernelSize, int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    const float kernelValue = static_cast<float>(1.0 / (kernelSize.w * kernelSize.h));
    int3        coord{x, y, batch_idx};

    const bool isInterior = x >= kernelAnchor.x && x < dstSize.w - (kernelSize.w - kernelAnchor.x - 1)
                         && y >= kernelAnchor.y && y < dstSize.h - (kernelSize.h - kernelAnchor.y - 1);

    if (isInterior)
    {
        for (int i = 0; i < kernelSize.h; ++i)
        {
            coord.y = y - kernelAnchor.y + i;

            for (int j = 0; j < kernelSize.w; ++j)
            {
                coord.x = x - kernelAnchor.x + j;

                res = res + src[coord] * kernelValue;
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize.h; ++i)
        {
            coord.y = y - kernelAnchor.y + i;

            for (int j = 0; j < kernelSize.w; ++j)
            {
                coord.x = x - kernelAnchor.x + j;

                res = res + borderSrc[coord] * kernelValue;
            }
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(res);
}

template<typename T, NVCVBorderType B>
ErrorCode AverageBlur2DCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                              Size2D kernelSize, int2 kernelAnchor, float borderValue, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(32, 8);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src    = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto srcRaw = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
        auto dst    = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        if constexpr (cuda::NumElements<T> == 1)
        {
            if (kernelSize.w == 5 && kernelSize.h == 5 && kernelAnchor.x == 2 && kernelAnchor.y == 2)
            {
                averageBlur2D5x5<<<grid, block, 0, stream>>>(src, srcRaw, dst, dstSize);
            }
            else
            {
                averageBlur2D<<<grid, block, 0, stream>>>(src, srcRaw, dst, dstSize, kernelSize, kernelAnchor);
            }
        }
        else
        {
            averageBlur2D<<<grid, block, 0, stream>>>(src, srcRaw, dst, dstSize, kernelSize, kernelAnchor);
        }
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<typename T>
ErrorCode AverageBlur2D(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Size2D kernelSize,
                        int2 kernelAnchor, NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_AVG_BLUR_CASE(BORDERTYPE) \
    case BORDERTYPE:                   \
        return AverageBlur2DCaller<T, BORDERTYPE>(inData, outData, kernelSize, kernelAnchor, borderValue, stream)

        NVCV_AVG_BLUR_CASE(NVCV_BORDER_CONSTANT);
        NVCV_AVG_BLUR_CASE(NVCV_BORDER_REPLICATE);
        NVCV_AVG_BLUR_CASE(NVCV_BORDER_REFLECT);
        NVCV_AVG_BLUR_CASE(NVCV_BORDER_WRAP);
        NVCV_AVG_BLUR_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_AVG_BLUR_CASE
    default:
        break;
    }
    return ErrorCode::SUCCESS;
}

ErrorCode AverageBlur::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(kernelSize.w > 0 && kernelSize.w % 2 == 1 && kernelSize.w <= m_maxKernelSize.w && kernelSize.h > 0
          && kernelSize.h % 2 == 1 && kernelSize.h <= m_maxKernelSize.h))
    {
        LOG_ERROR("Invalid ksize " << kernelSize.w << " " << kernelSize.h);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(kernelAnchor.x == -1 || (kernelAnchor.x >= 0 && kernelAnchor.x < kernelSize.w)))
    {
        LOG_ERROR("Invalid kernelAnchor.x " << kernelAnchor.x);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(kernelAnchor.y == -1 || (kernelAnchor.y >= 0 && kernelAnchor.y < kernelSize.h)))
    {
        LOG_ERROR("Invalid kernelAnchor.y " << kernelAnchor.y);
        return ErrorCode::INVALID_PARAMETER;
    }

    const int channels = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    normalizeAnchor(kernelAnchor, kernelSize);
    float borderValue = .0f;

    typedef ErrorCode (*filter2D_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode, float borderValue,
                                    cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { AverageBlur2D<uchar>, 0,  AverageBlur2D<uchar3>,  AverageBlur2D<uchar4>},
        {                    0, 0,                      0,                      0},
        {AverageBlur2D<ushort>, 0, AverageBlur2D<ushort3>, AverageBlur2D<ushort4>},
        { AverageBlur2D<short>, 0,  AverageBlur2D<short3>,  AverageBlur2D<short4>},
        {   AverageBlur2D<int>, 0,    AverageBlur2D<int3>,    AverageBlur2D<int4>},
        { AverageBlur2D<float>, 0,  AverageBlur2D<float3>,  AverageBlur2D<float4>},
    };

    if (isPlanar)
    {
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int64_t numSamples = inAccess->numSamples();
        if (outAccess->numSamples() != numSamples || outAccess->numChannels() != channels)
        {
            LOG_ERROR("Planar AverageBlur input and output must have matching sample and channel counts");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples > 1
            && (inAccess->sampleStride() != static_cast<int64_t>(channels) * inAccess->chStride()
                || outAccess->sampleStride() != static_cast<int64_t>(channels) * outAccess->chStride()))
        {
            LOG_ERROR("Planar AverageBlur of a batched tensor requires tightly packed channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples * channels > 65535)
        {
            LOG_ERROR("Planar AverageBlur requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto inView  = cvcuda::priv::PlanarAsSingleChannelView(inData, *inAccess);
        auto outView = cvcuda::priv::PlanarAsSingleChannelView(outData, *outAccess);
        return funcs[data_type][0](inView, outView, kernelSize, kernelAnchor, borderMode, borderValue, stream);
    }

    return funcs[data_type][channels - 1](inData, outData, kernelSize, kernelAnchor, borderMode, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op
