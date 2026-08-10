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
#include "random_resized_crop_common.cuh"

#include <cvcuda/cuda_tools/Compat.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>

#include <cmath>
#include <cstdlib>
#include <random>
#include <type_traits>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

#define BLOCK 32

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_v1(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                 const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

        fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
        sy = cuda::max(0, cuda::min(sy, height - 2));

        //row pointers
        const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

        { //compute source data position and weight for [x0] components
            float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

            fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
            sx = cuda::max(0, cuda::min(sx, width - 2));

            *dst.ptr(batch_idx, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
}

template<int NIX, typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_nix(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                  const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x0    = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if (dst_y < out_height)
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

        fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
        sy = cuda::max(0, cuda::min(sy, height - 2));

        const T *aPtr = src.ptr(batch_idx, sy, 0);
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0);

        T dstPack[NIX];
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int dst_x = dst_x0 + i;
            if (dst_x < out_width)
            {
                float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
                int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

                fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
                sx = cuda::max(0, cuda::min(sx, width - 2));

                dstPack[i] = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                                   + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
            }
        }

        T *dstRow = dst.ptr(batch_idx, dst_y, 0);
        // dst_x0 advances by NIX, so an aligned row base keeps the packed write aligned.
        if (dst_x0 + NIX - 1 < out_width && RRCCheckRowAlign(dstRow))
        {
            RRCWritePack(dstRow[dst_x0], dstPack);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i)
            {
                const int dst_x = dst_x0 + i;
                if (dst_x < out_width)
                {
                    dstRow[dst_x] = dstPack[i];
                }
            }
        }
    }
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void resize_nearest_v1(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                  const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    { //generic copy pixel to pixel
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x) + left, srcSize.x - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y) + top, srcSize.y - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_cubic_v1(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) & (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;

        const float A = -0.75f;

        float cY[4];
        cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        cY[3] = 1.f - cY[0] - cY[1] - cY[2];

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;

        float cX[4];
        cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
        cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
        cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
        cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

        work_type accum = cuda::SetAll<work_type>(0);

        if constexpr (RRC_USE_NIX<T>)
        {
            int csy[4];
            int csx[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                csy[k] = cuda::clamp(sy + k - 1, 0, height - 1);
                csx[k] = cuda::clamp(sx + k - 1, 0, width - 1);
            }

#pragma unroll
            for (int ky = 0; ky < 4; ++ky)
            {
                const T *srcRow = src.ptr(batch_idx, csy[ky], 0);
#pragma unroll
                for (int kx = 0; kx < 4; ++kx)
                {
                    accum += cY[ky] * cX[kx] * srcRow[csx[kx]];
                }
            }
        }
        else
        {
#pragma unroll
            for (int ky = 0; ky < 4; ++ky)
            {
                int csy = cuda::clamp(sy + ky - 1, 0, height - 1);
#pragma unroll
                for (int kx = 0; kx < 4; ++kx)
                {
                    int csx = cuda::clamp(sx + kx - 1, 0, width - 1);
                    accum += cY[ky] * cX[kx] * *src.ptr(batch_idx, csy, csx);
                }
            }
        }

        *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
    }
}

template<typename SrcWrapper, typename DstWrapper>
void resize(const SrcWrapper &src, const DstWrapper &dst, const NVCVInterpolationType interpolation,
            cudaStream_t stream, const int *top, const int *left, const float *scale_x, const float *scale_y,
            int2 srcSize, int2 dstSize, int batchSize)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), batchSize);

    using T = typename DstWrapper::ValueType;

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        constexpr int INIX = RRC_USE_NIX<T> ? RRC_NIX<T> : 1;
        if constexpr (INIX > 1)
        {
            dim3 linearGrid(divUp(dstSize.x, blockSize.x * INIX), divUp(dstSize.y, blockSize.y), batchSize);
            resize_linear_nix<INIX>
                <<<linearGrid, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y);
        }
        else
        {
            resize_linear_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x,
                                                                 scale_y);
        }
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        resize_nearest_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y);
        checkKernelErrors();
    }
    else
    {
        resize_cubic_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_planar(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                     const int *left_, const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

        fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
        sy = cuda::max(0, cuda::min(sy, height - 2));

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

        fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
        sx = cuda::max(0, cuda::min(sx, width - 2));

        for (int plane = 0; plane < channels; ++plane)
        {
            const T *aPtr = src.ptr(batch_idx, plane, sy, 0);
            const T *bPtr = src.ptr(batch_idx, plane, sy + 1, 0);

            *dst.ptr(batch_idx, plane, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
}

template<int NIX, typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_planar_nix(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize,
                                         const int *top_, const int *left_, const float *scale_x_,
                                         const float *scale_y_, int channels)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if (dst_y < out_height)
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

        fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
        sy = cuda::max(0, cuda::min(sy, height - 2));

        const int xBase = blockIdx.x * blockDim.x * NIX + threadIdx.x;

        int   dx[NIX], sxA[NIX];
        float fxA[NIX];
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            dx[i]    = xBase + i * static_cast<int>(blockDim.x);
            float fx = (float)((dx[i] + 0.5f) * scale_x - 0.5f + left);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

            fxA[i] = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
            sxA[i] = cuda::max(0, cuda::min(sx, width - 2));
        }

        for (int plane = 0; plane < channels; ++plane)
        {
            const T *aPtr = src.ptr(batch_idx, plane, sy, 0);
            const T *bPtr = src.ptr(batch_idx, plane, sy + 1, 0);

#pragma unroll
            for (int i = 0; i < NIX; ++i)
            {
                if (dx[i] >= out_width)
                    continue;

                const int   sx = sxA[i];
                const float fx = fxA[i];

                *dst.ptr(batch_idx, plane, dst_y, dx[i])
                    = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                            + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
            }
        }
    }
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void resize_nearest_planar(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                      const int *left_, const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x) + left, srcSize.x - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y) + top, srcSize.y - 1);

        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(batch_idx, plane, dst_y, dst_x) = *src.ptr(batch_idx, plane, sy, sx);
        }
    }
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_cubic_planar(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                    const int *left_, const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) & (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;

        const float A = -0.75f;

        float cY[4];
        cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        cY[3] = 1.f - cY[0] - cY[1] - cY[2];

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;

        float cX[4];
        cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
        cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
        cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
        cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

        int csy[4];
        int csx[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            csy[k] = cuda::clamp(sy + k - 1, 0, height - 1);
            csx[k] = cuda::clamp(sx + k - 1, 0, width - 1);
        }

        for (int plane = 0; plane < channels; ++plane)
        {
            work_type accum = cuda::SetAll<work_type>(0);

#pragma unroll
            for (int ky = 0; ky < 4; ++ky)
            {
#pragma unroll
                for (int kx = 0; kx < 4; ++kx)
                {
                    accum += cY[ky] * cX[kx] * *src.ptr(batch_idx, plane, csy[ky], csx[kx]);
                }
            }

            *dst.ptr(batch_idx, plane, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
        }
    }
}

template<typename T>
ErrorCode resize_planar(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                        const NVCVInterpolationType interpolation, cudaStream_t stream, const int *top, const int *left,
                        const float *scale_x, const float *scale_y)
{
    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 srcSize{inAccess->numCols(), inAccess->numRows()};
    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int  batchSize{static_cast<int>(outAccess->numSamples())};
    const int  channels{inAccess->numChannels()};

    int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto src = cuda::CreateTensorWrapNCHW<T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNCHW<T, int32_t>(outData);

    const int THREADS_PER_BLOCK = 256;
    int       planarWidth       = 32 / static_cast<int>(sizeof(T));
    if (planarWidth < 1)
    {
        planarWidth = 1;
    }

    dim3 blockSize(planarWidth, THREADS_PER_BLOCK / planarWidth, 1);
    dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), batchSize);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        constexpr int PNIX = (sizeof(T) == 1) ? 4 : 1;
        if constexpr (PNIX > 1)
        {
            dim3 linearGrid(divUp(dstSize.x, blockSize.x * PNIX), divUp(dstSize.y, blockSize.y), batchSize);
            resize_linear_planar_nix<PNIX><<<linearGrid, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left,
                                                                                 scale_x, scale_y, channels);
        }
        else
        {
            resize_linear_planar<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x,
                                                                     scale_y, channels);
        }
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        resize_nearest_planar<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x,
                                                                  scale_y, channels);
        checkKernelErrors();
    }
    else
    {
        resize_cubic_planar<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y,
                                                                channels);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    return ErrorCode::SUCCESS;
}

template<typename T>
ErrorCode resize(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                 const NVCVInterpolationType interpolation, cudaStream_t stream, const int *top, const int *left,
                 const float *scale_x, const float *scale_y)
{
    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 srcSize{inAccess->numCols(), inAccess->numRows()};
    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int  batchSize{static_cast<int>(outAccess->numSamples())};

    int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateTensorWrapNHW<T, int32_t>(inData);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        resize(src, dst, interpolation, stream, top, left, scale_x, scale_y, srcSize, dstSize, batchSize);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

RandomResizedCrop::RandomResizedCrop(DataShape max_input_shape, DataShape max_output_shape, const double min_scale,
                                     const double max_scale, const double min_ratio, const double max_ratio,
                                     int32_t maxBatchSize, uint32_t seed)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , min_scale_(min_scale)
    , max_scale_(max_scale)
    , min_ratio_(min_ratio)
    , max_ratio_(max_ratio)
    , m_maxBatchSize(maxBatchSize)
{
    if (min_scale_ > max_scale_ || min_ratio_ > max_ratio_)
    {
        LOG_ERROR("Invalid Parameter: scale and ratio should be of kind (min, max)");
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Parameter error!");
    }
    if (maxBatchSize > 0)
    {
        size_t bufferSize = (sizeof(int) * 2 + sizeof(float) * 2) * maxBatchSize;
        NVCV_CHECK_LOG(cudaMalloc(reinterpret_cast<void **>(&m_gpuCropParams), bufferSize));
        m_cpuCropParams = static_cast<std::byte *>(std::malloc(bufferSize));
        if (!m_cpuCropParams)
        {
            LOG_ERROR("Memory allocation error of size: " << bufferSize);
            throw LegacyCudaAllocationError("Memory allocation error!");
        }
    }
    if (seed == 0)
    {
        std::random_device rand_dev;
        generator_ = std::mt19937(rand_dev());
    }
    else
    {
        generator_ = std::mt19937(seed);
    }
}

RandomResizedCrop::~RandomResizedCrop()
{
    NVCV_CHECK_LOG(cudaFree(m_gpuCropParams));
    std::free(m_cpuCropParams);
}

size_t RandomResizedCrop::calBufferSize(int batch_size)
{
    // buffer size for batch oftop index, left index, scale y, scale x
    return (sizeof(int) * 2 + sizeof(float) * 2) * batch_size;
}

int32_t RandomResizedCrop::maxBatchSize() const noexcept
{
    return m_maxBatchSize;
}

RandomResizedCrop::CropParamBuffers RandomResizedCrop::hostCropParams(int batch) noexcept
{
    return {
        reinterpret_cast<float *>(m_cpuCropParams),
        reinterpret_cast<float *>(m_cpuCropParams + sizeof(float) * batch),
        reinterpret_cast<int *>(m_cpuCropParams + sizeof(float) * 2 * batch),
        reinterpret_cast<int *>(m_cpuCropParams + (sizeof(float) * 2 + sizeof(int)) * batch),
    };
}

RandomResizedCrop::CropParamBuffers RandomResizedCrop::deviceCropParams(int batch) noexcept
{
    return {
        reinterpret_cast<float *>(m_gpuCropParams),
        reinterpret_cast<float *>(m_gpuCropParams + sizeof(float) * batch),
        reinterpret_cast<int *>(m_gpuCropParams + sizeof(float) * 2 * batch),
        reinterpret_cast<int *>(m_gpuCropParams + (sizeof(float) * 2 + sizeof(int)) * batch),
    };
}

std::byte *RandomResizedCrop::hostCropParamStorage() noexcept
{
    return m_cpuCropParams;
}

std::byte *RandomResizedCrop::deviceCropParamStorage() noexcept
{
    return m_gpuCropParams;
}

void RandomResizedCrop::getCropParams(int input_rows, int input_cols, int *top_indices, int *left_indices,
                                      int *crop_rows, int *crop_cols)
{
    int                                    rows          = input_rows;
    int                                    cols          = input_cols;
    double                                 area          = rows * cols;
    const double                           log_min_ratio = std::log(min_ratio_);
    const double                           log_max_ratio = std::log(max_ratio_);
    std::uniform_real_distribution<double> scale_dist(min_scale_, max_scale_);
    std::uniform_real_distribution<double> ratio_dist(log_min_ratio, log_max_ratio);
    bool                                   got_params = false;
    for (int i = 0; i < 10; ++i)
    {
        if (got_params)
            return;
        int    target_area  = area * scale_dist(generator_);
        double aspect_ratio = std::exp(ratio_dist(generator_));

        *crop_cols = int(std::round(std::sqrt(target_area * aspect_ratio)));
        *crop_rows = int(std::round(std::sqrt(target_area / aspect_ratio)));

        if (*crop_cols > 0 && *crop_cols <= cols && *crop_rows > 0 && *crop_rows <= rows)
        {
            std::uniform_int_distribution<int> row_uni(0, rows - *crop_rows);
            std::uniform_int_distribution<int> col_uni(0, cols - *crop_cols);
            *top_indices  = row_uni(generator_);
            *left_indices = col_uni(generator_);
            got_params    = true;
        }
    }
    // Fallback to central crop
    if (!got_params)
    {
        double in_ratio = double(cols) / double(rows);
        if (in_ratio < min_ratio_)
        {
            *crop_cols = cols;
            *crop_rows = int(std::round(*crop_cols / min_ratio_));
        }
        else if (in_ratio > max_ratio_)
        {
            *crop_rows = rows;
            *crop_cols = int(std::round(*crop_rows * max_ratio_));
        }
        else // whole image
        {
            *crop_cols = cols;
            *crop_rows = rows;
        }
        *top_indices  = (rows - *crop_rows) / 2;
        *left_indices = (cols - *crop_cols) / 2;
    }
}

ErrorCode RandomResizedCrop::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

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

    const bool isPlanar = (format == kNCHW || format == kCHW);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  in_data_type  = helpers::GetLegacyDataType(inData.dtype());
    DataType  out_data_type = helpers::GetLegacyDataType(outData.dtype());
    DataShape input_shape   = helpers::GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(in_data_type == kCV_8U || in_data_type == kCV_16U || in_data_type == kCV_16S || in_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << in_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (in_data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << in_data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    int out_cols = outAccess->numCols();
    int out_rows = outAccess->numRows();

    int batch   = inAccess->numSamples();
    int in_cols = inAccess->numCols();
    int in_rows = inAccess->numRows();

    if (maxBatchSize() <= 0 || batch > maxBatchSize())
    {
        LOG_ERROR("Invalid maximum batch size " << maxBatchSize());
        return ErrorCode::INVALID_PARAMETER;
    }

    CropParamBuffers hostParams = hostCropParams(batch);

    for (int i = 0; i < batch; ++i)
    {
        int top, left, crop_rows, crop_cols;
        getCropParams(in_rows, in_cols, &top, &left, &crop_rows, &crop_cols);
        hostParams.scaleX[i] = ((float)crop_cols) / out_cols;
        hostParams.scaleY[i] = ((float)crop_rows) / out_rows;
        hostParams.tops[i]   = top;
        hostParams.lefts[i]  = left;
    }

    CropParamBuffers deviceParams = deviceCropParams(batch);

    size_t buffer_size = calBufferSize(batch);
    checkCudaErrors(
        cudaMemcpyAsync(deviceCropParamStorage(), hostCropParamStorage(), buffer_size, cudaMemcpyHostToDevice, stream));

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const NVCVInterpolationType interpolation, cudaStream_t stream, const int *top,
                                const int *left, const float *scale_x, const float *scale_y);

    static const func_t funcs[6][4] = {
        {      resize<uchar>,  0 /*resize<uchar2>*/,      resize<uchar3>,      resize<uchar4>},
        {0 /*resize<schar>*/,   0 /*resize<char2>*/, 0 /*resize<char3>*/, 0 /*resize<char4>*/},
        {     resize<ushort>, 0 /*resize<ushort2>*/,     resize<ushort3>,     resize<ushort4>},
        {      resize<short>,  0 /*resize<short2>*/,      resize<short3>,      resize<short4>},
        {  0 /*resize<int>*/,    0 /*resize<int2>*/,  0 /*resize<int3>*/,  0 /*resize<int4>*/},
        {      resize<float>,  0 /*resize<float2>*/,      resize<float3>,      resize<float4>}
    };

    if (isPlanar)
    {
        if (static_cast<int64_t>(batch) > 65535)
        {
            LOG_ERROR("Planar random resized crop requires numSamples <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        static const func_t planar_funcs[6] = {
            resize_planar<uchar>, 0 /*resize_planar<schar>*/, resize_planar<ushort>,
            resize_planar<short>, 0 /*resize_planar<int>*/,   resize_planar<float>,
        };

        const func_t planar_func = planar_funcs[in_data_type];
        NVCV_ASSERT(planar_func != 0);
        return planar_func(inData, outData, interpolation, stream, deviceParams.tops, deviceParams.lefts,
                           deviceParams.scaleX, deviceParams.scaleY);
    }

    const func_t func = funcs[in_data_type][channels - 1];
    NVCV_ASSERT(func != 0);
    return func(inData, outData, interpolation, stream, deviceParams.tops, deviceParams.lefts, deviceParams.scaleX,
                deviceParams.scaleY);
}

} // namespace nvcv::legacy::cuda_op
