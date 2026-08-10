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
#include <random>
#include <type_traits>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

#define BLOCK 32

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_v1(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                 const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) && (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

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
__global__ void resize_linear_nix(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                  const float *scale_x_, const float *scale_y_)
{
    const int dst_x0    = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if (dst_y < dstHeight)
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

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
            if (dst_x < dstWidth)
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
        if (dst_x0 + NIX - 1 < dstWidth && RRCCheckRowAlign(dstRow))
        {
            RRCWritePack(dstRow[dst_x0], dstPack);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i)
            {
                const int dst_x = dst_x0 + i;
                if (dst_x < dstWidth)
                {
                    dstRow[dst_x] = dstPack[i];
                }
            }
        }
    }
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void resize_nearest_v1(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                  const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) && (dst_y < dstHeight))
    { //generic copy pixel to pixel
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x) + left, width - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y) + top, height - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_cubic_v1(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) & (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

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

template<typename T>
void resize(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
            const NVCVInterpolationType interpolation, cudaStream_t stream, float *scale_y, float *scale_x, int *top,
            int *left)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    Size2D outMaxSize = out.maxSize();
    dim3   blockSize(BLOCK, BLOCK / 4, 1);
    dim3   gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    cuda::ImageBatchVarShapeWrap<T> src(in);
    cuda::ImageBatchVarShapeWrap<T> dst(out);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        constexpr int INIX = RRC_USE_NIX<T> ? RRC_NIX<T> : 1;
        if constexpr (INIX > 1)
        {
            dim3 linearGrid(divUp(outMaxSize.w, blockSize.x * INIX), divUp(outMaxSize.h, blockSize.y), in.numImages());
            resize_linear_nix<INIX><<<linearGrid, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y);
        }
        else
        {
            resize_linear_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y);
        }
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        resize_nearest_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y);
        checkKernelErrors();
    }
    else
    {
        resize_cubic_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_planar(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                     const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) && (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

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
__global__ void resize_linear_planar_nix(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                         const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if (dst_y < dstHeight)
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

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
                if (dx[i] >= dstWidth)
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
__global__ void resize_nearest_planar(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                      const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) && (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x) + left, width - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y) + top, height - 1);

        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(batch_idx, plane, dst_y, dst_x) = *src.ptr(batch_idx, plane, sy, sx);
        }
    }
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_cubic_planar(const SrcWrapper src, DstWrapper dst, const int *top_, const int *left_,
                                    const float *scale_x_, const float *scale_y_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) & (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

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
void resize_planar(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                   const NVCVInterpolationType interpolation, cudaStream_t stream, float *scale_y, float *scale_x,
                   int *top, int *left, int channels)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    Size2D outMaxSize = out.maxSize();

    const int THREADS_PER_BLOCK = 256;
    int       planarWidth       = 32 / static_cast<int>(sizeof(T));
    if (planarWidth < 1)
    {
        planarWidth = 1;
    }

    dim3 blockSize(planarWidth, THREADS_PER_BLOCK / planarWidth, 1);
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    cuda::ImageBatchVarShapeWrap<T> src(in);
    cuda::ImageBatchVarShapeWrap<T> dst(out);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        constexpr int PNIX = (sizeof(T) == 1) ? 4 : 1;
        if constexpr (PNIX > 1)
        {
            dim3 linearGrid(divUp(outMaxSize.w, blockSize.x * PNIX), divUp(outMaxSize.h, blockSize.y), in.numImages());
            resize_linear_planar_nix<PNIX>
                <<<linearGrid, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y, channels);
        }
        else
        {
            resize_linear_planar<<<gridSize, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y, channels);
        }
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        resize_nearest_planar<<<gridSize, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y, channels);
        checkKernelErrors();
    }
    else
    {
        resize_cubic_planar<<<gridSize, blockSize, 0, stream>>>(src, dst, top, left, scale_x, scale_y, channels);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

RandomResizedCropVarShape::RandomResizedCropVarShape(DataShape max_input_shape, DataShape max_output_shape,
                                                     const double min_scale, const double max_scale,
                                                     const double min_ratio, const double max_ratio,
                                                     int32_t maxBatchSize, uint32_t seed)
    : RandomResizedCrop(max_input_shape, max_output_shape, min_scale, max_scale, min_ratio, max_ratio, maxBatchSize,
                        seed)
{
}

ErrorCode RandomResizedCropVarShape::infer(const ImageBatchVarShape &in, const ImageBatchVarShape &out,
                                           const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    auto inDataPtr = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inDataPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outDataPtr = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outDataPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    const ImageBatchVarShapeDataStridedCuda &inData  = *inDataPtr;
    const ImageBatchVarShapeDataStridedCuda &outData = *outDataPtr;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in input batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in output batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (maxBatchSize() <= 0 || inData.numImages() > maxBatchSize())
    {
        LOG_ERROR("Invalid maximum batch size " << maxBatchSize());
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

    const bool isPlanar = (format == kNCHW || format == kCHW);

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType in_data_type  = helpers::GetLegacyDataType(inData.uniqueFormat());
    DataType out_data_type = helpers::GetLegacyDataType(outData.uniqueFormat());

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

    int batch = inData.numImages();

    CropParamBuffers hostParams = hostCropParams(batch);

    for (int i = 0; i < batch; ++i)
    {
        if (channels != in[i].format().numChannels())
        {
            LOG_ERROR("Invalid Input");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        int top, left, crop_rows, crop_cols;
        getCropParams(in[i].size().h, in[i].size().w, &top, &left, &crop_rows, &crop_cols);
        hostParams.scaleX[i] = ((float)crop_cols) / out[i].size().w;
        hostParams.scaleY[i] = ((float)crop_rows) / out[i].size().h;
        hostParams.tops[i]   = top;
        hostParams.lefts[i]  = left;
    }

    CropParamBuffers deviceParams = deviceCropParams(batch);

    size_t buffer_size = calBufferSize(batch);
    checkCudaErrors(
        cudaMemcpyAsync(deviceCropParamStorage(), hostCropParamStorage(), buffer_size, cudaMemcpyHostToDevice, stream));

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                           const NVCVInterpolationType interpolation, cudaStream_t stream, float *scale_y,
                           float *scale_x, int *top, int *left);

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
            LOG_ERROR("Planar random resized crop requires numImages <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        typedef void (*planar_func_t)(const ImageBatchVarShapeDataStridedCuda &in,
                                      const ImageBatchVarShapeDataStridedCuda &out,
                                      const NVCVInterpolationType interpolation, cudaStream_t stream, float *scale_y,
                                      float *scale_x, int *top, int *left, int channels);

        static const planar_func_t planar_funcs[6] = {
            resize_planar<uchar>, 0 /*resize_planar<schar>*/, resize_planar<ushort>,
            resize_planar<short>, 0 /*resize_planar<int>*/,   resize_planar<float>,
        };

        const planar_func_t planar_func = planar_funcs[in_data_type];
        NVCV_ASSERT(planar_func != 0);
        planar_func(inData, outData, interpolation, stream, deviceParams.scaleY, deviceParams.scaleX, deviceParams.tops,
                    deviceParams.lefts, channels);
        return SUCCESS;
    }

    const func_t func = funcs[in_data_type][channels - 1];
    NVCV_ASSERT(func != 0);
    func(inData, outData, interpolation, stream, deviceParams.scaleY, deviceParams.scaleX, deviceParams.tops,
         deviceParams.lefts);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
