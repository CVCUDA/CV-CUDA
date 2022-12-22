/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/SaturateCast.hpp>

#define BLOCK 32

#define USE_OCV_CPU_ALIGN_VERSION

namespace nvcv::legacy::cuda_op {

namespace {

template<typename T>
__global__ void resize_linear_ocv_align(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int dstWidth  = dst.at_cols(batch_idx);
    int dstHeight = dst.at_rows(batch_idx);

    if (dst_x >= dstWidth || dst_y >= dstHeight)
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type out   = {0};

    float fy = (float)((dst_y + 0.5) * scale_y - 0.5);
    int   sy = __float2int_rd(fy);
    fy -= sy;
    sy = min(sy, height - 2);
    sy = max(0, sy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    float fx = (float)((dst_x + 0.5) * scale_x - 0.5);
    int   sx = __float2int_rd(fx);
    fx -= sx;

    if (sx < 0)
    {
        fx = 0, sx = 0;
    }
    if (sx >= width - 1)
    {
        fx = 0, sx = width - 2;
    }

    float cbufx[2];
    cbufx[0] = 1.f - fx;
    cbufx[1] = fx;

    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<cuda::BaseType<T>>(
        (*src.ptr(batch_idx, sy, sx) * cbufx[0] * cbufy[0] + *src.ptr(batch_idx, sy + 1, sx) * cbufx[0] * cbufy[1]
         + *src.ptr(batch_idx, sy, sx + 1) * cbufx[1] * cbufy[0]
         + *src.ptr(batch_idx, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
}

template<typename T>
__global__ void resize_linear_v2(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst,
                                 const float *scale_xs, const float *scale_ys)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int dstWidth  = dst.at_cols(batch_idx);
    int dstHeight = dst.at_rows(batch_idx);

    if (dst_x >= dstWidth || dst_y >= dstHeight)
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    float src_x = dst_x * scale_x;
    float src_y = dst_y * scale_y;

    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type out   = {0};

    const int x1      = __float2int_rd(src_x);
    const int y1      = __float2int_rd(src_y);
    const int x2      = x1 + 1;
    const int y2      = y1 + 1;
    const int x2_read = min(x2, width - 1);
    const int y2_read = min(y2, height - 1);

    T src_reg = *src.ptr(batch_idx, y1, x1);
    out       = out + src_reg * ((x2 - src_x) * (y2 - src_y));

    src_reg = *src.ptr(batch_idx, y1, x2_read);
    out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

    src_reg = *src.ptr(batch_idx, y2_read, x1);
    out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

    src_reg = *src.ptr(batch_idx, y2_read, x2_read);
    out     = out + src_reg * ((src_x - x1) * (src_y - y1));

    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(out);
}

template<typename T>
__global__ void resize_nearest_ocv_align(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       dstWidth  = dst.at_cols(batch_idx);
    int       dstHeight = dst.at_rows(batch_idx);

    if (dst_x >= dstWidth || dst_y >= dstHeight)
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    int sx = __float2int_rd(dst_x * scale_x);
    sx     = min(sx, src.at_cols(batch_idx) - 1);

    int sy                            = __float2int_rd(dst_y * scale_y);
    sy                                = min(sy, src.at_rows(batch_idx) - 1);
    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
}

template<typename T>
__global__ void resize_nearest_v2(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst,
                                  const float *scale_xs, const float *scale_ys)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       dstWidth  = dst.at_cols(batch_idx);
    int       dstHeight = dst.at_rows(batch_idx);

    if (dst_x >= dstWidth || dst_y >= dstHeight)
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    float src_x = dst_x * scale_x;
    float src_y = dst_y * scale_y;

    const int x1 = __float2int_rz(src_x);
    const int y1 = __float2int_rz(src_y);

    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, y1, x1);
}

template<typename T>
__global__ void resize_cubic_ocv_align(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst)
{
    int       x         = blockIdx.x * blockDim.x + threadIdx.x;
    int       y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int dstWidth  = dst.at_cols(batch_idx);
    int dstHeight = dst.at_rows(batch_idx);

    if (x >= dstWidth || y >= dstHeight)
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    int iscale_x = cuda::SaturateCast<int>(scale_x);
    int iscale_y = cuda::SaturateCast<int>(scale_y);

    float fy = (float)((y + 0.5) * scale_y - 0.5);
    int   sy = __float2int_rd(fy);
    fy -= sy;
    sy = min(sy, src.at_rows(batch_idx) - 3);
    sy = max(1, sy);

    const float A = -0.75f;

    float coeffsY[4];
    coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
    coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
    coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
    coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

    float fx = (float)((x + 0.5) * scale_x - 0.5);
    int   sx = __float2int_rd(fx);
    fx -= sx;

    if (sx < 1)
    {
        fx = 0, sx = 1;
    }
    if (sx >= src.at_cols(batch_idx) - 3)
    {
        fx = 0, sx = src.at_cols(batch_idx) - 3;
    }

    float coeffsX[4];
    coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
    coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
    coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
    coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

    if (sx < 1)
    {
        sx = 1;
    }
    if (sx > src.at_cols(batch_idx) - 3)
    {
        sx = src.at_cols(batch_idx) - 3;
    }
    if (sy < 1)
    {
        sy = 1;
    }
    if (sy > src.at_rows(batch_idx) - 3)
    {
        sy = src.at_rows(batch_idx) - 3;
    }

    using cuda::abs;

    *dst.ptr(batch_idx, y, x)
        = cuda::SaturateCast<cuda::BaseType<T>>(abs(*src.ptr(batch_idx, sy - 1, sx - 1) * coeffsX[0] * coeffsY[0]
                                                    + *src.ptr(batch_idx, sy, sx - 1) * coeffsX[0] * coeffsY[1]
                                                    + *src.ptr(batch_idx, sy + 1, sx - 1) * coeffsX[0] * coeffsY[2]
                                                    + *src.ptr(batch_idx, sy + 2, sx - 1) * coeffsX[0] * coeffsY[3]
                                                    + *src.ptr(batch_idx, sy - 1, sx) * coeffsX[1] * coeffsY[0]
                                                    + *src.ptr(batch_idx, sy, sx) * coeffsX[1] * coeffsY[1]
                                                    + *src.ptr(batch_idx, sy + 1, sx) * coeffsX[1] * coeffsY[2]
                                                    + *src.ptr(batch_idx, sy + 2, sx) * coeffsX[1] * coeffsY[3]
                                                    + *src.ptr(batch_idx, sy - 1, sx + 1) * coeffsX[2] * coeffsY[0]
                                                    + *src.ptr(batch_idx, sy, sx + 1) * coeffsX[2] * coeffsY[1]
                                                    + *src.ptr(batch_idx, sy + 1, sx + 1) * coeffsX[2] * coeffsY[2]
                                                    + *src.ptr(batch_idx, sy + 2, sx + 1) * coeffsX[2] * coeffsY[3]
                                                    + *src.ptr(batch_idx, sy - 1, sx + 2) * coeffsX[3] * coeffsY[0]
                                                    + *src.ptr(batch_idx, sy, sx + 2) * coeffsX[3] * coeffsY[1]
                                                    + *src.ptr(batch_idx, sy + 1, sx + 2) * coeffsX[3] * coeffsY[2]
                                                    + *src.ptr(batch_idx, sy + 2, sx + 2) * coeffsX[3] * coeffsY[3]));
}

template<typename T>
__global__ void resize_cubic_v2(
    cuda_op::CubicFilter<cuda_op::BorderReader<cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdReplicate<T>>> filteredSrc,
    cuda_op::Ptr2dVarShapeNHWC<T>                                                                        dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int dstWidth  = dst.at_cols(batch_idx);
    int dstHeight = dst.at_rows(batch_idx);

    if (dst_x >= dstWidth || dst_y >= dstHeight)
        return;
    int height = filteredSrc.at_rows(batch_idx), width = filteredSrc.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    float src_x = dst_x * scale_x;
    float src_y = dst_y * scale_y;

    *dst.ptr(batch_idx, dst_y, dst_x) = filteredSrc(batch_idx, src_y, src_x);
}

template<typename T, typename BorderReader>
__global__ void resize_area_ocv_align(const cuda_op::Ptr2dVarShapeNHWC<T> src, const BorderReader brd_src,
                                      cuda_op::Ptr2dVarShapeNHWC<T> dst)
{
    const int x         = blockDim.x * blockIdx.x + threadIdx.x;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int dstWidth  = dst.at_cols(batch_idx);
    int dstHeight = dst.at_rows(batch_idx);

    if (x >= dstWidth || y >= dstHeight)
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    double inv_scale_x  = 1. / scale_x;
    double inv_scale_y  = 1. / scale_y;
    int    iscale_x     = cuda::SaturateCast<int>(scale_x);
    int    iscale_y     = cuda::SaturateCast<int>(scale_y);
    bool   is_area_fast = abs(scale_x - iscale_x) < DBL_EPSILON && abs(scale_y - iscale_y) < DBL_EPSILON;

    if (scale_x >= 1.0f && scale_y >= 1.0f) // zoom out
    {
        if (is_area_fast) // integer multiples
        {
            float scale = 1.f / (scale_x * scale_y);
            float fsx1  = x * scale_x;
            float fsx2  = fsx1 + scale_x;

            int sx1 = __float2int_ru(fsx1);
            int sx2 = __float2int_rd(fsx2);

            float fsy1 = y * scale_y;
            float fsy2 = fsy1 + scale_y;

            int sy1 = __float2int_ru(fsy1);
            int sy2 = __float2int_rd(fsy2);

            using work_type = cuda::ConvertBaseTypeTo<float, T>;
            work_type out   = {0};

            for (int dy = sy1; dy < sy2; ++dy)
            {
                for (int dx = sx1; dx < sx2; ++dx)
                {
                    out = out + brd_src(batch_idx, dy, dx) * scale;
                }
            }
            *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<T>>(out);
            return;
        }

        float fsx1 = x * scale_x;
        float fsx2 = fsx1 + scale_x;

        int sx1 = __float2int_ru(fsx1);
        int sx2 = __float2int_rd(fsx2);

        float fsy1 = y * scale_y;
        float fsy2 = fsy1 + scale_y;

        int sy1 = __float2int_ru(fsy1);
        int sy2 = __float2int_rd(fsy2);

        float scale
            = 1.f / (fminf(scale_x, src.at_cols(batch_idx) - fsx1) * fminf(scale_y, src.at_rows(batch_idx) - fsy1));

        using work_type = cuda::ConvertBaseTypeTo<float, T>;
        work_type out   = {0};

        for (int dy = sy1; dy < sy2; ++dy)
        {
            for (int dx = sx1; dx < sx2; ++dx) out = out + brd_src(batch_idx, dy, dx) * scale;

            if (sx1 > fsx1)
                out = out + brd_src(batch_idx, dy, (sx1 - 1)) * ((sx1 - fsx1) * scale);

            if (sx2 < fsx2)
                out = out + brd_src(batch_idx, dy, sx2) * ((fsx2 - sx2) * scale);
        }

        if (sy1 > fsy1)
            for (int dx = sx1; dx < sx2; ++dx) out = out + brd_src(batch_idx, (sy1 - 1), dx) * ((sy1 - fsy1) * scale);

        if (sy2 < fsy2)
            for (int dx = sx1; dx < sx2; ++dx) out = out + brd_src(batch_idx, sy2, dx) * ((fsy2 - sy2) * scale);

        if ((sy1 > fsy1) && (sx1 > fsx1))
            out = out + brd_src(batch_idx, (sy1 - 1), (sx1 - 1)) * ((sy1 - fsy1) * (sx1 - fsx1) * scale);

        if ((sy1 > fsy1) && (sx2 < fsx2))
            out = out + brd_src(batch_idx, (sy1 - 1), sx2) * ((sy1 - fsy1) * (fsx2 - sx2) * scale);

        if ((sy2 < fsy2) && (sx2 < fsx2))
            out = out + brd_src(batch_idx, sy2, sx2) * ((fsy2 - sy2) * (fsx2 - sx2) * scale);

        if ((sy2 < fsy2) && (sx1 > fsx1))
            out = out + brd_src(batch_idx, sy2, (sx1 - 1)) * ((fsy2 - sy2) * (sx1 - fsx1) * scale);

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<T>>(out);
        return;
    }

    // zoom in, it is emulated using some variant of bilinear interpolation
    int   sy = __float2int_rd(y * scale_y);
    float fy = (float)((y + 1) - (sy + 1) * inv_scale_y);
    fy       = fy <= 0 ? 0.f : fy - __float2int_rd(fy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    int   sx = __float2int_rd(x * scale_x);
    float fx = (float)((x + 1) - (sx + 1) * inv_scale_x);
    fx       = fx < 0 ? 0.f : fx - __float2int_rd(fx);

    if (sx < 0)
    {
        fx = 0, sx = 0;
    }

    if (sx >= src.at_cols(batch_idx) - 1)
    {
        fx = 0, sx = src.at_cols(batch_idx) - 2;
    }
    if (sy >= src.at_rows(batch_idx) - 1)
    {
        sy = src.at_rows(batch_idx) - 2;
    }

    float cbufx[2];
    cbufx[0] = 1.f - fx;
    cbufx[1] = fx;

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<T>>(
        (*src.ptr(batch_idx, sy, sx) * cbufx[0] * cbufy[0] + *src.ptr(batch_idx, sy + 1, sx) * cbufx[0] * cbufy[1]
         + *src.ptr(batch_idx, sy, sx + 1) * cbufx[1] * cbufy[0]
         + *src.ptr(batch_idx, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
}

template<class Filter, typename T>
__global__ void resize_area_v2(const Filter src, cuda_op::Ptr2dVarShapeNHWC<T> dst)
{
    int       dst_x     = blockDim.x * blockIdx.x + threadIdx.x;
    int       dst_y     = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.cols[batch_idx] || dst_y >= dst.rows[batch_idx])
        return;

    *dst.ptr(batch_idx, dst_y, dst_x) = src(batch_idx, dst_y, dst_x);
}

template<typename T>
void resize(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
            const int interpolation, cudaStream_t stream)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);

    Size2D outMaxSize = out.maxSize();

    NVCV_ASSERT(in.numImages() == out.numImages());

    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(in);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(out);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        resize_linear_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
#else
        resize_linear_v2<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
#endif
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        resize_nearest_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
#else
        resize_nearest_v2<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
#endif
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_CUBIC) // NVCV_INTERP_CUBIC
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        resize_cubic_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        checkKernelErrors();
#else
        cuda_op::BrdReplicate<T>                                                       brd(0, 0);
        cuda_op::BorderReader<cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdReplicate<T>> brdSrc(src_ptr, brd);
        cuda_op::CubicFilter<cuda_op::BorderReader<cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdReplicate<T>>>
            filteredSrc(brdSrc);

        resize_cubic_v2<T><<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr);
        checkKernelErrors();
#endif
    }
    else if (interpolation == NVCV_INTERP_AREA)
    {
        cuda_op::BrdConstant<T>                                                       brd(0, 0);
        cuda_op::BorderReader<cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdConstant<T>> brdSrc(src_ptr, brd);
        resize_area_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, brdSrc, dst_ptr);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

} // namespace

ErrorCode ResizeVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                const IImageBatchVarShapeDataStridedCuda &outData,
                                const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in input batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC || interpolation == NVCV_INTERP_AREA))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
                           const int interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      resize<uchar>,  0 /*resize<uchar2>*/,      resize<uchar3>,      resize<uchar4>},
        {0 /*resize<schar>*/,   0 /*resize<char2>*/, 0 /*resize<char3>*/, 0 /*resize<char4>*/},
        {     resize<ushort>, 0 /*resize<ushort2>*/,     resize<ushort3>,     resize<ushort4>},
        {      resize<short>,  0 /*resize<short2>*/,      resize<short3>,      resize<short4>},
        {  0 /*resize<int>*/,    0 /*resize<int2>*/,  0 /*resize<int3>*/,  0 /*resize<int4>*/},
        {      resize<float>,  0 /*resize<float2>*/,      resize<float3>,      resize<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];

    assert(func != 0);
    func(inData, outData, interpolation, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
