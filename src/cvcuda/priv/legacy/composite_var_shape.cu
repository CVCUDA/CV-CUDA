/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using namespace nvcv;
using namespace nvcv::legacy::helpers;
using namespace nvcv::legacy::cuda_op;

namespace {

__device__ __forceinline__ int AlphaLerp(uint8_t c0, uint8_t c1, uint8_t alpha)
{
    int value = (int)c0 * 255 + ((int)c1 - (int)c0) * (int)alpha + 128;
    // This is exact rounded division by 255 over the full uint8 input range.
    return (value + (value >> 8)) >> 8;
}

static bool IsInterleaved(DataFormat format)
{
    return format == kNHWC || format == kHWC;
}

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

static bool IsImageLayout(DataFormat format)
{
    return IsInterleaved(format) || IsPlanar(format);
}

} // namespace

template<typename T, typename U, typename D>
__global__ void composite_kernel(const cuda::ImageBatchVarShapeWrap<T> fg, const cuda::ImageBatchVarShapeWrap<T> bg,
                                 const cuda::ImageBatchVarShapeWrap<U> fgMask, cuda::ImageBatchVarShapeWrap<D> dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    // Compile-time channels let each instantiation fold the loop and avoid wrapper metadata loads.
    constexpr int dst_ch = cuda::NumElements<D>;
    constexpr int src_ch = cuda::NumElements<T>;

    U mask_val = *fgMask.ptr(batch_idx, dst_y, dst_x);
    T bg_val   = *bg.ptr(batch_idx, dst_y, dst_x);
    T fg_val   = *fg.ptr(batch_idx, dst_y, dst_x);
    D out;

    for (int i = 0; i < src_ch; i++)
    {
        uint8_t c0               = cuda::GetElement(bg_val, i);
        uint8_t c1               = cuda::GetElement(fg_val, i);
        cuda::GetElement(out, i) = AlphaLerp(c0, c1, mask_val);
    }
    if (src_ch == 3 && dst_ch == 4)
        cuda::GetElement(out, 3) = 255;
    *dst.ptr(batch_idx, dst_y, dst_x) = out;
}

template<int dcn>
__global__ void composite_planar_kernel(const cuda::ImageBatchVarShapeWrap<uchar> fg,
                                        const cuda::ImageBatchVarShapeWrap<uchar> bg,
                                        const cuda::ImageBatchVarShapeWrap<uchar> fgMask,
                                        cuda::ImageBatchVarShapeWrap<uchar>       dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    uchar mask_val = *fgMask.ptr(batch_idx, 0, dst_y, dst_x);

#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        const uchar c0                       = *bg.ptr(batch_idx, c, dst_y, dst_x);
        const uchar c1                       = *fg.ptr(batch_idx, c, dst_y, dst_x);
        *dst.ptr(batch_idx, c, dst_y, dst_x) = AlphaLerp(c0, c1, mask_val);
    }
    if constexpr (dcn == 4)
    {
        *dst.ptr(batch_idx, 3, dst_y, dst_x) = 255;
    }
}

template<typename T, int scn, int dcn> // uchar
void composite(const nvcv::ImageBatchVarShapeDataStridedCuda &foregroundData,
               const nvcv::ImageBatchVarShapeDataStridedCuda &backgroundData,
               const nvcv::ImageBatchVarShapeDataStridedCuda &fgMaskData,
               const nvcv::ImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    typedef typename cuda::MakeType<T, scn> src_type;
    typedef typename cuda::MakeType<T, dcn> dst_type;

    cuda::ImageBatchVarShapeWrap<src_type> fg_ptr(foregroundData);
    cuda::ImageBatchVarShapeWrap<src_type> bg_ptr(backgroundData);
    cuda::ImageBatchVarShapeWrap<T>        fgMask_ptr(fgMaskData);
    cuda::ImageBatchVarShapeWrap<dst_type> dst_ptr(outData);

    const int batch_size = outData.numImages();
    Size2D    outMaxSize = outData.maxSize();

    dim3 blockSize(dcn == 3 ? 32 : 16, dcn == 3 ? 8 : 16, 1);
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), batch_size);

    composite_kernel<<<gridSize, blockSize, 0, stream>>>(fg_ptr, bg_ptr, fgMask_ptr, dst_ptr);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<int dcn>
void composite_planar(const nvcv::ImageBatchVarShapeDataStridedCuda &foregroundData,
                      const nvcv::ImageBatchVarShapeDataStridedCuda &backgroundData,
                      const nvcv::ImageBatchVarShapeDataStridedCuda &fgMaskData,
                      const nvcv::ImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    cuda::ImageBatchVarShapeWrap<uchar> fg_ptr(foregroundData);
    cuda::ImageBatchVarShapeWrap<uchar> bg_ptr(backgroundData);
    cuda::ImageBatchVarShapeWrap<uchar> fgMask_ptr(fgMaskData);
    cuda::ImageBatchVarShapeWrap<uchar> dst_ptr(outData);

    const int batch_size = outData.numImages();
    Size2D    outMaxSize = outData.maxSize();

    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), batch_size);

    composite_planar_kernel<dcn><<<gridSize, blockSize, 0, stream>>>(fg_ptr, bg_ptr, fgMask_ptr, dst_ptr);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

ErrorCode CompositeVarShape::infer(const ImageBatchVarShapeDataStridedCuda &foreground,
                                   const ImageBatchVarShapeDataStridedCuda &background,
                                   const ImageBatchVarShapeDataStridedCuda &fgMask,
                                   const ImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream)
{
    if (!(foreground.uniqueFormat() && background.uniqueFormat() && fgMask.uniqueFormat() && outData.uniqueFormat()))
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat background_format = helpers::GetLegacyDataFormat(background);
    DataFormat foreground_format = helpers::GetLegacyDataFormat(foreground);
    DataFormat fgMask_format     = helpers::GetLegacyDataFormat(fgMask);
    DataFormat output_format     = helpers::GetLegacyDataFormat(outData);

    if (!((foreground_format == background_format) && (foreground_format == output_format)
          && IsImageLayout(foreground_format) && IsImageLayout(fgMask_format)))
    {
        LOG_ERROR("Invalid DataFormat between foreground ("
                  << foreground_format << "), background (" << background_format << "), foreground mask ("
                  << fgMask_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = foreground_format;

    DataType foreground_data_type = helpers::GetLegacyDataType(foreground.uniqueFormat());
    DataType background_data_type = helpers::GetLegacyDataType(background.uniqueFormat());
    DataType fgMask_data_type     = helpers::GetLegacyDataType(fgMask.uniqueFormat());
    DataType output_data_type     = helpers::GetLegacyDataType(outData.uniqueFormat());

    if (!((foreground_data_type == kCV_8U) && (background_data_type == kCV_8U) && (fgMask_data_type == kCV_8U)
          && (output_data_type == kCV_8U)))
    {
        LOG_ERROR("Invalid DataType " << foreground_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int foreground_channels = foreground.uniqueFormat().numChannels();
    const int background_channels = background.uniqueFormat().numChannels();
    const int fgMask_channels     = fgMask.uniqueFormat().numChannels();
    const int output_channels     = outData.uniqueFormat().numChannels();
    const int channels            = output_channels;

    if (!((foreground_channels == 3) && (background_channels == 3) && (fgMask_channels == 1)
          && (output_channels == 3 || output_channels == 4)))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!((foreground.numImages() == background.numImages()) && (foreground.numImages() == fgMask.numImages())
          && (foreground.numImages() == outData.numImages())))
    {
        LOG_ERROR("Invalid input/output batch size: foreground "
                  << foreground.numImages() << ", background " << background.numImages() << ", foreground mask "
                  << fgMask.numImages() << ", output " << outData.numImages());
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const bool isPlanar = IsPlanar(format);
    if (isPlanar)
    {
        if (outData.numImages() > 65535)
        {
            LOG_ERROR("Planar Composite requires number of images <= 65535");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (output_channels == 3)
        {
            composite_planar<3>(foreground, background, fgMask, outData, stream);
        }
        else
        {
            composite_planar<4>(foreground, background, fgMask, outData, stream);
        }
        return SUCCESS;
    }

    typedef void (*func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &foregroundData,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &backgroundData,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &fgMaskData,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &outData, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        { 0 /*composite<uchar,1,1>*/,  0 /*composite<uchar,2,2>*/,             composite<uchar,3, 3>, composite<uchar, 3, 4>},
        { 0 /*composite<schar,1,1>*/,  0 /*composite<schar,2,2>*/,  0 /*composite<schar,3,3>*/,
         0 /*composite<schar,3,4>*/   },
        {0 /*composite<ushort,1,1>*/, 0 /*composite<ushort,2,2>*/, 0 /*composite<ushort,3,3>*/,
         0 /*composite<ushort,3,4>*/   },
        { 0 /*composite<short,1,1>*/,  0 /*composite<short,2,2>*/,  0 /*composite<short,3,3>*/,
         0 /*composite<short,3,4>*/   },
        {   0 /*composite<int,1,1>*/,    0 /*composite<int,2,2>*/,    0 /*composite<int,3,3>*/, 0 /*composite<int,3,4>*/   },
        { 0 /*composite<float,1,1>*/,  0 /*composite<float,2,2>*/,  0 /*composite<float,3,3>*/,
         0 /*composite<float,3,4>*/   },
    };

    const func_t func = funcs[foreground_data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(foreground, background, fgMask, outData, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
