/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

struct TensorU8ImageAccess
{
    uint8_t *base;
    int64_t  sampleStride;
    int64_t  rowStride;
    int64_t  colStride;
    int64_t  chStride;
    int      rows;
    int      cols;
};

static TensorU8ImageAccess MakeTensorU8Access(const TensorDataAccessStridedImagePlanar &access)
{
    return {reinterpret_cast<uint8_t *>(access.sampleData(0)),
            access.sampleStride(),
            access.rowStride(),
            access.colStride(),
            access.chStride(),
            access.numRows(),
            access.numCols()};
}

__device__ __forceinline__ uint8_t *At(const TensorU8ImageAccess &access, int sample, int channel, int y, int x)
{
    return access.base + sample * access.sampleStride + channel * access.chStride + y * access.rowStride
         + x * access.colStride;
}

} // namespace

template<typename T, typename U, typename D>
__global__ void composite_kernel(const Ptr2dNHWC<T> fg, const Ptr2dNHWC<T> bg, const Ptr2dNHWC<U> fgMask,
                                 Ptr2dNHWC<D> dst)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst.cols || dst_y >= dst.rows)
        return;

    const int batch_idx = get_batch_idx();

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
__global__ void composite_planar_kernel(TensorU8ImageAccess fg, TensorU8ImageAccess bg, TensorU8ImageAccess fgMask,
                                        TensorU8ImageAccess dst)
{
    int dst_x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst.cols || dst_y >= dst.rows)
        return;

    const int batch_idx = get_batch_idx();
    for (int dx = 0; dx < 2; ++dx)
    {
        const int x = dst_x + dx * blockDim.x;
        if (x < dst.cols)
        {
            uint8_t mask_val = __ldg(At(fgMask, batch_idx, 0, dst_y, x));

#pragma unroll
            for (int c = 0; c < 3; ++c)
            {
                const uint8_t c0                 = __ldg(At(bg, batch_idx, c, dst_y, x));
                const uint8_t c1                 = __ldg(At(fg, batch_idx, c, dst_y, x));
                *At(dst, batch_idx, c, dst_y, x) = AlphaLerp(c0, c1, mask_val);
            }
            if constexpr (dcn == 4)
            {
                *At(dst, batch_idx, 3, dst_y, x) = 255;
            }
        }
    }
}

template<typename T, int scn, int dcn> // uchar
void composite(const nvcv::TensorDataAccessStridedImagePlanar &foregroundData,
               const nvcv::TensorDataAccessStridedImagePlanar &backgroundData,
               const nvcv::TensorDataAccessStridedImagePlanar &fgMaskData,
               const nvcv::TensorDataAccessStridedImagePlanar &outData, cudaStream_t stream)
{
    const int batch_size = foregroundData.numSamples();
    const int out_width  = outData.numCols();
    const int out_height = outData.numRows();

    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    typedef typename cuda::MakeType<T, scn> src_type;
    typedef typename cuda::MakeType<T, dcn> dst_type;

    Ptr2dNHWC<src_type> fg_ptr(foregroundData);
    Ptr2dNHWC<src_type> bg_ptr(backgroundData);
    Ptr2dNHWC<T>        fgMask_ptr(fgMaskData);
    Ptr2dNHWC<dst_type> dst_ptr(outData);

    composite_kernel<<<gridSize, blockSize, 0, stream>>>(fg_ptr, bg_ptr, fgMask_ptr, dst_ptr);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<int dcn>
void composite_planar(const nvcv::TensorDataAccessStridedImagePlanar &foregroundData,
                      const nvcv::TensorDataAccessStridedImagePlanar &backgroundData,
                      const nvcv::TensorDataAccessStridedImagePlanar &fgMaskData,
                      const nvcv::TensorDataAccessStridedImagePlanar &outData, cudaStream_t stream)
{
    const int batch_size = foregroundData.numSamples();
    const int out_width  = outData.numCols();
    const int out_height = outData.numRows();

    dim3 blockSize(64, 4, 1);
    dim3 gridSize(divUp(out_width, blockSize.x * 2), divUp(out_height, blockSize.y), batch_size);

    composite_planar_kernel<dcn>
        <<<gridSize, blockSize, 0, stream>>>(MakeTensorU8Access(foregroundData), MakeTensorU8Access(backgroundData),
                                             MakeTensorU8Access(fgMaskData), MakeTensorU8Access(outData));
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

ErrorCode Composite::infer(const TensorDataStridedCuda &foreground, const TensorDataStridedCuda &background,
                           const TensorDataStridedCuda &fgMask, const TensorDataStridedCuda &outData,
                           cudaStream_t stream)
{
    DataFormat background_format = GetLegacyDataFormat(background.layout());
    DataFormat foreground_format = GetLegacyDataFormat(foreground.layout());
    DataFormat fgMask_format     = GetLegacyDataFormat(fgMask.layout());
    DataFormat output_format     = GetLegacyDataFormat(outData.layout());

    if (!((foreground_format == background_format) && (foreground_format == output_format)
          && IsImageLayout(foreground_format) && IsImageLayout(fgMask_format)))
    {
        LOG_ERROR("Invalid DataFormat between foreground ("
                  << foreground_format << "), background (" << background_format << "), foreground mask ("
                  << fgMask_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = foreground_format;

    auto foregroundAccess = TensorDataAccessStridedImagePlanar::Create(foreground);
    NVCV_ASSERT(foregroundAccess);

    auto backgroundAccess = TensorDataAccessStridedImagePlanar::Create(background);
    NVCV_ASSERT(backgroundAccess);

    auto fgMaskAccess = TensorDataAccessStridedImagePlanar::Create(fgMask);
    NVCV_ASSERT(fgMaskAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataType foreground_data_type = GetLegacyDataType(foreground.dtype());
    DataType background_data_type = GetLegacyDataType(background.dtype());
    DataType fgMask_data_type     = GetLegacyDataType(fgMask.dtype());
    DataType output_data_type     = GetLegacyDataType(outData.dtype());

    DataShape foreground_shape = GetLegacyDataShape(foregroundAccess->infoShape());
    DataShape background_shape = GetLegacyDataShape(backgroundAccess->infoShape());
    DataShape fgMask_shape     = GetLegacyDataShape(fgMaskAccess->infoShape());
    DataShape output_shape     = GetLegacyDataShape(outAccess->infoShape());

    if (!((foreground_shape.N == background_shape.N) && (foreground_shape.N == fgMask_shape.N)
          && (foreground_shape.N == output_shape.N) && (foreground_shape.H == background_shape.H)
          && (foreground_shape.H == fgMask_shape.H) && (foreground_shape.H == output_shape.H)
          && (foreground_shape.W == background_shape.W) && (foreground_shape.W == fgMask_shape.W)
          && (foreground_shape.W == output_shape.W)))
    {
        LOG_ERROR("Invalid input/output shape: foreground " << foreground_shape << ", background " << background_shape
                                                            << ", foreground mask " << fgMask_shape << ", output "
                                                            << output_shape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int foreground_channels = foreground_shape.C;
    int background_channels = background_shape.C;
    int fgMask_channels     = fgMask_shape.C;
    int output_channels     = output_shape.C;
    int channels            = output_channels;

    if (!((foreground_channels == 3) && (background_channels == 3) && (fgMask_channels == 1)
          && (output_channels == 3 || output_channels == 4)))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!((foreground_data_type == kCV_8U) && (background_data_type == kCV_8U) && (fgMask_data_type == kCV_8U)
          && (output_data_type == kCV_8U)))
    {
        LOG_ERROR("Invalid DataType " << foreground_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const bool isPlanar = IsPlanar(format);
    if (isPlanar)
    {
        if (foregroundAccess->numSamples() > 65535)
        {
            LOG_ERROR("Planar Composite requires number of images <= 65535");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (output_channels == 3)
        {
            composite_planar<3>(*foregroundAccess, *backgroundAccess, *fgMaskAccess, *outAccess, stream);
        }
        else
        {
            composite_planar<4>(*foregroundAccess, *backgroundAccess, *fgMaskAccess, *outAccess, stream);
        }
        return SUCCESS;
    }

    typedef void (*func_t)(const nvcv::TensorDataAccessStridedImagePlanar &foregroundData,
                           const nvcv::TensorDataAccessStridedImagePlanar &backgroundData,
                           const nvcv::TensorDataAccessStridedImagePlanar &fgMaskData,
                           const nvcv::TensorDataAccessStridedImagePlanar &outData, cudaStream_t stream);

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

    func(*foregroundAccess, *backgroundAccess, *fgMaskAccess, *outAccess, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
