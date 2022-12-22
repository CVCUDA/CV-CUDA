/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

#define BLOCK 32

template<typename BrdRd, typename Ptr2D>
__global__ void copyMakeBorderKernel(const BrdRd src, Ptr2D dst, const int left, const int top)
{
    const int x          = blockIdx.x * blockDim.x + threadIdx.x;
    const int y          = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_shift    = x - left;
    const int y_shift    = y - top;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, y, x) = src(batch_idx, y_shift, x_shift);
    }
}

template<template<typename> class B, typename T>
struct copyMakeBorderDispatcher
{
    static void call(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const T &borderValue, const int left, const int top,
                     cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);
        dim3 gridSize(divUp(dst.cols, blockSize.x), divUp(dst.rows, blockSize.y), dst.batches);

        B<T>                             brd(src.rows, src.cols, borderValue);
        BorderReader<Ptr2dNHWC<T>, B<T>> brdSrc(src, brd);

        copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top);
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }
};

template<typename T, int cn> // uchar3 float3 uchar float
void copyMakeBorder(const TensorDataAccessStridedImagePlanar &d_in, const TensorDataAccessStridedImagePlanar &d_out,
                    const int batch_size, const int height, const int width, const int top, const int left,
                    const NVCVBorderType border_type, const float4 value, cudaStream_t stream)
{
    typedef typename cuda::MakeType<T, cn> src_type;

    src_type brdVal;
#pragma unroll
    for (int i = 0; i < cn; i++) cuda::GetElement(brdVal, i) = cuda::GetElement(value, i);

    Ptr2dNHWC<src_type> src_ptr(d_in);
    Ptr2dNHWC<src_type> dst_ptr(d_out);

    typedef void (*func_t)(const Ptr2dNHWC<src_type> src, Ptr2dNHWC<src_type> dst, const src_type &borderValue,
                           const int left, const int top, cudaStream_t stream);

    static const func_t funcs[]
        = {copyMakeBorderDispatcher<BrdConstant, src_type>::call,
           copyMakeBorderDispatcher<BrdReplicate, src_type>::call, copyMakeBorderDispatcher<BrdReflect, src_type>::call,
           copyMakeBorderDispatcher<BrdWrap, src_type>::call, copyMakeBorderDispatcher<BrdReflect101, src_type>::call};

    funcs[border_type](src_ptr, dst_ptr, brdVal, left, top, stream);
}

namespace nvcv::legacy::cuda_op {

ErrorCode CopyMakeBorder::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                                const int top, const int left, const NVCVBorderType border_type, const float4 value,
                                cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    int    batch     = input_shape.N;
    int    channels  = input_shape.C;
    int    rows      = input_shape.H;
    int    cols      = input_shape.W;
    size_t data_size = DataSize(data_type);

    cuda_op::DataShape output_shape = GetLegacyDataShape(outAccess->infoShape());
    int                out_rows     = output_shape.H;
    int                out_cols     = output_shape.W;

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(border_type == NVCVBorderType::NVCV_BORDER_CONSTANT || border_type == NVCVBorderType::NVCV_BORDER_REPLICATE
          || border_type == NVCVBorderType::NVCV_BORDER_REFLECT || border_type == NVCVBorderType::NVCV_BORDER_REFLECT101
          || border_type == NVCVBorderType::NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderType " << border_type);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(top >= 0 && out_rows >= top + rows && left >= 0 && out_cols >= left + cols))
    {
        LOG_ERROR("Invalid border " << top << " " << out_rows - top - rows << " " << left << " "
                                    << out_cols - left - cols
                                    << ", top >= 0 && bottom >= 0 && left >= 0 && right >= 0, in resolution: " << rows
                                    << "x" << cols << ", out resolution: " << out_rows << "x" << out_cols);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const TensorDataAccessStridedImagePlanar &d_in,
                           const TensorDataAccessStridedImagePlanar &d_out, const int batch_size, const int rows,
                           const int cols, const int top, const int left, const NVCVBorderType border_type,
                           const float4 value, cudaStream_t stream);

    // clang-format off
    static const func_t funcs[6][4] = {
        {copyMakeBorder<uchar, 1>      , copyMakeBorder<uchar, 2>       , copyMakeBorder<uchar, 3>      , copyMakeBorder<uchar, 4>      },
        {0 /*copyMakeBorder<schar, 1>*/, 0 /*copyMakeBorder<schar, 2>*/ , 0 /*copyMakeBorder<schar, 3>*/, 0 /*copyMakeBorder<schar, 4>*/},
        {copyMakeBorder<ushort, 1>     , 0 /*copyMakeBorder<ushort, 2>*/, copyMakeBorder<ushort, 3>     , copyMakeBorder<ushort, 4>     },
        {copyMakeBorder<short, 1>      , 0 /*copyMakeBorder<short , 2>*/, copyMakeBorder<short, 3>      , copyMakeBorder<short, 4>      },
        {0 /*copyMakeBorder<int, 1>*/  , 0 /*copyMakeBorder<int, 2>*/   , 0 /*copyMakeBorder<int, 3>*/  , 0 /*copyMakeBorder<int, 4>*/  },
        {copyMakeBorder<float, 1>      , 0 /*copyMakeBorder<float , 2>*/, copyMakeBorder<float, 3>      , copyMakeBorder<float, 4>      }
    };
    // clang-format on

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(*inAccess, *outAccess, batch, rows, cols, top, left, border_type, value, stream);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
