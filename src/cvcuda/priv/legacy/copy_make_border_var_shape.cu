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

#define BLOCK 32

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {
namespace {

template<typename BrdRd, typename T>
__global__ void copyMakeBorderKernel(const BrdRd src, cuda::Tensor3DWrap<T> dst, const cuda::Tensor3DWrap<int> left_,
                                     const cuda::Tensor3DWrap<int> top_, int out_height, int out_width)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int       left    = *left_.ptr(0, 0, batch_idx);
    int       top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, y, x) = src(batch_idx, y_shift, x_shift);
    }
}

template<typename BrdRd, typename T>
__global__ void copyMakeBorderKernel(const BrdRd src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor3DWrap<int> left_,
                                     const cuda::Tensor3DWrap<int> top_)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int left    = *left_.ptr(0, 0, batch_idx);
    const int top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, y, x) = src(batch_idx, y_shift, x_shift);
    }
}

template<template<typename> class B, typename T>
struct copyMakeBorderDispatcher
{
    static void call(const Ptr2dVarShapeNHWC<T> &src, cuda::Tensor3DWrap<T> dst, const T &borderValue,
                     const cuda::Tensor3DWrap<int> &left, const cuda::Tensor3DWrap<int> &top, int max_height,
                     int max_width, cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);
        dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), src.batches);

        B<T>                                     brd(0, 0, borderValue);
        BorderReader<Ptr2dVarShapeNHWC<T>, B<T>> brdSrc(src, brd);

        copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top, max_height, max_width);
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }

    static void call(const Ptr2dVarShapeNHWC<T> &src, Ptr2dVarShapeNHWC<T> dst, const T &borderValue,
                     const cuda::Tensor3DWrap<int> &left, const cuda::Tensor3DWrap<int> &top, int max_height,
                     int max_width, cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);
        dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), src.batches);

        B<T>                                     brd(0, 0, borderValue);
        BorderReader<Ptr2dVarShapeNHWC<T>, B<T>> brdSrc(src, brd);

        copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top);
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }
};

template<typename T, int cn, typename OutType> // uchar3 float3 uchar float
void copyMakeBorder(const IImageBatchVarShapeDataStridedCuda &inData, const OutType &outData,
                    const ITensorDataStridedCuda &top, const ITensorDataStridedCuda &left,
                    const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    typedef cuda::MakeType<T, cn> src_type;
    src_type                      brdVal;
#pragma unroll
    for (int i = 0; i < cn; i++) cuda::GetElement(brdVal, i) = cuda::GetElement(value, i);

    Ptr2dVarShapeNHWC<src_type> srcWrap(inData);
    cuda::Tensor3DWrap<int>     topVec(top);
    cuda::Tensor3DWrap<int>     leftVec(left);

    auto outSize = GetMaxImageSize(outData);

    using out_type = typename std::conditional<std::is_same<OutType, ITensorDataStridedCuda>::value,
                                               cuda::Tensor3DWrap<src_type>, Ptr2dVarShapeNHWC<src_type>>::type;

    out_type dstWrap(outData);

    typedef void (*func_t)(const Ptr2dVarShapeNHWC<src_type> &src, out_type dst, const src_type &borderValue,
                           const cuda::Tensor3DWrap<int> &left, const cuda::Tensor3DWrap<int> &top, int max_height,
                           int max_width, cudaStream_t stream);

    static const func_t funcs[]
        = {copyMakeBorderDispatcher<BrdConstant, src_type>::call,
           copyMakeBorderDispatcher<BrdReplicate, src_type>::call, copyMakeBorderDispatcher<BrdReflect, src_type>::call,
           copyMakeBorderDispatcher<BrdWrap, src_type>::call, copyMakeBorderDispatcher<BrdReflect101, src_type>::call};

    funcs[borderType](srcWrap, dstWrap, brdVal, leftVec, topVec, outSize.h, outSize.w, stream);
}
} // namespace

template<class OutType>
ErrorCode CopyMakeBorderVarShape::inferWarp(const IImageBatchVarShapeDataStridedCuda &data_in, const OutType &data_out,
                                            const ITensorDataStridedCuda &top, const ITensorDataStridedCuda &left,
                                            const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(data_in);
    DataFormat output_format = GetLegacyDataFormat(data_out);
    if (std::is_same<decltype(data_in), decltype(data_out)>::value)
    {
        if (input_format != output_format)
        {
            LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
    }

    auto format = input_format;
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = data_in.uniqueFormat().numChannels();
    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type = GetLegacyDataType(data_in.uniqueFormat());
    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(borderType == NVCV_BORDER_CONSTANT || borderType == NVCV_BORDER_REPLICATE || borderType == NVCV_BORDER_REFLECT
          || borderType == NVCV_BORDER_REFLECT101 || borderType == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderType " << borderType);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType   left_data_type = GetLegacyDataType(left.dtype());
    DataFormat left_format    = GetLegacyDataFormat(left.layout());
    if (left_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Left DataType " << left_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(left_format == kNHWC || left_format == kHWC))
    {
        LOG_ERROR("Invalid Left DataFormat " << left_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType   top_data_type = GetLegacyDataType(top.dtype());
    DataFormat top_format    = GetLegacyDataFormat(top.layout());
    if (top_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Top DataType " << top_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(top_format == kNHWC || top_format == kHWC))
    {
        LOG_ERROR("Invalid Top DataFormat " << top_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &d_in, const OutType &d_out,
                           const ITensorDataStridedCuda &top, const ITensorDataStridedCuda &left,
                           const NVCVBorderType borderType, const float4 value, cudaStream_t stream);

    // clang-format off
    static const func_t funcs[6][4] = {
        {copyMakeBorder<uchar, 1>,        copyMakeBorder<uchar, 2>,        copyMakeBorder<uchar, 3>,        copyMakeBorder<uchar, 4>       },
        {0 /*copyMakeBorder<schar , 1>*/, 0 /*copyMakeBorder<schar , 2>*/, 0 /*copyMakeBorder<schar , 3>*/, 0 /*copyMakeBorder<schar , 4>*/},
        {copyMakeBorder<ushort, 1>,       0 /*copyMakeBorder<ushort, 2>*/, copyMakeBorder<ushort, 3>,       copyMakeBorder<ushort, 4>      },
        {copyMakeBorder<short, 1>,        0 /*copyMakeBorder<short , 2>*/, copyMakeBorder<short, 3>,        copyMakeBorder<short, 4>       },
        {0 /*copyMakeBorder<int   , 1>*/, 0 /*copyMakeBorder<int   , 2>*/, 0 /*copyMakeBorder<int   , 3>*/, 0 /*copyMakeBorder<int   , 4>*/},
        {copyMakeBorder<float, 1>,        0 /*copyMakeBorder<float , 2>*/, copyMakeBorder<float, 3>,        copyMakeBorder<float, 4>       }
    };
    // clang-format on

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(data_in, data_out, top, left, borderType, value, stream);

    return SUCCESS;
}

ErrorCode CopyMakeBorderVarShape::infer(const IImageBatchVarShapeDataStridedCuda &data_in,
                                        const IImageBatchVarShapeDataStridedCuda &data_out,
                                        const ITensorDataStridedCuda &top, const ITensorDataStridedCuda &left,
                                        const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    return inferWarp(data_in, data_out, top, left, borderType, value, stream);
}

ErrorCode CopyMakeBorderVarShape::infer(const IImageBatchVarShapeDataStridedCuda &data_in,
                                        const ITensorDataStridedCuda &data_out, const ITensorDataStridedCuda &top,
                                        const ITensorDataStridedCuda &left, const NVCVBorderType borderType,
                                        const float4 value, cudaStream_t stream)
{
    return inferWarp(data_in, data_out, top, left, borderType, value, stream);
}

} // namespace nvcv::legacy::cuda_op
