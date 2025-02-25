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

#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>

#include <cassert>
#include <cstdio>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

template<typename SRC_TYPE, typename DST_TYPE, typename S>
struct Convertor
{
    S alpha;
    S beta;

    __device__ __forceinline__ DST_TYPE operator()(SRC_TYPE src) const
    {
        return nvcv::cuda::SaturateCast<DST_TYPE>(alpha * src + beta);
    }
};

template<class SrcWrapper, class DstWrapper, class UnOp>
__global__ void convertFormat(SrcWrapper src, DstWrapper dst, UnOp op, int2 size)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= size.x || src_y >= size.y)
        return;

    *dst.ptr(batch_idx, src_y, src_x) = op(*src.ptr(batch_idx, src_y, src_x));
}

template<typename DT_SOURCE, typename DT_DEST, int NC>
ErrorCode convertToScaleCN(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           const double alpha, const double beta, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 size       = {inAccess->numCols(), inAccess->numRows()};
    const int  batch_size = inAccess->numSamples();

    dim3 block(32, 8);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch_size);

    using DT_AB         = decltype(float() * DT_SOURCE() * DT_DEST()); //pick correct scalar
    using SRC_DATA_TYPE = nvcv::cuda::MakeType<DT_SOURCE, NC>;
    using DST_DATA_TYPE = nvcv::cuda::MakeType<DT_DEST, NC>;

    Convertor<SRC_DATA_TYPE, DST_DATA_TYPE, DT_AB> op;

    op.alpha = nvcv::cuda::SaturateCast<DT_AB>(alpha);
    op.beta  = nvcv::cuda::SaturateCast<DT_AB>(beta);

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= nvcv::cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHW<SRC_DATA_TYPE, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHW<DST_DATA_TYPE, int32_t>(outData);

        convertFormat<<<grid, block, 0, stream>>>(src, dst, op, size);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

template<typename DT_SOURCE, typename DT_DEST> // <uchar, float> <float double>
ErrorCode convertToScale(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                         int numChannels, const double alpha, const double beta, cudaStream_t stream)
{
    switch (numChannels)
    {
    case 1:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 1>(inData, outData, alpha, beta, stream);

    case 2:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 2>(inData, outData, alpha, beta, stream);

    case 3:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 3>(inData, outData, alpha, beta, stream);

    case 4:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 4>(inData, outData, alpha, beta, stream);

    default:
        LOG_ERROR("Unknown number of channels");
        return ErrorCode::INVALID_PARAMETER;
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

ErrorCode ConvertTo::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                           const double alpha, const double beta, cudaStream_t stream)
{
    cuda_op::DataFormat input_format    = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format   = GetLegacyDataFormat(outData.layout());
    cuda_op::DataType   input_datatype  = GetLegacyDataType(inData.dtype());
    cuda_op::DataType   output_datatype = GetLegacyDataType(outData.dtype());

    if (!(input_format == kNHWC || output_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat format must be kHWC/kNHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataShape inputShape  = helpers::GetLegacyDataShape(inAccess->infoShape());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C)
    {
        LOG_ERROR("input/output shape is different " << inputShape << "/" << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(input_datatype == kCV_8U || input_datatype == kCV_8S || input_datatype == kCV_16U || input_datatype == kCV_16S
          || input_datatype == kCV_32S || input_datatype == kCV_32F || input_datatype == kCV_64F))
    {
        LOG_ERROR("Invalid DataType " << input_datatype);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(output_datatype == kCV_8U || output_datatype == kCV_8S || output_datatype == kCV_16U
          || output_datatype == kCV_16S || output_datatype == kCV_32S || output_datatype == kCV_32F
          || output_datatype == kCV_64F))
    {
        LOG_ERROR("Invalid Converted DataType " << output_datatype);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef ErrorCode (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                int numChannels, const double alpha, const double beta, cudaStream_t stream);

    // clang-format off
    static const func_t funcs[7][7] = {
        { convertToScale<uchar, uchar>,  convertToScale<uchar, schar>,  convertToScale<uchar, ushort>,  convertToScale<uchar, short>,  convertToScale<uchar, int>,       convertToScale<uchar, float>,    convertToScale<uchar, double>   },
        { convertToScale<schar, uchar>,  convertToScale<schar, schar>,  convertToScale<schar, ushort>,  convertToScale<schar, short>,  convertToScale<schar, int>,       convertToScale<schar, float>,    convertToScale<schar, double>   },
        { convertToScale<ushort, uchar>, convertToScale<ushort, schar>, convertToScale<ushort, ushort>, convertToScale<ushort, short>, convertToScale<ushort, int>,      convertToScale<ushort, float>,   convertToScale<ushort, double>  },
        { convertToScale<short, uchar>,  convertToScale<short, schar>,  convertToScale<short, ushort>,  convertToScale<short, short>,  convertToScale<short, int>,       convertToScale<short, float>,    convertToScale<short, double>   },
        { convertToScale<int, uchar>,    convertToScale<int, schar>,    convertToScale<int, ushort>,    convertToScale<int, short>,    convertToScale<int, int>,         convertToScale<int, float>,      convertToScale<int, double>     },
        { convertToScale<float, uchar>,  convertToScale<float, schar>,  convertToScale<float, ushort>,  convertToScale<float, short>,  convertToScale<float, int>,       convertToScale<float, float>,    convertToScale<float, double>   },
        { convertToScale<double, uchar>, convertToScale<double, schar>, convertToScale<double, ushort>, convertToScale<double, short>, convertToScale<double, int>,      convertToScale<double, float>,   convertToScale<double, double>  }
    };

    // clang-format on
    const func_t func = funcs[input_datatype][output_datatype];
    return func(inData, outData, channels, alpha, beta, stream);
}

} // namespace nvcv::legacy::cuda_op
