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

#include <nvcv/IImage.hpp>
#include <nvcv/IImageData.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

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
        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<DST_TYPE>>(alpha * src + beta);
    }
};

template<typename Ptr2DSrc, typename Ptr2DDst, class UnOp>
__global__ void convertFormat(Ptr2DSrc src, Ptr2DDst dst, UnOp op, int2 size)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= size.x || src_y >= size.y)
        return;

    *dst.ptr(batch_idx, src_y, src_x) = op(*src.ptr(batch_idx, src_y, src_x));
}

template<typename DT_SOURCE, typename DT_DEST, int NC>
void convertToScaleCN(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
                      const double alpha, const double beta, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    const int2 size       = {inAccess->numCols(), inAccess->numRows()};
    const int  batch_size = inAccess->numSamples();

    dim3 block(32, 8);
    dim3 grid(divUp(size.x, block.x), divUp(size.y, block.y), batch_size);

    using DT_AB         = decltype(float() * DT_SOURCE() * DT_DEST()); //pick correct scalar
    using SRC_DATA_TYPE = nvcv::cuda::MakeType<DT_SOURCE, NC>;
    using DST_DATA_TYPE = nvcv::cuda::MakeType<DT_DEST, NC>;

    Convertor<SRC_DATA_TYPE, DST_DATA_TYPE, DT_AB> op;
    nvcv::cuda::Tensor3DWrap<SRC_DATA_TYPE>        src(inData);
    nvcv::cuda::Tensor3DWrap<DST_DATA_TYPE>        dst(outData);

    op.alpha = nvcv::cuda::SaturateCast<DT_AB>(alpha);
    op.beta  = nvcv::cuda::SaturateCast<DT_AB>(beta);
    convertFormat<<<grid, block, 0, stream>>>(src, dst, op, size);
}

template<typename DT_SOURCE, typename DT_DEST> // <uchar, float> <float double>
void convertToScale(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
                    int numChannels, const double alpha, const double beta, cudaStream_t stream)
{
    switch (numChannels)
    {
    case 1:
        convertToScaleCN<DT_SOURCE, DT_DEST, 1>(inData, outData, alpha, beta, stream);
        break;

    case 2:
        convertToScaleCN<DT_SOURCE, DT_DEST, 2>(inData, outData, alpha, beta, stream);
        break;

    case 3:
        convertToScaleCN<DT_SOURCE, DT_DEST, 3>(inData, outData, alpha, beta, stream);
        break;

    case 4:
        convertToScaleCN<DT_SOURCE, DT_DEST, 4>(inData, outData, alpha, beta, stream);
        break;

    default:
        LOG_ERROR("Unknown number of channels");
        return;
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

size_t ConvertTo::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode ConvertTo::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
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

    typedef void (*func_t)(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
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
    func(inData, outData, channels, alpha, beta, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
