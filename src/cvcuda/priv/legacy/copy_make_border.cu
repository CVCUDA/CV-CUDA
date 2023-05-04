/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvcv::legacy::cuda_op {

template<class SrcWrapper, class DstWrapper>
__global__ void copyMakeBorderKernel(SrcWrapper src, DstWrapper dst, int2 dstSize, int left, int top)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    int3 srcCoord = {dstCoord.x - left, dstCoord.y - top, dstCoord.z};

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        dst[dstCoord] = src[srcCoord];
    }
}

template<typename T, NVCVBorderType B>
struct copyMakeBorderDispatcher
{
    static void call(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const T &borderValue,
                     const int left, const int top, cudaStream_t stream)
    {
        auto src = cuda::CreateBorderWrapNHW<const T, B>(inData, borderValue);
        auto dst = cuda::CreateTensorWrapNHW<T>(outData);

        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        int2 dstSize{outAccess->numCols(), outAccess->numRows()};

        dim3 blockSize(BLOCK, BLOCK / 4, 1);
        dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), outAccess->numSamples());

        copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, left, top);
        checkKernelErrors();
    }
};

template<typename T> // uchar3 float3 uchar float
void copyMakeBorder(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const int top,
                    const int left, const NVCVBorderType border_type, const float4 &borderValue, cudaStream_t stream)
{
    const T bvalue = cuda::DropCast<cuda::NumElements<T>>(cuda::StaticCast<cuda::BaseType<T>>(borderValue));

    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                           const T &borderValue, const int left, const int top, cudaStream_t stream);

    static const func_t funcs[]
        = {copyMakeBorderDispatcher<T, NVCV_BORDER_CONSTANT>::call,
           copyMakeBorderDispatcher<T, NVCV_BORDER_REPLICATE>::call,
           copyMakeBorderDispatcher<T, NVCV_BORDER_REFLECT>::call, copyMakeBorderDispatcher<T, NVCV_BORDER_WRAP>::call,
           copyMakeBorderDispatcher<T, NVCV_BORDER_REFLECT101>::call};

    funcs[border_type](inData, outData, bvalue, left, top, stream);
}

ErrorCode CopyMakeBorder::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const int top, const int left, const NVCVBorderType border_type,
                                const float4 &borderValue, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(input_format == kNHWC || input_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << input_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType data_type = helpers::GetLegacyDataType(inData.dtype());

    const int channels = inAccess->numChannels();

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

    const int rows     = inAccess->numRows();
    const int cols     = inAccess->numCols();
    const int out_rows = outAccess->numRows();
    const int out_cols = outAccess->numCols();

    if (!(top >= 0 && out_rows >= top + rows && left >= 0 && out_cols >= left + cols))
    {
        LOG_ERROR("Invalid border " << top << " " << out_rows - top - rows << " " << left << " "
                                    << out_cols - left - cols
                                    << ", top >= 0 && bottom >= 0 && left >= 0 && right >= 0, in resolution: " << rows
                                    << "x" << cols << ", out resolution: " << out_rows << "x" << out_cols);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const int top,
                           const int left, const NVCVBorderType border_type, const float4 &borderValue,
                           cudaStream_t stream);

    // clang-format off
    static const func_t funcs[6][4] = {
        {copyMakeBorder<uchar1>      , copyMakeBorder<uchar2>       , copyMakeBorder<uchar3>      , copyMakeBorder<uchar4>      },
        {0 /*copyMakeBorder<char1>*/, 0 /*copyMakeBorder<char2>*/ , 0 /*copyMakeBorder<char3>*/, 0 /*copyMakeBorder<char4>*/},
        {copyMakeBorder<ushort1>     , 0 /*copyMakeBorder<ushort2>*/, copyMakeBorder<ushort3>     , copyMakeBorder<ushort4>     },
        {copyMakeBorder<short1>      , 0 /*copyMakeBorder<short2>*/, copyMakeBorder<short3>      , copyMakeBorder<short4>      },
        {0 /*copyMakeBorder<int, 1>*/  , 0 /*copyMakeBorder<int, 2>*/   , 0 /*copyMakeBorder<int, 3>*/  , 0 /*copyMakeBorder<int, 4>*/  },
        {copyMakeBorder<float1>      , 0 /*copyMakeBorder<float2>*/, copyMakeBorder<float3>      , copyMakeBorder<float4>      }
    };
    // clang-format on

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, top, left, border_type, borderValue, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
