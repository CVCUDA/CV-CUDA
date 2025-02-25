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

namespace nvcv::legacy::cuda_op {

template<class SrcWrapper, class DstWrapper, class VecWrapper>
__global__ void padAndStack(SrcWrapper src, DstWrapper dst, VecWrapper topVec, VecWrapper leftVec, int2 dstSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    const int top  = *topVec.ptr(0, 0, dstCoord.z);
    const int left = *leftVec.ptr(0, 0, dstCoord.z);

    int3 srcCoord = {dstCoord.x - left, dstCoord.y - top, dstCoord.z};

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        dst[dstCoord] = src[srcCoord];
    }
}

template<typename T, NVCVBorderType B>
ErrorCode padAndStackCaller(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                            const float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));

    auto topVec  = cuda::CreateTensorWrapNHW<const int>(top);
    auto leftVec = cuda::CreateTensorWrapNHW<const int>(left);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    int2 dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(16, 16);
    dim3 grid(divUp(dstSize.x, block.x), divUp(dstSize.y, block.y), outAccess->numSamples());

    if (outAccess->sampleStride() * outAccess->numSamples() <= cuda::TypeTraits<int32_t>::max)
    {
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        padAndStack<<<grid, block, 0, stream>>>(src, dst, topVec, leftVec, dstSize);
    }
    else
    {
        LOG_ERROR("Output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

template<typename T>
ErrorCode padAndStack(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                      const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                      const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream)
{
    typedef ErrorCode (*padAndStack_caller)(const ImageBatchVarShapeDataStridedCuda &inData,
                                            const TensorDataStridedCuda &outData, const TensorDataStridedCuda &top,
                                            const TensorDataStridedCuda &left, const float borderValue,
                                            cudaStream_t stream);

    static const padAndStack_caller funcs[]
        = {padAndStackCaller<T, NVCV_BORDER_CONSTANT>, padAndStackCaller<T, NVCV_BORDER_REPLICATE>,
           padAndStackCaller<T, NVCV_BORDER_REFLECT>, padAndStackCaller<T, NVCV_BORDER_WRAP>,
           padAndStackCaller<T, NVCV_BORDER_REFLECT101>};

    return funcs[borderMode](inData, outData, top, left, borderValue, stream);
}

ErrorCode PadAndStack::infer(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                             const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format    = helpers::GetLegacyDataFormat(inData);
    DataType   input_data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    DataFormat format    = helpers::GetLegacyDataFormat(outData.layout());
    DataType   data_type = helpers::GetLegacyDataType(outData.dtype());

    if (!(input_format == kNHWC || input_format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << input_format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid output DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid output DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(input_data_type == kCV_8U || input_data_type == kCV_16U || input_data_type == kCV_16S
          || input_data_type == kCV_32S || input_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid input DataType " << input_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataType   left_data_type = helpers::GetLegacyDataType(left.dtype());
    DataFormat left_format    = helpers::GetLegacyDataFormat(left.layout());
    if (!(left_format == kNHWC || left_format == kHWC))
    {
        LOG_ERROR("Invalid Left DataFormat " << left_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (left_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Left DataType " << left_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto leftAccess = TensorDataAccessStridedImagePlanar::Create(left);
    if (!leftAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   top_data_type = helpers::GetLegacyDataType(top.dtype());
    DataFormat top_format    = helpers::GetLegacyDataFormat(top.layout());
    if (!(top_format == kNHWC || top_format == kHWC))
    {
        LOG_ERROR("Invalid Top DataFormat " << top_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (top_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Top DataType " << top_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto topAccess = TensorDataAccessStridedImagePlanar::Create(top);
    if (!topAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = outAccess->numChannels();
    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    typedef ErrorCode (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                                const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        { padAndStack<uchar1>, padAndStack<uchar2>,  padAndStack<uchar3>,  padAndStack<uchar4>},
        {                   0,                   0,                    0,                    0},
        {padAndStack<ushort1>,                   0, padAndStack<ushort3>, padAndStack<ushort4>},
        { padAndStack<short1>,                   0,  padAndStack<short3>,  padAndStack<short4>},
        {   padAndStack<int1>,                   0,    padAndStack<int3>,    padAndStack<int4>},
        { padAndStack<float1>,                   0,  padAndStack<float3>,  padAndStack<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    return func(inData, outData, top, left, borderMode, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op
