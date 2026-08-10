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

namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

static bool SameLayoutFamily(DataFormat lhs, DataFormat rhs)
{
    return IsPlanar(lhs) == IsPlanar(rhs);
}

static TensorDataStridedCuda PlanarTensorView(const TensorDataStridedCuda &data)
{
    auto access = TensorDataAccessStridedImagePlanar::Create(data);
    NVCV_ASSERT(access);

    TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr());
    buf.strides[0] = access->sampleStride();
    buf.strides[1] = access->chStride();
    buf.strides[2] = access->rowStride();
    buf.strides[3] = access->colStride();

    return TensorDataStridedCuda{
        TensorShape{{access->numSamples(), access->numChannels(), access->numRows(), access->numCols()}, "NCHW"},
        data.dtype(), buf
    };
}

} // namespace

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

template<class SrcWrapper, typename T, class VecWrapper>
__global__ void padAndStackPlanar(SrcWrapper src, cuda::Tensor4DWrap<T> dst, VecWrapper topVec, VecWrapper leftVec,
                                  int2 dstSize, int channels)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    const int top  = *topVec.ptr(0, 0, dstCoord.z);
    const int left = *leftVec.ptr(0, 0, dstCoord.z);

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(dstCoord.z, plane, dstCoord.y, dstCoord.x)
                = src[int4{dstCoord.x - left, dstCoord.y - top, plane, dstCoord.z}];
        }
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

template<typename T, NVCVBorderType B>
ErrorCode padAndStackPlanarCaller(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                  const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                                  const float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData, static_cast<T>(borderValue));

    auto topVec  = cuda::CreateTensorWrapNHW<const int>(top);
    auto leftVec = cuda::CreateTensorWrapNHW<const int>(left);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int  channels = outAccess->numChannels();
    const auto samples  = static_cast<int64_t>(outAccess->numSamples());
    if (samples > 65535)
    {
        LOG_ERROR("Planar batch size " << samples << " exceeds CUDA grid z limit 65535.");
        return ErrorCode::INVALID_PARAMETER;
    }

    int2 dstSize{outAccess->numCols(), outAccess->numRows()};

    auto outView = PlanarTensorView(outData);
    auto dst     = cuda::Tensor4DWrap<T>(outView);

    dim3 block(32, sizeof(T) == 1 ? 4 : 8);
    dim3 grid(divUp(dstSize.x, block.x), divUp(dstSize.y, block.y), outAccess->numSamples());

    padAndStackPlanar<<<grid, block, 0, stream>>>(src, dst, topVec, leftVec, dstSize, channels);
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

template<typename T>
ErrorCode padAndStackPlanar(const ImageBatchVarShapeDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                            const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream)
{
    typedef ErrorCode (*padAndStack_caller)(const ImageBatchVarShapeDataStridedCuda &inData,
                                            const TensorDataStridedCuda &outData, const TensorDataStridedCuda &top,
                                            const TensorDataStridedCuda &left, const float borderValue,
                                            cudaStream_t stream);

    static const padAndStack_caller funcs[]
        = {padAndStackPlanarCaller<T, NVCV_BORDER_CONSTANT>, padAndStackPlanarCaller<T, NVCV_BORDER_REPLICATE>,
           padAndStackPlanarCaller<T, NVCV_BORDER_REFLECT>, padAndStackPlanarCaller<T, NVCV_BORDER_WRAP>,
           padAndStackPlanarCaller<T, NVCV_BORDER_REFLECT101>};

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

    if (!(input_format == kNHWC || input_format == kHWC || input_format == kNCHW || input_format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << input_format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid output DataFormat " << format
                                               << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!SameLayoutFamily(input_format, format))
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << format << ")");
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

    const int  channels      = outAccess->numChannels();
    const int  inputChannels = inData.uniqueFormat().numChannels();
    const bool isPlanar      = IsPlanar(format);
    if (channels != inputChannels)
    {
        LOG_ERROR("Input and output channel counts must match, but got " << inputChannels << " and " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (data_type != input_data_type)
    {
        LOG_ERROR("Input and output DataTypes must match, but got " << input_data_type << " and " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!isPlanar && channels == 2 && data_type != kCV_8U)
    {
        LOG_ERROR("Invalid channel number " << channels << " for DataType " << data_type
                                            << ", 2 channels only supported for 8bit Unsigned");
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

    if (isPlanar)
    {
        static const func_t planarFuncs[6]
            = {padAndStackPlanar<uchar>, 0, padAndStackPlanar<ushort>, padAndStackPlanar<short>, padAndStackPlanar<int>,
               padAndStackPlanar<float>};

        const func_t planarFunc = planarFuncs[data_type];
        NVCV_ASSERT(planarFunc != 0);
        return planarFunc(inData, outData, top, left, borderMode, borderValue, stream);
    }

    return func(inData, outData, top, left, borderMode, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op
