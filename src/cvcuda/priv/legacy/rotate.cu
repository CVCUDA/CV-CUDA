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
#define PI    3.1415926535897932384626433832795

namespace nvcv::legacy::cuda_op {

__global__ void compute_warpAffine(const double angle, const double xShift, const double yShift, double *aCoeffs)
{
    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

template<class SrcWrapper, class DstWrapper>
__global__ void rotate(SrcWrapper src, DstWrapper dst, int2 dstSize, const double *d_aCoeffs)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y)
    {
        return;
    }

    const double dst_x_shift = dstCoord.x - d_aCoeffs[2];
    const double dst_y_shift = dstCoord.y - d_aCoeffs[5];
    const float3 srcCoord{static_cast<float>(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1])),
                          static_cast<float>(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]),
                          static_cast<float>(dstCoord.z)};

    const int2 srcSize{src.borderWrap().tensorShape()[1], src.borderWrap().tensorShape()[0]};

    if (srcCoord.x > -0.5 && srcCoord.x < srcSize.x && srcCoord.y > -0.5 && srcCoord.y < srcSize.y)
    {
        dst[dstCoord] = src[srcCoord];
    }
}

template<typename T, NVCVInterpolationType I>
void rotate(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, double *d_aCoeffs,
            const double angleDeg, const double2 shift, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int  batchSize{static_cast<int>(outAccess->numSamples())};

    compute_warpAffine<<<1, 1, 0, stream>>>(angleDeg, shift.x, shift.y, d_aCoeffs);
    checkKernelErrors();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), batchSize);

    auto src = cuda::CreateInterpolationWrapNHW<const T, NVCV_BORDER_REPLICATE, I>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T>(outData);

    rotate<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, d_aCoeffs);

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T> // uchar3 float3 uchar1 float3
void rotate(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, double *d_aCoeffs,
            const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        rotate<T, NVCV_INTERP_NEAREST>(inData, outData, d_aCoeffs, angleDeg, shift, stream);
        break;

    case NVCV_INTERP_LINEAR:
        rotate<T, NVCV_INTERP_LINEAR>(inData, outData, d_aCoeffs, angleDeg, shift, stream);
        break;

    case NVCV_INTERP_CUBIC:
        rotate<T, NVCV_INTERP_CUBIC>(inData, outData, d_aCoeffs, angleDeg, shift, stream);
        break;

    default:
        LOG_ERROR("Invalid rotate interpolation " << interpolation);
        break;
    }
}

Rotate::Rotate(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , d_aCoeffs(nullptr)
{
    size_t      bufferSize = calBufferSize(max_input_shape, max_output_shape, DataType::kCV_8U /*not in use*/);
    cudaError_t err        = cudaMalloc(&d_aCoeffs, bufferSize);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << bufferSize);
        throw std::runtime_error("CUDA memory allocation error!");
    }
}

Rotate::~Rotate()
{
    if (d_aCoeffs != nullptr)
    {
        cudaError_t err = cudaFree(d_aCoeffs);
        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDA memory free error, possible memory leak!");
        }
    }
    d_aCoeffs = nullptr;
}

size_t Rotate::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 6 * sizeof(double);
}

ErrorCode Rotate::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                        const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation,
                        cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

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

    DataType  data_type   = helpers::GetLegacyDataType(inData.dtype());
    DataShape input_shape = helpers::GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

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

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, double *d_aCoeffs,
                           const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation,
                           cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      rotate<uchar>,  0 /*rotate<uchar2>*/,      rotate<uchar3>,      rotate<uchar4>},
        {0 /*rotate<schar>*/,   0 /*rotate<char2>*/, 0 /*rotate<char3>*/, 0 /*rotate<char4>*/},
        {     rotate<ushort>, 0 /*rotate<ushort2>*/,     rotate<ushort3>,     rotate<ushort4>},
        {      rotate<short>,  0 /*rotate<short2>*/,      rotate<short3>,      rotate<short4>},
        {  0 /*rotate<int>*/,    0 /*rotate<int2>*/,  0 /*rotate<int3>*/,  0 /*rotate<int4>*/},
        {      rotate<float>,  0 /*rotate<float2>*/,      rotate<float3>,      rotate<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, d_aCoeffs, angleDeg, shift, interpolation, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
