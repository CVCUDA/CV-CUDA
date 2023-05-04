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

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

__global__ void compute_warpAffine(const int numImages, const cuda::Tensor1DWrap<double> angleDeg,
                                   const cuda::Tensor2DWrap<double> shift, double *d_aCoeffs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    double *aCoeffs = (double *)((char *)d_aCoeffs + (sizeof(double) * 6) * index);

    double angle  = angleDeg[index];
    double xShift = *shift.ptr(index, 0);
    double yShift = *shift.ptr(index, 1);

    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

template<class SrcWrapper, class DstWrapper>
__global__ void rotate(SrcWrapper src, DstWrapper dst, const double *d_aCoeffs_)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    if (dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z))
        return;

    const double *d_aCoeffs   = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * dstCoord.z);
    const double  dst_x_shift = dstCoord.x - d_aCoeffs[2];
    const double  dst_y_shift = dstCoord.y - d_aCoeffs[5];
    float         src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float         src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    const int width  = src.borderWrap().imageBatchWrap().width(dstCoord.z);
    const int height = src.borderWrap().imageBatchWrap().height(dstCoord.z);

    if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        const float3 srcCoord{src_x, src_y, static_cast<float>(dstCoord.z)};

        dst[dstCoord] = src[srcCoord];
    }
}

template<typename T, NVCVInterpolationType I>
void rotate(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
            double *d_aCoeffs, cudaStream_t stream)
{
    Size2D outMaxSize = out.maxSize();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    cuda::InterpolationVarShapeWrap<const T, NVCV_BORDER_REPLICATE, I> src(in);
    cuda::ImageBatchVarShapeWrap<T>                                    dst(out);

    rotate<<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
    checkKernelErrors();
}

template<typename T> // uchar3 float3 uchar1 float3
void rotate(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
            double *d_aCoeffs, const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        rotate<T, NVCV_INTERP_NEAREST>(in, out, d_aCoeffs, stream);
        break;

    case NVCV_INTERP_LINEAR:
        rotate<T, NVCV_INTERP_LINEAR>(in, out, d_aCoeffs, stream);
        break;

    case NVCV_INTERP_CUBIC:
        rotate<T, NVCV_INTERP_CUBIC>(in, out, d_aCoeffs, stream);
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation type");
    }
}

RotateVarShape::RotateVarShape(const int maxBatchSize)
    : CudaBaseOp()
    , d_aCoeffs(nullptr)
    , m_maxBatchSize(maxBatchSize)
{
    if (m_maxBatchSize > 0)
    {
        size_t      bufferSize = sizeof(double) * 6 * m_maxBatchSize;
        cudaError_t err        = cudaMalloc(&d_aCoeffs, bufferSize);
        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDA memory allocation error of size: " << bufferSize);
            throw std::runtime_error("CUDA memory allocation error!");
        }
    }
}

RotateVarShape::~RotateVarShape()
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

ErrorCode RotateVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                const ImageBatchVarShapeDataStridedCuda &outData, const TensorDataStridedCuda &angleDeg,
                                const TensorDataStridedCuda &shift, const NVCVInterpolationType interpolation,
                                cudaStream_t stream)
{
    if (m_maxBatchSize <= 0)
    {
        LOG_ERROR("Operator rotate var shape is not initialized properly, maxVarShapeBatchSize: " << m_maxBatchSize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_maxBatchSize < inData.numImages())
    {
        LOG_ERROR("Invalid number of images, it should not exceed " << m_maxBatchSize);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

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

    cuda::Tensor1DWrap<double> angleDecPtr(angleDeg);
    cuda::Tensor2DWrap<double> shiftPtr(shift);

    compute_warpAffine<<<1, inData.numImages(), 0, stream>>>(inData.numImages(), angleDecPtr, shiftPtr, d_aCoeffs);
    checkKernelErrors();

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                           double *d_aCoeffs, const NVCVInterpolationType interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      rotate<uchar>,  0 /*rotate<uchar2>*/,      rotate<uchar3>,      rotate<uchar4>},
        {0 /*rotate<schar>*/,   0 /*rotate<char2>*/, 0 /*rotate<char3>*/, 0 /*rotate<char4>*/},
        {     rotate<ushort>, 0 /*rotate<ushort2>*/,     rotate<ushort3>,     rotate<ushort4>},
        {      rotate<short>,  0 /*rotate<short2>*/,      rotate<short3>,      rotate<short4>},
        {  0 /*rotate<int>*/,    0 /*rotate<int2>*/,  0 /*rotate<int3>*/,  0 /*rotate<int4>*/},
        {      rotate<float>,  0 /*rotate<float2>*/,      rotate<float3>,      rotate<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];

    func(inData, outData, d_aCoeffs, interpolation, stream);
    assert(func != 0);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
