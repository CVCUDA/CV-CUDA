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

    double angle  = *angleDeg.ptr(index);
    double xShift = *shift.ptr(index, 0);
    double yShift = *shift.ptr(index, 1);

    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

template<typename T>
__global__ void rotate_linear(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, const double *d_aCoeffs_)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    const double *d_aCoeffs   = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * batch_idx);
    const double  dst_x_shift = dst_x - d_aCoeffs[2];
    const double  dst_y_shift = dst_y - d_aCoeffs[5];
    float         src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float         src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        using work_type = nvcv::cuda::ConvertBaseTypeTo<float, T>;
        work_type out   = nvcv::cuda::SetAll<work_type>(0);

        const int x1      = __float2int_rz(src_x);
        const int y1      = __float2int_rz(src_y);
        const int x2      = x1 + 1;
        const int y2      = y1 + 1;
        const int x2_read = min(x2, width - 1);
        const int y2_read = min(y2, height - 1);

        T src_reg = *src.ptr(batch_idx, y1, x1);
        out       = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *src.ptr(batch_idx, y1, x2_read);
        out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *src.ptr(batch_idx, y2_read, x1);
        out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *src.ptr(batch_idx, y2_read, x2_read);
        out     = out + src_reg * ((src_x - x1) * (src_y - y1));

        *dst.ptr(batch_idx, dst_y, dst_x) = nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<T>>(out);
    }
}

template<typename T>
__global__ void rotate_nearest(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, const double *d_aCoeffs_)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;
    int height = src.at_rows(batch_idx), width = src.at_cols(batch_idx);

    const double *d_aCoeffs   = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * batch_idx);
    const double  dst_x_shift = dst_x - d_aCoeffs[2];
    const double  dst_y_shift = dst_y - d_aCoeffs[5];
    float         src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float         src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        const int x1 = min(__float2int_rz(src_x + 0.5), width - 1);
        const int y1 = min(__float2int_rz(src_y + 0.5), height - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, y1, x1);
    }
}

template<typename T>
__global__ void rotate_cubic(CubicFilter<BorderReader<Ptr2dVarShapeNHWC<T>, BrdReplicate<T>>> filteredSrc,
                             Ptr2dVarShapeNHWC<T> dst, const double *d_aCoeffs_)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;
    int height = filteredSrc.src.ptr.at_rows(batch_idx), width = filteredSrc.src.ptr.at_cols(batch_idx);

    const double *d_aCoeffs   = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * batch_idx);
    const double  dst_x_shift = dst_x - d_aCoeffs[2];
    const double  dst_y_shift = dst_y - d_aCoeffs[5];
    float         src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float         src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        *dst.ptr(batch_idx, dst_y, dst_x) = filteredSrc(batch_idx, src_y, src_x);
    }
}

template<typename T> // uchar3 float3 uchar1 float3
void rotate(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
            double *d_aCoeffs, const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);

    Size2D outMaxSize = out.maxSize();

    NVCV_ASSERT(in.numImages() == out.numImages());

    dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    Ptr2dVarShapeNHWC<T> src_ptr(in);  //batch_size, height, width, channels, (T **) d_in);
    Ptr2dVarShapeNHWC<T> dst_ptr(out); //batch_size, out_height, out_width, channels, (T **) d_out);
    if (interpolation == NVCV_INTERP_LINEAR)
    {
        rotate_linear<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        rotate_nearest<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_CUBIC)
    {
        BrdReplicate<T>                                                  brd(0, 0);
        BorderReader<Ptr2dVarShapeNHWC<T>, BrdReplicate<T>>              brdSrc(src_ptr, brd);
        CubicFilter<BorderReader<Ptr2dVarShapeNHWC<T>, BrdReplicate<T>>> filteredSrc(brdSrc);

        rotate_cubic<T><<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr, d_aCoeffs);
        checkKernelErrors();
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

ErrorCode RotateVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                const IImageBatchVarShapeDataStridedCuda &outData,
                                const ITensorDataStridedCuda &angleDeg, const ITensorDataStridedCuda &shift,
                                const NVCVInterpolationType interpolation, cudaStream_t stream)
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

    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
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
