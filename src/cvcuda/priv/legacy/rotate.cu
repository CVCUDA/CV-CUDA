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

__global__ void compute_warpAffine(const double angle, const double xShift, const double yShift, double *aCoeffs)
{
    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

template<typename T>
__global__ void rotate_linear(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const double *d_aCoeffs)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst.cols || dst_y >= dst.rows)
        return;
    const int batch_idx = get_batch_idx();
    int       height = src.rows, width = src.cols;

    const double dst_x_shift = dst_x - d_aCoeffs[2];
    const double dst_y_shift = dst_y - d_aCoeffs[5];
    float        src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float        src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

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
__global__ void rotate_nearest(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const double *d_aCoeffs)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst.cols || dst_y >= dst.rows)
        return;
    const int batch_idx = get_batch_idx();
    int       height = src.rows, width = src.cols;

    const double dst_x_shift = dst_x - d_aCoeffs[2];
    const double dst_y_shift = dst_y - d_aCoeffs[5];

    float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        const int x1 = min(__float2int_rz(src_x + 0.5), width - 1);
        const int y1 = min(__float2int_rz(src_y + 0.5), height - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, y1, x1);
    }
}

template<typename T>
__global__ void rotate_cubic(CubicFilter<BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>> filteredSrc, Ptr2dNHWC<T> dst,
                             const double *d_aCoeffs)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst.cols || dst_y >= dst.rows)
        return;
    const int batch_idx = get_batch_idx();
    int       height = filteredSrc.src.ptr.rows, width = filteredSrc.src.ptr.cols;

    const double dst_x_shift = dst_x - d_aCoeffs[2];
    const double dst_y_shift = dst_y - d_aCoeffs[5];

    float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        *dst.ptr(batch_idx, dst_y, dst_x) = filteredSrc(batch_idx, src_y, src_x);
    }
}

template<typename T> // uchar3 float3 uchar1 float3
void rotate(const nvcv::TensorDataAccessStridedImagePlanar &inData,
            const nvcv::TensorDataAccessStridedImagePlanar &outData, double *d_aCoeffs, const double angleDeg,
            const double2 shift, const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    const int batch_size = inData.numSamples();
    const int in_width   = inData.numCols();
    const int in_height  = inData.numRows();
    const int out_width  = outData.numCols();
    const int out_height = outData.numRows();

    compute_warpAffine<<<1, 1, 0, stream>>>(angleDeg, shift.x, shift.y, d_aCoeffs);
    checkKernelErrors();

    dim3         blockSize(BLOCK, BLOCK / 4, 1);
    dim3         gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);
    Ptr2dNHWC<T> src_ptr(inData);  //batch_size, height, width, channels, (T *) d_in);
    Ptr2dNHWC<T> dst_ptr(outData); //batch_size, out_height, out_width, channels, (T *) d_out);

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
        BrdReplicate<T>                                          brd(src_ptr.rows, src_ptr.cols);
        BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>              brdSrc(src_ptr, brd);
        CubicFilter<BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>> filteredSrc(brdSrc);

        rotate_cubic<T><<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

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

ErrorCode Rotate::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                        const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation,
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

    DataType  data_type   = GetLegacyDataType(inData.dtype());
    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

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

    typedef void (*func_t)(const nvcv::TensorDataAccessStridedImagePlanar &inData,
                           const nvcv::TensorDataAccessStridedImagePlanar &outData, double *d_aCoeffs,
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

    func(*inAccess, *outAccess, d_aCoeffs, angleDeg, shift, interpolation, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
