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
#include <nvcv/ITensorData.hpp>

#define BLOCK 32

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

template<typename Ptr2D>
__global__ void center_crop_kernel_nhwc(Ptr2D src_ptr, Ptr2D dst_ptr, const int left_indices, const int top_indices,
                                        const int crop_rows, const int crop_columns)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if ((dst_x < crop_columns) && (dst_y < crop_rows))
    {
        *dst_ptr.ptr(batch_idx, dst_y, dst_x) = *src_ptr.ptr(batch_idx, dst_y + top_indices, dst_x + left_indices);
    }
}

template<typename T>
void center_crop(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData, int crop_rows,
                 int crop_columns, const int batch_size, const int rows, const int columns, cudaStream_t stream)
{
    int top_indices  = (rows - crop_rows) / 2;
    int left_indices = (columns - crop_columns) / 2;

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(crop_columns, (int)blockSize.x), divUp(crop_rows, (int)blockSize.y), batch_size);

    nvcv::cuda::Tensor3DWrap<T> src_ptr(inData);
    nvcv::cuda::Tensor3DWrap<T> dst_ptr(outData);

    center_crop_kernel_nhwc<<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, left_indices, top_indices, crop_rows,
                                                                crop_columns);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

size_t CenterCrop::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode CenterCrop::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData, int crop_rows,
                            int crop_columns, cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat both Input and Output must be kHWC or kNHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int columns  = inAccess->numCols();
    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if ((batch != outAccess->numSamples()) || (crop_rows > outAccess->numRows())
        || (crop_columns > outAccess->numCols()))
    {
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    typedef void (*func_t)(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
                           int crop_rows, int crop_columns, const int batch_size, const int rows, const int columns,
                           cudaStream_t stream);
    static const func_t funcs[5][4] = {
        { center_crop<uchar>,  center_crop<uchar2>,  center_crop<uchar3>,  center_crop<uchar4>},
        {center_crop<ushort>, center_crop<ushort2>, center_crop<ushort3>, center_crop<ushort4>},
        {   center_crop<int>,    center_crop<int2>,    center_crop<int3>,    center_crop<int4>},
        {                  0,                    0,                    0,                    0},
        {center_crop<double>, center_crop<double2>, center_crop<double3>, center_crop<double4>},
    };
    int data_size = DataSize(GetLegacyDataType(inData.dtype()));
    funcs[data_size / 2][channels - 1](inData, outData, crop_rows, crop_columns, batch, rows, columns, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
