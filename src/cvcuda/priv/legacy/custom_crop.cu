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

#include <cstdio>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

template<typename Ptr2D>
__global__ void custom_crop_kernel(const Ptr2D src, Ptr2D dst, int start_x, int start_y, int width, int height)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (x >= width || y >= height)
        return;

    *dst.ptr(batch_idx, y, x) = *src.ptr(batch_idx, y + start_y, x + start_x);
}

template<typename T>
void customCrop(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData, NVCVRectI roi,
                cudaStream_t stream)
{
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    nvcv::cuda::Tensor3DWrap<T> src(inData);
    nvcv::cuda::Tensor3DWrap<T> dst(outData);

    dim3 block(16, 16);
    dim3 grid(divUp(roi.width, block.x), divUp(roi.height, block.y), outAccess->numSamples());

    custom_crop_kernel<<<grid, block, 0, stream>>>(src, dst, roi.x, roi.y, roi.width, roi.height);
    checkKernelErrors();
}

namespace nvcv::legacy::cuda_op {

size_t CustomCrop::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode CustomCrop::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData, NVCVRectI roi,
                            cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invliad DataFormat both Input and Output must be kHWC or kHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
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
    int cols     = inAccess->numCols();

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

    if (roi.height > outAccess->size().h || roi.width > outAccess->size().w)
    {
        LOG_ERROR("ROI larger than dst buffer");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int data_size = DataSize(GetLegacyDataType(inData.dtype()));
    int start_x   = roi.x;
    int start_y   = roi.y;
    int end_x     = start_x + roi.width - 1;
    int end_y     = start_y + roi.height - 1;
#ifdef CUDA_DEBUG_LOG
    LOG_ERROR("x " << roi.x << " y " << roi.y << " width " << roi.width << " height " << roi.height);
#endif

    if (start_x < 0 || start_y < 0 || end_x >= cols || end_y >= rows)
    {
        LOG_ERROR("Invliad Roi range x " << roi.x << " y " << roi.y << " width " << roi.width << " height "
                                         << roi.height);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
                           NVCVRectI roi, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {customCrop<uchar1>,  customCrop<uchar2>,  customCrop<uchar3>,  customCrop<uchar4>},
        {customCrop<ushort>, customCrop<ushort2>, customCrop<ushort3>, customCrop<ushort4>},
        {   customCrop<int>,    customCrop<int2>,    customCrop<int3>,    customCrop<int4>},
        {                 0,                   0,                   0,                   0},
        {customCrop<double>, customCrop<double2>, customCrop<double3>, customCrop<double4>}
    };

    funcs[data_size / 2][channels - 1](inData, outData, roi, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
