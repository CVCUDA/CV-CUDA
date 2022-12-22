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

#include <cassert>

namespace cuda    = nvcv::cuda;
namespace cuda_op = nvcv::legacy::cuda_op;

template<bool srcIsNCHW, typename Ptr2DSrc, typename Ptr2DDst>
__global__ void transformFormat(const Ptr2DSrc src, Ptr2DDst dst, int3 inout_size)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= inout_size.x || src_y >= inout_size.y)
        return;

    for (int c = 0; c < inout_size.z; c++)
    {
        if constexpr (srcIsNCHW)
        {
            *dst.ptr(batch_idx, src_y, src_x, c) = *src.ptr(batch_idx, c, src_y, src_x);
        }
        else
        {
            *dst.ptr(batch_idx, c, src_y, src_x) = *src.ptr(batch_idx, src_y, src_x, c);
        }
    }
}

template<typename data_type> // uchar float
void transform(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
               cuda_op::DataFormat input_format, cuda_op::DataFormat output_format, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int3 inout_size = {inAccess->numCols(), inAccess->numRows(), outAccess->numChannels()};

    dim3 block(32, 8);
    dim3 grid(cuda_op::divUp(inout_size.x, block.x), cuda_op::divUp(inout_size.y, block.y), inAccess->numSamples());

    cuda::Tensor4DWrap<data_type> src_ptr(inData);
    cuda::Tensor4DWrap<data_type> dst_ptr(outData);

    if ((input_format == cuda_op::kNHWC || input_format == cuda_op::kHWC)
        && (output_format == cuda_op::kNCHW || output_format == cuda_op::kCHW))
    {
        transformFormat<false><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, inout_size);
    }
    else if ((input_format == cuda_op::kNCHW || input_format == cuda_op::kCHW)
             && (output_format == cuda_op::kNHWC || output_format == cuda_op::kHWC))
    {
        transformFormat<true><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, inout_size);
    }

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

void Reformat::checkDataFormat(DataFormat format)
{
    NVCV_ASSERT(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW);
}

size_t Reformat::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode Reformat::infer(const nvcv::ITensorDataStridedCuda &inData, const nvcv::ITensorDataStridedCuda &outData,
                          cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

    checkDataFormat(input_format);
    checkDataFormat(output_format);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    if (inData.dtype() == outData.dtype() && inData.shape() == outData.shape())
    {
#ifdef CUDA_DEBUG_LOG
        LOG_ERROR("input_format == output_format, copy outputs from inputs");
#endif

        for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
        {
            nvcv::Byte *inSampData  = inAccess->sampleData(i);
            nvcv::Byte *outSampData = outAccess->sampleData(i);

            for (int p = 0; p < inAccess->numPlanes(); ++p)
            {
                checkCudaErrors(cudaMemcpy2DAsync(outAccess->planeData(p, outSampData), outAccess->rowStride(),
                                                  inAccess->planeData(p, inSampData), inAccess->rowStride(),
                                                  inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                                  cudaMemcpyDeviceToDevice, stream));
            }
        }
        return SUCCESS;
    }

    if (((input_format == cuda_op::kNHWC || input_format == cuda_op::kHWC)
         && !(output_format == cuda_op::kNCHW || output_format == cuda_op::kCHW))
        || ((input_format == cuda_op::kNCHW || input_format == cuda_op::kCHW)
            && !(output_format == cuda_op::kNHWC || output_format == cuda_op::kHWC)))
    {
        LOG_ERROR("Invalid combination of input format " << input_format << " and output format " << output_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.dtype());

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F || data_type == kCV_64F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef void (*transform_t)(const ITensorDataStridedCuda &input, const ITensorDataStridedCuda &output,
                                DataFormat in_format, DataFormat out_format, cudaStream_t stream);

    static const transform_t funcs[7] = {transform<uchar>, transform<schar>, transform<ushort>, transform<short>,
                                         transform<int>,   transform<float>, transform<double>};

    transform_t func = funcs[data_type];
    func(inData, outData, input_format, output_format, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
