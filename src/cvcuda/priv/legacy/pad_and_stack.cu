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

using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

template<typename Ptr2D, typename D, typename BrdRd, class Ptr2DVec>
__global__ void padAndStack(const BrdRd src, Ptr2D dst, const Ptr2DVec topVec, const Ptr2DVec leftVec, int out_rows,
                            int out_cols)
{
    const int x_out     = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int top  = *topVec.ptr(0, 0, batch_idx);
    const int left = *leftVec.ptr(0, 0, batch_idx);

    const int x_in = x_out - left;
    const int y_in = y_out - top;

    if (x_out < out_cols && y_out < out_rows)
    {
        *dst.ptr(batch_idx, y_out, x_out) = src(batch_idx, y_in, x_in);
    }
}

template<typename D, template<typename> class Brd>
void padAndStackCaller(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::TensorDataAccessStridedImagePlanar &outData,
                       const nvcv::TensorDataAccessStridedImagePlanar &top,
                       const nvcv::TensorDataAccessStridedImagePlanar &left, const float borderValue,
                       cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(inData);

    Ptr2dNHWC<D> dst(outData);

    Ptr2dNHWC<int> topVec(top);
    Ptr2dNHWC<int> leftVec(left);

    dim3 block(16, 16);
    dim3 grid(divUp(outData.size().w, block.x), divUp(outData.size().h, block.y), outData.numSamples());

    Brd<D> brd(0, 0, nvcv::cuda::SetAll<D>(borderValue));

    BorderReader<Ptr2dVarShapeNHWC<D>, Brd<D>> brdSrc(src, brd);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    padAndStack<Ptr2dNHWC<D>, D, BorderReader<Ptr2dVarShapeNHWC<D>, Brd<D>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, topVec, leftVec, outData.numRows(), outData.numCols());

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void padAndStack(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                 const nvcv::TensorDataAccessStridedImagePlanar &outData,
                 const nvcv::TensorDataAccessStridedImagePlanar &top,
                 const nvcv::TensorDataAccessStridedImagePlanar &left, const NVCVBorderType borderMode,
                 const float borderValue, cudaStream_t stream)
{
    typedef void (*padAndStack_caller)(
        const nvcv::IImageBatchVarShapeDataStridedCuda &inData, const nvcv::TensorDataAccessStridedImagePlanar &outData,
        const nvcv::TensorDataAccessStridedImagePlanar &top, const nvcv::TensorDataAccessStridedImagePlanar &left,
        const float borderValue, cudaStream_t stream);

    static const padAndStack_caller funcs[]
        = {padAndStackCaller<D, BrdConstant>, padAndStackCaller<D, BrdReplicate>, padAndStackCaller<D, BrdReflect>,
           padAndStackCaller<D, BrdWrap>, padAndStackCaller<D, BrdReflect101>};

    funcs[borderMode](inData, outData, top, left, borderValue, stream);
}

namespace nvcv::legacy::cuda_op {

size_t PadAndStack::calBufferSize(int batch_size)
{
    return 0;
}

ErrorCode PadAndStack::infer(const IImageBatchVarShapeDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                             const ITensorDataStridedCuda &top, const ITensorDataStridedCuda &left,
                             const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream)
{
    DataFormat format    = GetLegacyDataFormat(outData.layout());
    DataType   data_type = GetLegacyDataType(outData.dtype());

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
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
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataType   left_data_type = GetLegacyDataType(left.dtype());
    DataFormat left_format    = GetLegacyDataFormat(left.layout());
    if (left_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Left DataType " << left_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(left_format == kNHWC || left_format == kHWC))
    {
        LOG_ERROR("Invalid Left DataFormat " << left_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto leftAccess = TensorDataAccessStridedImagePlanar::Create(left);
    if (!leftAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   top_data_type = GetLegacyDataType(top.dtype());
    DataFormat top_format    = GetLegacyDataFormat(top.layout());
    if (top_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Top DataType " << top_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(top_format == kNHWC || top_format == kHWC))
    {
        LOG_ERROR("Invalid Top DataFormat " << top_format);
        return ErrorCode::INVALID_DATA_FORMAT;
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

    typedef void (*func_t)(
        const nvcv::IImageBatchVarShapeDataStridedCuda &inData, const TensorDataAccessStridedImagePlanar &outData,
        const TensorDataAccessStridedImagePlanar &top, const TensorDataAccessStridedImagePlanar &left,
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

    func(inData, *outAccess, *topAccess, *leftAccess, borderMode, borderValue, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
