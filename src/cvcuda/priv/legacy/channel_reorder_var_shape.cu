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

#include "../Assert.h"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <util/CheckError.hpp>

#define BLOCK 32

namespace nvcv::legacy::cuda_op {

template<typename T>
__global__ void channel_reorder_kernel(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst,
                                       const cuda::Tensor2DWrap<int> orders)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);
    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int *chOrder = orders.ptr(batch_idx);

    for (int ch = 0; ch < dst.nch; ch++)
    {
        int src_ch = chOrder[ch];
        if (src_ch < 0)
        {
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = 0;
        }
        else
        {
            NVCV_CUDA_ASSERT(0 <= src_ch && src_ch < src.nch, "Index to source channel %d is out of bounds (%d)",
                             src_ch, src.nch);
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = *src.ptr(batch_idx, dst_y, dst_x, src_ch);
        }
    }
}

template<typename T>
void reorder(const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData,
             const ITensorDataStridedCuda &orderData, int numChannels, cudaStream_t stream)
{
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(inData.maxSize().w, blockSize.x), divUp(inData.maxSize().h, blockSize.y), batch_size);

    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(inData, numChannels);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(outData, numChannels);
    cuda::Tensor2DWrap<int>       order_ptr(orderData);

    channel_reorder_kernel<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

ErrorCode ChannelReorderVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                        const IImageBatchVarShapeDataStridedCuda &outData,
                                        const ITensorDataStridedCuda &orderData, cudaStream_t stream)
{
    if (inData.numImages() != outData.numImages())
    {
        LOG_ERROR("Input and output batches must have the same number of images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type;
    int      channels;
    {
        nvcv::ImageFormat fmt(inData.hostFormatList()[0]);
        data_type = helpers::GetLegacyDataType(fmt);
        channels  = fmt.numChannels();
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (orderData.rank() != 2)
    {
        LOG_ERROR("order tensor must have 2 dimensions, not " << orderData.rank());
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (orderData.layout()[0] != nvcv::LABEL_BATCH)
    {
        LOG_ERROR("Label of the first dimension of order tensor must be " << nvcv::LABEL_BATCH);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (orderData.shape(0) != inData.numImages())
    {
        LOG_ERROR("Order tensor must have same number of samples as number of input images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (orderData.shape(1) > 4)
    {
        LOG_ERROR("Second dimension of order tensor must be at most 4, not " << orderData.shape(1));
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    for (int i = 0; i < inData.numImages(); ++i)
    {
        nvcv::ImageFormat inFmt(inData.hostFormatList()[i]);
        nvcv::ImageFormat outFmt(outData.hostFormatList()[i]);

        if (inFmt.numPlanes() != 1)
        {
            LOG_ERROR("Format of input image #" << i << " must have only 1 plane");
            return ErrorCode::INVALID_DATA_FORMAT;
        }

        if (outFmt.numPlanes() != 1)
        {
            LOG_ERROR("Format of input image #" << i << " must have only 1 plane");
            return ErrorCode::INVALID_DATA_FORMAT;
        }

        // Legacy code has this check, let's stick to it.
        if (inFmt.numChannels() != channels)
        {
            LOG_ERROR("Invalid input");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (inFmt.numChannels() > 4)
        {
            LOG_ERROR("Invalid input channel number " << inFmt.numChannels());
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        if (outFmt.numChannels() > orderData.shape(1))
        {
            LOG_ERROR("Invalid output channel number " << outFmt.numChannels() << ", must be at most "
                                                       << orderData.shape(1));
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        // TODO: we can't check if order index is < channels, like legacy does. It'd incur in
        // perf penalty as the data is currently on device. Instead, we added an assertion in
        // the cuda kernel, but it's only triggered in debug builds, and leads to an unrecoverable
        // error (cuda kernel errors are sticky), process must be restarted.

        if (helpers::GetLegacyDataType(inFmt) != data_type)
        {
            LOG_ERROR("Format of input images must all have the same data type");
            return ErrorCode::INVALID_DATA_TYPE;
        }

        if (helpers::GetLegacyDataType(outFmt) != data_type)
        {
            LOG_ERROR("Format of output images must all have the same data type");
            return ErrorCode::INVALID_DATA_TYPE;
        }
    }

    if (inData.numImages() == 0)
    {
        // nothing to do
        return ErrorCode::SUCCESS;
    }

    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &inData,
                           const IImageBatchVarShapeDataStridedCuda &outData, const ITensorDataStridedCuda &orderData,
                           int numChannels, cudaStream_t stream);

    static const func_t funcs[6] = {reorder<uchar>, 0, reorder<ushort>, reorder<short>, reorder<int>, reorder<float>};

    const func_t func = funcs[data_type];
    NVCV_ASSERT(func != 0);

    func(inData, outData, orderData, channels, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
