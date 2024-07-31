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

#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>

#include <cstdio>
#include <iomanip>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

template<class SrcWrapper, class DstWrapper>
__global__ void hist_kernel(const SrcWrapper src, DstWrapper histogram, int channels, nvcv::Size2D dstSize)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int local_id  = threadIdx.y * blockDim.x + threadIdx.x;

    //initialize the block local memory
    extern __shared__ int shist[];
    if (local_id < 256 * channels)
    {
        shist[local_id] = 0;
    }
    __syncthreads(); // wait for all threads to finish initialization

    //check if we are in the image.
    if (src_x < dstSize.w && src_y < dstSize.h)
    {
        for (int ch = 0; ch < channels; ch++)
        {
            int4  coordImg{ch, src_x, src_y, batch_idx};
            uchar out = src[coordImg];
            int   idx = out + (256 * ch);
            atomicAdd(&shist[idx], 1);
        }
    }
    __syncthreads();

    // copy to the final destination histogram from block local memory
    if (local_id < 256 * channels)
    {
        int hist_val = shist[local_id];
        if (hist_val > 0)
        {
            int2 coordHisto{local_id, batch_idx};
            atomicAdd(&histogram[coordHisto], hist_val);
        }
    }
}

template<class CdfWrapper>
__global__ void prefix_sum_with_norm_kernel(CdfWrapper histogram, nvcv::Size2D dstSize)
{
    const int tid       = threadIdx.x; // thread id in the block 0-255
    const int batch_idx = get_batch_idx();

    const int  hist_idx = threadIdx.x + (blockIdx.x * 256); // index into the histogram 0-255*channels
    const int2 coordHisto{hist_idx, batch_idx};             // index into the histogram 0-255*channels

    __shared__ int temp[256 * 2]; // temp block shared buffer
    int           *reduce_buf = &temp[256];

    // Set block shared memory
    temp[tid] = histogram[coordHisto]; // copy the histogram for this thred to the shared buffer

    // check if there is a histogram value for this index
    if (temp[tid])
    {
        reduce_buf[tid] = tid; // set the reduce buffer to the index of this histogram bin
    }
    else
    {
        reduce_buf[tid] = 255;
    }
    __syncthreads();

    //min-reduce
    for (int s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            reduce_buf[tid] = min(reduce_buf[tid], reduce_buf[tid + s]);
        }
        __syncthreads(); //this can probably be moved prior to the min call.
    }

    const int min_idx = reduce_buf[0]; // this is the first non-zero element in the histogram

    const int total_pixels = dstSize.w * dstSize.h;

    // compute and normalize cdf put into histogram
    if (temp[min_idx] == total_pixels)
    {
        // all pixels have same value
        histogram[coordHisto] = min_idx;
    }
    else
    {
        int pout = 0, pin = 1;
        temp[min_idx] = 0;
        for (int offset = 1; offset < 256; offset *= 2)
        {
            pout = 1 - pout;
            pin  = 1 - pout;
            if (tid >= offset)
                temp[pout * 256 + tid] = temp[pin * 256 + tid] + temp[pin * 256 + tid - offset];
            else
                temp[pout * 256 + tid] = temp[pin * 256 + tid];
            __syncthreads();
        }
        histogram[coordHisto]
            = nvcv::cuda::SaturateCast<int>(1.0 * temp[pout * 256 + tid] / temp[pout * 256 + 255] * 255);
    }
}

template<class SrcWrapper, class DstWrapper, class CdfWrapper>
__global__ void lookup(const SrcWrapper src, DstWrapper dst, CdfWrapper cdf, int channels, nvcv::Size2D dstSize)

{
    const int             src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int             src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int             batch_idx = get_batch_idx();
    const int             local_id  = threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ int temp[];

    //copy the cdf to the block shared memory
    if (local_id < 256 * channels)
    {
        int2 coordHisto{local_id, batch_idx};
        temp[local_id] = cdf[coordHisto];
    }
    __syncthreads();

    //check if we are in the image.
    if (src_x < dstSize.w && src_y < dstSize.h)
    {
        int offset = 0;
        for (int ch = 0; ch < channels; ch++)
        {
            offset = 256 * ch;
            int4 coordImg{ch, src_x, src_y, batch_idx};
            int2 coordHisto{src[coordImg] + offset, batch_idx};
            dst[coordImg] = nvcv::cuda::SaturateCast<uchar>((temp[src[coordImg] + offset]));
        }
    }
}

namespace nvcv::legacy::cuda_op {

HistogramEq::HistogramEq(int maxBatchSize)
{
    if (maxBatchSize < 0)
    {
        LOG_ERROR("Invalid num of max batch size " << maxBatchSize);
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 0");
    }

    m_maxBatchSize    = maxBatchSize;
    m_maxChannelCount = 4;
    //histogram is 256 * channels per image
    m_sizeOfHisto = m_maxBatchSize * m_maxChannelCount * 256 * sizeof(int);

    if (m_maxBatchSize > 0 && m_maxChannelCount > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_histoArray, m_sizeOfHisto));
    }
}

HistogramEq::~HistogramEq()
{
    if (m_histoArray)
    {
        cudaFree(m_histoArray);
        m_histoArray = nullptr;
    }
}

template<typename SrcWrap, typename DstWrap, typename HistWrap>
ErrorCode infer_histogram(SrcWrap src, DstWrap dst, HistWrap histo, int batch, nvcv::Size2D dstSize, int channels,
                          cudaStream_t stream)
{
    {
        //compute the histogram for each image in the batch into m_histoArray
        int bsX = 32; //1024 ( 4 ch of 256 bins)
        int bsY = 32;

        switch (channels)
        {
        case 1:
            bsX = 16; // 256 (1 ch)
            bsY = 16;
            break;
        case 2:
            bsX = 32; // 512 (2 ch)
            bsY = 16;
            break;
        case 3:
            bsX = 32; // 768 (3 ch)
            bsY = 24;
            break;
        default:
            break;
        }

        // each block is going to be 256bins * channels = threads
        dim3   histBlockSize(bsX, bsY, 1);
        dim3   histGridSize(divUp(dstSize.w, histBlockSize.x), divUp(dstSize.h, histBlockSize.y), batch);
        size_t sharedMemSize = 256 * channels * sizeof(int);
        hist_kernel<<<histGridSize, histBlockSize, sharedMemSize, stream>>>(src, histo, channels, dstSize);
        checkKernelErrors();
    }

    //compute cfd
    {
        int  bsX = 256;
        int  bsY = 1;
        int  bsZ = 1;
        dim3 prefixSumBlockSize(bsX, bsY, bsZ);
        dim3 prefixSumGridSize(channels, 1, batch);
        prefix_sum_with_norm_kernel<<<prefixSumGridSize, prefixSumBlockSize, 0, stream>>>(histo, dstSize);
        checkKernelErrors();
    }

    {
        dim3 lookupBlockSize(32, 32, 1);
        dim3 lookupGridSize(divUp(dstSize.w, lookupBlockSize.x), divUp(dstSize.h, lookupBlockSize.y), batch);
        lookup<<<lookupGridSize, lookupBlockSize, 256 * channels * sizeof(int), stream>>>(src, dst, histo, channels,
                                                                                          dstSize);
        checkKernelErrors();
    }

    return ErrorCode::SUCCESS;
}

ErrorCode HistogramEq::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());
    DataType            data_type     = helpers::GetLegacyDataType(inData.dtype());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invliad DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(data_type == kCV_8U))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    int          channels = inAccess->numChannels();
    int          batch    = inAccess->numSamples();
    int          width    = inAccess->numCols();
    int          height   = inAccess->numRows();
    nvcv::Size2D dstSize{width, height};

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

    //clear the histogram.
    checkCudaErrors(cudaMemsetAsync(m_histoArray, 0, m_sizeOfHisto, stream));

    // 2d wrap since its an array of [256 * channels] = width, height = samples
    auto histo = nvcv::cuda::Tensor2DWrap<int, int32_t>(m_histoArray, (int)(256 * channels * sizeof(int)));

    int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHWC<uchar, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHWC<uchar, int32_t>(outData);

        return infer_histogram(src, dst, histo, batch, dstSize, channels, stream);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
}

} // namespace nvcv::legacy::cuda_op
