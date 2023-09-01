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

#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/SaturateCast.hpp>

using namespace nvcv::legacy::helpers;
using namespace nvcv::legacy::cuda_op;

template<class SrcWrapper, class DstWrapper>
__global__ void calc_hist_kernel(const SrcWrapper src, DstWrapper histogram, int numPixels, int width)
{
    extern __shared__ int shist[]; //size 256 * sizeof(int)

    int  batch_idx = get_batch_idx();                       //Z this will be the batch index.
    int  tid       = blockIdx.x * blockDim.x + threadIdx.x; // this is the thread index in the block
    int  x         = tid % width;                           // this is the x index of the pixel assigned to this tid
    int  y         = tid / width;                           // this is the y index in the pixel assigned to this tid
    int3 coord{x, y, batch_idx};

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (tid < numPixels)
    {
        atomicAdd(&shist[src[coord]], 1);
    }
    __syncthreads(); // wait for all of the threads in this block to finish

    int hist_val = shist[sub_tid]; // get the bin value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0)
    {
        atomicAdd(histogram.ptr(batch_idx, sub_tid), hist_val);
    }
}

template<class SrcWrapper, class MaskWrapper, class DstWrapper>
__global__ void calc_hist_kernel(const SrcWrapper src, DstWrapper histogram, MaskWrapper mask, int numPixels, int width)
{
    extern __shared__ int shist[];                                           //size 256 * sizeof(int)
    int                   batch_idx = get_batch_idx();                       //Z this will be the batch index.
    int                   tid       = blockIdx.x * blockDim.x + threadIdx.x; // this is the thread index in the block
    int                   x         = tid % width; // this is the x index of the pixel assigned to this tid
    int                   y         = tid / width; // this is the y index in the pixel assigned to this tid
    int3                  coord{x, y, batch_idx};

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (tid < numPixels)
    {
        if (mask[coord])
            atomicAdd(&shist[src[coord]], 1);
    }
    __syncthreads(); // wait for all of the threads in this block to finish

    int hist_val = shist[sub_tid]; // get the bin value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0)
    {
        atomicAdd(histogram.ptr(batch_idx, sub_tid), hist_val);
    }
}

namespace nvcv::legacy::cuda_op {

ErrorCode Histogram::infer(const TensorDataStridedCuda &inData, OptionalTensorConstRef mask,
                           const TensorDataStridedCuda &histogram, cudaStream_t stream)
{
    DataFormat input_format = GetLegacyDataFormat(inData.layout());
    DataFormat histo_format = GetLegacyDataFormat(histogram.layout());
    DataType   data_type    = GetLegacyDataType(inData.dtype());

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    auto histoAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(histogram);
    NVCV_ASSERT(histoAccess);

    if (!(input_format == kNHWC || input_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat for calculating histogram" << input_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(histo_format == kNHWC || histo_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat for calculating histogram" << input_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (data_type != kCV_8U)
    {
        LOG_ERROR("Invalid DataType for calculating histogram" << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());
    DataShape histo_shape = GetLegacyDataShape(histoAccess->infoShape());

    if (input_shape.N != histo_shape.H)
    {
        LOG_ERROR(
            "Historgram tensor does not contain enough rows for an input batch tensor of N = " << (input_shape.N));
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int batch    = input_shape.N;
    int channels = input_shape.C;
    int rows     = input_shape.H;
    int cols     = input_shape.W;

    if (channels != 1)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // clear the histogram since we will just add to it only 1 sample in thi HW tensor by definition
    checkCudaErrors(cudaMemset2DAsync(histoAccess->sampleData(0), histoAccess->rowStride(), 0, histoAccess->numCols(),
                                      histoAccess->numRows(), stream));

    auto src   = nvcv::cuda::CreateTensorWrapNHW<uchar>(inData);
    auto histo = nvcv::cuda::Tensor2DWrap<int>(histogram);

    int threads_block = 256;

    // Setup 1 thread / pixel slpit into blocks of 256 threads for local binning.
    // grid in y could be colors.
    dim3 grid_size((rows * cols + threads_block - 1) / threads_block, 1, batch);
    int  smem_size = 256 * sizeof(int);

    if (mask == nullptr)
    {
        calc_hist_kernel<<<grid_size, threads_block, smem_size, stream>>>(src, histo, rows * cols, cols);
        checkKernelErrors();
    }
    else
    {
        // below retruns optional
        auto maskTensorData = mask->get().exportData<nvcv::TensorDataStridedCuda>();
        NVCV_ASSERT(maskTensorData);
        // just check if the mask is formatted correcty.
        auto inMask = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskTensorData);
        NVCV_ASSERT(inMask);

        if (GetLegacyDataShape(inMask->infoShape()) != GetLegacyDataShape(inAccess->infoShape()))
        {
            LOG_ERROR("Mask tensor does not match input tensor shape");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto maskAccess = nvcv::cuda::CreateTensorWrapNHW<uchar>(*maskTensorData);

        calc_hist_kernel<<<grid_size, threads_block, smem_size, stream>>>(src, histo, maskAccess, rows * cols, cols);
        checkKernelErrors();
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
