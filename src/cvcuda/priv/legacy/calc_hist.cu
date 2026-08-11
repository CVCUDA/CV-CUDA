/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../CudaDeviceUtils.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>

#include <limits>

using namespace nvcv::legacy::helpers;
using namespace nvcv::legacy::cuda_op;

namespace {

constexpr bool UseOnePixelHistogramKernel(int sm, int64_t numPixels)
{
    return sm == 75 && numPixels <= std::numeric_limits<int>::max();
}

static_assert(UseOnePixelHistogramKernel(75, std::numeric_limits<int>::max()));
static_assert(!UseOnePixelHistogramKernel(75, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1));
static_assert(!UseOnePixelHistogramKernel(80, 1));
static_assert(!UseOnePixelHistogramKernel(90, 1));

} // namespace

template<class SrcWrapper, class DstWrapper>
__global__ void calc_hist_one_pixel_kernel(const SrcWrapper src, DstWrapper histogram, int numPixels, int width)
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
__global__ void calc_hist_one_pixel_kernel(const SrcWrapper src, DstWrapper histogram, MaskWrapper mask, int numPixels,
                                           int width)
{
    extern __shared__ int shist[];                                           //size 256 * sizeof(int)
    int                   batch_idx = get_batch_idx();                       //Z this will be the batch index.
    int                   tid       = blockIdx.x * blockDim.x + threadIdx.x; // this is the thread index in the block
    int                   x         = tid % width; // this is the x index in the pixel assigned to this tid
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

template<class SrcWrapper, class DstWrapper>
__global__ void calc_hist_kernel(const SrcWrapper src, DstWrapper histogram, int height, int width)
{
    extern __shared__ int shist[]; //size 256 * sizeof(int)

    int batch_idx = get_batch_idx(); //Z this will be the batch index.
    int x         = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int y         = blockIdx.y * blockDim.y + threadIdx.y;

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.y * blockDim.x + threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (y < height)
    {
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (x + i < width)
            {
                atomicAdd(&shist[src[int3{x + i, y, batch_idx}]], 1);
            }
        }
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
__global__ void calc_hist_kernel(const SrcWrapper src, DstWrapper histogram, MaskWrapper mask, int height, int width)
{
    extern __shared__ int shist[];                     //size 256 * sizeof(int)
    int                   batch_idx = get_batch_idx(); //Z this will be the batch index.
    int                   x         = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int                   y         = blockIdx.y * blockDim.y + threadIdx.y;

    // histogram index only used to sub the output histogram
    int sub_tid = threadIdx.y * blockDim.x + threadIdx.x; //histogram index

    shist[sub_tid] = 0; //initialize the histogram for this bin in this block

    __syncthreads();

    if (y < height)
    {
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (x + i < width)
            {
                int3 coord{x + i, y, batch_idx};
                if (mask[coord])
                    atomicAdd(&shist[src[coord]], 1);
            }
        }
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
    DataType   histo_type   = GetLegacyDataType(histogram.dtype());

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    auto histoAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(histogram);
    NVCV_ASSERT(histoAccess);

    if (!(input_format == kNHWC || input_format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat for calculating histogram " << input_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(histo_format == kNHWC || histo_format == kHWC))
    {
        LOG_ERROR("Invalid histogram DataFormat for calculating histogram " << histo_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (data_type != kCV_8U)
    {
        LOG_ERROR("Invalid DataType for calculating histogram " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (histo_type != kCV_32S)
    {
        LOG_ERROR("Invalid histogram DataType for calculating histogram " << histo_type);
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

    if (histo_shape.W < 256)
    {
        LOG_ERROR("Histogram tensor must have at least 256 columns, got " << histo_shape.W);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (histo_shape.C != 1)
    {
        LOG_ERROR("Invalid histogram channel number " << histo_shape.C);
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
    checkCudaErrors(cudaMemset2DAsync(histoAccess->sampleData(0), histoAccess->rowStride(), 0,
                                      histoAccess->numCols() * histoAccess->colStride(), histoAccess->numRows(),
                                      stream));

    auto src   = nvcv::cuda::CreateTensorWrapNHW<uchar>(inData);
    auto histo = nvcv::cuda::Tensor2DWrap<int>(histogram);

    const int64_t num_pixels = static_cast<int64_t>(rows) * cols;

    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));
    const bool use_one_pixel_kernel = UseOnePixelHistogramKernel(sm, num_pixels);

    constexpr int one_pixel_block_size = 256;
    int           one_pixel_num_pixels = 0;
    dim3          one_pixel_grid_size;
    if (use_one_pixel_kernel)
    {
        one_pixel_num_pixels = static_cast<int>(num_pixels);
        one_pixel_grid_size.x
            = one_pixel_num_pixels / one_pixel_block_size + (one_pixel_num_pixels % one_pixel_block_size != 0);
        one_pixel_grid_size.z = batch;
    }

    dim3 block_size(32, 8);

    // Amortize each block-local histogram across sixteen horizontal pixels per lane.
    dim3 grid_size((cols + block_size.x * 16 - 1) / (block_size.x * 16), (rows + block_size.y - 1) / block_size.y,
                   batch);
    int  smem_size = 256 * sizeof(int);

    if (mask == nullptr)
    {
        if (use_one_pixel_kernel)
        {
            calc_hist_one_pixel_kernel<<<one_pixel_grid_size, one_pixel_block_size, smem_size, stream>>>(
                src, histo, one_pixel_num_pixels, cols);
        }
        else
        {
            calc_hist_kernel<<<grid_size, block_size, smem_size, stream>>>(src, histo, rows, cols);
        }
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

        DataFormat mask_format = GetLegacyDataFormat(maskTensorData->layout());
        DataType   mask_type   = GetLegacyDataType(maskTensorData->dtype());
        DataShape  mask_shape  = GetLegacyDataShape(inMask->infoShape());

        if (!(mask_format == kNHWC || mask_format == kHWC))
        {
            LOG_ERROR("Invalid mask DataFormat for calculating histogram " << mask_format);
            return ErrorCode::INVALID_DATA_FORMAT;
        }

        if (mask_type != kCV_8U)
        {
            LOG_ERROR("Invalid mask DataType for calculating histogram " << mask_type);
            return ErrorCode::INVALID_DATA_TYPE;
        }

        if (mask_shape != input_shape)
        {
            LOG_ERROR("Mask tensor does not match input tensor shape");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto maskAccess = nvcv::cuda::CreateTensorWrapNHW<uchar>(*maskTensorData);

        if (use_one_pixel_kernel)
        {
            calc_hist_one_pixel_kernel<<<one_pixel_grid_size, one_pixel_block_size, smem_size, stream>>>(
                src, histo, maskAccess, one_pixel_num_pixels, cols);
        }
        else
        {
            calc_hist_kernel<<<grid_size, block_size, smem_size, stream>>>(src, histo, maskAccess, rows, cols);
        }
        checkKernelErrors();
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
