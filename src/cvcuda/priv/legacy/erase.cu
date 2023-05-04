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
#include "cub/cub.cuh"

using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

static __device__ int erase_hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template<class Wrapper, typename T = typename Wrapper::ValueType>
__global__ void erase(Wrapper img, int imgH, int imgW, nvcv::cuda::Tensor1DWrap<int2> anchorVec,
                      nvcv::cuda::Tensor1DWrap<int3> erasingVec, nvcv::cuda::Tensor1DWrap<float> valuesVec,
                      nvcv::cuda::Tensor1DWrap<int> imgIdxVec, int channels, int random, unsigned int seed)
{
    unsigned int id      = threadIdx.x + blockIdx.x * blockDim.x;
    int          c       = blockIdx.y;
    int          eraseId = blockIdx.z;
    int2         anchor  = anchorVec[eraseId];
    int3         erasing = erasingVec[eraseId];
    float        value   = valuesVec[eraseId * channels + c];
    int          batchId = imgIdxVec[eraseId];
    if (id < erasing.y * erasing.x && (0x1 & (erasing.z >> c)) == 1)
    {
        int x = id % erasing.x;
        int y = id / erasing.x;
        if (anchor.x + x < imgW && anchor.y + y < imgH)
        {
            if (random)
            {
                unsigned int hashValue = seed + threadIdx.x
                                       + 0x26AD0C9 * blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + 1)
                                             * (blockIdx.y + 1) * (blockIdx.z + 1);
                *img.ptr(batchId, anchor.y + y, anchor.x + x, c)
                    = nvcv::cuda::SaturateCast<T>(erase_hash(hashValue) % 256);
            }
            else
            {
                *img.ptr(batchId, anchor.y + y, anchor.x + x, c) = nvcv::cuda::SaturateCast<T>(value);
            }
        }
    }
}

template<typename T>
void eraseCaller(const nvcv::TensorDataStridedCuda &imgs, const nvcv::TensorDataStridedCuda &anchor,
                 const nvcv::TensorDataStridedCuda &erasing, const nvcv::TensorDataStridedCuda &imgIdx,
                 const nvcv::TensorDataStridedCuda &values, int max_eh, int max_ew, int num_erasing_area, bool random,
                 unsigned int seed, int rows, int cols, int channels, cudaStream_t stream)
{
    auto wrap = nvcv::cuda::CreateTensorWrapNHWC<T>(imgs);

    nvcv::cuda::Tensor1DWrap<int2>  anchorVec(anchor);
    nvcv::cuda::Tensor1DWrap<int3>  erasingVec(erasing);
    nvcv::cuda::Tensor1DWrap<int>   imgIdxVec(imgIdx);
    nvcv::cuda::Tensor1DWrap<float> valuesVec(values);

    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, channels, num_erasing_area);
    erase<<<grid, block, 0, stream>>>(wrap, rows, cols, anchorVec, erasingVec, valuesVec, imgIdxVec, channels, random,
                                      seed);
}

struct MaxWH
{
    __device__ __forceinline__ int3 operator()(const int3 &a, const int3 &b) const
    {
        return int3{max(a.x, b.x), max(a.y, b.y), 0};
    }
};

namespace nvcv::legacy::cuda_op {

Erase::Erase(DataShape max_input_shape, DataShape max_output_shape, int num_erasing_area)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , d_max_values(nullptr)
    , temp_storage(nullptr)
{
    cudaError_t err = cudaMalloc(&d_max_values, sizeof(int3));
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << sizeof(int3));
        throw std::runtime_error("CUDA memory allocation error!");
    }

    max_num_erasing_area = num_erasing_area;
    if (max_num_erasing_area < 0)
    {
        cudaFree(d_max_values);
        LOG_ERROR("Invalid num of erasing area" << max_num_erasing_area);
        throw std::runtime_error("Parameter error!");
    }
    temp_storage  = NULL;
    storage_bytes = 0;
    MaxWH mwh;
    int3  init = {0, 0, 0};
    cub::DeviceReduce::Reduce(temp_storage, storage_bytes, (int3 *)nullptr, (int3 *)nullptr, max_num_erasing_area, mwh,
                              init);

    err = cudaMalloc(&temp_storage, storage_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_max_values);
        LOG_ERROR("CUDA memory allocation error of size: " << storage_bytes);
        throw std::runtime_error("CUDA memory allocation error!");
    }
}

Erase::~Erase()
{
    cudaError_t err0 = cudaFree(d_max_values);
    cudaError_t err1 = cudaFree(temp_storage);
    if (err0 != cudaSuccess || err1 != cudaSuccess)
    {
        LOG_ERROR("CUDA memory free error, possible memory leak!");
    }
    d_max_values = nullptr;
    temp_storage = nullptr;
}

ErrorCode Erase::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                       const TensorDataStridedCuda &anchor, const TensorDataStridedCuda &erasing,
                       const TensorDataStridedCuda &values, const TensorDataStridedCuda &imgIdx, bool random,
                       unsigned int seed, bool inplace, cudaStream_t stream)
{
    DataFormat format    = GetLegacyDataFormat(inData.layout());
    DataType   data_type = GetLegacyDataType(inData.dtype());

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType anchor_data_type = GetLegacyDataType(anchor.dtype());
    if (anchor_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid anchor DataType " << anchor_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int anchor_dim = anchor.layout().rank();
    if (anchor_dim != 1)
    {
        LOG_ERROR("Invalid anchor Dim " << anchor_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int num_erasing_area = anchor.shape()[0];
    if (num_erasing_area < 0)
    {
        LOG_ERROR("Invalid num of erasing area " << num_erasing_area);
        return ErrorCode::INVALID_PARAMETER;
    }
    if (num_erasing_area > max_num_erasing_area)
    {
        LOG_ERROR("Invalid num of erasing area " << num_erasing_area);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType erasing_data_type = GetLegacyDataType(erasing.dtype());
    if (erasing_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid erasing_w DataType " << erasing_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int erasing_dim = erasing.layout().rank();
    if (erasing_dim != 1)
    {
        LOG_ERROR("Invalid erasing Dim " << erasing_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType imgidx_data_type = GetLegacyDataType(imgIdx.dtype());
    if (imgidx_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid imgIdx DataType " << imgidx_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int imgidx_dim = imgIdx.layout().rank();
    if (imgidx_dim != 1)
    {
        LOG_ERROR("Invalid imgIdx Dim " << imgidx_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType values_data_type = GetLegacyDataType(values.dtype());
    if (values_data_type != kCV_32F)
    {
        LOG_ERROR("Invalid values DataType " << values_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int values_dim = values.layout().rank();
    if (values_dim != 1)
    {
        LOG_ERROR("Invalid values Dim " << values_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    if (!inplace)
    {
        for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
        {
            void *inSampData  = inAccess->sampleData(i);
            void *outSampData = outAccess->sampleData(i);

            checkCudaErrors(cudaMemcpy2DAsync(outSampData, outAccess->rowStride(), inSampData, inAccess->rowStride(),
                                              inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                              cudaMemcpyDeviceToDevice, stream));
        }
    }

    if (num_erasing_area == 0)
    {
        return SUCCESS;
    }

    int3 *d_erasing = (int3 *)erasing.basePtr();
    int3  h_max_values;
    MaxWH maxwh;
    int3  init = {0, 0, 0};

    cub::DeviceReduce::Reduce(temp_storage, storage_bytes, d_erasing, d_max_values, num_erasing_area, maxwh, init,
                              stream);
    checkCudaErrors(cudaMemcpyAsync(&h_max_values, d_max_values, sizeof(int3), cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    int max_ew = h_max_values.x, max_eh = h_max_values.y;

    // All areas as empty? Weird, but valid nonetheless.
    if (max_ew == 0 || max_eh == 0)
    {
        return SUCCESS;
    }

    typedef void (*erase_t)(const TensorDataStridedCuda &imgs, const TensorDataStridedCuda &anchor,
                            const TensorDataStridedCuda &erasing, const TensorDataStridedCuda &imgIdx,
                            const TensorDataStridedCuda &values, int max_eh, int max_ew, int num_erasing_area,
                            bool random, unsigned int seed, int rows, int cols, int channels, cudaStream_t stream);

    static const erase_t funcs[6] = {eraseCaller<uchar>, eraseCaller<char>, eraseCaller<ushort>,
                                     eraseCaller<short>, eraseCaller<int>,  eraseCaller<float>};

    if (inplace)
        funcs[data_type](inData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed,
                         inAccess->numRows(), inAccess->numCols(), inAccess->numChannels(), stream);
    else
        funcs[data_type](outData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed,
                         outAccess->numRows(), outAccess->numCols(), outAccess->numChannels(), stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
