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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cub/device/device_reduce.cuh>

#include <vector>

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

template<class Wrapper, typename T = typename Wrapper::ValueType>
__global__ void erase_planar(Wrapper img, int imgH, int imgW, nvcv::cuda::Tensor1DWrap<int2> anchorVec,
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
                *img.ptr(batchId, c, anchor.y + y, anchor.x + x)
                    = nvcv::cuda::SaturateCast<T>(erase_hash(hashValue) % 256);
            }
            else
            {
                *img.ptr(batchId, c, anchor.y + y, anchor.x + x) = nvcv::cuda::SaturateCast<T>(value);
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

template<typename T>
void erasePlanarCaller(const nvcv::TensorDataStridedCuda &imgs, const nvcv::TensorDataStridedCuda &anchor,
                       const nvcv::TensorDataStridedCuda &erasing, const nvcv::TensorDataStridedCuda &imgIdx,
                       const nvcv::TensorDataStridedCuda &values, int max_eh, int max_ew, int num_erasing_area,
                       bool random, unsigned int seed, int rows, int cols, int channels, cudaStream_t stream)
{
    auto wrap = nvcv::cuda::CreateTensorWrapNCHW<T, int32_t>(imgs);

    nvcv::cuda::Tensor1DWrap<int2>  anchorVec(anchor);
    nvcv::cuda::Tensor1DWrap<int3>  erasingVec(erasing);
    nvcv::cuda::Tensor1DWrap<int>   imgIdxVec(imgIdx);
    nvcv::cuda::Tensor1DWrap<float> valuesVec(values);

    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, channels, num_erasing_area);
    erase_planar<<<grid, block, 0, stream>>>(wrap, rows, cols, anchorVec, erasingVec, valuesVec, imgIdxVec, channels,
                                             random, seed);
}

namespace {
static ErrorCode validateParameterLength(const nvcv::TensorDataStridedCuda &tensor, int expectedLength,
                                         const char *name)
{
    if (tensor.shape()[0] < expectedLength)
    {
        LOG_ERROR("Invalid " << name << " length " << tensor.shape()[0] << ", expected at least " << expectedLength);
        return ErrorCode::INVALID_PARAMETER;
    }

    return ErrorCode::SUCCESS;
}

// Per-element host-side validation of anchor / imgIdx. The kernel indexes
// img.ptr(imgIdx[i], ...) without a bound check on the batch axis, so an
// out-of-range imgIdx is an unchecked OOB write — this validation closes
// that hole. It necessarily issues a D→H copy and synchronizes the user's
// stream before launching the erase kernel; the sync is intentional and
// is the cost of throwing a clean INVALID_PARAMETER exception instead of
// silently corrupting memory.
static ErrorCode validateEraseAreaData(const nvcv::TensorDataStridedCuda &anchor,
                                       const nvcv::TensorDataStridedCuda &imgIdx, int numErasingArea, int numSamples,
                                       cudaStream_t stream)
{
    std::vector<int2> hostAnchor(numErasingArea);
    std::vector<int>  hostImgIdx(numErasingArea);

    // Use cudaMemcpy2DAsync to honor the source tensor's element stride; a
    // rank-1 view of a larger tensor is not guaranteed to be packed.
    checkCudaErrors(cudaMemcpy2DAsync(hostAnchor.data(), sizeof(int2), anchor.basePtr(), anchor.stride(0), sizeof(int2),
                                      numErasingArea, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpy2DAsync(hostImgIdx.data(), sizeof(int), imgIdx.basePtr(), imgIdx.stride(0), sizeof(int),
                                      numErasingArea, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    for (int i = 0; i < numErasingArea; ++i)
    {
        if (hostAnchor[i].x < 0 || hostAnchor[i].y < 0)
        {
            LOG_ERROR("Invalid anchor at erase area " << i << ": (" << hostAnchor[i].x << ", " << hostAnchor[i].y
                                                      << ")");
            return ErrorCode::INVALID_PARAMETER;
        }
        if (hostImgIdx[i] < 0 || hostImgIdx[i] >= numSamples)
        {
            LOG_ERROR("Invalid imgIdx at erase area " << i << ": " << hostImgIdx[i] << ", expected [0, " << numSamples
                                                      << ")");
            return ErrorCode::INVALID_PARAMETER;
        }
    }

    return ErrorCode::SUCCESS;
}

struct MaxWH
{
    __device__ __forceinline__ int3 operator()(const int3 &a, const int3 &b) const
    {
        return int3{max(a.x, b.x), max(a.y, b.y), 0};
    }
};

static bool copyContiguousTensor(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                 const nvcv::TensorDataAccessStridedImagePlanar &outAccess, bool isPlanar, int channels,
                                 cudaStream_t stream)
{
    if (inAccess.numSamples() <= 0 || inAccess.numSamples() != outAccess.numSamples()
        || inAccess.numRows() != outAccess.numRows() || inAccess.numCols() != outAccess.numCols()
        || inAccess.numChannels() != outAccess.numChannels() || channels != inAccess.numChannels()
        || inAccess.colStride() != outAccess.colStride())
    {
        return false;
    }

    const int64_t rowBytes = static_cast<int64_t>(inAccess.numCols()) * inAccess.colStride();
    if (rowBytes <= 0 || inAccess.rowStride() != rowBytes || outAccess.rowStride() != rowBytes)
    {
        return false;
    }

    const int64_t planeBytes = static_cast<int64_t>(inAccess.numRows()) * inAccess.rowStride();
    int64_t       sampleBytes;
    if (isPlanar)
    {
        if (inAccess.chStride() != planeBytes || outAccess.chStride() != planeBytes)
        {
            return false;
        }
        sampleBytes = static_cast<int64_t>(channels) * planeBytes;
    }
    else
    {
        sampleBytes = planeBytes;
    }

    if (sampleBytes <= 0)
    {
        return false;
    }

    if (inAccess.numSamples() > 1
        && (inAccess.sampleStride() != sampleBytes || outAccess.sampleStride() != sampleBytes))
    {
        return false;
    }

    const size_t totalBytes = static_cast<size_t>(sampleBytes) * static_cast<size_t>(inAccess.numSamples());
    checkCudaErrors(
        cudaMemcpyAsync(outAccess.sampleData(0), inAccess.sampleData(0), totalBytes, cudaMemcpyDeviceToDevice, stream));
    return true;
}
} // namespace

namespace nvcv::legacy::cuda_op {

Erase::Erase(DataShape max_input_shape, DataShape max_output_shape, int num_erasing_area, bool useBulkCopy)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , d_max_values(nullptr)
    , temp_storage(nullptr)
    , m_useBulkCopy(useBulkCopy)
{
    cudaError_t err = cudaMalloc(&d_max_values, sizeof(int3));
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << sizeof(int3));
        throw LegacyCudaAllocationError("CUDA memory allocation error!");
    }

    max_num_erasing_area = num_erasing_area;
    if (max_num_erasing_area < 0)
    {
        cudaFree(d_max_values);
        LOG_ERROR("Invalid num of erasing area" << max_num_erasing_area);
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "max_num_erasing_area must be >= 0");
    }
    temp_storage  = nullptr;
    storage_bytes = 0;
    MaxWH mwh;
    int3  init = {0, 0, 0};
    cub::DeviceReduce::Reduce(temp_storage, storage_bytes, (int3 *)nullptr, (int3 *)nullptr, max_num_erasing_area, mwh,
                              init);

    void *raw_storage = nullptr;
    err               = cudaMalloc(&raw_storage, storage_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_max_values);
        LOG_ERROR("CUDA memory allocation error of size: " << storage_bytes);
        throw LegacyCudaAllocationError("CUDA memory allocation error!");
    }
    temp_storage = static_cast<std::byte *>(raw_storage);
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
    DataFormat format        = GetLegacyDataFormat(inData.layout());
    DataFormat out_format    = GetLegacyDataFormat(outData.layout());
    DataType   data_type     = GetLegacyDataType(inData.dtype());
    DataType   out_data_type = GetLegacyDataType(outData.dtype());

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(out_format == kNHWC || out_format == kHWC || out_format == kNCHW || out_format == kCHW))
    {
        LOG_ERROR("Invalid output DataFormat " << out_format
                                               << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (format != out_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << format << ") and output (" << out_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    const bool isPlanar = (format == kNCHW || format == kCHW);

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << data_type << " and " << out_data_type);
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
    if (auto status = validateParameterLength(erasing, num_erasing_area, "erasing"); status != ErrorCode::SUCCESS)
    {
        return status;
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
    if (auto status = validateParameterLength(imgIdx, num_erasing_area, "imgIdx"); status != ErrorCode::SUCCESS)
    {
        return status;
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

    const int channels = inAccess->numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (auto status = validateParameterLength(values, num_erasing_area * channels, "values");
        status != ErrorCode::SUCCESS)
    {
        return status;
    }
    if (num_erasing_area > 0)
    {
        if (auto status = validateEraseAreaData(anchor, imgIdx, num_erasing_area, inAccess->numSamples(), stream);
            status != ErrorCode::SUCCESS)
        {
            return status;
        }
    }

    if (!inplace)
    {
        if (!m_useBulkCopy || !copyContiguousTensor(*inAccess, *outAccess, isPlanar, channels, stream))
        {
            for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
            {
                if (isPlanar)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        nvcv::Byte *inSampData  = inAccess->sampleData(i) + c * inAccess->chStride();
                        nvcv::Byte *outSampData = outAccess->sampleData(i) + c * outAccess->chStride();

                        checkCudaErrors(cudaMemcpy2DAsync(outSampData, outAccess->rowStride(), inSampData,
                                                          inAccess->rowStride(),
                                                          inAccess->numCols() * inAccess->colStride(),
                                                          inAccess->numRows(), cudaMemcpyDeviceToDevice, stream));
                    }
                }
                else
                {
                    void *inSampData  = inAccess->sampleData(i);
                    void *outSampData = outAccess->sampleData(i);

                    checkCudaErrors(cudaMemcpy2DAsync(outSampData, outAccess->rowStride(), inSampData,
                                                      inAccess->rowStride(),
                                                      inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                                      cudaMemcpyDeviceToDevice, stream));
                }
            }
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

    static const erase_t funcs[6]
        = {eraseCaller<uchar>, 0, eraseCaller<ushort>, eraseCaller<short>, eraseCaller<int>, eraseCaller<float>};
    static const erase_t planarFuncs[6]
        = {erasePlanarCaller<uchar>, 0, erasePlanarCaller<ushort>, erasePlanarCaller<short>, erasePlanarCaller<int>,
           erasePlanarCaller<float>};

    const erase_t func = isPlanar ? planarFuncs[data_type] : funcs[data_type];

    if (inplace)
        func(inData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed,
             inAccess->numRows(), inAccess->numCols(), inAccess->numChannels(), stream);
    else
        func(outData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed,
             outAccess->numRows(), outAccess->numCols(), outAccess->numChannels(), stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
