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

#include <cstdint>
#include <vector>

using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

static __device__ int erase_var_shape_hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template<typename D>
__global__ void erase(nvcv::cuda::ImageBatchVarShapeWrapNHWC<D> img, nvcv::cuda::Tensor1DWrap<int2> anchorVec,
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
        if ((anchor.x + x) < img.width(batchId) && (anchor.y + y) < img.height(batchId))
        {
            if (random)
            {
                unsigned int hashValue = seed + threadIdx.x
                                       + 0x26AD0C9 * blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + 1)
                                             * (blockIdx.y + 1) * (blockIdx.z + 1);
                *img.ptr(batchId, anchor.y + y, anchor.x + x, c)
                    = nvcv::cuda::SaturateCast<D>(erase_var_shape_hash(hashValue) % 256);
            }
            else
            {
                *img.ptr(batchId, anchor.y + y, anchor.x + x, c) = nvcv::cuda::SaturateCast<D>(value);
            }
        }
    }
}

template<typename D>
__global__ void erase_planar(nvcv::cuda::ImageBatchVarShapeWrap<D> img, nvcv::cuda::Tensor1DWrap<int2> anchorVec,
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
        if ((anchor.x + x) < img.width(batchId) && (anchor.y + y) < img.height(batchId))
        {
            if (random)
            {
                unsigned int hashValue = seed + threadIdx.x
                                       + 0x26AD0C9 * blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + 1)
                                             * (blockIdx.y + 1) * (blockIdx.z + 1);
                *img.ptr(batchId, c, anchor.y + y, anchor.x + x)
                    = nvcv::cuda::SaturateCast<D>(erase_var_shape_hash(hashValue) % 256);
            }
            else
            {
                *img.ptr(batchId, c, anchor.y + y, anchor.x + x) = nvcv::cuda::SaturateCast<D>(value);
            }
        }
    }
}

constexpr int kCopyVarShapeNIX = 4;

template<int NIX>
__global__ void copyVarShapeRows(nvcv::cuda::ImageBatchVarShapeWrap<uchar> src,
                                 nvcv::cuda::ImageBatchVarShapeWrap<uchar> dst, int channels, int elementBytes,
                                 bool isPlanar)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int x0        = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride    = gridDim.x * blockDim.x;
    const int planes    = isPlanar ? channels : 1;

    for (int p = 0; p < planes; ++p)
    {
        if (y >= dst.height(batch_idx, p))
        {
            continue;
        }

        const int    rowVecs = dst.width(batch_idx, p) * elementBytes * (isPlanar ? 1 : channels) / sizeof(uint4);
        const uint4 *srcRow  = reinterpret_cast<const uint4 *>(src.ptr(batch_idx, p, y, 0));
        uint4       *dstRow  = reinterpret_cast<uint4 *>(dst.ptr(batch_idx, p, y, 0));

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int x = x0 + i * stride;
            if (x < rowVecs)
            {
                dstRow[x] = srcRow[x];
            }
        }
    }
}

template<typename D>
void eraseCaller(const nvcv::ImageBatchVarShapeDataStridedCuda &imgs, const nvcv::TensorDataStridedCuda &anchor,
                 const nvcv::TensorDataStridedCuda &erasing, const nvcv::TensorDataStridedCuda &imgIdx,
                 const nvcv::TensorDataStridedCuda &values, int max_eh, int max_ew, int num_erasing_area, bool random,
                 unsigned int seed, cudaStream_t stream)
{
    nvcv::cuda::ImageBatchVarShapeWrapNHWC<D> src(imgs, imgs.uniqueFormat().numChannels());

    nvcv::cuda::Tensor1DWrap<int2>  anchorVec(anchor);
    nvcv::cuda::Tensor1DWrap<int3>  erasingVec(erasing);
    nvcv::cuda::Tensor1DWrap<int>   imgIdxVec(imgIdx);
    nvcv::cuda::Tensor1DWrap<float> valuesVec(values);

    int  channel   = imgs.uniqueFormat().numChannels();
    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, channel, num_erasing_area);
    erase<D><<<grid, block, 0, stream>>>(src, anchorVec, erasingVec, valuesVec, imgIdxVec, channel, random, seed);
}

template<typename D>
void erasePlanarCaller(const nvcv::ImageBatchVarShapeDataStridedCuda &imgs, const nvcv::TensorDataStridedCuda &anchor,
                       const nvcv::TensorDataStridedCuda &erasing, const nvcv::TensorDataStridedCuda &imgIdx,
                       const nvcv::TensorDataStridedCuda &values, int max_eh, int max_ew, int num_erasing_area,
                       bool random, unsigned int seed, cudaStream_t stream)
{
    nvcv::cuda::ImageBatchVarShapeWrap<D> src(imgs);

    nvcv::cuda::Tensor1DWrap<int2>  anchorVec(anchor);
    nvcv::cuda::Tensor1DWrap<int3>  erasingVec(erasing);
    nvcv::cuda::Tensor1DWrap<int>   imgIdxVec(imgIdx);
    nvcv::cuda::Tensor1DWrap<float> valuesVec(values);

    int  channel   = imgs.uniqueFormat().numChannels();
    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, channel, num_erasing_area);
    erase_planar<D>
        <<<grid, block, 0, stream>>>(src, anchorVec, erasingVec, valuesVec, imgIdxVec, channel, random, seed);
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

static int elementSizeBytes(DataType dataType)
{
    switch (dataType)
    {
    case kCV_8U:
        return 1;
    case kCV_16U:
    case kCV_16S:
        return 2;
    case kCV_32S:
    case kCV_32F:
        return 4;
    default:
        return 0;
    }
}

static bool canUseVarShapeRowCopy(const nvcv::ImageBatchVarShape &inbatch, const nvcv::ImageBatchVarShape &outbatch,
                                  bool isPlanar, int channels, int elementBytes)
{
    const int expectedPlanes = isPlanar ? channels : 1;
    if (expectedPlanes <= 0 || elementBytes <= 0)
    {
        return false;
    }

    for (auto init = inbatch.begin(), outit = outbatch.begin(); init != inbatch.end() && outit != outbatch.end();
         ++init, ++outit)
    {
        const nvcv::Image &inimg      = *init;
        const nvcv::Image &outimg     = *outit;
        auto               inimgdata  = inimg.exportData<nvcv::ImageDataStridedCuda>();
        auto               outimgdata = outimg.exportData<nvcv::ImageDataStridedCuda>();
        if (inimgdata->numPlanes() != expectedPlanes || outimgdata->numPlanes() != expectedPlanes)
        {
            return false;
        }

        for (int p = 0; p < expectedPlanes; ++p)
        {
            const nvcv::ImagePlaneStrided &inplane  = inimgdata->plane(p);
            const nvcv::ImagePlaneStrided &outplane = outimgdata->plane(p);
            if (inplane.width != outplane.width || inplane.height != outplane.height)
            {
                return false;
            }

            const int64_t rowBytes = static_cast<int64_t>(inplane.width) * elementBytes * (isPlanar ? 1 : channels);
            if (rowBytes <= 0 || rowBytes % static_cast<int>(sizeof(uint4)) != 0 || inplane.rowStride < rowBytes
                || outplane.rowStride < rowBytes || inplane.rowStride % static_cast<int>(sizeof(uint4)) != 0
                || outplane.rowStride % static_cast<int>(sizeof(uint4)) != 0
                || (reinterpret_cast<std::uintptr_t>(inplane.basePtr) & (sizeof(uint4) - 1)) != 0
                || (reinterpret_cast<std::uintptr_t>(outplane.basePtr) & (sizeof(uint4) - 1)) != 0)
            {
                return false;
            }
        }
    }

    return true;
}

static bool copyVarShapeBatch(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &outData, bool isPlanar, int channels,
                              int elementBytes, cudaStream_t stream)
{
    if (outData.numImages() <= 0 || outData.numImages() > 65535)
    {
        return false;
    }

    const nvcv::Size2D maxSize = outData.maxSize();
    if (maxSize.w <= 0 || maxSize.h <= 0)
    {
        return false;
    }

    const int rowBytes = maxSize.w * elementBytes * (isPlanar ? 1 : channels);
    if (rowBytes <= 0 || rowBytes % static_cast<int>(sizeof(uint4)) != 0)
    {
        return false;
    }

    const int rowVecs = rowBytes / sizeof(uint4);
    dim3      blockSize(32, 8, 1);
    dim3      gridSize(divUp(divUp(rowVecs, kCopyVarShapeNIX), static_cast<int>(blockSize.x)),
                       divUp(maxSize.h, static_cast<int>(blockSize.y)), outData.numImages());

    nvcv::cuda::ImageBatchVarShapeWrap<uchar> src(inData);
    nvcv::cuda::ImageBatchVarShapeWrap<uchar> dst(outData);
    copyVarShapeRows<kCopyVarShapeNIX><<<gridSize, blockSize, 0, stream>>>(src, dst, channels, elementBytes, isPlanar);
    checkKernelErrors();
    return true;
}
} // namespace

namespace nvcv::legacy::cuda_op {

EraseVarShape::EraseVarShape(DataShape max_input_shape, DataShape max_output_shape, int num_erasing_area,
                             bool useBulkCopy)
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

EraseVarShape::~EraseVarShape()
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

ErrorCode EraseVarShape::infer(const nvcv::ImageBatchVarShape &inbatch, const nvcv::ImageBatchVarShape &outbatch,
                               const TensorDataStridedCuda &anchor, const TensorDataStridedCuda &erasing,
                               const TensorDataStridedCuda &values, const TensorDataStridedCuda &imgIdx, bool random,
                               unsigned int seed, bool inplace, cudaStream_t stream)
{
    auto inData = inbatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        LOG_ERROR("Input must be varshape image batch");
        return ErrorCode::INVALID_PARAMETER;
    }
    auto outData = outbatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        LOG_ERROR("Output must be varshape image batch");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (inData->numImages() != outData->numImages())
    {
        LOG_ERROR("Input and output batches must have the same number of images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!inData->uniqueFormat())
    {
        LOG_ERROR("Images in input batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!outData->uniqueFormat())
    {
        LOG_ERROR("Images in output batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format     = helpers::GetLegacyDataFormat(*inData);
    DataFormat out_format = helpers::GetLegacyDataFormat(*outData);
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

    DataType data_type     = helpers::GetLegacyDataType(inData->uniqueFormat());
    DataType out_data_type = helpers::GetLegacyDataType(outData->uniqueFormat());
    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    const int elementBytes = elementSizeBytes(data_type);
    if (data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData->uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
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
        LOG_ERROR("Invalid erasing DataType " << erasing_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int erasing_dim = erasing.layout().rank();
    if (erasing_dim != 1)
    {
        LOG_ERROR("Invalid erasing_w Dim " << erasing_dim);
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
    if (auto status = validateParameterLength(values, num_erasing_area * channels, "values");
        status != ErrorCode::SUCCESS)
    {
        return status;
    }
    if (num_erasing_area > 0)
    {
        if (auto status = validateEraseAreaData(anchor, imgIdx, num_erasing_area, inbatch.numImages(), stream);
            status != ErrorCode::SUCCESS)
        {
            return status;
        }
    }

    if (!inplace)
    {
        if (!m_useBulkCopy || !canUseVarShapeRowCopy(inbatch, outbatch, isPlanar, channels, elementBytes)
            || !copyVarShapeBatch(*inData, *outData, isPlanar, channels, elementBytes, stream))
        {
            for (auto init = inbatch.begin(), outit = outbatch.begin();
                 init != inbatch.end() && outit != outbatch.end(); ++init, ++outit)
            {
                const Image &inimg      = *init;
                const Image &outimg     = *outit;
                auto         inimgdata  = inimg.exportData<ImageDataStridedCuda>();
                auto         outimgdata = outimg.exportData<ImageDataStridedCuda>();
                for (int p = 0; p < inimgdata->numPlanes(); ++p)
                {
                    const ImagePlaneStrided &inplane  = inimgdata->plane(p);
                    const ImagePlaneStrided &outplane = outimgdata->plane(p);
                    const size_t             rowBytes
                        = static_cast<size_t>(inplane.width) * inimgdata->format().planePixelStrideBytes(p);
                    checkCudaErrors(cudaMemcpy2DAsync(outplane.basePtr, outplane.rowStride, inplane.basePtr,
                                                      inplane.rowStride, rowBytes, inplane.height,
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

    typedef void (*erase_t)(const ImageBatchVarShapeDataStridedCuda &imgs, const TensorDataStridedCuda &anchor,
                            const TensorDataStridedCuda &erasing, const TensorDataStridedCuda &imgIdx,
                            const TensorDataStridedCuda &values, int max_eh, int max_ew, int num_erasing_area,
                            bool random, unsigned int seed, cudaStream_t stream);

    static const erase_t funcs[6]
        = {eraseCaller<uchar>, 0, eraseCaller<ushort>, eraseCaller<short>, eraseCaller<int>, eraseCaller<float>};
    static const erase_t planarFuncs[6]
        = {erasePlanarCaller<uchar>, 0, erasePlanarCaller<ushort>, erasePlanarCaller<short>, erasePlanarCaller<int>,
           erasePlanarCaller<float>};

    const erase_t func = isPlanar ? planarFuncs[data_type] : funcs[data_type];

    if (inplace)
        func(*inData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed, stream);
    else
        func(*outData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed, stream);

    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
