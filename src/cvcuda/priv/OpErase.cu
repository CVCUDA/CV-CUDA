/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "EraseCopyPolicy.hpp"
#include "Nvtx.hpp"
#include "OpErase.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cub/device/device_reduce.cuh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cvcuda::priv {

// Declared inside cvcuda::priv because the CUDA toolkit headers pulled in by cub already own a
// global ::cuda namespace.
namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

inline bool UseBulkEraseCopyOnDevice(int deviceId)
{
    cudaDeviceProp properties{};
    NVCV_CHECK_THROW(cudaGetDeviceProperties(&properties, deviceId));

    const int sm = properties.major * 10 + properties.minor;
    return UseBulkEraseCopyForDevice(sm, properties.name);
}

// Element classification by data kind plus bits-per-channel.  Packed types therefore land on the
// same entry as their scalar sibling (TYPE_3U8 classifies as kU8, which is what lets the anchor
// tensor be read as int2 while still classifying as kS32), and any (kind, width) pair without an
// entry here is rejected before the operator's own supported-type list is consulted.
enum class EraseDataType
{
    kU8,
    kS8,
    kU16,
    kS16,
    kS32,
    kF16,
    kF32,
    kF64,
};

// Image layouts the operator accepts.  Anything else is rejected while classifying.
enum class EraseFormat
{
    kNHWC,
    kHWC,
    kNCHW,
    kCHW,
};

inline bool IsPlanarFormat(EraseFormat format)
{
    return format == EraseFormat::kNCHW || format == EraseFormat::kCHW;
}

inline EraseDataType ClassifyDataType(nvcv::DataType dtype)
{
    auto bpc = dtype.bitsPerChannel();
    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
        if (bpc[0] == 8)
        {
            return EraseDataType::kU8;
        }
        if (bpc[0] == 16)
        {
            return EraseDataType::kU16;
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ",
                              bpc[0]);

    case nvcv::DataKind::SIGNED:
        if (bpc[0] == 8)
        {
            return EraseDataType::kS8;
        }
        if (bpc[0] == 16)
        {
            return EraseDataType::kS16;
        }
        if (bpc[0] == 32)
        {
            return EraseDataType::kS32;
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc[0]);

    case nvcv::DataKind::FLOAT:
        if (bpc[0] == 64)
        {
            return EraseDataType::kF64;
        }
        if (bpc[0] == 32)
        {
            return EraseDataType::kF32;
        }
        if (bpc[0] == 16)
        {
            return EraseDataType::kF16;
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc[0]);

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only floating-point, signed integer and unsigned integer data kinds are supported ");
    }
}

inline EraseDataType ClassifyDataType(nvcv::ImageFormat fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    return ClassifyDataType(fmt.planeDataType(0));
}

inline bool IsSupportedEraseDataType(EraseDataType dataType)
{
    return dataType == EraseDataType::kU8 || dataType == EraseDataType::kU16 || dataType == EraseDataType::kS16
        || dataType == EraseDataType::kS32 || dataType == EraseDataType::kF32 || dataType == EraseDataType::kF16;
}

inline int ElementSizeBytes(EraseDataType dataType)
{
    switch (dataType)
    {
    case EraseDataType::kU8:
        return 1;
    case EraseDataType::kU16:
    case EraseDataType::kS16:
    case EraseDataType::kF16:
        return 2;
    case EraseDataType::kS32:
    case EraseDataType::kF32:
        return 4;
    default:
        return 0;
    }
}

inline EraseFormat ClassifyLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return EraseFormat::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return EraseFormat::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return EraseFormat::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return EraseFormat::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline EraseFormat ClassifyBatchFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &imgBatch)
{
    nvcv::ImageFormat fmt = imgBatch.uniqueFormat();
    if (!fmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar images must have one channel per plane");
        }

        return imgBatch.numImages() >= 2 ? EraseFormat::kNCHW : EraseFormat::kCHW;
    }

    return imgBatch.numImages() >= 2 ? EraseFormat::kNHWC : EraseFormat::kHWC;
}

// The erase-area descriptors, wrapped once on the host and handed to whichever kernel runs.
struct EraseAreaParams
{
    cuda::Tensor1DWrap<int2>  anchor;
    cuda::Tensor1DWrap<int3>  erasing;
    cuda::Tensor1DWrap<float> values;
    cuda::Tensor1DWrap<int>   imgIdx;
    int                       numErasingArea;
    int                       maxEraseW;
    int                       maxEraseH;
    int                       random;
    unsigned int              seed;
};

__device__ inline int EraseHash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

// The channel axis is the trailing coordinate for interleaved wrappers and the second one for
// planar wrappers; both tensor and var-shape wrappers agree on that spelling.
template<bool IsPlanar, class Wrapper>
__device__ __forceinline__ auto ErasePixelPtr(const Wrapper &img, int batchId, int c, int y, int x)
{
    if constexpr (IsPlanar)
    {
        return img.ptr(batchId, c, y, x);
    }
    else
    {
        return img.ptr(batchId, y, x, c);
    }
}

// The random fill is a pure function of the launch coordinates and the seed, so both the value
// and the mapping from thread to erased pixel must stay exactly as they were for a given seed to
// keep producing the same image.
__device__ __forceinline__ unsigned int EraseRandomHashValue(unsigned int seed)
{
    return seed + threadIdx.x
         + 0x26AD0C9 * blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + 1) * (blockIdx.y + 1) * (blockIdx.z + 1);
}

// Writes one erased pixel.  Shared by both kernels so the fill value -- the bit-exactness-critical
// part -- cannot drift between the tensor and var-shape paths.
template<bool IsPlanar, typename T, class Wrapper>
__device__ __forceinline__ void EraseWritePixel(const Wrapper &img, const EraseAreaParams &params, int batchId, int c,
                                                int y, int x, float value)
{
    if (params.random)
    {
        unsigned int hashValue                          = EraseRandomHashValue(params.seed);
        *ErasePixelPtr<IsPlanar>(img, batchId, c, y, x) = cuda::SaturateCast<T>(EraseHash(hashValue) % 256);
    }
    else
    {
        *ErasePixelPtr<IsPlanar>(img, batchId, c, y, x) = cuda::SaturateCast<T>(value);
    }
}

template<bool IsPlanar, typename T, class Wrapper>
__global__ void EraseTensorKernel(Wrapper img, int imgH, int imgW, EraseAreaParams params, int channels)
{
    unsigned int id      = threadIdx.x + blockIdx.x * blockDim.x;
    int          c       = blockIdx.y;
    int          eraseId = blockIdx.z;
    int2         anchor  = params.anchor[eraseId];
    int3         erasing = params.erasing[eraseId];
    float        value   = params.values[eraseId * channels + c];
    int          batchId = params.imgIdx[eraseId];
    if (id < erasing.y * erasing.x && (0x1 & (erasing.z >> c)) == 1)
    {
        int x = id % erasing.x;
        int y = id / erasing.x;
        if (anchor.x + x < imgW && anchor.y + y < imgH)
        {
            EraseWritePixel<IsPlanar, T>(img, params, batchId, c, anchor.y + y, anchor.x + x, value);
        }
    }
}

template<bool IsPlanar, typename T, class Wrapper>
__global__ void EraseVarShapeKernel(Wrapper img, EraseAreaParams params, int channels)
{
    unsigned int id      = threadIdx.x + blockIdx.x * blockDim.x;
    int          c       = blockIdx.y;
    int          eraseId = blockIdx.z;
    int2         anchor  = params.anchor[eraseId];
    int3         erasing = params.erasing[eraseId];
    float        value   = params.values[eraseId * channels + c];
    int          batchId = params.imgIdx[eraseId];
    if (id < erasing.y * erasing.x && (0x1 & (erasing.z >> c)) == 1)
    {
        int x = id % erasing.x;
        int y = id / erasing.x;
        if ((anchor.x + x) < img.width(batchId) && (anchor.y + y) < img.height(batchId))
        {
            EraseWritePixel<IsPlanar, T>(img, params, batchId, c, anchor.y + y, anchor.x + x, value);
        }
    }
}

// One thread block covers up to this many erased pixels of one (erase area, channel) pair; the
// grid's y and z axes carry the channel and the erase area.  Do not retune: the random fill value
// is a function of blockDim/blockIdx/threadIdx, so any change to the launch geometry changes the
// output image for a given seed.
constexpr int kEraseBlockPixels = 1024;

inline dim3 EraseBlock(const EraseAreaParams &params)
{
    const int area = params.maxEraseH * params.maxEraseW;
    return dim3(area < kEraseBlockPixels ? area : kEraseBlockPixels);
}

inline dim3 EraseGrid(const EraseAreaParams &params, int channels)
{
    return dim3(util::DivUp(params.maxEraseH * params.maxEraseW, kEraseBlockPixels), channels, params.numErasingArea);
}

// Resolves the element type to the compile-time type the kernels are instantiated on and hands a
// value of it to `cb`.  The two types with no arm here (8-bit signed and 64-bit float) are exactly
// the ones validation has already rejected.  F16 keeps a real __half instantiation rather than a
// 16-bit integer alias because the fill value is converted to the image type -- a single
// round-to-nearest step from the float values tensor, or from the integer hash in random mode.
template<class Cb>
inline void DispatchEraseType(EraseDataType dataType, const Cb &cb)
{
    switch (dataType)
    {
    case EraseDataType::kU8:
        return cb(static_cast<unsigned char>(0));
    case EraseDataType::kU16:
        return cb(static_cast<unsigned short>(0));
    case EraseDataType::kS16:
        return cb(static_cast<short>(0));
    case EraseDataType::kS32:
        return cb(0);
    case EraseDataType::kF32:
        return cb(0.0f);
    case EraseDataType::kF16:
        return cb(__half{});
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }
}

template<bool IsPlanar>
inline void DispatchEraseTensor(EraseDataType dataType, const nvcv::TensorDataStridedCuda &imgs,
                                const EraseAreaParams &params, int rows, int cols, int channels, cudaStream_t stream)
{
    const dim3 block = EraseBlock(params);
    const dim3 grid  = EraseGrid(params, channels);

    DispatchEraseType(dataType,
                      [&](auto sample)
                      {
                          using T   = decltype(sample);
                          auto wrap = [&]
                          {
                              if constexpr (IsPlanar)
                              {
                                  return cuda::CreateTensorWrapNCHW<T, int32_t>(imgs);
                              }
                              else
                              {
                                  return cuda::CreateTensorWrapNHWC<T>(imgs);
                              }
                          }();
                          EraseTensorKernel<IsPlanar, T>
                              <<<grid, block, 0, stream>>>(wrap, rows, cols, params, channels);
                      });
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<bool IsPlanar>
inline void DispatchEraseVarShape(EraseDataType dataType, const nvcv::ImageBatchVarShapeDataStridedCuda &imgs,
                                  const EraseAreaParams &params, cudaStream_t stream)
{
    const int  channels = imgs.uniqueFormat().numChannels();
    const dim3 block    = EraseBlock(params);
    const dim3 grid     = EraseGrid(params, channels);

    DispatchEraseType(dataType,
                      [&](auto sample)
                      {
                          using T  = decltype(sample);
                          auto src = [&]
                          {
                              if constexpr (IsPlanar)
                              {
                                  return cuda::ImageBatchVarShapeWrap<T>(imgs);
                              }
                              else
                              {
                                  return cuda::ImageBatchVarShapeWrapNHWC<T>(imgs, channels);
                              }
                          }();
                          EraseVarShapeKernel<IsPlanar, T><<<grid, block, 0, stream>>>(src, params, channels);
                      });
}

inline void ValidateParameterLength(const nvcv::TensorDataStridedCuda &tensor, int expectedLength, const char *name)
{
    if (tensor.shape()[0] < expectedLength)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s length %ld, expected at least %d", name,
                              static_cast<long>(tensor.shape()[0]), expectedLength);
    }
}

// Per-element host-side validation of anchor / imgIdx. The kernel indexes
// img.ptr(imgIdx[i], ...) without a bound check on the batch axis, so an
// out-of-range imgIdx is an unchecked OOB write -- this validation closes
// that hole. It necessarily issues a D->H copy and synchronizes the user's
// stream before launching the erase kernel; the sync is intentional and
// is the cost of throwing a clean invalid-argument exception instead of
// silently corrupting memory.
inline void ValidateEraseAreaData(const nvcv::TensorDataStridedCuda &anchor, const nvcv::TensorDataStridedCuda &imgIdx,
                                  int numErasingArea, int numSamples, cudaStream_t stream)
{
    std::vector<int2> hostAnchor(numErasingArea);
    std::vector<int>  hostImgIdx(numErasingArea);

    // Use cudaMemcpy2DAsync to honor the source tensor's element stride; a
    // rank-1 view of a larger tensor is not guaranteed to be packed.
    NVCV_CHECK_THROW(cudaMemcpy2DAsync(hostAnchor.data(), sizeof(int2), anchor.basePtr(), anchor.stride(0),
                                       sizeof(int2), numErasingArea, cudaMemcpyDeviceToHost, stream));
    NVCV_CHECK_THROW(cudaMemcpy2DAsync(hostImgIdx.data(), sizeof(int), imgIdx.basePtr(), imgIdx.stride(0), sizeof(int),
                                       numErasingArea, cudaMemcpyDeviceToHost, stream));
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream));

    for (int i = 0; i < numErasingArea; ++i)
    {
        if (hostAnchor[i].x < 0 || hostAnchor[i].y < 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid anchor at erase area %d: (%d, %d)", i,
                                  hostAnchor[i].x, hostAnchor[i].y);
        }
        if (hostImgIdx[i] < 0 || hostImgIdx[i] >= numSamples)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid imgIdx at erase area %d: %d, expected [0, %d)", i, hostImgIdx[i],
                                  numSamples);
        }
    }
}

struct MaxWH
{
    __device__ __forceinline__ int3 operator()(const int3 &a, const int3 &b) const
    {
        return int3{max(a.x, b.x), max(a.y, b.y), 0};
    }
};

// Finds the largest erase extent so the kernel launch can be sized once for the whole batch.
inline int2 ReduceMaxEraseExtent(const EraseDeviceScratch &scratch, const nvcv::TensorDataStridedCuda &erasing,
                                 int numErasingArea, cudaStream_t stream)
{
    int3 *d_erasing = reinterpret_cast<int3 *>(erasing.basePtr());
    int3  h_max_values;
    MaxWH maxwh;
    int3  init = {0, 0, 0};

    size_t storageBytes = scratch.storageBytes;
    cub::DeviceReduce::Reduce(scratch.tempStorage, storageBytes, d_erasing, scratch.maxValues, numErasingArea, maxwh,
                              init, stream);
    NVCV_CHECK_THROW(cudaMemcpyAsync(&h_max_values, scratch.maxValues, sizeof(int3), cudaMemcpyDeviceToHost, stream));

    NVCV_CHECK_THROW(cudaStreamSynchronize(stream));

    return int2{h_max_values.x, h_max_values.y};
}

// Single-copy fast path: when both tensors are fully dense the whole batch is one contiguous
// block, so it moves in one cudaMemcpyAsync instead of a 2D copy per sample (per plane, when
// planar).  Returns false when any stride makes that unsound, leaving the caller on the general
// path.
inline bool CopyContiguousTensor(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
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
    NVCV_CHECK_THROW(
        cudaMemcpyAsync(outAccess.sampleData(0), inAccess.sampleData(0), totalBytes, cudaMemcpyDeviceToDevice, stream));
    return true;
}

inline void CopyTensor(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                       const nvcv::TensorDataAccessStridedImagePlanar &outAccess, bool isPlanar, int channels,
                       bool useBulkCopy, cudaStream_t stream)
{
    if (useBulkCopy && CopyContiguousTensor(inAccess, outAccess, isPlanar, channels, stream))
    {
        return;
    }

    // An interleaved sample is a single "plane", so plane 0 lands on the sample base pointer and
    // the loop collapses to the same 2D copy either way.
    const int numPlanes = isPlanar ? channels : 1;

    for (int32_t i = 0; i < inAccess.numSamples(); ++i)
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            nvcv::Byte *inSampData  = inAccess.sampleData(i) + p * inAccess.chStride();
            nvcv::Byte *outSampData = outAccess.sampleData(i) + p * outAccess.chStride();

            NVCV_CHECK_THROW(cudaMemcpy2DAsync(outSampData, outAccess.rowStride(), inSampData, inAccess.rowStride(),
                                               inAccess.numCols() * inAccess.colStride(), inAccess.numRows(),
                                               cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// Four 16-byte vectors per thread; the grid is sized for this so the unrolled loop stays fully
// resident rather than tail-heavy.
constexpr int kCopyVarShapeNIX = 4;

template<int NIX>
__global__ void CopyVarShapeRows(cuda::ImageBatchVarShapeWrap<unsigned char> src,
                                 cuda::ImageBatchVarShapeWrap<unsigned char> dst, int channels, int elementBytes,
                                 bool isPlanar)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
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

// The vectorized row copy needs every row -- and every row stride -- to be a whole number of
// 16-byte vectors, on a 16-byte aligned base, for both batches.
inline bool CanUseVarShapeRowCopy(const nvcv::ImageBatchVarShape &inbatch, const nvcv::ImageBatchVarShape &outbatch,
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

inline bool CopyVarShapeBatch(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
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
    dim3      gridSize(util::DivUp(util::DivUp(rowVecs, kCopyVarShapeNIX), static_cast<int>(blockSize.x)),
                       util::DivUp(maxSize.h, static_cast<int>(blockSize.y)), outData.numImages());

    cuda::ImageBatchVarShapeWrap<unsigned char> src(inData);
    cuda::ImageBatchVarShapeWrap<unsigned char> dst(outData);
    CopyVarShapeRows<kCopyVarShapeNIX><<<gridSize, blockSize, 0, stream>>>(src, dst, channels, elementBytes, isPlanar);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

inline void CopyVarShape(const nvcv::ImageBatchVarShape &inbatch, const nvcv::ImageBatchVarShape &outbatch,
                         const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                         const nvcv::ImageBatchVarShapeDataStridedCuda &outData, bool isPlanar, int channels,
                         int elementBytes, bool useBulkCopy, cudaStream_t stream)
{
    if (useBulkCopy && CanUseVarShapeRowCopy(inbatch, outbatch, isPlanar, channels, elementBytes)
        && CopyVarShapeBatch(inData, outData, isPlanar, channels, elementBytes, stream))
    {
        return;
    }

    for (auto init = inbatch.begin(), outit = outbatch.begin(); init != inbatch.end() && outit != outbatch.end();
         ++init, ++outit)
    {
        const nvcv::Image &inimg      = *init;
        const nvcv::Image &outimg     = *outit;
        auto               inimgdata  = inimg.exportData<nvcv::ImageDataStridedCuda>();
        auto               outimgdata = outimg.exportData<nvcv::ImageDataStridedCuda>();
        for (int p = 0; p < inimgdata->numPlanes(); ++p)
        {
            const nvcv::ImagePlaneStrided &inplane  = inimgdata->plane(p);
            const nvcv::ImagePlaneStrided &outplane = outimgdata->plane(p);
            const size_t rowBytes = static_cast<size_t>(inplane.width) * inimgdata->format().planePixelStrideBytes(p);
            NVCV_CHECK_THROW(cudaMemcpy2DAsync(outplane.basePtr, outplane.rowStride, inplane.basePtr, inplane.rowStride,
                                               rowBytes, inplane.height, cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// Descriptor tensors are validated identically on both paths: 32-bit signed coordinates, 32-bit
// float fill values, all rank 1.
inline void ValidateDescriptorTensor(const nvcv::TensorDataStridedCuda &tensor, EraseDataType expected,
                                     const char *name)
{
    if (ClassifyDataType(tensor.dtype()) != expected)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s DataType", name);
    }
    if (tensor.layout().rank() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s Dim %d", name, tensor.layout().rank());
    }
}

// The order of these checks decides which exception a multiply-invalid call sees, so both paths
// share this one sequence instead of each spelling it out.
inline int ValidateEraseDescriptors(const nvcv::TensorDataStridedCuda &anchor,
                                    const nvcv::TensorDataStridedCuda &erasing,
                                    const nvcv::TensorDataStridedCuda &values,
                                    const nvcv::TensorDataStridedCuda &imgIdx, int maxNumErasingArea)
{
    ValidateDescriptorTensor(anchor, EraseDataType::kS32, "anchor");

    const int numErasingArea = anchor.shape()[0];
    if (numErasingArea < 0 || numErasingArea > maxNumErasingArea)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid num of erasing area %d", numErasingArea);
    }

    ValidateDescriptorTensor(erasing, EraseDataType::kS32, "erasing");
    ValidateParameterLength(erasing, numErasingArea, "erasing");
    ValidateDescriptorTensor(imgIdx, EraseDataType::kS32, "imgIdx");
    ValidateParameterLength(imgIdx, numErasingArea, "imgIdx");
    ValidateDescriptorTensor(values, EraseDataType::kF32, "values");

    return numErasingArea;
}

// Wraps the descriptors and resolves the launch extent in one step, so the returned params are
// complete.  The extent comes from a device reduction, hence the stream.
inline EraseAreaParams MakeEraseAreaParams(const EraseDeviceScratch &scratch, const nvcv::TensorDataStridedCuda &anchor,
                                           const nvcv::TensorDataStridedCuda &erasing,
                                           const nvcv::TensorDataStridedCuda &values,
                                           const nvcv::TensorDataStridedCuda &imgIdx, int numErasingArea, bool random,
                                           unsigned int seed, cudaStream_t stream)
{
    const int2 maxExtent = ReduceMaxEraseExtent(scratch, erasing, numErasingArea, stream);

    EraseAreaParams params{};
    params.anchor         = cuda::Tensor1DWrap<int2>(anchor);
    params.erasing        = cuda::Tensor1DWrap<int3>(erasing);
    params.values         = cuda::Tensor1DWrap<float>(values);
    params.imgIdx         = cuda::Tensor1DWrap<int>(imgIdx);
    params.numErasingArea = numErasingArea;
    params.maxEraseW      = maxExtent.x;
    params.maxEraseH      = maxExtent.y;
    params.random         = random;
    params.seed           = seed;
    return params;
}

// The format/dtype/channel checks below run in a different order than in RunEraseVarShape (there
// the channel check precedes the descriptor checks, here it follows them).  Which exception a
// multiply-invalid call sees is observable and is inherited from the two legacy operators, so the
// two orders must not be "unified".
inline void RunEraseTensor(const EraseDeviceScratch &scratch, const nvcv::TensorDataStridedCuda &inData,
                           const nvcv::TensorDataStridedCuda &outData, const nvcv::TensorDataStridedCuda &anchor,
                           const nvcv::TensorDataStridedCuda &erasing, const nvcv::TensorDataStridedCuda &values,
                           const nvcv::TensorDataStridedCuda &imgIdx, bool random, unsigned int seed, bool inplace,
                           cudaStream_t stream)
{
    const EraseFormat   format      = ClassifyLayout(inData.layout());
    const EraseFormat   outFormat   = ClassifyLayout(outData.layout());
    const EraseDataType dataType    = ClassifyDataType(inData.dtype());
    const EraseDataType outDataType = ClassifyDataType(outData.dtype());

    if (format != outFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }
    const bool isPlanar = IsPlanarFormat(format);

    if (!IsSupportedEraseDataType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }
    if (dataType != outDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "DataType of input and output must be equal");
    }

    const int numErasingArea = ValidateEraseDescriptors(anchor, erasing, values, imgIdx, scratch.maxNumErasingArea);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int channels = inAccess->numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }
    ValidateParameterLength(values, numErasingArea * channels, "values");
    if (numErasingArea > 0)
    {
        ValidateEraseAreaData(anchor, imgIdx, numErasingArea, inAccess->numSamples(), stream);
    }

    if (!inplace)
    {
        CopyTensor(*inAccess, *outAccess, isPlanar, channels, scratch.useBulkCopy, stream);
    }

    if (numErasingArea == 0)
    {
        return;
    }

    const EraseAreaParams params
        = MakeEraseAreaParams(scratch, anchor, erasing, values, imgIdx, numErasingArea, random, seed, stream);

    // All areas as empty? Weird, but valid nonetheless.
    if (params.maxEraseW == 0 || params.maxEraseH == 0)
    {
        return;
    }

    const nvcv::TensorDataStridedCuda              &target       = inplace ? inData : outData;
    const nvcv::TensorDataAccessStridedImagePlanar &targetAccess = inplace ? *inAccess : *outAccess;

    if (isPlanar)
    {
        DispatchEraseTensor<true>(dataType, target, params, targetAccess.numRows(), targetAccess.numCols(),
                                  targetAccess.numChannels(), stream);
    }
    else
    {
        DispatchEraseTensor<false>(dataType, target, params, targetAccess.numRows(), targetAccess.numCols(),
                                   targetAccess.numChannels(), stream);
    }
}

// See the ordering note on RunEraseTensor: this path's channel check deliberately runs before the
// descriptor checks.
inline void RunEraseVarShape(const EraseDeviceScratch &scratch, const nvcv::ImageBatchVarShape &inbatch,
                             const nvcv::ImageBatchVarShape &outbatch, const nvcv::TensorDataStridedCuda &anchor,
                             const nvcv::TensorDataStridedCuda &erasing, const nvcv::TensorDataStridedCuda &values,
                             const nvcv::TensorDataStridedCuda &imgIdx, bool random, unsigned int seed, bool inplace,
                             cudaStream_t stream)
{
    auto inData = inbatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }
    auto outData = outbatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    if (inData->numImages() != outData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output batches must have the same number of images");
    }

    if (!inData->uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in input batch must all have the same format ");
    }
    if (!outData->uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in output batch must all have the same format ");
    }

    const EraseFormat format    = ClassifyBatchFormat(*inData);
    const EraseFormat outFormat = ClassifyBatchFormat(*outData);
    if (format != outFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }
    const bool isPlanar = IsPlanarFormat(format);

    const EraseDataType dataType    = ClassifyDataType(inData->uniqueFormat());
    const EraseDataType outDataType = ClassifyDataType(outData->uniqueFormat());
    if (!IsSupportedEraseDataType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }
    const int elementBytes = ElementSizeBytes(dataType);
    if (dataType != outDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "DataType of input and output must be equal");
    }

    const int channels = inData->uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    const int numErasingArea = ValidateEraseDescriptors(anchor, erasing, values, imgIdx, scratch.maxNumErasingArea);
    ValidateParameterLength(values, numErasingArea * channels, "values");
    if (numErasingArea > 0)
    {
        ValidateEraseAreaData(anchor, imgIdx, numErasingArea, inbatch.numImages(), stream);
    }

    if (!inplace)
    {
        CopyVarShape(inbatch, outbatch, *inData, *outData, isPlanar, channels, elementBytes, scratch.useBulkCopy,
                     stream);
    }

    if (numErasingArea == 0)
    {
        return;
    }

    const EraseAreaParams params
        = MakeEraseAreaParams(scratch, anchor, erasing, values, imgIdx, numErasingArea, random, seed, stream);

    // All areas as empty? Weird, but valid nonetheless.
    if (params.maxEraseW == 0 || params.maxEraseH == 0)
    {
        return;
    }

    const nvcv::ImageBatchVarShapeDataStridedCuda &target = inplace ? *inData : *outData;

    if (isPlanar)
    {
        DispatchEraseVarShape<true>(dataType, target, params, stream);
    }
    else
    {
        DispatchEraseVarShape<false>(dataType, target, params, stream);
    }
}

// The per-device scratch is single-device by design.  PerDeviceResource creates one instance per
// CUDA device for transparent multi-GPU support.  The tensor and var-shape paths keep separate
// scratches so a concurrent call on each does not share the cub temp buffer.
inline PerDeviceResource<EraseDeviceScratch>::Factory MakeEraseScratchFactory(int maxNumErasingArea)
{
    return [maxNumErasingArea](int deviceId)
    {
        return std::make_unique<EraseDeviceScratch>(maxNumErasingArea, UseBulkEraseCopyOnDevice(deviceId));
    };
}

} // namespace

EraseDeviceScratch::EraseDeviceScratch(int maxNumErasingArea, bool useBulkCopy)
    : maxNumErasingArea(maxNumErasingArea)
    , useBulkCopy(useBulkCopy)
{
    NVCV_CHECK_THROW(cudaMalloc(&maxValues, sizeof(int3)));

    MaxWH mwh;
    int3  init = {0, 0, 0};
    cub::DeviceReduce::Reduce(tempStorage, storageBytes, (int3 *)nullptr, (int3 *)nullptr, maxNumErasingArea, mwh,
                              init);

    void             *rawStorage = nullptr;
    const cudaError_t err        = cudaMalloc(&rawStorage, storageBytes);
    if (err != cudaSuccess)
    {
        cudaFree(maxValues);
        maxValues = nullptr;
        NVCV_CHECK_THROW(err);
    }
    tempStorage = static_cast<std::byte *>(rawStorage);
}

EraseDeviceScratch::~EraseDeviceScratch()
{
    cudaFree(maxValues);
    cudaFree(tempStorage);
    maxValues   = nullptr;
    tempStorage = nullptr;
}

Erase::Erase(int num_erasing_area)
    : m_tensorScratch(MakeEraseScratchFactory(num_erasing_area))
    , m_varShapeScratch(MakeEraseScratchFactory(num_erasing_area))
{
    if (num_erasing_area < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "num_erasing_area must be >= 0");
    }
}

void Erase::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &anchor,
                       const nvcv::Tensor &erasing, const nvcv::Tensor &values, const nvcv::Tensor &imgIdx, bool random,
                       unsigned int seed) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Erase::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto anchorData = anchor.exportData<nvcv::TensorDataStridedCuda>();
    if (anchorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "anchor must be cuda-accessible, pitch-linear tensor");
    }

    auto erasingData = erasing.exportData<nvcv::TensorDataStridedCuda>();
    if (erasingData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "erasing must be cuda-accessible, pitch-linear tensor");
    }

    auto valuesData = values.exportData<nvcv::TensorDataStridedCuda>();
    if (valuesData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "values must be cuda-accessible, pitch-linear tensor");
    }

    auto imgIdxData = imgIdx.exportData<nvcv::TensorDataStridedCuda>();
    if (imgIdxData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "imgIdx must be cuda-accessible, pitch-linear tensor");
    }

    bool inplace = (in.handle() == out.handle());
    RunEraseTensor(m_tensorScratch.get(), *inData, *outData, *anchorData, *erasingData, *valuesData, *imgIdxData,
                   random, seed, inplace, stream);
}

void Erase::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                       const nvcv::Tensor &anchor, const nvcv::Tensor &erasing, const nvcv::Tensor &values,
                       const nvcv::Tensor &imgIdx, bool random, unsigned int seed) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Erase::operator()[ImageBatchVarShape]");
    auto anchorData = anchor.exportData<nvcv::TensorDataStridedCuda>();
    if (anchorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "anchor must be cuda-accessible, pitch-linear tensor");
    }

    auto erasingData = erasing.exportData<nvcv::TensorDataStridedCuda>();
    if (erasingData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "erasing must be cuda-accessible, pitch-linear tensor");
    }

    auto valuesData = values.exportData<nvcv::TensorDataStridedCuda>();
    if (valuesData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "values must be cuda-accessible, pitch-linear tensor");
    }

    auto imgIdxData = imgIdx.exportData<nvcv::TensorDataStridedCuda>();
    if (imgIdxData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "imgIdx must be cuda-accessible, pitch-linear tensor");
    }

    bool inplace = (in.handle() == out.handle());
    RunEraseVarShape(m_varShapeScratch.get(), in, out, *anchorData, *erasingData, *valuesData, *imgIdxData, random,
                     seed, inplace, stream);
}

} // namespace cvcuda::priv
