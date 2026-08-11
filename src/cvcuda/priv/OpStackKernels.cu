/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "OpStackKernels.hpp"

#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstdint>

namespace cvcuda::priv {

namespace {

constexpr int kBlockSize      = 256;
constexpr int kBytesPerThread = 16;

static __device__ __forceinline__ void CopyBytes(unsigned char *dst, const unsigned char *src, int64_t offset,
                                                 int64_t rowBytes)
{
    if (offset + kBytesPerThread <= rowBytes
        && ((reinterpret_cast<uintptr_t>(dst + offset) | reinterpret_cast<uintptr_t>(src + offset))
            & (alignof(uint4) - 1))
               == 0)
    {
        *reinterpret_cast<uint4 *>(dst + offset) = *reinterpret_cast<const uint4 *>(src + offset);
        return;
    }

#pragma unroll
    for (int byte = 0; byte < kBytesPerThread; ++byte)
    {
        if (offset + byte < rowBytes)
        {
            dst[offset + byte] = src[offset + byte];
        }
    }
}

__global__ void StackTensorBatchRows(const NVCVTensorBatchElementStrided *in, nvcv::Byte *out, int64_t outSampleStride,
                                     int64_t outPlaneStride, int64_t outRowStride, int64_t rowBytes, int32_t numPlanes,
                                     bool isPlanar)
{
    const int32_t sample = blockIdx.z / numPlanes;
    const int32_t plane  = blockIdx.z - sample * numPlanes;
    const int32_t row    = blockIdx.y;
    const int64_t offset = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kBytesPerThread;

    const NVCVTensorBatchElementStrided &tensor       = in[sample];
    const int32_t                        rowDimension = isPlanar ? 1 : 0;
    const unsigned char                 *inRow        = reinterpret_cast<const unsigned char *>(
        tensor.data + static_cast<int64_t>(plane) * (isPlanar ? tensor.stride[0] : 0)
        + static_cast<int64_t>(row) * tensor.stride[rowDimension]);
    unsigned char *outRow = reinterpret_cast<unsigned char *>(out + static_cast<int64_t>(sample) * outSampleStride
                                                              + static_cast<int64_t>(plane) * outPlaneStride
                                                              + static_cast<int64_t>(row) * outRowStride);

    CopyBytes(outRow, inRow, offset, rowBytes);
}

__global__ void StackVarShapeRows(const NVCVImageBufferStrided *in, nvcv::Byte *out, int64_t outSampleStride,
                                  int64_t outPlaneStride, int64_t outRowStride, int64_t rowBytes, int32_t numPlanes)
{
    const int32_t sample = blockIdx.z / numPlanes;
    const int32_t plane  = blockIdx.z - sample * numPlanes;
    const int32_t row    = blockIdx.y;
    const int64_t offset = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kBytesPerThread;

    const NVCVImagePlaneStrided &inPlane = in[sample].planes[plane];
    const unsigned char         *inRow
        = reinterpret_cast<const unsigned char *>(inPlane.basePtr + static_cast<int64_t>(row) * inPlane.rowStride);
    unsigned char *outRow = reinterpret_cast<unsigned char *>(out + static_cast<int64_t>(sample) * outSampleStride
                                                              + static_cast<int64_t>(plane) * outPlaneStride
                                                              + static_cast<int64_t>(row) * outRowStride);

    CopyBytes(outRow, inRow, offset, rowBytes);
}

static bool MakeGrid(dim3 &grid, int64_t rowBytes, int32_t numRows, int64_t numSamplePlanes)
{
    constexpr int32_t kMaxGridYZ = 65535;
    if (rowBytes <= 0 || numRows <= 0 || numRows > kMaxGridYZ || numSamplePlanes <= 0 || numSamplePlanes > kMaxGridYZ)
    {
        return false;
    }

    const int64_t bytesPerBlock = static_cast<int64_t>(kBlockSize) * kBytesPerThread;
    const int64_t numBlocks     = (rowBytes + bytesPerBlock - 1) / bytesPerBlock;
    if (numBlocks > INT32_MAX)
    {
        return false;
    }

    grid = dim3(static_cast<uint32_t>(numBlocks), numRows, static_cast<uint32_t>(numSamplePlanes));
    return true;
}

} // namespace

bool RunStackTensorBatchKernel(cudaStream_t stream, const nvcv::TensorBatchDataStridedCuda &inData,
                               const nvcv::TensorDataStridedCuda &outData)
{
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!out || inData.rank() != 3)
    {
        return false;
    }

    const int32_t numSamples = inData.numTensors();
    const int32_t numPlanes  = out->numPlanes();
    const int64_t rowBytes   = static_cast<int64_t>(out->numCols()) * out->colStride();
    dim3          grid;
    if (!MakeGrid(grid, rowBytes, out->numRows(), static_cast<int64_t>(numSamples) * numPlanes))
    {
        return false;
    }

    const bool isPlanar = outData.layout() == nvcv::TENSOR_NCHW;
    StackTensorBatchRows<<<grid, kBlockSize, 0, stream>>>(inData.buffer().tensors, outData.basePtr(),
                                                          out->sampleStride(), out->planeStride(), out->rowStride(),
                                                          rowBytes, numPlanes, isPlanar);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

bool RunStackVarShapeKernel(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::TensorDataStridedCuda &outData)
{
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!out)
    {
        return false;
    }

    const int32_t numSamples = inData.numImages();
    const int32_t numPlanes  = out->numPlanes();
    const int64_t rowBytes   = static_cast<int64_t>(out->numCols()) * out->colStride();
    dim3          grid;
    if (!MakeGrid(grid, rowBytes, out->numRows(), static_cast<int64_t>(numSamples) * numPlanes))
    {
        return false;
    }

    StackVarShapeRows<<<grid, kBlockSize, 0, stream>>>(inData.imageList(), outData.basePtr(), out->sampleStride(),
                                                       out->planeStride(), out->rowStride(), rowBytes, numPlanes);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

} // namespace cvcuda::priv
