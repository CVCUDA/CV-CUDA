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

#include "Nvtx.hpp"
#include "OpCLAHE.hpp"

#include "legacy/CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace cvcuda::priv {

namespace {

constexpr int kHistBins = 256;

template<typename W>
struct IsBorderVarShapeWrap : std::false_type
{
};

template<typename T, NVCVBorderType B>
struct IsBorderVarShapeWrap<nvcv::cuda::BorderVarShapeWrap<T, B>> : std::true_type
{
};

template<typename W>
struct IsImageBatchVarShapeWrap : std::false_type
{
};

template<typename T>
struct IsImageBatchVarShapeWrap<nvcv::cuda::ImageBatchVarShapeWrap<T>> : std::true_type
{
};

template<class SrcWrapper>
__global__ void CLAHEBuildLUTKernel(SrcWrapper src, unsigned char *luts, float clipLimit, int32_t tilesX,
                                    int32_t tilesY)
{
    const int32_t batch = blockIdx.z;
    const int32_t tx    = blockIdx.x;
    const int32_t ty    = blockIdx.y;
    const int32_t tid   = threadIdx.x;

    using BlockReduce = cub::BlockReduce<unsigned int, kHistBins>;
    using BlockScan   = cub::BlockScan<unsigned int, kHistBins>;

    __shared__ unsigned int                      shHist[kHistBins];
    __shared__ typename BlockReduce::TempStorage shReduce;
    __shared__ typename BlockScan::TempStorage   shScan;
    __shared__ unsigned int                      shExcess;

    shHist[tid] = 0;
    __syncthreads();

    int32_t width;
    int32_t height;
    if constexpr (IsBorderVarShapeWrap<SrcWrapper>::value)
    {
        width  = src.imageBatchWrap().width(batch, 0);
        height = src.imageBatchWrap().height(batch, 0);
    }
    else
    {
        width  = src.tensorShape()[1];
        height = src.tensorShape()[0];
    }
    const int32_t padW  = (tilesX - (width % tilesX)) % tilesX;
    const int32_t padH  = (tilesY - (height % tilesY)) % tilesY;
    const int32_t extW  = width + padW;
    const int32_t extH  = height + padH;
    const int32_t tileW = extW / tilesX;
    const int32_t tileH = extH / tilesY;

    const int32_t x0       = tx * tileW;
    const int32_t y0       = ty * tileH;
    const int32_t tileArea = tileW * tileH;

    for (int32_t idx = tid; idx < tileArea; idx += blockDim.x)
    {
        const int32_t       localY = idx / tileW;
        const int32_t       localX = idx - localY * tileW;
        const unsigned char v      = *(src.ptr(batch, y0 + localY, x0 + localX));
        atomicAdd(&shHist[v], 1U);
    }
    __syncthreads();

    const unsigned int clipCount   = max((unsigned int)(clipLimit * (float)tileArea / (float)kHistBins), 1U);
    unsigned int       localExcess = 0;
    if (shHist[tid] > clipCount)
    {
        localExcess = shHist[tid] - clipCount;
        shHist[tid] = clipCount;
    }
    const unsigned int blockExcess = BlockReduce(shReduce).Sum(localExcess);
    if (tid == 0)
    {
        shExcess = blockExcess;
    }
    __syncthreads();

    const unsigned int redist = shExcess / (unsigned int)kHistBins;
    const unsigned int rem    = shExcess % (unsigned int)kHistBins;
    shHist[tid] += redist;
    if (rem > 0U)
    {
        const unsigned int residualStep = max((unsigned int)(kHistBins / rem), 1U);
        if (((unsigned int)tid % residualStep) == 0U && ((unsigned int)tid / residualStep) < rem)
        {
            shHist[tid] += 1U;
        }
    }
    __syncthreads();

    unsigned int cdf = 0;
    BlockScan(shScan).InclusiveSum(shHist[tid], cdf);
    unsigned int lutV               = (cdf * 255U + (unsigned int)(tileArea / 2)) / (unsigned int)max(tileArea, 1);
    lutV                            = min(lutV, 255U);
    const int32_t tileIdx           = (batch * tilesY + ty) * tilesX + tx;
    luts[tileIdx * kHistBins + tid] = (unsigned char)lutV;
}

template<class SrcWrapper, class DstWrapper>
__global__ void CLAHEApplyKernel(SrcWrapper src, DstWrapper dst, const unsigned char *luts, int32_t tilesX,
                                 int32_t tilesY)
{
    const int32_t batch = blockIdx.z;
    int32_t       width;
    int32_t       height;
    if constexpr (IsImageBatchVarShapeWrap<SrcWrapper>::value)
    {
        width  = src.width(batch);
        height = src.height(batch);
    }
    else
    {
        width  = src.tensorShape()[1];
        height = src.tensorShape()[0];
    }

    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const int32_t padW  = (tilesX - (width % tilesX)) % tilesX;
    const int32_t padH  = (tilesY - (height % tilesY)) % tilesY;
    const int32_t extW  = width + padW;
    const int32_t extH  = height + padH;
    const int32_t tileW = extW / tilesX;
    const int32_t tileH = extH / tilesY;

    float gx = (float)x / (float)tileW - 0.5f;
    float gy = (float)y / (float)tileH - 0.5f;
    gx       = fminf(fmaxf(gx, 0.0f), (float)(tilesX - 1));
    gy       = fminf(fmaxf(gy, 0.0f), (float)(tilesY - 1));

    const int32_t tx0 = (int32_t)floorf(gx);
    const int32_t ty0 = (int32_t)floorf(gy);
    const float   fx  = gx - (float)tx0;
    const float   fy  = gy - (float)ty0;
    const int32_t tx1 = min(tx0 + 1, tilesX - 1);
    const int32_t ty1 = min(ty0 + 1, tilesY - 1);
    const int32_t v   = *(src.ptr(batch, y, x));

    const int32_t base  = batch * tilesY * tilesX * kHistBins;
    const int32_t idx00 = base + ((ty0 * tilesX + tx0) * kHistBins + v);
    const int32_t idx10 = base + ((ty0 * tilesX + tx1) * kHistBins + v);
    const int32_t idx01 = base + ((ty1 * tilesX + tx0) * kHistBins + v);
    const int32_t idx11 = base + ((ty1 * tilesX + tx1) * kHistBins + v);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const float outV
        = w00 * (float)luts[idx00] + w10 * (float)luts[idx10] + w01 * (float)luts[idx01] + w11 * (float)luts[idx11];
    *(dst.ptr(batch, y, x)) = (unsigned char)(outV + 0.5f);
}

// Keep Tensor kernels separate so the original VarShape kernel definitions and code generation stay intact.
template<class SrcWrapper>
__global__ void CLAHETensorBuildLUTKernel(SrcWrapper src, unsigned char *luts, float clipLimit, int32_t tilesX,
                                          int32_t tilesY, int32_t width, int32_t height)
{
    const int32_t batch = blockIdx.z;
    const int32_t tx    = blockIdx.x;
    const int32_t ty    = blockIdx.y;
    const int32_t tid   = threadIdx.x;

    using BlockReduce = cub::BlockReduce<unsigned int, kHistBins>;
    using BlockScan   = cub::BlockScan<unsigned int, kHistBins>;

    __shared__ unsigned int                      shHist[kHistBins];
    __shared__ typename BlockReduce::TempStorage shReduce;
    __shared__ typename BlockScan::TempStorage   shScan;
    __shared__ unsigned int                      shExcess;

    shHist[tid] = 0;
    __syncthreads();

    const int32_t padW  = (tilesX - (width % tilesX)) % tilesX;
    const int32_t padH  = (tilesY - (height % tilesY)) % tilesY;
    const int32_t extW  = width + padW;
    const int32_t extH  = height + padH;
    const int32_t tileW = extW / tilesX;
    const int32_t tileH = extH / tilesY;

    const int32_t x0       = tx * tileW;
    const int32_t y0       = ty * tileH;
    const int32_t tileArea = tileW * tileH;

    for (int32_t idx = tid; idx < tileArea; idx += blockDim.x)
    {
        const int32_t       localY = idx / tileW;
        const int32_t       localX = idx - localY * tileW;
        const unsigned char v      = *(src.ptr(batch, y0 + localY, x0 + localX));
        atomicAdd(&shHist[v], 1U);
    }
    __syncthreads();

    const unsigned int clipCount   = max((unsigned int)(clipLimit * (float)tileArea / (float)kHistBins), 1U);
    unsigned int       localExcess = 0;
    if (shHist[tid] > clipCount)
    {
        localExcess = shHist[tid] - clipCount;
        shHist[tid] = clipCount;
    }
    const unsigned int blockExcess = BlockReduce(shReduce).Sum(localExcess);
    if (tid == 0)
    {
        shExcess = blockExcess;
    }
    __syncthreads();

    const unsigned int redist = shExcess / (unsigned int)kHistBins;
    const unsigned int rem    = shExcess % (unsigned int)kHistBins;
    shHist[tid] += redist;
    if (rem > 0U)
    {
        const unsigned int residualStep = max((unsigned int)(kHistBins / rem), 1U);
        if (((unsigned int)tid % residualStep) == 0U && ((unsigned int)tid / residualStep) < rem)
        {
            shHist[tid] += 1U;
        }
    }
    __syncthreads();

    unsigned int cdf = 0;
    BlockScan(shScan).InclusiveSum(shHist[tid], cdf);
    unsigned int lutV               = (cdf * 255U + (unsigned int)(tileArea / 2)) / (unsigned int)max(tileArea, 1);
    lutV                            = min(lutV, 255U);
    const int32_t tileIdx           = (batch * tilesY + ty) * tilesX + tx;
    luts[tileIdx * kHistBins + tid] = (unsigned char)lutV;
}

template<typename W>
struct IsTensorWrap : std::false_type
{
};

template<typename T, typename StrideT, StrideT... Strides>
struct IsTensorWrap<nvcv::cuda::TensorWrapT<T, StrideT, Strides...>> : std::true_type
{
};

template<class SrcWrapper>
__global__ void CLAHETensorBuildLUTDirectKernel(SrcWrapper src, unsigned char *luts, float clipLimit, int32_t tilesX,
                                                int32_t tilesY, int32_t width, int32_t height)
{
    static_assert(IsTensorWrap<SrcWrapper>::value);

    const int32_t batch = blockIdx.z;
    const int32_t tx    = blockIdx.x;
    const int32_t ty    = blockIdx.y;
    const int32_t tid   = threadIdx.x;

    using BlockReduce = cub::BlockReduce<unsigned int, kHistBins>;
    using BlockScan   = cub::BlockScan<unsigned int, kHistBins>;

    __shared__ unsigned int                      shHist[kHistBins];
    __shared__ typename BlockReduce::TempStorage shReduce;
    __shared__ typename BlockScan::TempStorage   shScan;
    __shared__ unsigned int                      shExcess;

    shHist[tid] = 0;
    __syncthreads();

    const int32_t padW  = (tilesX - (width % tilesX)) % tilesX;
    const int32_t padH  = (tilesY - (height % tilesY)) % tilesY;
    const int32_t extW  = width + padW;
    const int32_t extH  = height + padH;
    const int32_t tileW = extW / tilesX;
    const int32_t tileH = extH / tilesY;

    const int32_t x0       = tx * tileW;
    const int32_t y0       = ty * tileH;
    const int32_t tileArea = tileW * tileH;

    // Keep the 256-thread CUB block while mapping each warp onto one tile row for coalesced reads.
    constexpr int32_t kWarpWidth = 32;
    constexpr int32_t kWarpCount = kHistBins / kWarpWidth;
    const int32_t     virtualX   = tid % kWarpWidth;
    const int32_t     virtualY   = tid / kWarpWidth;

    if (virtualX < tileW && virtualY < tileH)
    {
        const int64_t        rowStride = src.strides()[1];
        const unsigned char *row       = src.ptr(batch, y0 + virtualY, x0);
        for (int32_t localY = virtualY; localY < tileH; localY += kWarpCount)
        {
            const int32_t scalarPrefix = min((int32_t)((-reinterpret_cast<uintptr_t>(row)) & 3U), tileW);
            const int32_t vectorEnd    = scalarPrefix + ((tileW - scalarPrefix) & ~3);

            if (virtualX < scalarPrefix)
            {
                atomicAdd(&shHist[row[virtualX]], 1U);
            }
            // Keep one packed word live per lane; NVCC's default unrolling increases register pressure.
#pragma unroll 1
            for (int32_t localX = scalarPrefix + 4 * virtualX; localX < vectorEnd; localX += 4 * kWarpWidth)
            {
                uint32_t pixels = *reinterpret_cast<const uint32_t *>(row + localX);
                atomicAdd(&shHist[pixels & 0xFFU], 1U);
                pixels >>= 8;
                atomicAdd(&shHist[pixels & 0xFFU], 1U);
                pixels >>= 8;
                atomicAdd(&shHist[pixels & 0xFFU], 1U);
                pixels >>= 8;
                atomicAdd(&shHist[pixels], 1U);
            }
            if (virtualX < tileW - vectorEnd)
            {
                atomicAdd(&shHist[row[vectorEnd + virtualX]], 1U);
            }
            if (localY + kWarpCount < tileH)
            {
                row += kWarpCount * rowStride;
            }
        }
    }
    __syncthreads();

    const unsigned int clipCount   = max((unsigned int)(clipLimit * (float)tileArea / (float)kHistBins), 1U);
    unsigned int       localExcess = 0;
    if (shHist[tid] > clipCount)
    {
        localExcess = shHist[tid] - clipCount;
        shHist[tid] = clipCount;
    }
    const unsigned int blockExcess = BlockReduce(shReduce).Sum(localExcess);
    if (tid == 0)
    {
        shExcess = blockExcess;
    }
    __syncthreads();

    const unsigned int redist = shExcess / (unsigned int)kHistBins;
    const unsigned int rem    = shExcess % (unsigned int)kHistBins;
    shHist[tid] += redist;
    if (rem > 0U)
    {
        const unsigned int residualStep = max((unsigned int)(kHistBins / rem), 1U);
        if (((unsigned int)tid % residualStep) == 0U && ((unsigned int)tid / residualStep) < rem)
        {
            shHist[tid] += 1U;
        }
    }
    __syncthreads();

    unsigned int cdf = 0;
    BlockScan(shScan).InclusiveSum(shHist[tid], cdf);
    unsigned int lutV               = (cdf * 255U + (unsigned int)(tileArea / 2)) / (unsigned int)max(tileArea, 1);
    lutV                            = min(lutV, 255U);
    const int32_t tileIdx           = (batch * tilesY + ty) * tilesX + tx;
    luts[tileIdx * kHistBins + tid] = (unsigned char)lutV;
}

template<class SrcWrapper, class DstWrapper>
__global__ void CLAHETensorApplyKernel(SrcWrapper src, DstWrapper dst, const unsigned char *luts, int32_t tilesX,
                                       int32_t tilesY, int32_t width, int32_t height, float invTileW, float invTileH)
{
    const int32_t batch = blockIdx.z;
    const int32_t x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y     = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    float gx = __fmul_rn((float)x, invTileW) - 0.5f;
    float gy = __fmul_rn((float)y, invTileH) - 0.5f;
    gx       = fminf(fmaxf(gx, 0.0f), (float)(tilesX - 1));
    gy       = fminf(fmaxf(gy, 0.0f), (float)(tilesY - 1));

    const int32_t tx0 = (int32_t)floorf(gx);
    const int32_t ty0 = (int32_t)floorf(gy);
    const float   fx  = gx - (float)tx0;
    const float   fy  = gy - (float)ty0;
    const int32_t tx1 = min(tx0 + 1, tilesX - 1);
    const int32_t ty1 = min(ty0 + 1, tilesY - 1);
    const int32_t v   = *(src.ptr(batch, y, x));

    const int32_t base  = batch * tilesY * tilesX * kHistBins;
    const int32_t idx00 = base + ((ty0 * tilesX + tx0) * kHistBins + v);
    const int32_t idx10 = base + ((ty0 * tilesX + tx1) * kHistBins + v);
    const int32_t idx01 = base + ((ty1 * tilesX + tx0) * kHistBins + v);
    const int32_t idx11 = base + ((ty1 * tilesX + tx1) * kHistBins + v);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const float outV
        = w00 * (float)luts[idx00] + w10 * (float)luts[idx10] + w01 * (float)luts[idx01] + w11 * (float)luts[idx11];
    *(dst.ptr(batch, y, x)) = (unsigned char)(outV + 0.5f);
}

static void ValidateTileGrid(int32_t tilesX, int32_t tilesY)
{
    if (tilesX < 1 || tilesY < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "tilesX and tilesY must be >= 1");
    }
}

static void ValidateClipLimit(float clipLimit)
{
    if (!(clipLimit > 0.0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "clipLimit must be > 0");
    }
}

static void ValidateGridZ(int64_t z, const char *what)
{
    if (z > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be <= 65535", what);
    }
}

static void RunCLAHETensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           float clipLimit, int32_t tilesX, int32_t tilesY, unsigned char *dLUTs, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!inAccess || !outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must be pitch-linear image tensors");
    }

    if (inData.layout() != outData.layout() || inData.dtype() != outData.dtype()
        || inAccess->numSamples() != outAccess->numSamples() || inAccess->numRows() != outAccess->numRows()
        || inAccess->numCols() != outAccess->numCols() || inAccess->numChannels() != outAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output tensor must match");
    }

    const bool isPlanar      = inData.layout() == nvcv::TENSOR_NCHW || inData.layout() == nvcv::TENSOR_CHW;
    const bool isInterleaved = inData.layout() == nvcv::TENSOR_NHWC || inData.layout() == nvcv::TENSOR_HWC;
    if (!isInterleaved && !isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have (N)HWC or (N)CHW layout");
    }

    if (inAccess->numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "CLAHE supports only single-channel tensors");
    }

    if (inData.dtype() != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "CLAHE supports only U8 tensors");
    }

    auto src       = nvcv::cuda::CreateTensorWrapNHW<const unsigned char, int64_t>(inData);
    auto srcBorder = nvcv::cuda::CreateBorderWrapNHW<const unsigned char, NVCV_BORDER_REFLECT101, int64_t>(inData);
    auto dst       = nvcv::cuda::CreateTensorWrapNHW<unsigned char, int64_t>(outData);

    const int32_t batch  = inAccess->numSamples();
    const int32_t width  = inAccess->numCols();
    const int32_t height = inAccess->numRows();
    ValidateGridZ(batch, "Tensor batch");

    const int32_t padW     = (tilesX - (width % tilesX)) % tilesX;
    const int32_t padH     = (tilesY - (height % tilesY)) % tilesY;
    const int32_t tileW    = (width + padW) / tilesX;
    const int32_t tileH    = (height + padH) / tilesY;
    const float   invTileW = 1.0f / (float)tileW;
    const float   invTileH = 1.0f / (float)tileH;

    const dim3 lutGrid(tilesX, tilesY, batch);
    const dim3 lutBlock(kHistBins, 1, 1);
    if (width % tilesX == 0 && height % tilesY == 0)
    {
        CLAHETensorBuildLUTDirectKernel<<<lutGrid, lutBlock, 0, stream>>>(src, dLUTs, clipLimit, tilesX, tilesY, width,
                                                                          height);
    }
    else
    {
        CLAHETensorBuildLUTKernel<<<lutGrid, lutBlock, 0, stream>>>(srcBorder, dLUTs, clipLimit, tilesX, tilesY, width,
                                                                    height);
    }
    checkKernelErrors();

    const dim3 applyBlock(32, 8, 1);
    const dim3 applyGrid((width + applyBlock.x - 1) / applyBlock.x, (height + applyBlock.y - 1) / applyBlock.y, batch);
    CLAHETensorApplyKernel<<<applyGrid, applyBlock, 0, stream>>>(src, dst, dLUTs, tilesX, tilesY, width, height,
                                                                 invTileW, invTileH);
    checkKernelErrors();
}

static void ValidateVarShapeFormat(const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out)
{
    if (in.numImages() != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have same number of images");
    }

    for (int i = 0; i < in.numImages(); ++i)
    {
        if (in[i].size() != out[i].size() || in[i].format() != out[i].format())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output varshape images must match in size and format");
        }

        const auto fmt = in[i].format();
        if (fmt.planeNumChannels(0) != 1 || fmt.planeDataType(0) != nvcv::TYPE_U8)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "CLAHE varshape supports only single-channel U8");
        }
    }
}

static void RunCLAHEVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &outData, float clipLimit, int32_t tilesX,
                             int32_t tilesY, unsigned char *dLUTs, cudaStream_t stream)
{
    ValidateGridZ(inData.numImages(), "Varshape batch");

    nvcv::cuda::BorderVarShapeWrap<const unsigned char, NVCV_BORDER_REFLECT101> srcBorder(inData);
    nvcv::cuda::ImageBatchVarShapeWrap<const unsigned char>                     src(inData);
    nvcv::cuda::ImageBatchVarShapeWrap<unsigned char>                           dst(outData);

    const dim3 lutGrid(tilesX, tilesY, inData.numImages());
    const dim3 lutBlock(kHistBins, 1, 1);
    CLAHEBuildLUTKernel<<<lutGrid, lutBlock, 0, stream>>>(srcBorder, dLUTs, clipLimit, tilesX, tilesY);
    checkKernelErrors();

    const dim3 applyBlock(16, 16, 1);
    const dim3 applyGrid((inData.maxSize().w + applyBlock.x - 1) / applyBlock.x,
                         (inData.maxSize().h + applyBlock.y - 1) / applyBlock.y, inData.numImages());
    CLAHEApplyKernel<<<applyGrid, applyBlock, 0, stream>>>(src, dst, dLUTs, tilesX, tilesY);
    checkKernelErrors();
}

} // namespace

CLAHE::CLAHE(int32_t maxBatchSize, int32_t tilesX, int32_t tilesY)
    : m_maxBatchSize(maxBatchSize)
    , m_tilesX(tilesX)
    , m_tilesY(tilesY)
    , m_deviceBuffers([maxBatchSize, tilesX, tilesY](int)
                      { return std::make_unique<CLAHEDeviceBuffers>(maxBatchSize, tilesX, tilesY); })
{
    if (m_maxBatchSize < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 1");
    }
    ValidateTileGrid(m_tilesX, m_tilesY);
}

void CLAHE::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float clipLimit) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CLAHE::operator()[Tensor]");
    ValidateClipLimit(clipLimit);

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (!outData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be image-compatible tensor");
    }
    if (inAccess->numSamples() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    RunCLAHETensor(*inData, *outData, clipLimit, m_tilesX, m_tilesY, m_deviceBuffers.get().luts, stream);
}

void CLAHE::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                       float clipLimit) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CLAHE::operator()[ImageBatchVarShape]");
    ValidateClipLimit(clipLimit);

    if (in.numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    ValidateVarShapeFormat(in, out);

    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }
    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!outData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    RunCLAHEVarShape(*inData, *outData, clipLimit, m_tilesX, m_tilesY, m_deviceBuffers.get().luts, stream);
}

} // namespace cvcuda::priv
