/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpResize.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/Printer.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/Math.hpp>

namespace {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

// Destination pack type given the source and destination type T
template<typename T>
using DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;

// Row alignment mask to determine if data pointer is aligned for vectorized write:
// note that in CUDA, uint3 has 4-byte (uint) alignment.
template<typename T>
constexpr uint MSK = (sizeof(DPT<T>) == sizeof(uint3) ? sizeof(uint) : sizeof(DPT<T>)) - 1;

// Number of items in x written by each thread
template<typename T>
constexpr int NIX = sizeof(DPT<T>) / sizeof(T);

// Write a pack of N elements of type T as a different pack type DPT
template<typename T>
__device__ __forceinline__ void WritePack(T &u, const T (&v)[NIX<T>])
{
    reinterpret_cast<DPT<T> &>(u) = reinterpret_cast<const DPT<T> &>(v);
}

// Check if destination row pointer is aligned for vector writes.
template<typename T>
__device__ __forceinline__ bool CheckRowAlign(T *row)
{
    return (static_cast<uint>(reinterpret_cast<size_t>(row)) & MSK<T>) == 0;
}

// Nearest ---------------------------------------------------------------------

template<bool INTERSECT, typename T, class SrcWrapper>
inline __device__ void NearestInterpolatePack(T *dstRow, SrcWrapper src, int3 iSrcCoord, float srcCoordX, int srcSizeX,
                                              int dstCoordX, int dstSizeX, float scaleRatioX)
{
    int iPrevCoordX;
    T   srcPack;

    if (dstCoordX + NIX<T> - 1 < dstSizeX)
    {
        T dstPack[NIX<T>];
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            iSrcCoord.x = floor(srcCoordX + x * scaleRatioX);
            iSrcCoord.x = cuda::min(iSrcCoord.x, srcSizeX - 1);

            if constexpr (INTERSECT)
            {
                if (x == 0 || iSrcCoord.x != iPrevCoordX)
                {
                    srcPack = src[iSrcCoord];
                }

                dstPack[x] = srcPack;

                iPrevCoordX = iSrcCoord.x;
            }
            else
            {
                dstPack[x] = src[iSrcCoord];
            }
        }

        if (CheckRowAlign(dstRow))                 // Branch is the same for all threads in warp.
            WritePack(dstRow[dstCoordX], dstPack); // If row is aligned, write vector pack;
        else
        {
            T *dstPtr = dstRow + dstCoordX; // otherwise, write individual elements.
#pragma unroll
            for (uint i = 0; i < NIX<T>; ++i) dstPtr[i] = dstPack[i];
        }
        // writePack(dstRow + dstCoordX, dstPack);
    }
    else
    {
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            if (dstCoordX + x < dstSizeX)
            {
                iSrcCoord.x = floor(srcCoordX + x * scaleRatioX);
                iSrcCoord.x = cuda::min(iSrcCoord.x, srcSizeX - 1);

                if constexpr (INTERSECT)
                {
                    if (x == 0 || iSrcCoord.x != iPrevCoordX)
                    {
                        srcPack = src[iSrcCoord];
                    }

                    dstRow[dstCoordX + x] = srcPack;

                    iPrevCoordX = iSrcCoord.x;
                }
                else
                {
                    dstRow[dstCoordX + x] = src[iSrcCoord];
                }
            }
        }
    }
}

template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__global__ void NearestResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T = typename DstWrapper::ValueType;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (dstCoord.y < dstSize.y)
    {
        dstCoord.x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

        float2 srcCoord = (cuda::DropCast<2>(dstCoord) + 0.5f) * scaleRatio;
        int3   iSrcCoord{0, (int)floor(srcCoord.y), dstCoord.z};

        iSrcCoord.y = cuda::min(iSrcCoord.y, srcSize.y - 1);

        T *dstRow = dst.ptr(dstCoord.z, dstCoord.y);

        NearestInterpolatePack<INTERSECT>(dstRow, src, iSrcCoord, srcCoord.x, srcSize.x, dstCoord.x, dstSize.x,
                                          scaleRatio.x);
    }
}

// Linear ----------------------------------------------------------------------

template<class SrcWrapper, typename T>
inline __device__ void LinearReadPack(SrcWrapper src, T (&srcPack)[4], int3 iSrcCoord)
{
    srcPack[0] = src[int3{iSrcCoord.x, iSrcCoord.y, iSrcCoord.z}];
    srcPack[1] = src[int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}];
    srcPack[2] = src[int3{iSrcCoord.x, iSrcCoord.y + 1, iSrcCoord.z}];
    srcPack[3] = src[int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}];
}

template<bool INTERSECT, typename T, class SrcWrapper>
inline __device__ void LinearInterpolatePack(T *dstRow, SrcWrapper src, int3 iSrcCoord, float srcCoordX, int srcSizeX,
                                             int dstCoordX, int dstSizeX, float scaleRatioX, float2 w)
{
    float sx;
    int   iPrevCoordX;
    T     srcPack[4];

    if (dstCoordX + NIX<T> - 1 < dstSizeX)
    {
        T dstPack[NIX<T>];
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            sx          = srcCoordX + x * scaleRatioX;
            iSrcCoord.x = floor(sx);

            w.x = ((iSrcCoord.x < 0) ? 0 : ((iSrcCoord.x > srcSizeX - 2) ? 1 : sx - iSrcCoord.x));

            iSrcCoord.x = cuda::max(0, cuda::min(iSrcCoord.x, srcSizeX - 2));

            if constexpr (INTERSECT)
            {
                if (x == 0)
                {
                    LinearReadPack(src, srcPack, iSrcCoord);
                }
                else
                {
                    if (iSrcCoord.x != iPrevCoordX)
                    {
                        if (iSrcCoord.x == (iPrevCoordX + 1))
                        {
                            srcPack[0] = srcPack[1];
                            srcPack[2] = srcPack[3];
                            srcPack[1] = src[int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}];
                            srcPack[3] = src[int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}];
                        }
                        else
                        {
                            LinearReadPack(src, srcPack, iSrcCoord);
                        }
                    }
                }
                dstPack[x]
                    = cuda::SaturateCast<T>(srcPack[0] * ((1.f - w.x) * (1.f - w.y)) + srcPack[1] * (w.x * (1.f - w.y))
                                            + srcPack[2] * ((1.f - w.x) * w.y) + srcPack[3] * (w.x * w.y));

                iPrevCoordX = iSrcCoord.x;
            }
            else
            {
                dstPack[x] = cuda::SaturateCast<T>(
                    src[int3{iSrcCoord.x, iSrcCoord.y, iSrcCoord.z}] * ((1.f - w.x) * (1.f - w.y))
                    + src[int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}] * (w.x * (1.f - w.y))
                    + src[int3{iSrcCoord.x, iSrcCoord.y + 1, iSrcCoord.z}] * ((1.f - w.x) * w.y)
                    + src[int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}] * (w.x * w.y));
            }
        }

        if (CheckRowAlign(dstRow))                 // Branch is the same for all threads in warp.
            WritePack(dstRow[dstCoordX], dstPack); // If row is aligned, write vector pack;
        else
        {
            T *dstPtr = dstRow + dstCoordX; // otherwise, write individual elements.
#pragma unroll
            for (uint i = 0; i < NIX<T>; ++i) dstPtr[i] = dstPack[i];
        }
        // writePack<true>(dstRow + dstCoordX, dstPack, reinterpret_cast<uint>(dstRow) & DstMask) == 0);
    }
    else
    {
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            if (dstCoordX + x < dstSizeX)
            {
                sx          = srcCoordX + x * scaleRatioX;
                iSrcCoord.x = floor(sx);

                w.x = ((iSrcCoord.x < 0) ? 0 : ((iSrcCoord.x > srcSizeX - 2) ? 1 : sx - iSrcCoord.x));

                iSrcCoord.x = cuda::max(0, cuda::min(iSrcCoord.x, srcSizeX - 2));

                if constexpr (INTERSECT)
                {
                    if (x == 0)
                    {
                        LinearReadPack(src, srcPack, iSrcCoord);
                    }
                    else
                    {
                        if (iSrcCoord.x != iPrevCoordX)
                        {
                            if (iSrcCoord.x == (iPrevCoordX + 1))
                            {
                                srcPack[0] = srcPack[1];
                                srcPack[2] = srcPack[3];
                                srcPack[1] = src[int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}];
                                srcPack[3] = src[int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}];
                            }
                            else
                            {
                                LinearReadPack(src, srcPack, iSrcCoord);
                            }
                        }
                    }
                    dstRow[dstCoordX + x] = cuda::SaturateCast<T>(
                        srcPack[0] * ((1.f - w.x) * (1.f - w.y)) + srcPack[1] * (w.x * (1.f - w.y))
                        + srcPack[2] * ((1.f - w.x) * w.y) + srcPack[3] * (w.x * w.y));

                    iPrevCoordX = iSrcCoord.x;
                }
                else
                {
                    dstRow[dstCoordX + x] = cuda::SaturateCast<T>(
                        src[int3{iSrcCoord.x, iSrcCoord.y, iSrcCoord.z}] * ((1.f - w.x) * (1.f - w.y))
                        + src[int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}] * (w.x * (1.f - w.y))
                        + src[int3{iSrcCoord.x, iSrcCoord.y + 1, iSrcCoord.z}] * ((1.f - w.x) * w.y)
                        + src[int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}] * (w.x * w.y));
                }
            }
        }
    }
}

template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__global__ void LinearResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T = typename DstWrapper::ValueType;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (dstCoord.y < dstSize.y)
    {
        dstCoord.x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

        float2 srcCoord = (cuda::DropCast<2>(dstCoord) + .5f) * scaleRatio - .5f;
        int3   iSrcCoord{0, (int)floor(srcCoord.y), dstCoord.z};

        float2 w;

        w.y = ((iSrcCoord.y < 0) ? 0 : ((iSrcCoord.y > srcSize.y - 2) ? 1 : srcCoord.y - iSrcCoord.y));

        iSrcCoord.y = cuda::max(0, cuda::min(iSrcCoord.y, srcSize.y - 2));

        T *dstRow = dst.ptr(dstCoord.z, dstCoord.y);

        LinearInterpolatePack<INTERSECT>(dstRow, src, iSrcCoord, srcCoord.x, srcSize.x, dstCoord.x, dstSize.x,
                                         scaleRatio.x, w);
    }
}

// Cubic -----------------------------------------------------------------------

inline __device__ void GetCubicCoeffs(float delta, float &w0, float &w1, float &w2, float &w3)
{
    constexpr float A = -0.75f;

    w0 = ((A * (delta + 1) - 5 * A) * (delta + 1) + 8 * A) * (delta + 1) - 4 * A;
    w1 = ((A + 2) * delta - (A + 3)) * delta * delta + 1;
    w2 = ((A + 2) * (1 - delta) - (A + 3)) * (1 - delta) * (1 - delta) + 1;
    w3 = 1.f - w0 - w1 - w2;
}

template<class SrcWrapper, class DstWrapper>
__global__ void CubicResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = blockIdx.y * blockDim.y + threadIdx.y;
    dstCoord.x = blockIdx.x * blockDim.x + threadIdx.x;

    if (dstCoord.y < dstSize.y && dstCoord.x < dstSize.x)
    {
        const float2 srcCoord = (cuda::DropCast<2>(dstCoord) + .5f) * scaleRatio - .5f;
        int3         baseCoord{(int)floor(srcCoord.x), (int)floor(srcCoord.y), dstCoord.z};

        const float fx = srcCoord.x - baseCoord.x;
        const float fy = srcCoord.y - baseCoord.y;

        const int xMax = srcSize.x - 1;
        const int yMax = srcSize.y - 1;

        float wx[4];
        float wy[4];

        GetCubicCoeffs(fx, wx[0], wx[1], wx[2], wx[3]);
        GetCubicCoeffs(fy, wy[0], wy[1], wy[2], wy[3]);

        FT sum = FT{};

#pragma unroll
        for (int cy = -1; cy <= 2; cy++)
        {
            const int sy = cuda::min(cuda::max(baseCoord.y + cy, 0), yMax);
#pragma unroll
            for (int cx = -1; cx <= 2; cx++)
            {
                const int sx = cuda::min(cuda::max(baseCoord.x + cx, 0), xMax);

                sum += src[int3{sx, sy, baseCoord.z}] * (wx[cx + 1] * wy[cy + 1]);
            }
        }

        dst[dstCoord] = cuda::SaturateCast<T>(cuda::abs(sum));
    }
}

// Area ------------------------------------------------------------------------

template<class SrcWrapper, class DstWrapper>
__global__ void AreaResize(SrcWrapper src, DstWrapper dst, int2 dstSize)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= dstSize.x || y >= dstSize.y)
        return;

    int3 coord{x, y, z};

    dst[coord] = src[cuda::StaticCast<float>(coord)];
}

// Host run resize functions ---------------------------------------------------

template<typename T>
void RunResizeInterp(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                     const nvcv::TensorDataStridedCuda &dstData, int2 srcSize, int2 dstSize, int batchSize,
                     const NVCVInterpolationType interpolation)
{
    float2 scaleRatio{(float)srcSize.x / dstSize.x, (float)srcSize.y / dstSize.y};

    auto srcTW = cuda::CreateTensorWrapNHW<const T, int32_t>(srcData);
    auto dstTW = cuda::CreateTensorWrapNHW<T, int32_t>(dstData);
    auto srcIW = cuda::CreateInterpolationWrapNHW<const T, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA, int32_t>(
        srcData, T{}, scaleRatio.x, scaleRatio.y);

    dim3 threads1(32, 4, 1);
    dim3 blocks1(util::DivUp(dstSize.x, threads1.x * NIX<T>), util::DivUp(dstSize.y, threads1.y), batchSize);

    dim3 threads2(128, 1, 1);
    dim3 blocks2(util::DivUp(dstSize.x, threads2.x), util::DivUp(dstSize.y, threads2.y), batchSize);

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        if (scaleRatio.x < 1)
            NearestResize<true><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        else
            NearestResize<false><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        break;

    case NVCV_INTERP_LINEAR:
        if (scaleRatio.x < 2)
            LinearResize<true><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        else
            LinearResize<false><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        break;

    case NVCV_INTERP_CUBIC:
        CubicResize<<<blocks2, threads2, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        break;

    case NVCV_INTERP_AREA:
        AreaResize<<<blocks2, threads2, 0, stream>>>(srcIW, dstTW, dstSize);
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation");
    }
}

inline void RunResizeInterpType(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                const nvcv::TensorDataStridedCuda &dstData, int2 srcSize, int2 dstSize, int numChannels,
                                int batchSize, const NVCVInterpolationType interpolation)
{
    // The data type may contain the channels baked in or the number of channels is in the tensor shape

    // clang-format off

#define CVCUDA_RUN_RESIZE(BT, DT, T)                                             \
    ((srcData.dtype() == nvcv::TYPE_##BT && numChannels == cuda::NumElements<T>) \
     || (srcData.dtype() == nvcv::TYPE_##DT && numChannels == 1))                \
        RunResizeInterp<T>(stream, srcData, dstData, srcSize, dstSize, batchSize, interpolation);

    if CVCUDA_RUN_RESIZE(U8, U8, uchar1)
    else if CVCUDA_RUN_RESIZE(U8, 3U8, uchar3)
    else if CVCUDA_RUN_RESIZE(U8, 4U8, uchar4)
    else if CVCUDA_RUN_RESIZE(U16, U16, ushort)
    else if CVCUDA_RUN_RESIZE(U16, 3U16, ushort3)
    else if CVCUDA_RUN_RESIZE(U16, 4U16, ushort4)
    else if CVCUDA_RUN_RESIZE(S16, S16, short)
    else if CVCUDA_RUN_RESIZE(S16, 3S16, short3)
    else if CVCUDA_RUN_RESIZE(S16, 4S16, short4)
    else if CVCUDA_RUN_RESIZE(F32, F32, float)
    else if CVCUDA_RUN_RESIZE(F32, 3F32, float3)
    else if CVCUDA_RUN_RESIZE(F32, 4F32, float4)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }

#undef CVCUDA_RUN_RESIZE

    // clang-format on
}

} // anonymous namespace

namespace cvcuda::priv {

// Tensor operator -------------------------------------------------------------

void Resize::RunResize(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                       const nvcv::TensorDataStridedCuda &dstData, const NVCVInterpolationType interpolation) const
{
    if (srcData.dtype() != dstData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output data type are different");
    }
    if (srcData.layout() != dstData.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output data layout are different");
    }
    if (srcData.layout() != nvcv::TENSOR_HWC && srcData.layout() != nvcv::TENSOR_NHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC layout");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output samples are different");
    }
    if (srcAccess->numChannels() != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output channels are different");
    }
    if (srcAccess->numChannels() > 4 || srcAccess->numChannels() < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of channels");
    }

    constexpr int32_t kIntMax = cuda::TypeTraits<int32_t>::max;

    int64_t srcMaxStride = srcAccess->sampleStride() * srcAccess->numSamples();
    int64_t dstMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) > kIntMax || srcAccess->numSamples() > kIntMax
        || srcAccess->numCols() > kIntMax || srcAccess->numRows() > kIntMax || dstAccess->numCols() > kIntMax
        || dstAccess->numRows() > kIntMax)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input or output tensors are too large");
    }

    int  numChannels{(int)srcAccess->numChannels()};
    int  batchSize{(int)srcAccess->numSamples()};
    int2 srcSize{(int)srcAccess->numCols(), (int)srcAccess->numRows()};
    int2 dstSize{(int)dstAccess->numCols(), (int)dstAccess->numRows()};

    RunResizeInterpType(stream, srcData, dstData, srcSize, dstSize, numChannels, batchSize, interpolation);
}

} // namespace cvcuda::priv
