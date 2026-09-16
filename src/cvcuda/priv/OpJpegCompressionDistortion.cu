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

// Kernel structure ported from NVIDIA DALI's jpeg_distortion_gpu_impl.cuh (Apache-2.0,
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES), fixed to 4:2:0 chroma subsampling,
// with quantization tables built on device from a per-image or per-batch quality value and DALI's
// BlockSetup tiling replaced by a chroma-plane grid with one 32x16-chroma region per CUDA block.

#include "JpegDistortionMath.hpp"
#include "OpJpegCompressionDistortion.hpp"

#include "ChannelAxisCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>

namespace cuda         = nvcv::cuda;
namespace util         = nvcv::util;
namespace channel_axis = cvcuda::priv::channel_axis;
namespace jpeg         = cvcuda::priv::jpeg;

namespace {

// One 32x16-thread block covers a 32x16 chroma region = 64x32 luma pixels in 4:2:0.
constexpr int kBlockWidth  = 32;
constexpr int kBlockHeight = 16;

// Per-batch or per-image quality: a packed rank-1 S32 tensor when present, a scalar otherwise.
struct QualityParam
{
    cuda::Tensor1DWrap<const int32_t> perSample;
    int32_t                           scalar;
    bool                              hasTensor;

    inline __device__ int32_t at(int z) const
    {
        return hasTensor ? *perSample.ptr(z) : scalar;
    }
};

// Pixel accessors: coordinates are pre-clamped by the caller; z is the sample/image index.
// The VarShape accessors also expose per-image dimensions.

template<class SrcWrapper, class DstWrapper>
struct InterleavedColorIO
{
    SrcWrapper src;
    DstWrapper dst;

    inline __device__ uchar3 load(int z, int y, int x) const
    {
        return src[int3{x, y, z}];
    }

    inline __device__ void store(int z, int y, int x, uchar3 v) const
    {
        dst[int3{x, y, z}] = v;
    }
};

template<class SrcWrapper, class DstWrapper>
struct PlanarColorIO
{
    SrcWrapper src;
    DstWrapper dst;

    inline __device__ uchar3 load(int z, int y, int x) const
    {
        return uchar3{*src.ptr(z, 0, y, x), *src.ptr(z, 1, y, x), *src.ptr(z, 2, y, x)};
    }

    inline __device__ void store(int z, int y, int x, uchar3 v) const
    {
        *dst.ptr(z, 0, y, x) = v.x;
        *dst.ptr(z, 1, y, x) = v.y;
        *dst.ptr(z, 2, y, x) = v.z;
    }
};

template<class SrcWrapper, class DstWrapper>
struct InterleavedGrayIO
{
    SrcWrapper src;
    DstWrapper dst;

    inline __device__ uint8_t load(int z, int y, int x) const
    {
        return src[int3{x, y, z}].x;
    }

    inline __device__ void store(int z, int y, int x, uint8_t v) const
    {
        dst[int3{x, y, z}] = uchar1{v};
    }
};

template<class SrcWrapper, class DstWrapper>
struct PlanarGrayIO
{
    SrcWrapper src;
    DstWrapper dst;

    inline __device__ uint8_t load(int z, int y, int x) const
    {
        return *src.ptr(z, 0, y, x);
    }

    inline __device__ void store(int z, int y, int x, uint8_t v) const
    {
        *dst.ptr(z, 0, y, x) = v;
    }
};

// Color (4:2:0) block pipeline -----------------------------------------------------------
//
// Each thread owns one chroma pixel and its 2x2 luma quad. Loads are clamped to the image (edge
// replication, DALI's NN sampler + BorderClamp), so the whole region's 8x8 blocks are populated
// even past the image edge; stores are bounds-guarded per pixel.

template<class IO>
inline __device__ void JpegDistortionColorBlock(const IO &io, int2 size, int z, QualityParam quality)
{
    constexpr int kChromaPerRow = kBlockWidth / 8;  // 4 chroma blocks across
    constexpr int kChromaRows   = kBlockHeight / 8; // 2 chroma blocks down
    constexpr int kChromaBlocks = kChromaPerRow * kChromaRows;
    constexpr int kLumaPerRow   = 2 * kChromaPerRow;
    constexpr int kLumaBlocks   = 4 * kChromaBlocks;
    constexpr int kTotalBlocks  = 2 * kChromaBlocks + kLumaBlocks; // cb + cr + luma = 48

    // The 9-wide padding avoids shared-memory bank conflicts between row and column DCT passes.
    __shared__ float sBlocks[kTotalBlocks][8][9];
    __shared__ float sQuant[2][64]; // [0] luma, [1] chroma

    const int chromaW      = (size.x + 1) >> 1;
    const int chromaH      = (size.y + 1) >> 1;
    const int chromaStartX = static_cast<int>(blockIdx.x) * kBlockWidth;
    const int chromaStartY = static_cast<int>(blockIdx.y) * kBlockHeight;
    // The grid covers the batch's maximum extent; a block fully outside this image's chroma plane
    // has nothing to do. The condition is uniform across the block, before any __syncthreads().
    if (chromaStartX >= chromaW || chromaStartY >= chromaH)
    {
        return;
    }

    const int tx  = static_cast<int>(threadIdx.x);
    const int ty  = static_cast<int>(threadIdx.y);
    const int tid = ty * kBlockWidth + tx;

    // Scaled quantization tables for this sample (device-side equivalent of DALI's host build).
    const float scale = jpeg::QuantScale(quality.at(z));
    if (tid < 128)
    {
        const uint8_t base         = tid < 64 ? jpeg::kLumaQuantBase[tid] : jpeg::kChromaQuantBase[tid - 64];
        sQuant[tid >> 6][tid & 63] = jpeg::QuantTableEntry(scale, base);
    }

    // Clamped loads of the 2x2 luma quad; chroma from the RGB average (average first, convert after).
    const int lxg  = (chromaStartX + tx) << 1;
    const int lyg  = (chromaStartY + ty) << 1;
    const int xmax = size.x - 1;
    const int ymax = size.y - 1;

    const uchar3 p00 = io.load(z, ::min(lyg, ymax), ::min(lxg, xmax));
    const uchar3 p01 = io.load(z, ::min(lyg, ymax), ::min(lxg + 1, xmax));
    const uchar3 p10 = io.load(z, ::min(lyg + 1, ymax), ::min(lxg, xmax));
    const uchar3 p11 = io.load(z, ::min(lyg + 1, ymax), ::min(lxg + 1, xmax));

    const uchar3 avg = jpeg::Avg4(p00, p01, p10, p11);

    const int cBlk = (ty >> 3) * kChromaPerRow + (tx >> 3);
    const int cy   = ty & 7;
    const int cx   = tx & 7;
    const int lx2  = tx << 1;
    const int ly2  = ty << 1;
    const int lBlk = (ly2 >> 3) * kLumaPerRow + (lx2 >> 3);
    const int ly   = ly2 & 7;
    const int lx   = lx2 & 7;

    float(*cb)[8][9]   = &sBlocks[cBlk];
    float(*cr)[8][9]   = &sBlocks[kChromaBlocks + cBlk];
    float(*luma)[8][9] = &sBlocks[2 * kChromaBlocks + lBlk];

    // Level shift to [-128, 127] before the DCT.
    (*cb)[cy][cx]           = static_cast<float>(jpeg::RgbToCb(avg)) - 128.0f;
    (*cr)[cy][cx]           = static_cast<float>(jpeg::RgbToCr(avg)) - 128.0f;
    (*luma)[ly][lx]         = static_cast<float>(jpeg::RgbToY(p00)) - 128.0f;
    (*luma)[ly][lx + 1]     = static_cast<float>(jpeg::RgbToY(p01)) - 128.0f;
    (*luma)[ly + 1][lx]     = static_cast<float>(jpeg::RgbToY(p10)) - 128.0f;
    (*luma)[ly + 1][lx + 1] = static_cast<float>(jpeg::RgbToY(p11)) - 128.0f;

    __syncthreads();

    constexpr int kSlices   = kTotalBlocks * 8;
    constexpr int kNThreads = kBlockWidth * kBlockHeight;

    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::FwdDct8<1>(&sBlocks[s >> 3][s & 7][0]);
    }
    __syncthreads();
    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::FwdDct8<9>(&sBlocks[s >> 3][0][s & 7]);
    }
    __syncthreads();

    const float chromaQ = sQuant[1][(cy << 3) + cx];
    (*cb)[cy][cx]       = jpeg::Quantize((*cb)[cy][cx], chromaQ);
    (*cr)[cy][cx]       = jpeg::Quantize((*cr)[cy][cx], chromaQ);

    (*luma)[ly][lx]         = jpeg::Quantize((*luma)[ly][lx], sQuant[0][(ly << 3) + lx]);
    (*luma)[ly][lx + 1]     = jpeg::Quantize((*luma)[ly][lx + 1], sQuant[0][(ly << 3) + lx + 1]);
    (*luma)[ly + 1][lx]     = jpeg::Quantize((*luma)[ly + 1][lx], sQuant[0][((ly + 1) << 3) + lx]);
    (*luma)[ly + 1][lx + 1] = jpeg::Quantize((*luma)[ly + 1][lx + 1], sQuant[0][((ly + 1) << 3) + lx + 1]);
    __syncthreads();

    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::InvDct8<9>(&sBlocks[s >> 3][0][s & 7]);
    }
    __syncthreads();
    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::InvDct8<1>(&sBlocks[s >> 3][s & 7][0]);
    }
    __syncthreads();

    // Level shift back, replicate the reconstructed chroma over the quad, store in-bounds pixels.
    const uint8_t ocb = jpeg::SatCastU8((*cb)[cy][cx] + 128.0f);
    const uint8_t ocr = jpeg::SatCastU8((*cr)[cy][cx] + 128.0f);

    if (lyg <= ymax && lxg <= xmax)
    {
        io.store(z, lyg, lxg, jpeg::YCbCrToRgb(jpeg::SatCastU8((*luma)[ly][lx] + 128.0f), ocb, ocr));
    }
    if (lyg <= ymax && lxg + 1 <= xmax)
    {
        io.store(z, lyg, lxg + 1, jpeg::YCbCrToRgb(jpeg::SatCastU8((*luma)[ly][lx + 1] + 128.0f), ocb, ocr));
    }
    if (lyg + 1 <= ymax && lxg <= xmax)
    {
        io.store(z, lyg + 1, lxg, jpeg::YCbCrToRgb(jpeg::SatCastU8((*luma)[ly + 1][lx] + 128.0f), ocb, ocr));
    }
    if (lyg + 1 <= ymax && lxg + 1 <= xmax)
    {
        io.store(z, lyg + 1, lxg + 1, jpeg::YCbCrToRgb(jpeg::SatCastU8((*luma)[ly + 1][lx + 1] + 128.0f), ocb, ocr));
    }
}

// Grayscale block pipeline ---------------------------------------------------------------
//
// The single channel is the luma plane: level shift, DCT, luma-table quantization, inverse DCT.
// No color conversion and no chroma path. One thread per pixel over a 32x16 luma region.

template<class IO>
inline __device__ void JpegDistortionGrayBlock(const IO &io, int2 size, int z, QualityParam quality)
{
    constexpr int kBlocksPerRow = kBlockWidth / 8;                    // 4
    constexpr int kBlockCount   = kBlocksPerRow * (kBlockHeight / 8); // 8

    __shared__ float sBlocks[kBlockCount][8][9];
    __shared__ float sQuant[64];

    const int startX = static_cast<int>(blockIdx.x) * kBlockWidth;
    const int startY = static_cast<int>(blockIdx.y) * kBlockHeight;
    if (startX >= size.x || startY >= size.y)
    {
        return;
    }

    const int tx  = static_cast<int>(threadIdx.x);
    const int ty  = static_cast<int>(threadIdx.y);
    const int tid = ty * kBlockWidth + tx;

    const float scale = jpeg::QuantScale(quality.at(z));
    if (tid < 64)
    {
        sQuant[tid] = jpeg::QuantTableEntry(scale, jpeg::kLumaQuantBase[tid]);
    }

    const int px = startX + tx;
    const int py = startY + ty;

    const uint8_t v = io.load(z, ::min(py, size.y - 1), ::min(px, size.x - 1));

    const int blk = ((ty >> 3) * kBlocksPerRow) + (tx >> 3);
    const int by  = ty & 7;
    const int bx  = tx & 7;

    sBlocks[blk][by][bx] = static_cast<float>(v) - 128.0f;
    __syncthreads();

    constexpr int kSlices   = kBlockCount * 8;
    constexpr int kNThreads = kBlockWidth * kBlockHeight;

    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::FwdDct8<1>(&sBlocks[s >> 3][s & 7][0]);
    }
    __syncthreads();
    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::FwdDct8<9>(&sBlocks[s >> 3][0][s & 7]);
    }
    __syncthreads();

    sBlocks[blk][by][bx] = jpeg::Quantize(sBlocks[blk][by][bx], sQuant[(by << 3) + bx]);
    __syncthreads();

    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::InvDct8<9>(&sBlocks[s >> 3][0][s & 7]);
    }
    __syncthreads();
    for (int s = tid; s < kSlices; s += kNThreads)
    {
        jpeg::InvDct8<1>(&sBlocks[s >> 3][s & 7][0]);
    }
    __syncthreads();

    if (px < size.x && py < size.y)
    {
        io.store(z, py, px, jpeg::SatCastU8(sBlocks[blk][by][bx] + 128.0f));
    }
}

// Kernel shells ---------------------------------------------------------------------------

template<class IO>
__global__ void JpegDistortionColorKernel(IO io, int2 size, QualityParam quality)
{
    JpegDistortionColorBlock(io, size, static_cast<int>(blockIdx.z), quality);
}

template<class IO>
__global__ void JpegDistortionColorVarShapeKernel(IO io, QualityParam quality)
{
    const int  z = static_cast<int>(blockIdx.z);
    const int2 size{io.dst.width(z), io.dst.height(z)};
    JpegDistortionColorBlock(io, size, z, quality);
}

template<class IO>
__global__ void JpegDistortionGrayKernel(IO io, int2 size, QualityParam quality)
{
    JpegDistortionGrayBlock(io, size, static_cast<int>(blockIdx.z), quality);
}

template<class IO>
__global__ void JpegDistortionGrayVarShapeKernel(IO io, QualityParam quality)
{
    const int  z = static_cast<int>(blockIdx.z);
    const int2 size{io.dst.width(z), io.dst.height(z)};
    JpegDistortionGrayBlock(io, size, z, quality);
}

// Launchers -------------------------------------------------------------------------------

constexpr int kGridYZLimit = 65535;

inline void CheckBatchLimit(int numSamples)
{
    if (numSamples > kGridYZLimit)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }
}

// The color grid tiles the chroma plane (half resolution); the grayscale grid tiles the luma plane.
inline dim3 MakeGrid(int width, int height, int numSamples, bool isColor)
{
    const int gridW = isColor ? (width + 1) >> 1 : width;
    const int gridH = isColor ? (height + 1) >> 1 : height;
    const int gridY = util::DivUp(gridH, kBlockHeight);
    if (gridY > kGridYZLimit)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image height exceeds the CUDA grid.y limit of 65535");
    }
    return dim3(util::DivUp(gridW, kBlockWidth), gridY, static_cast<unsigned int>(numSamples));
}

inline void RunTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                      const nvcv::TensorDataStridedCuda &dstData, bool isPlanar, int numChannels,
                      const QualityParam &quality)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    const int2 size = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
    CheckBatchLimit(static_cast<int>(srcAccess->numSamples()));

    const int64_t inMaxStride  = srcAccess->sampleStride() * srcAccess->numSamples();
    const int64_t outMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    const dim3 block(kBlockWidth, kBlockHeight, 1);
    const dim3 grid = MakeGrid(size.x, size.y, static_cast<int>(srcAccess->numSamples()), numChannels == 3);

    if (numChannels == 3)
    {
        if (isPlanar)
        {
            auto src = cuda::CreateTensorWrapNCHW<const uint8_t, int32_t>(srcData);
            auto dst = cuda::CreateTensorWrapNCHW<uint8_t, int32_t>(dstData);
            PlanarColorIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionColorKernel<<<grid, block, 0, stream>>>(io, size, quality);
        }
        else
        {
            auto src = cuda::CreateTensorWrapNHW<const uchar3, int32_t>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<uchar3, int32_t>(dstData);
            InterleavedColorIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionColorKernel<<<grid, block, 0, stream>>>(io, size, quality);
        }
    }
    else
    {
        if (isPlanar)
        {
            auto src = cuda::CreateTensorWrapNCHW<const uint8_t, int32_t>(srcData);
            auto dst = cuda::CreateTensorWrapNCHW<uint8_t, int32_t>(dstData);
            PlanarGrayIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionGrayKernel<<<grid, block, 0, stream>>>(io, size, quality);
        }
        else
        {
            auto src = cuda::CreateTensorWrapNHW<const uchar1, int32_t>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<uchar1, int32_t>(dstData);
            InterleavedGrayIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionGrayKernel<<<grid, block, 0, stream>>>(io, size, quality);
        }
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void RunVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, bool isPlanar, int numChannels,
                        const QualityParam &quality)
{
    CheckBatchLimit(dstData.numImages());

    const int3 maxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
    const dim3 block(kBlockWidth, kBlockHeight, 1);
    const dim3 grid = MakeGrid(maxSize.x, maxSize.y, maxSize.z, numChannels == 3);

    if (numChannels == 3)
    {
        if (isPlanar)
        {
            cuda::ImageBatchVarShapeWrap<const uint8_t> src(srcData);
            cuda::ImageBatchVarShapeWrap<uint8_t>       dst(dstData);
            PlanarColorIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionColorVarShapeKernel<<<grid, block, 0, stream>>>(io, quality);
        }
        else
        {
            cuda::ImageBatchVarShapeWrap<const uchar3>       src(srcData);
            cuda::ImageBatchVarShapeWrap<uchar3>             dst(dstData);
            InterleavedColorIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionColorVarShapeKernel<<<grid, block, 0, stream>>>(io, quality);
        }
    }
    else
    {
        if (isPlanar)
        {
            cuda::ImageBatchVarShapeWrap<const uint8_t> src(srcData);
            cuda::ImageBatchVarShapeWrap<uint8_t>       dst(dstData);
            PlanarGrayIO<decltype(src), decltype(dst)>  io{src, dst};
            JpegDistortionGrayVarShapeKernel<<<grid, block, 0, stream>>>(io, quality);
        }
        else
        {
            cuda::ImageBatchVarShapeWrap<const uchar1>      src(srcData);
            cuda::ImageBatchVarShapeWrap<uchar1>            dst(dstData);
            InterleavedGrayIO<decltype(src), decltype(dst)> io{src, dst};
            JpegDistortionGrayVarShapeKernel<<<grid, block, 0, stream>>>(io, quality);
        }
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Validation ------------------------------------------------------------------------------

inline void ValidateScalarQuality(int quality)
{
    if (quality < 1 || quality > 100)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The quality must be in [1, 100]");
    }
}

// Per-image quality tensor: packed rank-1 S32 with exactly one value per image. Values are not
// validated on host (the data lives on device); the kernel clamps them to [1, 100].
inline QualityParam ValidateQualityTensor(const nvcv::Tensor &quality, int numSamples)
{
    auto qualityData = quality.exportData<nvcv::TensorDataStridedCuda>();
    if (!qualityData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Quality must be a cuda-accessible, pitch-linear tensor");
    }
    if (qualityData->rank() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Quality tensor must be rank-1");
    }
    if (qualityData->dtype() != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Quality tensor data type must be S32");
    }
    if (qualityData->shape(0) != numSamples)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Quality tensor must have one value per image: expected %d, got %d", numSamples,
                              static_cast<int>(qualityData->shape(0)));
    }
    if (qualityData->stride(0) != sizeof(int32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Quality tensor must be packed");
    }

    QualityParam param{};
    param.perSample = cuda::Tensor1DWrap<const int32_t>(reinterpret_cast<const int32_t *>(qualityData->basePtr()));
    param.scalar    = 0;
    param.hasTensor = true;
    return param;
}

inline QualityParam MakeScalarQuality(int quality)
{
    ValidateScalarQuality(quality);

    QualityParam param{};
    param.scalar    = quality;
    param.hasTensor = false;
    return param;
}

} // anonymous namespace

namespace cvcuda::priv {

JpegCompressionDistortion::JpegCompressionDistortion() {}

void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                           const nvcv::Tensor &quality) const
{
    bool           isEmpty;
    int            numChannels;
    int            numSamples;
    nvcv::DataType dtype;
    auto           srcData = in.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = out.exportData<nvcv::TensorDataStridedCuda>();
    const bool     isPlanar
        = channel_axis::ValidateSrcDstTensors(isEmpty, numChannels, dtype, numSamples, srcData, dstData);
    if (dtype != nvcv::TYPE_U8 && dtype != nvcv::TYPE_3U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: JpegCompressionDistortion supports 8-bit unsigned only");
    }

    const QualityParam param = ValidateQualityTensor(quality, numSamples);
    if (isEmpty)
    {
        return;
    }

    RunTensor(stream, *srcData, *dstData, isPlanar, numChannels, param);
}

void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                           int quality) const
{
    const QualityParam param = MakeScalarQuality(quality);

    bool           isEmpty;
    int            numChannels;
    int            numSamples;
    nvcv::DataType dtype;
    auto           srcData = in.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = out.exportData<nvcv::TensorDataStridedCuda>();
    const bool     isPlanar
        = channel_axis::ValidateSrcDstTensors(isEmpty, numChannels, dtype, numSamples, srcData, dstData);
    if (dtype != nvcv::TYPE_U8 && dtype != nvcv::TYPE_3U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: JpegCompressionDistortion supports 8-bit unsigned only");
    }
    if (isEmpty)
    {
        return;
    }

    RunTensor(stream, *srcData, *dstData, isPlanar, numChannels, param);
}

void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                           const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &quality) const
{
    bool           isEmpty;
    int            numChannels;
    nvcv::DataType dtype;
    auto           srcData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    auto           dstData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    const bool isPlanar = channel_axis::ValidateSrcDstVarBatch(isEmpty, numChannels, dtype, in, out, srcData, dstData);
    if (isEmpty)
    {
        return;
    }
    if (dtype != nvcv::TYPE_U8 && dtype != nvcv::TYPE_3U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: JpegCompressionDistortion supports 8-bit unsigned only");
    }

    const QualityParam param = ValidateQualityTensor(quality, srcData->numImages());

    RunVarShape(stream, *srcData, *dstData, isPlanar, numChannels, param);
}

void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                           const nvcv::ImageBatchVarShape &out, int quality) const
{
    const QualityParam param = MakeScalarQuality(quality);

    bool           isEmpty;
    int            numChannels;
    nvcv::DataType dtype;
    auto           srcData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    auto           dstData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    const bool isPlanar = channel_axis::ValidateSrcDstVarBatch(isEmpty, numChannels, dtype, in, out, srcData, dstData);
    if (isEmpty)
    {
        return;
    }
    if (dtype != nvcv::TYPE_U8 && dtype != nvcv::TYPE_3U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: JpegCompressionDistortion supports 8-bit unsigned only");
    }

    RunVarShape(stream, *srcData, *dstData, isPlanar, numChannels, param);
}

} // namespace cvcuda::priv
