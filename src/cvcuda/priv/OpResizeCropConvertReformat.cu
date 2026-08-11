/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "OpResizeCropConvertReformat.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/InterpolationVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/Math.hpp>

#include <limits> // for numeric_limits
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace cuda_op = nvcv::legacy::cuda_op;
namespace helpers = nvcv::legacy::helpers;

namespace {

// clang-format off

template <uint N>
uchar4 remapC(const NVCVChannelManip manip)
{
    static_assert(N > 0 && N <= 4, "Number of remap channels must be >= 1 and <= 4.");

    if (manip == NVCV_CHANNEL_REVERSE)
    {
        if constexpr (N == 1) return uchar4{0, 0, 0, 0};
        if constexpr (N == 2) return uchar4{1, 0, 0, 0};
        if constexpr (N == 3) return uchar4{2, 1, 0, 0};
        if constexpr (N == 4) return uchar4{3, 2, 1, 0};
    }
    return uchar4{0, 1, 2, 3};
}

template<typename DstT, uint N>
class DstMap {
public:
    using DstType = DstT;

    static_assert(N > 0 && N <= 4, "Number of DstMap channels must be >= 1 and <= 4.");

    DstMap(DstT *dst, size_t addN, int addH, int addW, size_t addC,
           const NVCVChannelManip manip, int width, int height)
          : m_dst {dst},
            m_addN{addN},
            m_addY{addH},
            m_addX{addW},
            m_wdth{width},
            m_hght{height} {_init(addC, remapC<N>(manip)); }

    __device__ __forceinline__
    int width() const { return m_wdth; }

    __device__ __forceinline__
    int height() const { return m_hght; }

    __device__ __forceinline__
    DstT *ptr(const uint n, const int y, const int x) { return m_dst + n * m_addN + (y * m_addY + x * m_addX); }

    template <typename SrcT, class = cuda::Require<cuda::HasTypeTraits<SrcT> > >
    __device__ __forceinline__
    void operator()(const uint n, const int y, const int x, const SrcT val)
    {
        static_assert(cuda::NumElements<SrcT> == N);

        // Set destination pointer to correct pixel (batch, row, & column).
        DstT *dst = ptr(n, y, x);

        // Shuffle pixel channels.
        if constexpr (cuda::NumComponents<SrcT> > 1) {
            dst[m_mapC[0]] = cuda::SaturateCast<DstT>(val.x);
            dst[m_mapC[1]] = cuda::SaturateCast<DstT>(val.y);
            if constexpr (N >= 3) dst[m_mapC[2]] = cuda::SaturateCast<DstT>(val.z);
            if constexpr (N == 4) dst[m_mapC[3]] = cuda::SaturateCast<DstT>(val.w);
        }
        else if constexpr (cuda::NumComponents<SrcT> == 1)
             *dst = cuda::SaturateCast<DstT>(val.x);
        else *dst = cuda::SaturateCast<DstT>(val);
    }

private:
    void _init(size_t addC, uchar4 mapC)
    {
        m_mapC[0] = mapC.x * addC;
        if constexpr (N >= 2) m_mapC[1] = mapC.y * addC;
        if constexpr (N >= 3) m_mapC[2] = mapC.z * addC;
        if constexpr (N == 4) m_mapC[3] = mapC.w * addC;
    }

    size_t m_mapC[N];
    size_t m_addN;
    int    m_addY, m_addX;
    int    m_wdth, m_hght;
    DstT  *m_dst;
};

template<typename PixelT, typename BaseT>
__device__ __forceinline__ PixelT MakePlanarPixel(const BaseT *plane0, int64_t planeStride)
{
    static_assert(cuda::NumElements<PixelT> == 1 || cuda::NumElements<PixelT> == 3);

    if constexpr (cuda::NumElements<PixelT> == 1)
    {
        return PixelT{plane0[0]};
    }
    else
    {
        return PixelT{plane0[0], plane0[planeStride], plane0[2 * planeStride]};
    }
}

template<typename SrcT>
class TensorPlanarSrc
{
public:
    using ValueType = SrcT;
    using BaseT     = cuda::BaseType<SrcT>;

    TensorPlanarSrc(const nvcv::TensorDataStridedCuda &srcData,
                    const nvcv::TensorDataAccessStridedImagePlanar &srcAccess)
        : m_src{reinterpret_cast<const BaseT *>(srcData.basePtr())}
        , m_sampleStride{srcAccess.sampleStride() / static_cast<int64_t>(sizeof(BaseT))}
        , m_planeStride{srcAccess.planeStride() / static_cast<int64_t>(sizeof(BaseT))}
        , m_rowStride{srcAccess.rowStride() / static_cast<int64_t>(sizeof(BaseT))}
        , m_colStride{srcAccess.colStride() / static_cast<int64_t>(sizeof(BaseT))}
    {
    }

    __device__ __forceinline__ SrcT read(int n, int y, int x) const
    {
        const BaseT *plane0 = m_src + n * m_sampleStride + y * m_rowStride + x * m_colStride;
        return MakePlanarPixel<SrcT>(plane0, m_planeStride);
    }

private:
    const BaseT *m_src;
    int64_t      m_sampleStride;
    int64_t      m_planeStride;
    int64_t      m_rowStride;
    int64_t      m_colStride;
};

template<typename SrcT>
class ImageBatchVarShapePlanarSrc
{
public:
    using ValueType = SrcT;
    using BaseT     = cuda::BaseType<SrcT>;

    explicit ImageBatchVarShapePlanarSrc(const nvcv::ImageBatchVarShapeDataStridedCuda &srcData)
        : m_src{srcData}
    {
    }

    __host__ __device__ int width(int s) const
    {
        return m_src.width(s, 0);
    }

    __host__ __device__ int height(int s) const
    {
        return m_src.height(s, 0);
    }

    __device__ __forceinline__ SrcT read(int n, int y, int x) const
    {
        const BaseT *plane0 = m_src.ptr(n, 0, y, x);

        if constexpr (cuda::NumElements<SrcT> == 1)
        {
            return SrcT{plane0[0]};
        }
        else
        {
            return SrcT{plane0[0], *m_src.ptr(n, 1, y, x), *m_src.ptr(n, 2, y, x)};
        }
    }

private:
    cuda::ImageBatchVarShapeWrap<const BaseT> m_src;
};

template<class SrcWrapper>
__device__ __forceinline__ std::remove_cv_t<typename SrcWrapper::ValueType>
ReadPixel(const SrcWrapper &src, int n, int y, int x)
{
    return *src.ptr(n, y, x);
}

template<typename SrcT>
__device__ __forceinline__ SrcT ReadPixel(const TensorPlanarSrc<SrcT> &src, int n, int y, int x)
{
    return src.read(n, y, x);
}

template<typename SrcT>
__device__ __forceinline__ SrcT ReadPixel(const ImageBatchVarShapePlanarSrc<SrcT> &src, int n, int y, int x)
{
    return src.read(n, y, x);
}

template<class DstMap, class SrcWrapper>
__device__ __forceinline__ void WriteBilinearPixel(DstMap dst, const SrcWrapper &src, int n, int dstY, int dstX,
                                                   float fx, float fy, int srcW, int srcH, float scale, float offset,
                                                   bool srcCast)
{
    using SrcT = std::remove_cv_t<typename SrcWrapper::ValueType>;

    int sx0 = __float2int_rd(fx);
    int sy0 = __float2int_rd(fy);
    int sx1 = cuda::min(sx0 + 1, srcW - 1);
    int sy1 = cuda::min(sy0 + 1, srcH - 1);

    fx -= sx0;
    fy -= sy0;
    sx0 = cuda::max(0, sx0);
    sy0 = cuda::max(0, sy0);
    sx1 = sx1 > sx0 ? sx1 : sx0;

    const SrcT src00 = ReadPixel(src, n, sy0, sx0);
    const SrcT src01 = ReadPixel(src, n, sy0, sx1);
    const SrcT src10 = ReadPixel(src, n, sy1, sx0);
    const SrcT src11 = ReadPixel(src, n, sy1, sx1);

    if (srcCast)
    {
        dst(n, dstY, dstX, scale * cuda::SaturateCast<SrcT>((1-fy) * ((1-fx) * src00 + src01 * fx)
                                                            + fy  * ((1-fx) * src10 + src11 * fx)) + offset);
    }
    else
    {
        dst(n, dstY, dstX, scale * ((1-fy) * ((1-fx) * src00 + src01 * fx)
                                    + fy  * ((1-fx) * src10 + src11 * fx)) + offset);
    }
}

//******************** Tensor Source ********************//

//******************** NN = Nearest Neighbor (TensorWrap)
template<class DstMap, class SrcWrapper>
__global__ void resizeCrop_NN(DstMap dst, SrcWrapper src,
                              const float2 resize, const int2 crop,
                              const float scale, const float offset)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst.width() && dst_y < dst.height())
    {
        // Copy nearest pixel to pixel.
        // Compute source pixel positions.
        const int sx = __float2int_rd((dst_x + crop.x + 0.5f) * resize.x);
        const int sy = __float2int_rd((dst_y + crop.y + 0.5f) * resize.y);

        // Rescale, channel manipulation, convert type, and reformat.
        dst(blockIdx.z, dst_y, dst_x, scale * ReadPixel(src, (int)blockIdx.z, sy, sx) + offset);
    }
} // resizeCrop_NN

//******************** Bilinear (TensorWrap; WITH normalization)
template<class DstMap, class SrcWrapper>
__global__ void resizeCrop_bilinear(DstMap dst, SrcWrapper src, const int src_w, const int src_h,
                                    const float2 resize, const int2 crop,
                                    const float scale, const float offset, bool src_cast)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst.width() && dst_y < dst.height())
    {
        // Use floating-point space for bi-linear interpolation computation.
        // Compute x and y coordinates, source data position, and weights.
        float fx = (dst_x + crop.x + 0.5f) * resize.x - 0.5f;
        float fy = (dst_y + crop.y + 0.5f) * resize.y - 0.5f;

        // Bi-linear interpolation, rescale, channel manipulation, convert type, and reformat.
        WriteBilinearPixel(dst, src, (int)blockIdx.z, dst_y, dst_x, fx, fy, src_w, src_h, scale, offset, src_cast);
    }
} // resizeCrop_bilinear

//******************** ImageBatchVarShape Source ********************//

//******************** NN = Nearest Neighbor (ImageBatchVarShapeWrap)
template<class DstMap, class SrcWrapper>
__global__ void resizeCrop_NN_varShape(DstMap dst, SrcWrapper src,
                                       const NVCVSize2D resize, const int2 crop,
                                       float scale, float offset)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst.width() && dst_y < dst.height())
    { // Generic copy pixel to pixel.
        const int src_w = src.width (blockIdx.z);
        const int src_h = src.height(blockIdx.z);

        // Compute scale factors.
        const float resize_x = static_cast<float>(src_w) / resize.w;
        const float resize_y = static_cast<float>(src_h) / resize.h;

        // Compute source pixel positions.
        const int sx = __float2int_rd((dst_x + crop.x + 0.5f) * resize_x);
        const int sy = __float2int_rd((dst_y + crop.y + 0.5f) * resize_y);

        // Rescale, channel manipulation, convert type, and reformat.
        dst(blockIdx.z, dst_y, dst_x, scale * ReadPixel(src, (int)blockIdx.z, sy, sx) + offset);
    }
} // resizeCrop_NN_varShape

//******************** Bilinear (ImageBatchVarShapeWrap; WITH normalization)
template<class DstMap, class SrcWrapper>
__global__ void resizeCrop_bilinear_varShape(DstMap dst, SrcWrapper src,
                                             const NVCVSize2D resize, const int2 crop,
                                             float scale, float offset, bool src_cast)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst.width() && dst_y < dst.height())
    {
        const int src_w = src.width (blockIdx.z);
        const int src_h = src.height(blockIdx.z);

        // Compute resize scale factors.
        float resize_x = static_cast<float>(src_w) / resize.w;
        float resize_y = static_cast<float>(src_h) / resize.h;

        // Use floating-point space for bi-linear interpolation computation.
        // Compute x and y coordinates, source data position, and weights.
        float fx = (dst_x + crop.x + 0.5f) * resize_x - 0.5f;
        float fy = (dst_y + crop.y + 0.5f) * resize_y - 0.5f;

        // Bi-linear interpolation, rescale, channel manipulation, convert type, and reformat.
        WriteBilinearPixel(dst, src, (int)blockIdx.z, dst_y, dst_x, fx, fy, src_w, src_h, scale, offset, src_cast);
    }
} // resizeCrop_bilinear_varShape

template<class DstMap, class SrcWrapper>
void LaunchTensorResizeCrop(DstMap dst, SrcWrapper src, const dim3 &gridSize, const dim3 &blockSize,
                            cudaStream_t stream, NVCVInterpolationType interp, int src_w, int src_h, float2 resize,
                            int2 cropPos, float scale, float offset, bool srcCast)
{
    switch (interp)
    {
    case NVCV_INTERP_NEAREST:
        resizeCrop_NN<<<gridSize, blockSize, 0, stream>>>(dst, src, resize, cropPos, scale, offset);
        break;

    case NVCV_INTERP_LINEAR:
        resizeCrop_bilinear<<<gridSize, blockSize, 0, stream>>>(dst, src, src_w, src_h, resize, cropPos, scale, offset,
                                                                srcCast);
        break;
    default:
        break;
    } // switch
}

template<class DstMap, class SrcWrapper>
void LaunchVarShapeResizeCrop(DstMap dst, SrcWrapper src, const dim3 &gridSize, const dim3 &blockSize,
                              cudaStream_t stream, NVCVInterpolationType interp, NVCVSize2D resizeDim, int2 cropPos,
                              float scale, float offset, bool srcCast)
{
    switch (interp)
    {
    case NVCV_INTERP_NEAREST:
        resizeCrop_NN_varShape<<<gridSize, blockSize, 0, stream>>>(dst, src, resizeDim, cropPos, scale, offset);
        break;

    case NVCV_INTERP_LINEAR:
        resizeCrop_bilinear_varShape<<<gridSize, blockSize, 0, stream>>>(dst, src, resizeDim, cropPos, scale, offset,
                                                                         srcCast);
        break;
    default:
        break;
    } // switch
}

// clang-format on

template<typename SrcT, typename DstT>
void resizeCropConvertReformat(const nvcv::TensorDataStridedCuda &srcData, const nvcv::TensorDataStridedCuda &dstData,
                               const NVCVSize2D resizeDim, NVCVInterpolationType interp, const int2 cropPos,
                               const NVCVChannelManip manip, float scale, float offset, bool srcCast,
                               cudaStream_t stream)

{
    constexpr uint NumElems = cuda::NumElements<SrcT>;

    using SrcBaseT = cuda::BaseType<SrcT>;
    using DstBaseT = cuda::BaseType<DstT>;
    using DstMapT  = DstMap<DstBaseT, NumElems>;
    using StrideT  = int32_t;

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    NVCV_ASSERT(srcAccess);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(dstAccess);

    const int samples  = srcAccess->numSamples();
    const int channels = srcAccess->numChannels();
    const int src_w    = srcAccess->numCols();
    const int src_h    = srcAccess->numRows();
    const int dst_w    = dstAccess->numCols();
    const int dst_h    = dstAccess->numRows();

    NVCV_ASSERT(samples == dstAccess->numSamples());
    NVCV_ASSERT(channels == dstAccess->numChannels());

    float2 resize{(float)src_w / resizeDim.w, (float)src_h / resizeDim.h};

    const int planes = dstAccess->numPlanes();

    const size_t addC = (planes > 1 ? dstAccess->planeStride() / sizeof(DstBaseT) : 1);
    const int    addW = channels / planes; // 1 if planar; channels if not.
    const int    addH = static_cast<int>(dstAccess->rowStride() / sizeof(DstBaseT));
    const size_t addN = dstAccess->rowStride() * dst_h * dstAccess->numPlanes() / sizeof(DstBaseT);

    DstBaseT *dstPtr = reinterpret_cast<DstBaseT *>(dstData.basePtr());

    DstMapT dst{dstPtr, addN, addH, addW, addC, manip, dst_w, dst_h};

    const int THREADS_PER_BLOCK = 256; // 256?  64?
    const int BLOCK_WIDTH       = 16;  // as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(util::DivUp(dst_w, blockSize.x), util::DivUp(dst_h, blockSize.y), samples);

    // Note: resize is fundamentally a gather memory operation, with a little bit of compute
    //       our goals are to (a) maximize throughput, and (b) minimize occupancy for the same performance
    // The channel dispatcher uses scalar SrcT only for a single-channel input, which has one plane.
    if constexpr (NumElems > 1)
    {
        if (srcAccess->numPlanes() > 1)
        {
            TensorPlanarSrc<SrcT> src{srcData, *srcAccess};
            LaunchTensorResizeCrop(dst, src, gridSize, blockSize, stream, interp, src_w, src_h, resize, cropPos, scale,
                                   offset, srcCast);
            return;
        }
    }

    auto src = cuda::CreateTensorWrapNHW<const SrcT, StrideT>(srcData);
    LaunchTensorResizeCrop(dst, src, gridSize, blockSize, stream, interp, src_w, src_h, resize, cropPos, scale, offset,
                           srcCast);
} //resize

template<typename SrcT, typename DstT>
void resizeCropConvertReformat(const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                               const nvcv::TensorDataStridedCuda &dstData, const NVCVSize2D resizeDim,
                               const NVCVInterpolationType interp, const int2 cropPos, const NVCVChannelManip manip,
                               float scale, float offset, bool srcCast, cudaStream_t stream)
{
    constexpr uint NumElems = cuda::NumElements<SrcT>;

    using SrcBaseT = cuda::BaseType<SrcT>;
    using DstBaseT = cuda::BaseType<DstT>;
    using DstMapT  = DstMap<DstBaseT, NumElems>;

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(dstAccess);

    const nvcv::ImageFormat srcFrmt = srcData.uniqueFormat();
    NVCV_ASSERT(srcFrmt);

    const int samples  = srcData.numImages();
    const int channels = srcFrmt.numChannels();
    const int dst_w    = dstAccess->numCols();
    const int dst_h    = dstAccess->numRows();

    NVCV_ASSERT(samples == dstAccess->numSamples());
    NVCV_ASSERT(channels == dstAccess->numChannels());

    const int planes = dstAccess->numPlanes();

    const size_t addC = (planes > 1 ? dstAccess->planeStride() / sizeof(DstBaseT) : 1);
    const int    addW = channels / planes; // 1 if planar; channels if not.
    const int    addH = static_cast<int>(dstAccess->rowStride() / sizeof(DstBaseT));
    const size_t addN = dstAccess->rowStride() * dst_h * dstAccess->numPlanes() / sizeof(DstBaseT);

    DstBaseT *dstPtr = reinterpret_cast<DstBaseT *>(dstData.basePtr());

    DstMapT dst{dstPtr, addN, addH, addW, addC, manip, dst_w, dst_h};

    const int THREADS_PER_BLOCK = 256; // Performance degrades above 256 and below 16 (GMEM speed limited)
    const int BLOCK_WIDTH       = 32;  // Keep each warp on one contiguous output row.

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(util::DivUp(dst_w, blockSize.x), util::DivUp(dst_h, blockSize.y), samples);

    // The channel dispatcher uses scalar SrcT only for a single-channel input, which has one plane.
    if constexpr (NumElems > 1)
    {
        if (srcFrmt.numPlanes() > 1)
        {
            ImageBatchVarShapePlanarSrc<SrcT> src(srcData);
            LaunchVarShapeResizeCrop(dst, src, gridSize, blockSize, stream, interp, resizeDim, cropPos, scale, offset,
                                     srcCast);
            return;
        }
    }

    cuda::ImageBatchVarShapeWrap<const SrcT> src(srcData);
    LaunchVarShapeResizeCrop(dst, src, gridSize, blockSize, stream, interp, resizeDim, cropPos, scale, offset, srcCast);
}

} // anonymous namespace

// clang-format off
namespace cvcuda::priv {
ResizeCropConvertReformat::ResizeCropConvertReformat() { }

// clang-format on

void ResizeCropConvertReformat::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                           const NVCVSize2D resizeDim, const NVCVInterpolationType interp,
                                           const int2 cropPos, const NVCVChannelManip manip, float scale, float offset,
                                           bool srcCast) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ResizeCropConvertReformat::operator()[Tensor]");
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be a cuda-accessible, pitch-linear tensor");
    }

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a cuda-accessible, pitch-linear tensor");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);

    const int samples  = srcAccess->numSamples();
    const int channels = srcAccess->numChannels();

    const int dst_w = dstAccess->numCols();
    const int dst_h = dstAccess->numRows();

    if (samples != dstAccess->numSamples())
    {
        std::string msg = "Input and output must have the same batch size (i.e., same number of images): Provided "
                        + std::to_string(samples) + " input and " + std::to_string(dstAccess->numSamples())
                        + " output images / samples";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (channels != dstAccess->numChannels())
    {
        std::string msg = "Input and output must have same number of channels: Provided " + std::to_string(channels)
                        + " input and " + std::to_string(dstAccess->numChannels()) + " output channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (channels != 1 && channels != 3)
    {
        std::string msg
            = "Only 1- or 3-channel input is supported: Provided " + std::to_string(channels) + " input channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    cuda_op::DataType srcType = helpers::GetLegacyDataType((*srcData).dtype());
    cuda_op::DataType dstType = helpers::GetLegacyDataType((*dstData).dtype());

    if (srcType != cuda_op::kCV_8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Input must be of data type uchar.");
    }

    if (dstType != cuda_op::kCV_8U && dstType != cuda_op::kCV_32F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Output must be of data type uchar or float.");
    }

    nvcv::TensorLayout srcLayout = srcData->layout();
    nvcv::TensorLayout dstLayout = dstData->layout();

    if (!(srcLayout == NVCV_TENSOR_NHWC || srcLayout == NVCV_TENSOR_HWC || srcLayout == NVCV_TENSOR_NCHW
          || srcLayout == NVCV_TENSOR_CHW))
    {
        const char *layout = nvcvTensorLayoutGetName(&srcLayout.m_layout);
        std::string msg
            = "Input tensor must have 'NHWC', 'NCHW', 'HWC', or 'CHW' layout: Layout provided " + std::string(layout);
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (!(dstLayout == NVCV_TENSOR_NHWC || dstLayout == NVCV_TENSOR_HWC || dstLayout == NVCV_TENSOR_NCHW
          || dstLayout == NVCV_TENSOR_CHW))
    {
        const char *layout = nvcvTensorLayoutGetName(&dstLayout.m_layout);
        std::string msg
            = "Output tensor must have 'NHWC', 'NCHW', 'HWC', or 'CHW' layout: Layout provided " + std::string(layout);
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (resizeDim.w <= 1 || resizeDim.h <= 1)
    {
        std::string msg = "Invalid resize dimensions: width x hight = " + std::to_string(resizeDim.w) + " x "
                        + std::to_string(resizeDim.h) + " dimensions must be larger than 1.";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (cropPos.x < 0 || cropPos.y < 0 || cropPos.x + dst_w > resizeDim.w || cropPos.y + dst_h > resizeDim.h)
    {
        std::string msg = "Invalid crop region: crop region(x, y, w, h) = (" + std::to_string(cropPos.x) + ", "
                        + std::to_string(cropPos.y) + ", " + std::to_string(dst_w) + ", " + std::to_string(dst_h)
                        + ") extends beyond bounds of resized tensor";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (srcAccess->sampleStride() * samples > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    if (interp != NVCV_INTERP_NEAREST && interp != NVCV_INTERP_LINEAR)
    {
        switch (interp)
        {
        case NVCV_INTERP_CUBIC:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Interpolation not implemented: NVCV_INTERP_CUBIC");
            break;

        case NVCV_INTERP_AREA:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Interpolation not implemented: NVCV_INTERP_AREA");
            break;

        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation");
            break;
        } // switch
    }

    if (srcType == cuda_op::kCV_8U)
    {
        if (channels == 1)
        {
            if (dstType == cuda_op::kCV_8U)
                resizeCropConvertReformat<uchar1, uint8_t>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                           offset, srcCast, stream);
            else if (dstType == cuda_op::kCV_32F)
                resizeCropConvertReformat<uchar1, float>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                         offset, srcCast, stream);
        }
        else
        {
            if (dstType == cuda_op::kCV_8U)
                resizeCropConvertReformat<uchar3, uint8_t>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                           offset, srcCast, stream);
            else if (dstType == cuda_op::kCV_32F)
                resizeCropConvertReformat<uchar3, float>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                         offset, srcCast, stream);
        }
    }
}

void ResizeCropConvertReformat::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                           const nvcv::Tensor &dst, const NVCVSize2D resizeDim,
                                           const NVCVInterpolationType interp, const int2 cropPos,
                                           const NVCVChannelManip manip, float scale, float offset, bool srcCast) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ResizeCropConvertReformat::operator()[ImageBatchVarShape]");
    auto srcData = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data must be a cuda-accessible, varshape image batch");
    }

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a cuda-accessible, pitch-linear tensor");
    }

    const nvcv::ImageFormat srcFrmt = src.uniqueFormat();

    if (!srcFrmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "All input images in a batch must have the same format (including number of channels)");
    }

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);

    const int samples  = srcData->numImages();
    const int channels = srcFrmt.numChannels();

    const int dst_w = dstAccess->numCols();
    const int dst_h = dstAccess->numRows();

    if (samples != dstAccess->numSamples())
    {
        std::string msg = "Input and output must have the same batch size (i.e., same number of images): Provided "
                        + std::to_string(samples) + " input and " + std::to_string(dstAccess->numChannels())
                        + " output channels";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (channels != dstAccess->numChannels())
    {
        std::string msg = "Input and output must have same number of channels: Provided " + std::to_string(channels)
                        + " input and " + std::to_string(dstAccess->numChannels()) + " output channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (channels != 1 && channels != 3)
    {
        std::string msg
            = "Only 1- or 3-channel input is supported: Provided " + std::to_string(channels) + " input channels";
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    cuda_op::DataType srcType = helpers::GetLegacyDataType(srcFrmt);
    cuda_op::DataType dstType = helpers::GetLegacyDataType((*dstData).dtype());

    if (srcType != cuda_op::kCV_8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Input must be of data type uchar.");
    }

    if (dstType != cuda_op::kCV_8U && dstType != cuda_op::kCV_32F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "Output must be of data type uchar or float.");
    }

    nvcv::TensorLayout dstLayout = dstData->layout();

    if (!(dstLayout == NVCV_TENSOR_NHWC || dstLayout == NVCV_TENSOR_HWC || dstLayout == NVCV_TENSOR_NCHW
          || dstLayout == NVCV_TENSOR_CHW))
    {
        const char *layout = nvcvTensorLayoutGetName(&dstLayout.m_layout);
        std::string msg
            = "Output tensor must have 'NHWC', 'NCHW', 'HWC', or 'CHW' layout: Layout provided " + std::string(layout);
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", msg.c_str());
    }

    if (cropPos.x < 0 || cropPos.y < 0 || cropPos.x + dst_w > abs(resizeDim.w) || cropPos.y + dst_h > abs(resizeDim.h))
    {
        std::string msg = "Invalid crop region: crop region(x, y, w, h) = (" + std::to_string(cropPos.x) + ", "
                        + std::to_string(cropPos.y) + ", " + std::to_string(dst_w) + ", " + std::to_string(dst_h)
                        + ") extends beyond bounds of resized tensor";
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", msg.c_str());
    }

    if (interp != NVCV_INTERP_NEAREST && interp != NVCV_INTERP_LINEAR)
    {
        switch (interp)
        {
        case NVCV_INTERP_CUBIC:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Interpolation not implemented: NVCV_INTERP_CUBIC");
            break;

        case NVCV_INTERP_AREA:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Interpolation not implemented: NVCV_INTERP_AREA");
            break;

        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation");
            break;
        } // switch
    }

    if (srcType == cuda_op::kCV_8U)
    {
        if (channels == 1)
        {
            if (dstType == cuda_op::kCV_8U)
                resizeCropConvertReformat<uchar1, uint8_t>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                           offset, srcCast, stream);
            else if (dstType == cuda_op::kCV_32F)
                resizeCropConvertReformat<uchar1, float>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                         offset, srcCast, stream);
        }
        else
        {
            if (dstType == cuda_op::kCV_8U)
                resizeCropConvertReformat<uchar3, uint8_t>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                           offset, srcCast, stream);
            else if (dstType == cuda_op::kCV_32F)
                resizeCropConvertReformat<uchar3, float>(*srcData, *dstData, resizeDim, interp, cropPos, manip, scale,
                                                         offset, srcCast, stream);
        }
    }
}
} // namespace cvcuda::priv
