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

#include "OpHQResize.hpp"
#include "WorkspaceUtil.hpp"
#include "cvcuda/Workspace.hpp"

#include "OpHQResizeBatchWrap.cuh"
#include "OpHQResizeFilter.cuh"

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <tuple>
#include <type_traits>

namespace {

namespace cuda          = nvcv::cuda;
namespace filter        = cvcuda::priv::hq_resize::filter;
namespace batch_wrapper = cvcuda::priv::hq_resize::batch_wrapper;

template<typename T, int N>
using Vec = typename cuda::MakeType<T, N>;

template<int N>
using VecI = Vec<int, N>;

template<int N>
using VecF = Vec<float, N>;

namespace utils {

template<typename T, class = cuda::Require<cuda::IsCompound<T>>>
inline std::enable_if_t<std::is_integral_v<cuda::BaseType<T>>, int64_t> Volume(const T &v)
{
    int64_t vol = 1;
    for (int i = 0; i < cuda::NumComponents<T>; i++)
    {
        vol *= cuda::GetElement(v, i);
    }
    return vol;
}

template<typename T, typename = std::enable_if<std::is_integral_v<T>>>
auto DivCeil(const T &a, const T &b)
{
    return (a + b - 1) / b;
}
} // namespace utils

namespace resampling {

template<int _kSpatialNDim>
struct SampleDesc
{
    static constexpr int kSpatialNDim = _kSpatialNDim;

    // input, output and the intermediate buffers
    static constexpr int kNumBuffers = kSpatialNDim + 1;

    // shapes[0] - input shape, consecutive intermediate results shapes,
    // shapes[kSpatialNDim] - output shape
    VecI<kSpatialNDim> shapes[kNumBuffers];

    // the number of channels in the sample, common for input,
    // intermediate and output sample
    int channels;

    // describes which axis to processes in a given resampling pass, e.g.
    // if processingOrder.x = 2, then in the first pass the z axis
    // will be resampled
    VecI<kSpatialNDim> processingOrder;

    // resampling origin and scale in pass order, i.e.
    // origin.x and scale.x describe origin and scale for resampling
    // in the first pass
    VecF<kSpatialNDim> origin, scale;

    // what type of filter to use (NN, Linear, Support based)
    // in pass order (i.e. filterKind[0] refers to filter used in the first pass)
    filter::FilterTypeKind filterKind[kSpatialNDim];

    // filter description (support, coefficients etc.)
    // in pass order (i.e. filter[0] refers to filter used in the first pass)
    filter::ResamplingFilter filter[kSpatialNDim];

    // spatial offset in the input sample based on the input ROI
    // and filter support
    VecI<kSpatialNDim> inRoiOffset;

    // describes the logical block shape, i.e. a size of a slice
    // that a single gpu block will process in a given pass
    VecI<kSpatialNDim> blockShape[kSpatialNDim];
};

/**
 * @brief Helper structure to indicate the static number of channels
 * dynamic number of channels that may differ between samples.
 */
template<int _kStaticChannels>
struct NumChannels
{
    constexpr int __forceinline__ __device__ operator()() const
    {
        return kStaticChannels;
    }

    static constexpr bool kHasStaticChannels = true;
    static constexpr int  kStaticChannels    = _kStaticChannels;
};

template<>
struct NumChannels<-1>
{
    int __forceinline__ __device__ operator()() const
    {
        return dynamicChannels;
    }

    static constexpr bool kHasStaticChannels = false;
    static constexpr int  kStaticChannels    = -1;
    int                   dynamicChannels;
};

template<int kNumStaticChannels, typename Cb>
__forceinline__ __device__ void WithChannels(const int dynamicChannels, Cb &&cb)
{
    if constexpr (kNumStaticChannels == -1)
    {
        cb(NumChannels<-1>{dynamicChannels});
    }
    else if constexpr (kNumStaticChannels != -1)
    {
        static_assert(kNumStaticChannels > 0);
        cb(NumChannels<kNumStaticChannels>{});
    }
}

/**
 * @brief Each threadblock will cover `lanes * volume(blockDim)`
 * elements of the output sample. More lanes result in:
 * 1. smaller grid launched (possibly reducing parallelism for small images),
 * 2. better resuing of the filter's coefficients
 *    (they are computed once for all lanes).
 *
 * @return int - the number of lanes for a single threadblock
 * to cover in the output image
 */
inline int GetResizeBlockLanesEnv()
{
    char *env = getenv("CVCUDA_HQ_RESIZE_BLOCK_LANES");
    if (env)
    {
        int lanes = atoi(env);
        if (lanes < 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The CVCUDA_HQ_RESIZE_BLOCK_LANES must be a positive integer");
        }
        return lanes;
    }
    else
    {
        return 8;
    }
}

inline int GetResizeBlockLanes()
{
    static int lanes = GetResizeBlockLanesEnv();
    return lanes;
}

template<int kSpatialNDim>
struct GridHelperDevice
{
};

/**
 * @brief Maps cuda blockIdx to sample and bounds of the sample region
 * to be processed be the threadblock for 2D resampling
 */
template<>
struct GridHelperDevice<2>
{
    GridHelperDevice(VecI<2> numBlocks)
        : m_numBlocksX{numBlocks.x}
    {
    }

    int __forceinline__ __device__ CurrentSample() const
    {
        return blockIdx.y;
    }

    void __forceinline__ __device__ CurrentBlock(VecI<2> &lo, VecI<2> &hi, const VecI<2> blockShape) const

    {
        VecI<2> currentBlock;
        {
            int block      = blockIdx.x;
            currentBlock.x = block % m_numBlocksX;
            currentBlock.y = block / m_numBlocksX;
        }
        lo = blockShape * currentBlock;
        hi = lo + blockShape;
    }

private:
    int m_numBlocksX;
};

/**
 * @brief Maps cuda blockIdx to sample and bounds of the sample region
 * to be processed be the threadblock for 3D resampling
 */
template<>
struct GridHelperDevice<3>
{
    GridHelperDevice(VecI<3> numBlocks)
        : m_numBlocksX{numBlocks.x}
        , m_numBlocksY{numBlocks.y}
    {
    }

    int __forceinline__ __device__ CurrentSample() const
    {
        return blockIdx.y;
    }

    void __forceinline__ __device__ CurrentBlock(VecI<3> &lo, VecI<3> &hi, const VecI<3> blockShape) const

    {
        VecI<3> currentBlock;
        {
            int block      = blockIdx.x;
            currentBlock.x = block % m_numBlocksX;
            block          = block / m_numBlocksX;
            currentBlock.y = block % m_numBlocksY;
            currentBlock.z = block / m_numBlocksY;
        }
        lo = blockShape * currentBlock;
        hi = lo + blockShape;
    }

private:
    int m_numBlocksX, m_numBlocksY;
};

/**
 * @brief Maps the logical blocks and the number of samples into cuda grid and back.
 */
template<int kSpatialNDim>
struct GridHelper
{
    GridHelper(VecI<kSpatialNDim> numBlocks, int numSamples)
        : m_numBlocks{numBlocks}
        , m_numSamples{numSamples}
    {
    }

    template<int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2, dim3> GetKernelGrid() const
    {
        static_assert(kSpatialNDim == 2);
        return dim3(m_numBlocks.x * m_numBlocks.y, m_numSamples, 1);
    }

    template<int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 3, dim3> GetKernelGrid() const
    {
        static_assert(kSpatialNDim == 3);
        return dim3(m_numBlocks.x * m_numBlocks.y * m_numBlocks.z, m_numSamples, 1);
    }

    GridHelperDevice<kSpatialNDim> GetDeviceGridHelper()
    {
        return {m_numBlocks};
    }

private:
    VecI<kSpatialNDim> m_numBlocks;
    int                m_numSamples;
};

// The namespace contains implementation of different resampling
// methods in device code.
namespace interpolate {

template<typename Wrap, typename... Idxs>
auto __forceinline__ __device__ GetWrapPtr(const Wrap wrap, const VecI<2> yx, const Idxs... idxs)
{
    return wrap.ptr(yx.y, yx.x, idxs...);
}

template<typename Wrap, typename... Idxs>
auto __forceinline__ __device__ GetWrapPtr(const Wrap wrap, const VecI<3> zyx, const Idxs... idxs)
{
    return wrap.ptr(zyx.z, zyx.y, zyx.x, idxs...);
}

template<typename Wrap, typename NumChannelsT, typename... Idxs>
std::enable_if_t<NumChannelsT::kHasStaticChannels, typename Wrap::ValueType> __forceinline__ __device__
    LoadPixelLdg(const Wrap wrap, const NumChannelsT numChannels, const Idxs... idxs)
{
    using T                       = std::remove_const_t<typename Wrap::ValueType>;
    using BT                      = cuda::BaseType<T>;
    constexpr int kStaticChannels = NumChannelsT::kStaticChannels;
    static_assert(kStaticChannels == cuda::NumElements<T>);

    constexpr bool kSupportsLdg = kStaticChannels == 2 || kStaticChannels == 4;

    if constexpr (kSupportsLdg)
    {
        return __ldg(GetWrapPtr(wrap, idxs...));
    }
    else if constexpr (!kSupportsLdg)
    {
        const BT *basePtr = reinterpret_cast<const BT *>(GetWrapPtr(wrap, idxs...));
        T         value;
#pragma unroll
        for (int c = 0; c < kStaticChannels; c++)
        {
            cuda::GetElement(value, c) = __ldg(basePtr + c);
        }
        return value;
    }
}

template<typename Wrap, typename NumChannelsT, typename... Idxs>
std::enable_if_t<!NumChannelsT::kHasStaticChannels, typename Wrap::ValueType> __forceinline__ __device__
    LoadPixelLdg(const Wrap wrap, const NumChannelsT numChannels, const Idxs... idxs)
{
    static_assert(!cuda::IsCompound<typename Wrap::ValueType>);
    return __ldg(GetWrapPtr(wrap, idxs...));
}

namespace nn {

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllPixels(const VecI<2> lo, const VecI<2> hi, ProcessPixel &&processPixel)
{
    for (int y = lo.y + threadIdx.y; y < hi.y; y += blockDim.y)
    {
        for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
        {
            processPixel(VecI<2>{x, y});
        }
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllPixels(const VecI<3> lo, const VecI<3> hi, ProcessPixel &&processPixel)
{
    for (int z = lo.z + threadIdx.z; z < hi.z; z += blockDim.z)
    {
        for (int y = lo.y + threadIdx.y; y < hi.y; y += blockDim.y)
        {
            for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
            {
                processPixel(VecI<3>{x, y, z});
            }
        }
    }
}

/**
 * @brief Nearest neighbor resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param origin - source coordinates corresponding to output's (0, 0)
 * @param scale - step, in source coordinates, for one pixel in output coordinates
 * @param inShape - shape of the input (x, y) order
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 */
template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ Resample(const PassOutWrap outWrap, const PassInWrap inWrap,
                                         const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi,
                                         VecF<kSpatialNDim> origin, const VecF<kSpatialNDim> scale,
                                         const VecI<kSpatialNDim> inShape, const NumChannelsT numChannels)
{
    using OutT = typename PassOutWrap::ValueType;
    using InT  = typename PassInWrap::ValueType;
    // spatial extents and optional channels extent
    constexpr int kNDim = kSpatialNDim + !NumChannelsT::kHasStaticChannels;

    static_assert(!NumChannelsT::kHasStaticChannels || NumChannelsT::kStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);
    static_assert(PassOutWrap::kNumDimensions == kNDim);
    static_assert(PassInWrap::kNumDimensions == kNDim);

    origin += 0.5f * scale;
    ForAllPixels(lo, hi,
                 [=](const VecI<kSpatialNDim> outIdxs)
                 {
                     VecI<kSpatialNDim> inIdxs = cuda::round<cuda::RoundMode::DOWN, int>(outIdxs * scale + origin);
                     inIdxs                    = cuda::clamp(inIdxs, cuda::SetAll<VecI<kSpatialNDim>>(0), inShape - 1);

                     if constexpr (NumChannelsT::kHasStaticChannels)
                     {
                         const InT in  = LoadPixelLdg(inWrap, numChannels, inIdxs);
                         OutT     &out = *GetWrapPtr(outWrap, outIdxs);
                         out           = cuda::SaturateCast<OutT>(in);
                     }
                     else if constexpr (!NumChannelsT::kHasStaticChannels)
                     {
                         for (int c = 0; c < numChannels(); c++)
                         {
                             const InT in  = LoadPixelLdg(inWrap, numChannels, inIdxs, c);
                             OutT     &out = *GetWrapPtr(outWrap, outIdxs, c);
                             out           = cuda::SaturateCast<OutT>(in);
                         }
                     }
                 });
}

} // namespace nn

namespace linear {

template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ Linear(const PassOutWrap outWrap, const PassInWrap inWrap,
                                       const NumChannelsT numChannels, const VecI<kSpatialNDim> inIdx0,
                                       const VecI<kSpatialNDim> inIdx1, const float q, const VecI<kSpatialNDim> outIdx)
{
    using OutT = typename PassOutWrap::ValueType;
    using InT  = std::remove_const_t<typename PassInWrap::ValueType>;
    // spatial extents and optional channels extent
    constexpr int kNDim = kSpatialNDim + !NumChannelsT::kHasStaticChannels;

    static_assert(!NumChannelsT::kHasStaticChannels || NumChannelsT::kStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);
    static_assert(PassOutWrap::kNumDimensions == kNDim);
    static_assert(PassInWrap::kNumDimensions == kNDim);

    if constexpr (NumChannelsT::kHasStaticChannels)
    {
        using FloatT     = cuda::ConvertBaseTypeTo<float, InT>;
        const FloatT a   = cuda::StaticCast<float>(LoadPixelLdg(inWrap, numChannels, inIdx0));
        const FloatT b   = cuda::StaticCast<float>(LoadPixelLdg(inWrap, numChannels, inIdx1));
        FloatT       tmp = b - a;
#pragma unroll
        for (int c = 0; c < NumChannelsT::kStaticChannels; c++)
        {
            cuda::GetElement(tmp, c) = fmaf(cuda::GetElement(tmp, c), q, cuda::GetElement(a, c));
        }
        OutT &out = *GetWrapPtr(outWrap, outIdx);
        out       = cuda::SaturateCast<OutT>(tmp);
    }
    else if constexpr (!NumChannelsT::kHasStaticChannels)
    {
        for (int c = 0; c < numChannels(); c++)
        {
            const float a   = LoadPixelLdg(inWrap, numChannels, inIdx0, c);
            const float b   = LoadPixelLdg(inWrap, numChannels, inIdx1, c);
            const float tmp = fmaf(b - a, q, a);
            OutT       &out = *GetWrapPtr(outWrap, outIdx, c);
            out             = cuda::SaturateCast<OutT>(tmp);
        }
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllPixelsHorz(const VecI<2> lo, const VecI<2> hi, ProcessPixel &&processPixel)
{
    for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
    {
        for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
        {
            processPixel(VecI<2>{x, y});
        }
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllPixelsHorz(const VecI<3> lo, const VecI<3> hi, ProcessPixel &&processPixel)
{
    for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
    {
        for (int z = threadIdx.z + lo.z; z < hi.z; z += blockDim.z)
        {
            for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
            {
                processPixel(VecI<3>{x, y, z});
            }
        }
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllPixelsVert(const VecI<2> lo, const VecI<2> hi, ProcessPixel &&processPixel)
{
    for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
    {
        for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
        {
            processPixel(VecI<2>{x, y});
        }
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllPixelsVert(const VecI<3> lo, const VecI<3> hi, ProcessPixel &&processPixel)
{
    for (int z = threadIdx.z + lo.z; z < hi.z; z += blockDim.z)
    {
        for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
        {
            for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
            {
                processPixel(VecI<3>{x, y, z});
            }
        }
    }
}

/**
 * @brief Implements horizontal resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param srcX0 - X coordinate in the source image corresponding to output 0
 * @param scale - step, in source X, for one pixel in output X (may be negative)
 * @param inShape - shape of the input (x, y[, z]) order
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 *
 * The input region of interest is defined in terms of origin/scale, which are relative to
 * output (0, 0).
 * The lo/hi parameters are not output RoI - they merely indicate the output slice processed
 * by current block.
 */
template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ ResampleHorz(const PassOutWrap outWrap, const PassInWrap inWrap,
                                             const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, float srcX0,
                                             const float scale, const VecI<kSpatialNDim> inShape,
                                             const NumChannelsT numChannels)
{
    srcX0 += 0.5f * scale - 0.5f;
    ForAllPixelsHorz(lo, hi,
                     [=](const VecI<kSpatialNDim> outIdx)
                     {
                         const float sx0f = outIdx.x * scale + srcX0;
                         const int   sx0i = cuda::round<cuda::RoundMode::DOWN, int>(sx0f);
                         const float q    = sx0f - sx0i;
                         const int   sx0  = cuda::clamp(sx0i, 0, inShape.x - 1);
                         const int   sx1  = cuda::clamp(sx0i + 1, 0, inShape.x - 1);

                         VecI<kSpatialNDim> inIdx0 = outIdx;
                         VecI<kSpatialNDim> inIdx1 = outIdx;
                         inIdx0.x                  = sx0;
                         inIdx1.x                  = sx1;

                         Linear<kSpatialNDim>(outWrap, inWrap, numChannels, inIdx0, inIdx1, q, outIdx);
                     });
}

/**
 * @brief Implements vertical resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param srcY0 - Y coordinate in the source image corresponding to output 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param inShape - shape of the input (x, y[, z]) order
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 */
template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ ResampleVert(const PassOutWrap outWrap, const PassInWrap inWrap,
                                             const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, float srcY0,
                                             const float scale, const VecI<kSpatialNDim> inShape,
                                             const NumChannelsT numChannels)
{
    srcY0 += 0.5f * scale - 0.5f;
    ForAllPixelsVert(lo, hi,
                     [=](const VecI<kSpatialNDim> outIdx)
                     {
                         const float sy0f = outIdx.y * scale + srcY0;
                         const int   sy0i = cuda::round<cuda::RoundMode::DOWN, int>(sy0f);
                         const float q    = sy0f - sy0i;
                         const int   sy0  = cuda::clamp(sy0i, 0, inShape.y - 1);
                         const int   sy1  = cuda::clamp(sy0i + 1, 0, inShape.y - 1);

                         VecI<kSpatialNDim> inIdx0 = outIdx;
                         VecI<kSpatialNDim> inIdx1 = outIdx;
                         inIdx0.y                  = sy0;
                         inIdx1.y                  = sy1;

                         Linear<kSpatialNDim>(outWrap, inWrap, numChannels, inIdx0, inIdx1, q, outIdx);
                     });
}

/**
 * @brief Implements depthwise resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param srcZ0 - Z coordinate in the source image corresponding to output's 0
 * @param scale - step, in source Z, for one pixel in output Z (may be negative)
 * @param inShape - shape of the input (x, y[, z]) order
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 */
template<typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ ResampleDepth(const PassOutWrap outWrap, const PassInWrap inWrap, const VecI<3> lo,
                                              const VecI<3> hi, float srcZ0, const float scale, const VecI<3> inShape,
                                              const NumChannelsT numChannels)
{
    srcZ0 += 0.5f * scale - 0.5f;
    // threadIdx.y is used to traverse Z axis
    for (int z = lo.z + threadIdx.y; z < hi.z; z += blockDim.y)
    {
        const float sz0f = z * scale + srcZ0;
        const int   sz0i = cuda::round<cuda::RoundMode::DOWN, int>(sz0f);
        const float q    = sz0f - sz0i;
        const int   sz0  = cuda::clamp(sz0i, 0, inShape.z - 1);
        const int   sz1  = cuda::clamp(sz0i + 1, 0, inShape.z - 1);

        for (int y = lo.y + threadIdx.z; y < hi.y; y += blockDim.z)
        {
            for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x)
            {
                VecI<3> inIdx0{x, y, sz0};
                VecI<3> inIdx1{x, y, sz1};
                VecI<3> outIdx{x, y, z};
                Linear<3>(outWrap, inWrap, numChannels, inIdx0, inIdx1, q, outIdx);
            }
        }
    }
}

} // namespace linear

namespace filter_support {

constexpr int kMaxGPUFilterSupport = 8192;

bool __forceinline__ __host__ __device__ CanComputeCoefPerThread(const int support, const int resamplingAxisBlockSize)
{
    return support * resamplingAxisBlockSize <= kMaxGPUFilterSupport;
}

inline int RequiredSharedMemoryElements(const int support, const int resamplingAxisBlockSize)
{
    if (CanComputeCoefPerThread(support, resamplingAxisBlockSize))
    {
        return support * resamplingAxisBlockSize;
    }
    else
    {
        return support;
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllOrthogonalToHorz(const VecI<2> lo, const VecI<2> hi, ProcessPixel &&processPixel)
{
    for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
    {
        processPixel(VecI<2>{0, y});
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllOrthogonalToHorz(const VecI<3> lo, const VecI<3> hi, ProcessPixel &&processPixel)
{
    for (int z = threadIdx.z + lo.z; z < hi.z; z += blockDim.z)
    {
        for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
        {
            processPixel(VecI<3>{0, y, z});
        }
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllOrthogonalToVert(const VecI<2> lo, const VecI<2> hi, ProcessPixel &&processPixel)
{
    for (int x = threadIdx.x + lo.x; x < hi.x; x += blockDim.x)
    {
        processPixel(VecI<2>{x, 0});
    }
}

template<typename ProcessPixel>
void __forceinline__ __device__ ForAllOrthogonalToVert(const VecI<3> lo, const VecI<3> hi, ProcessPixel &&processPixel)
{
    for (int z = threadIdx.z + lo.z; z < hi.z; z += blockDim.z)
    {
        for (int x = threadIdx.x + lo.x; x < hi.x; x += blockDim.x)
        {
            processPixel(VecI<3>{x, 0, z});
        }
    }
}

/**
 * @brief Implements horizontal resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param srcX0 - X coordinate in the source image corresponding to output's 0
 * @param scale - step, in source X, for one pixel in output X (may be negative)
 * @param support - size of the resampling kernel, in source pixels
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 *
 * The function fills the output in block-sized vertical spans.
 * Block horizontal size is warp-aligned.
 * Filter coefficients are pre-calculated for each vertical span to avoid
 * recalculating them for each row, and stored in a shared memory block.
 *
 * The function follows different code paths for static and dynamic number of channels.
 * For the dynamic, the innermost loop goes over filter taps, which eliminates the need
 * for thread-local memory to store intermediate sums. This allows processing arbitrary
 * number of channels.
 * For static number of channels, the run-time parameter `channels` is ignored and
 * there's also a local temporary storage for a tap sum for each channel. This is faster,
 * but requires extra registers for the intermediate sums.
 */
template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ ResampleHorz(const PassOutWrap outWrap, const PassInWrap inWrap,
                                             const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, float srcX0,
                                             const float scale, const VecI<kSpatialNDim> inShape,
                                             const filter::ResamplingFilter filter, const NumChannelsT numChannels)
{
    extern __shared__ float coeffs[];

    using OutT = typename PassOutWrap::ValueType;
    using InT  = std::remove_const_t<typename PassInWrap::ValueType>;
    // spatial extents and optional channels extent
    constexpr int kNDim = kSpatialNDim + !NumChannelsT::kHasStaticChannels;

    static_assert(!NumChannelsT::kHasStaticChannels || NumChannelsT::kStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);
    static_assert(PassOutWrap::kNumDimensions == kNDim);
    static_assert(PassInWrap::kNumDimensions == kNDim);

    const int   support    = filter.support();
    const float filterStep = filter.scale;
    // If the support is small enough (for blockDim.x = 32 and kMaxGPUFilterSupport = 8192, it's 256),
    // we can fit `support` x `blockDim.x` elements into shm, so that for each output_x mapped to input_x,
    // we take into account the exact error that comes from rounding the input_x from float to integer.
    // For larger supports, we just compute `support` elements common for all threads.
    const bool  hugeSupport = !CanComputeCoefPerThread(support, blockDim.x);
    const int   coeffBase   = hugeSupport ? 0 : threadIdx.x;
    const int   coeffStride = hugeSupport ? 1 : blockDim.x;

    srcX0 += 0.5f * scale - 0.5f - filter.anchor;

    for (int j = lo.x; j < hi.x; j += blockDim.x)
    {
        const int   x    = j + threadIdx.x;
        const float sx0f = x * scale + srcX0;
        const int   sx0  = hugeSupport ? cuda::round<cuda::RoundMode::NEAREST, int>(sx0f)
                                       : cuda::round<cuda::RoundMode::UP, int>(sx0f);
        const float f    = (sx0 - sx0f) * filterStep;
        __syncthreads();
        if (hugeSupport)
        {
            for (int k = threadIdx.x + blockDim.x * threadIdx.y; k < support; k += blockDim.x * blockDim.y)
            {
                float flt = filter(f + k * filterStep);
                coeffs[k] = flt;
            }
        }
        else
        {
            for (int k = threadIdx.y; k < support; k += blockDim.y)
            {
                float flt                           = filter(f + k * filterStep);
                coeffs[coeffBase + coeffStride * k] = flt;
            }
        }
        __syncthreads();

        if (x >= hi.x)
            continue;

        float norm = 0;
        for (int k = 0; k < support; k++)
        {
            norm += coeffs[coeffBase + coeffStride * k];
        }
        norm = 1.0f / norm;

        ForAllOrthogonalToHorz(
            lo, hi,
            [=](VecI<kSpatialNDim> outIdx)
            {
                VecI<kSpatialNDim> inIdx = outIdx;
                outIdx.x                 = x;

                if constexpr (NumChannelsT::kHasStaticChannels)
                {
                    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;
                    FloatT tmp{};

                    for (int k = 0, coeffIdx = coeffBase; k < support; k++, coeffIdx += coeffStride)
                    {
                        inIdx.x         = cuda::clamp(sx0 + k, 0, inShape.x - 1);
                        const float flt = coeffs[coeffIdx];
                        const InT   px  = LoadPixelLdg(inWrap, numChannels, inIdx);
#pragma unroll
                        for (int c = 0; c < NumChannelsT::kStaticChannels; c++)
                        {
                            cuda::GetElement(tmp, c) = fmaf(cuda::GetElement(px, c), flt, cuda::GetElement(tmp, c));
                        }
                    }

                    OutT &out = *GetWrapPtr(outWrap, outIdx);
                    out       = cuda::SaturateCast<OutT>(tmp * norm);
                }
                else if constexpr (!NumChannelsT::kHasStaticChannels)
                {
                    for (int c = 0; c < numChannels(); c++)
                    {
                        float tmp = 0;

                        for (int k = 0, coeffIdx = coeffBase; k < support; k++, coeffIdx += coeffStride)
                        {
                            inIdx.x         = cuda::clamp(sx0 + k, 0, inShape.x - 1);
                            const float flt = coeffs[coeffIdx];
                            const InT   px  = LoadPixelLdg(inWrap, numChannels, inIdx, c);
                            tmp             = fmaf(px, flt, tmp);
                        }

                        OutT &out = *GetWrapPtr(outWrap, outIdx, c);
                        out       = cuda::SaturateCast<OutT>(tmp * norm);
                    }
                }
            });
    }
}

/**
 * @brief Implements vertical resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param srcY0 - Y coordinate in the source image corresponding to output's 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param support - size of the resampling kernel, in source pixels
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 *
 * The function fills the output in block-sized horizontal spans.
 * Filter coefficients are pre-calculated for each horizontal span to avoid
 * recalculating them for each column, and stored in a shared memory block.
 */
template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ ResampleVert(const PassOutWrap outWrap, const PassInWrap inWrap,
                                             const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, float srcY0,
                                             const float scale, const VecI<kSpatialNDim> inShape,
                                             const filter::ResamplingFilter filter, const NumChannelsT numChannels)
{
    extern __shared__ float coeffs[];

    using OutT = typename PassOutWrap::ValueType;
    using InT  = std::remove_const_t<typename PassInWrap::ValueType>;
    // spatial extents and optional channels extent
    constexpr int kNDim = kSpatialNDim + !NumChannelsT::kHasStaticChannels;

    static_assert(!NumChannelsT::kHasStaticChannels || NumChannelsT::kStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);
    static_assert(PassOutWrap::kNumDimensions == kNDim);
    static_assert(PassInWrap::kNumDimensions == kNDim);

    const int   support    = filter.support();
    const float filterStep = filter.scale;
    // If the support is small enough, we can fit `blockDim.y` x `support` elements into shm, so that
    // for each output_y mapped to input_y, we take into account the exact error that comes from
    // rounding the input_y from float to integer. For larger supports, we just compute `support`
    // elements common for all threads.
    const bool  hugeSupport = !CanComputeCoefPerThread(support, blockDim.y);
    const int   coeffBase   = hugeSupport ? 0 : support * threadIdx.y;

    srcY0 += 0.5f * scale - 0.5f - filter.anchor;

    for (int i = lo.y; i < hi.y; i += blockDim.y)
    {
        const int   y    = i + threadIdx.y;
        const float sy0f = y * scale + srcY0;
        const int   sy0  = hugeSupport ? cuda::round<cuda::RoundMode::NEAREST, int>(sy0f)
                                       : cuda::round<cuda::RoundMode::UP, int>(sy0f);
        float       f    = (sy0 - sy0f) * filterStep;
        __syncthreads();
        // fills `support`
        if (hugeSupport)
        {
            for (int k = threadIdx.x + blockDim.x * threadIdx.y; k < support; k += blockDim.x * blockDim.y)
            {
                float flt = filter(f + k * filterStep);
                coeffs[k] = flt;
            }
        }
        else
        {
            for (int k = threadIdx.x; k < support; k += blockDim.x)
            {
                float flt             = filter(f + k * filterStep);
                coeffs[coeffBase + k] = flt;
            }
        }
        __syncthreads();

        if (y >= hi.y)
            continue;

        float norm = 0;
        for (int k = 0; k < support; k++)
        {
            norm += coeffs[coeffBase + k];
        }
        norm = 1.0f / norm;

        ForAllOrthogonalToVert(lo, hi,
                               [=](VecI<kSpatialNDim> outIdx)
                               {
                                   VecI<kSpatialNDim> inIdx = outIdx;
                                   outIdx.y                 = y;

                                   if constexpr (NumChannelsT::kHasStaticChannels)
                                   {
                                       using FloatT = cuda::ConvertBaseTypeTo<float, InT>;
                                       FloatT tmp{};

                                       for (int k = 0; k < support; k++)
                                       {
                                           inIdx.y         = cuda::clamp(sy0 + k, 0, inShape.y - 1);
                                           const float flt = coeffs[coeffBase + k];
                                           const InT   px  = LoadPixelLdg(inWrap, numChannels, inIdx);
#pragma unroll
                                           for (int c = 0; c < NumChannelsT::kStaticChannels; c++)
                                           {
                                               cuda::GetElement(tmp, c)
                                                   = fmaf(cuda::GetElement(px, c), flt, cuda::GetElement(tmp, c));
                                           }
                                       }

                                       OutT &out = *GetWrapPtr(outWrap, outIdx);
                                       out       = cuda::SaturateCast<OutT>(tmp * norm);
                                   }
                                   else if constexpr (!NumChannelsT::kHasStaticChannels)
                                   {
                                       for (int c = 0; c < numChannels(); c++)
                                       {
                                           float tmp = 0;

                                           for (int k = 0; k < support; k++)
                                           {
                                               inIdx.y         = cuda::clamp(sy0 + k, 0, inShape.y - 1);
                                               const float flt = coeffs[coeffBase + k];
                                               const InT   px  = LoadPixelLdg(inWrap, numChannels, inIdx, c);
                                               tmp             = fmaf(px, flt, tmp);
                                           }

                                           OutT &out = *GetWrapPtr(outWrap, outIdx, c);
                                           out       = cuda::SaturateCast<OutT>(tmp * norm);
                                       }
                                   }
                               });
    }
}

/**
 * @brief Implements depth resampling
 *
 * @param outWrap - the wrapper for accessing output data
 * @param inWrap - the wrapper for accessing input data
 * @param lo - inclusive lower bound output coordinates of the block processed by the threadblock
 * @param hi - exclusive upper bound output coordinates of the block processed by the threadblock
 * @param srcZ0 - Y coordinate in the source image corresponding to output's 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param support - size of the resampling kernel, in source pixels
 * @param numChannels - the NumChannels specialization describing the number of interleaved
 *                      channels in the input and output sample.
 *
 * The function fills the output in block-sized horizontal spans.
 * Filter coefficients are pre-calculated for each horizontal span to avoid
 * recalculating them for each column, and stored in a shared memory block.
 */
template<typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ ResampleDepth(const PassOutWrap outWrap, const PassInWrap inWrap, const VecI<3> lo,
                                              const VecI<3> hi, float srcZ0, const float scale, const VecI<3> inShape,
                                              const filter::ResamplingFilter filter, const NumChannelsT numChannels)
{
    extern __shared__ float coeffs[];

    using OutT = typename PassOutWrap::ValueType;
    using InT  = std::remove_const_t<typename PassInWrap::ValueType>;
    // spatial extents and optional channels extent
    constexpr int kNDim = 3 + !NumChannelsT::kHasStaticChannels;

    static_assert(!NumChannelsT::kHasStaticChannels || NumChannelsT::kStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);
    static_assert(PassOutWrap::kNumDimensions == kNDim);
    static_assert(PassInWrap::kNumDimensions == kNDim);

    const int   support    = filter.support();
    const float filterStep = filter.scale;
    // If the support is small enough, we can fit `blockDim.y` x `support` elements into shm,
    // so that for each output_z mapped to input_z, we take into account the exact error that
    // comes from rounding the input_z from float to integer. For larger supports, we just
    // compute `support` elements common for all threads.
    const bool  hugeSupport = !CanComputeCoefPerThread(support, blockDim.y);
    const int   coeffBase   = hugeSupport ? 0 : support * threadIdx.y;

    srcZ0 += 0.5f * scale - 0.5f - filter.anchor;

    for (int i = lo.z; i < hi.z; i += blockDim.y)
    {
        // threadIdx.y is used to traverse Z axis
        const int   z    = i + threadIdx.y;
        const float sz0f = z * scale + srcZ0;
        const int   sz0  = hugeSupport ? cuda::round<cuda::RoundMode::NEAREST, int>(sz0f)
                                       : cuda::round<cuda::RoundMode::UP, int>(sz0f);
        float       f    = (sz0 - sz0f) * filterStep;
        __syncthreads();
        if (hugeSupport)
        {
            for (int k = threadIdx.x + blockDim.x * threadIdx.y; k < support; k += blockDim.x * blockDim.y)
            {
                float flt = filter(f + k * filterStep);
                coeffs[k] = flt;
            }
        }
        else
        {
            for (int k = threadIdx.x; k < support; k += blockDim.x)
            {
                float flt             = filter(f + k * filterStep);
                coeffs[coeffBase + k] = flt;
            }
        }
        __syncthreads();

        if (z >= hi.z)
            continue;

        float norm = 0;
        for (int k = 0; k < support; k++)
        {
            norm += coeffs[coeffBase + k];
        }
        norm = 1.0f / norm;

        for (int y = threadIdx.z + lo.y; y < hi.y; y += blockDim.z)
        {
            for (int x = threadIdx.x + lo.x; x < hi.x; x += blockDim.x)
            {
                const VecI<3> outIdx{x, y, z};
                VecI<3>       inIdx = outIdx;

                if constexpr (NumChannelsT::kHasStaticChannels)
                {
                    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;
                    FloatT tmp{};

                    for (int k = 0; k < support; k++)
                    {
                        inIdx.z         = cuda::clamp(sz0 + k, 0, inShape.z - 1);
                        const float flt = coeffs[coeffBase + k];
                        const InT   px  = LoadPixelLdg(inWrap, numChannels, inIdx);
#pragma unroll
                        for (int c = 0; c < NumChannelsT::kStaticChannels; c++)
                        {
                            cuda::GetElement(tmp, c) = fmaf(cuda::GetElement(px, c), flt, cuda::GetElement(tmp, c));
                        }
                    }

                    OutT &out = *GetWrapPtr(outWrap, outIdx);
                    out       = cuda::SaturateCast<OutT>(tmp * norm);
                }
                else if constexpr (!NumChannelsT::kHasStaticChannels)
                {
                    for (int c = 0; c < numChannels(); c++)
                    {
                        float tmp = 0;

                        for (int k = 0; k < support; k++)
                        {
                            inIdx.z         = cuda::clamp(sz0 + k, 0, inShape.z - 1);
                            const float flt = coeffs[coeffBase + k];
                            const InT   px  = LoadPixelLdg(inWrap, numChannels, inIdx, c);
                            tmp             = fmaf(px, flt, tmp);
                        }

                        OutT &out = *GetWrapPtr(outWrap, outIdx, c);
                        out       = cuda::SaturateCast<OutT>(tmp * norm);
                    }
                }
            }
        }
    }
}
} // namespace filter_support

template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ RunNN(const PassOutWrap outWrap, const PassInWrap inWrap, const VecI<kSpatialNDim> lo,
                                      const VecI<kSpatialNDim> hi, int axis, const VecI<kSpatialNDim> inShape,
                                      const float origin, const float scale, const NumChannelsT numChannels)
{
    auto originV                    = cuda::SetAll<VecF<kSpatialNDim>>(0.f);
    auto scaleV                     = cuda::SetAll<VecF<kSpatialNDim>>(1.f);
    cuda::GetElement(originV, axis) = origin;
    cuda::GetElement(scaleV, axis)  = scale;
    nn::Resample<kSpatialNDim>(outWrap, inWrap, lo, hi, originV, scaleV, inShape, numChannels);
}

template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ RunLinear(const PassOutWrap outWrap, const PassInWrap inWrap,
                                          const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, int axis,
                                          const VecI<kSpatialNDim> inShape, const float origin, const float scale,
                                          const NumChannelsT numChannels)
{
    if (axis == 0)
    {
        linear::ResampleHorz<kSpatialNDim>(outWrap, inWrap, lo, hi, origin, scale, inShape, numChannels);
    }
    else if (axis == 1)
    {
        linear::ResampleVert<kSpatialNDim>(outWrap, inWrap, lo, hi, origin, scale, inShape, numChannels);
    }
    else if (axis == 2)
    {
        if constexpr (kSpatialNDim == 3)
        {
            linear::ResampleDepth(outWrap, inWrap, lo, hi, origin, scale, inShape, numChannels);
        }
    }
}

template<int kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT>
void __forceinline__ __device__ RunFilter(const PassOutWrap outWrap, const PassInWrap inWrap,
                                          const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, int axis,
                                          const VecI<kSpatialNDim> inShape, const float origin, const float scale,
                                          const filter::ResamplingFilter filter, const NumChannelsT numChannels)
{
    if (axis == 0)
    {
        filter_support::ResampleHorz<kSpatialNDim>(outWrap, inWrap, lo, hi, origin, scale, inShape, filter,
                                                   numChannels);
    }
    else if (axis == 1)
    {
        filter_support::ResampleVert<kSpatialNDim>(outWrap, inWrap, lo, hi, origin, scale, inShape, filter,
                                                   numChannels);
    }
    else if (axis == 2)
    {
        if constexpr (kSpatialNDim == 3)
        {
            filter_support::ResampleDepth(outWrap, inWrap, lo, hi, origin, scale, inShape, filter, numChannels);
        }
    }
}
} // namespace interpolate

template<int kWhichPass, typename PassOutWrap, typename PassInWrap, int kSpatialNDim, typename NumChannelsT>
void __forceinline__ __device__ RunResamplingPass(const SampleDesc<kSpatialNDim> sampleDesc, const PassOutWrap outWrap,
                                                  const PassInWrap inWrap, const VecI<kSpatialNDim> lo,
                                                  const VecI<kSpatialNDim> hi, const NumChannelsT numChannels)
{
    VecI<kSpatialNDim> inShape = sampleDesc.shapes[kWhichPass];
    int         axis   = cuda::GetElement(sampleDesc.processingOrder, kWhichPass); // vec-order: 0 = X, 1 = Y, 2 = Z
    const float origin = cuda::GetElement(sampleDesc.origin, kWhichPass);
    const float scale  = cuda::GetElement(sampleDesc.scale, kWhichPass);

    switch (sampleDesc.filterKind[kWhichPass])
    {
    case filter::FilterTypeKind::Nearest:
        interpolate::RunNN<kSpatialNDim>(outWrap, inWrap, lo, hi, axis, inShape, origin, scale, numChannels);
        break;
    case filter::FilterTypeKind::Linear:
        interpolate::RunLinear<kSpatialNDim>(outWrap, inWrap, lo, hi, axis, inShape, origin, scale, numChannels);
        break;
    default:
        interpolate::RunFilter<kSpatialNDim>(outWrap, inWrap, lo, hi, axis, inShape, origin, scale,
                                             sampleDesc.filter[kWhichPass], numChannels);
        break;
    }
}

// Tensor variant (unfirom batch)
template<int kNumStaticChannels, int kWhichPass, typename PassOutWrap, typename PassInWrap, int kSpatialNDim>
__global__ void SeparableResamplingKernel(const SampleDesc<kSpatialNDim> sampleDesc, const PassOutWrap outWrap,
                                          const PassInWrap inWrap, const GridHelperDevice<kSpatialNDim> gridHelper)

{
    constexpr bool kHasDynamicChannels = kNumStaticChannels == -1;
    static_assert(PassInWrap::kNumDimensions == 1 + kSpatialNDim + kHasDynamicChannels);
    static_assert(PassOutWrap::kNumDimensions == 1 + kSpatialNDim + kHasDynamicChannels);
    // Get sample idx and the region of the output image that
    // the current threadblock has to process
    int                sampleIdx = gridHelper.CurrentSample();
    VecI<kSpatialNDim> lo, hi;
    gridHelper.CurrentBlock(lo, hi, sampleDesc.blockShape[kWhichPass]);
    hi = cuda::min(hi, sampleDesc.shapes[kWhichPass + 1]);

    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    WithChannels<kNumStaticChannels>(
        sampleDesc.channels, [=](const NumChannels<kNumStaticChannels> numChannels)
        { RunResamplingPass<kWhichPass>(sampleDesc, outSampleView, inSampleView, lo, hi, numChannels); });
}

// Batch variant (ImageBatchVarShape, TensorBatch)
template<int kNumStaticChannels, int kWhichPass, typename PassOutWrap, typename PassInWrap, int kSpatialNDim>
__global__ void SeparableResamplingKernel(const SampleDesc<kSpatialNDim> *__restrict__ samples,
                                          const PassOutWrap outWrap, const PassInWrap inWrap,
                                          const GridHelperDevice<kSpatialNDim> gridHelper)
{
    constexpr bool kHasDynamicChannels = kNumStaticChannels == -1;
    static_assert(PassInWrap::kNumDimensions == 1 + kSpatialNDim + kHasDynamicChannels);
    static_assert(PassOutWrap::kNumDimensions == 1 + kSpatialNDim + kHasDynamicChannels);
    // Get sample idx and the region of the output image that
    // the current threadblock has to process
    const int                sampleIdx  = gridHelper.CurrentSample();
    const auto               sampleDesc = samples[sampleIdx];
    const VecI<kSpatialNDim> outShape   = sampleDesc.shapes[kWhichPass + 1];
    VecI<kSpatialNDim>       lo, hi;
    gridHelper.CurrentBlock(lo, hi, sampleDesc.blockShape[kWhichPass]);

    // exit early for smaller samples
    if (lo.x >= outShape.x || lo.y >= outShape.y)
    {
        return;
    }
    if constexpr (kSpatialNDim == 3)
    {
        if (lo.z >= outShape.z)
        {
            return;
        }
    }
    hi = cuda::min(hi, outShape);

    const auto outSampleView = outWrap.GetSampleView(sampleIdx);
    WithChannels<kNumStaticChannels>(
        sampleDesc.channels,
        [=](const NumChannels<kNumStaticChannels> numChannels)
        {
            if constexpr (kWhichPass == 0)
            {
                const auto inSampleView = inWrap.GetSampleView(sampleIdx, sampleDesc.inRoiOffset);
                RunResamplingPass<kWhichPass>(sampleDesc, outSampleView, inSampleView, lo, hi, numChannels);
            }
            else if constexpr (kWhichPass != 0)
            {
                const auto inSampleView = inWrap.GetSampleView(sampleIdx);
                RunResamplingPass<kWhichPass>(sampleDesc, outSampleView, inSampleView, lo, hi, numChannels);
            }
        });
}

} // namespace resampling

namespace validate {
inline auto srcDst(const nvcv::Tensor &src, const nvcv::Tensor &dst)
{
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();

    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be cuda-accessible tensor");
    }

    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be cuda-accessible tensor");
    }

    using maybeTensorAccess = nvcv::Optional<nvcv::TensorDataAccessStridedImagePlanar>;
    std::tuple<maybeTensorAccess, maybeTensorAccess, int, int, nvcv::DataType, nvcv::DataType> ret;

    auto &[srcAccess, dstAccess, numSamples, numChannels, srcDtype, dstDtype] = ret;

    srcDtype = srcData->dtype();
    dstDtype = dstData->dtype();

    srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    numSamples = srcAccess->numSamples();
    if (numSamples != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    if (srcDtype.numChannels() > 1 || dstDtype.numChannels() > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The tensor channels should be explicit part of the shape, not tensor type");
    }

    numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    if (numChannels <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Number of channels must be positive");
    }

    auto numPlanes = srcAccess->numPlanes();
    if (numPlanes != dstAccess->numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }

    if (numPlanes > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Plannar images are not supported");
    }

    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    return ret;
}

inline void srcDst(int &numSamples, int &uniqueNumChannels, nvcv::DataType &srcDtype, nvcv::DataType &dstDtype,
                   const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst)
{
    numSamples = src.numImages();
    if (numSamples != dst.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    const auto &srcFormat = src.uniqueFormat();
    const auto &dstFormat = dst.uniqueFormat();

    if (!srcFormat || !dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "All images in a batch must have the same format (including number of channels)");
    }

    auto numPlanes = srcFormat.numPlanes();
    if (numPlanes != dstFormat.numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }

    if (numPlanes > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Plannar images are not supported");
    }

    srcDtype = srcFormat.planeDataType(0);
    dstDtype = dstFormat.planeDataType(0);

    uniqueNumChannels = srcFormat.numChannels();
    if (uniqueNumChannels != dstFormat.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
}

inline void srcDst(int &numSamples, int &uniqueNumChannels, nvcv::DataType &srcDtype, nvcv::DataType &dstDtype,
                   const nvcv::TensorBatch &src, const nvcv::TensorBatch &dst)
{
    numSamples = src.numTensors();
    if (numSamples != dst.numTensors())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    uniqueNumChannels = -1;
    srcDtype          = src.dtype();
    dstDtype          = dst.dtype();

    if (srcDtype.numChannels() > 1 || dstDtype.numChannels() > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The tensor channels should be explicit part of the shape, not tensor type");
    }

    if (src.layout() != dst.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output layouts");
    }

    if (src.layout() != nvcv::TENSOR_HW && src.layout() != nvcv::TENSOR_HWC && src.layout() != nvcv::TENSOR_DHW
        && src.layout() != nvcv::TENSOR_DHWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The tensor batch must contain [D]HW[C] samples");
    }
}

inline void inOutNumberOfChannels(const HQResizeTensorShapeI &inShape, const HQResizeTensorShapeI &outShape)
{
    if (inShape.numChannels != outShape.numChannels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Incompatible input/output number of channels in one of the samples");
    }
    if (inShape.numChannels <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The number of channels must be positive");
    }
}

inline void sameInOutNdim(const HQResizeTensorShapeI &inShape, const HQResizeTensorShapeI &outShape)
{
    if (inShape.ndim != outShape.ndim)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Incompatible input/output number extents to resize");
    }
}

inline void inOutShapes(int numSamples, const HQResizeTensorShapesI &inShapes, const HQResizeTensorShapesI &outShapes)
{
    if (inShapes.ndim != outShapes.ndim)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The dimensionality of input and output shapes does not match");
    }

    if (numSamples != inShapes.size || numSamples != outShapes.size)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    if (inShapes.ndim != outShapes.ndim)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of extents");
    }

    if (inShapes.numChannels != outShapes.numChannels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    if (inShapes.numChannels < 0)
    {
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
        {
            inOutNumberOfChannels(inShapes.shape[sampleIdx], outShapes.shape[sampleIdx]);
        }
    }
    else if (inShapes.numChannels == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The number of channels cannot be 0");
    }
}

inline void roiBatch(int numSamples, int ndim, const HQResizeRoisF &rois)
{
    auto numRois = rois.size;
    if (numRois != 0 && numRois != 1 && numRois != numSamples)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The resize ROI list, if specified, must contain a single element to be used across all "
                              "samples in a batch or its length must match the batch size.");
    }
    if (numRois != 0)
    {
        if (rois.ndim != ndim)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The number of ROI extents does not match the numebr of extents in the input");
        }
    }
}
} // namespace validate

namespace shape {

template<typename T, int kSpatialNDim>
struct Roi
{
    Vec<T, kSpatialNDim> Size() const
    {
        return hi - lo;
    }

    Vec<T, kSpatialNDim> lo, hi;
};

inline HQResizeRoiF *SampleRoi(const HQResizeRoisF &rois, int sampleIdx)
{
    if (rois.size == 0)
    {
        return nullptr;
    }
    else if (rois.size == 1)
    {
        return rois.roi;
    }
    else
    {
        return rois.roi + sampleIdx;
    }
}

template<int kSpatialNDim>
inline VecI<kSpatialNDim> TensorShape(const HQResizeTensorShapeI &shape)
{
    VecI<kSpatialNDim> shapeVec;
    for (int d = 0; d < kSpatialNDim; d++)
    {
        cuda::GetElement(shapeVec, d) = shape.extent[kSpatialNDim - d - 1];
    }
    return shapeVec;
}

template<int kSpatialNDim>
inline VecI<kSpatialNDim> SampleShape(const HQResizeTensorShapesI &shapes, int sampleIdx)
{
    return TensorShape<kSpatialNDim>(shapes.shape[shapes.size == 1 ? 0 : sampleIdx]);
}

template<int kSpatialNDim>
inline VecI<kSpatialNDim> TensorShape(const nvcv::Tensor &tensor)
{
    static_assert(kSpatialNDim == 2 || kSpatialNDim == 3);
    const auto        &shape             = tensor.shape();
    const auto        &layout            = tensor.layout();
    char               shapeArgLayout[4] = "WHD";
    VecI<kSpatialNDim> tensorShape;
    for (int d = 0; d < kSpatialNDim; d++)
    {
        int axis = layout.find(shapeArgLayout[d]);
        if (axis < 0)
        {
            throw std::runtime_error(
                "The layout of an input tensor to the resize operator must contain HW extents in the layout (for "
                "images) or DHW extents (for 3D resampling). Some extents are missing in the input tensor.");
        }
        cuda::GetElement(tensorShape, d) = shape[axis];
    }
    return tensorShape;
}

template<int kSpatialNDim>
inline VecI<kSpatialNDim> SampleShape(const nvcv::ImageBatchVarShape &batch, int sampleIdx)
{
    static_assert(kSpatialNDim == 2);
    VecI<kSpatialNDim> sampleShape;
    const nvcv::Image &image     = batch[sampleIdx];
    const auto        &imageSize = image.size();
    sampleShape.x                = imageSize.w;
    sampleShape.y                = imageSize.h;
    return sampleShape;
}

template<int kSpatialNDim>
inline VecI<kSpatialNDim> SampleShape(const nvcv::TensorBatch &batch, int sampleIdx)
{
    return TensorShape<kSpatialNDim>(batch[sampleIdx]);
}

inline int TensorNumChannels(const nvcv::Tensor &tensor)
{
    const auto &shape       = tensor.shape();
    const auto &layout      = tensor.layout();
    int         channelAxis = layout.find('C');
    if (channelAxis < 0)
    {
        return 1;
    }
    return shape[channelAxis];
}

inline int64_t TensorByteSize(const nvcv::Tensor &tensor)
{
    auto data = tensor.exportData().cast<nvcv::TensorDataStrided>();
    assert(data);
    return data->stride(0) * data->shape(0);
}

inline int64_t ImageByteSize(const nvcv::Image &image)
{
    auto data = image.exportData<nvcv::ImageDataStrided>();
    assert(data);
    auto plane = data->plane(0); // only single-plane images are supported
    return plane.rowStride * plane.height;
}

inline int SampleNumChannels(const nvcv::TensorBatch &src, const nvcv::TensorBatch &dst, int sampleIdx)
{
    const auto &srcSample   = src[sampleIdx];
    const auto &dstSample   = dst[sampleIdx];
    int         numChannels = TensorNumChannels(srcSample);
    if (numChannels != TensorNumChannels(dstSample))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Number of channels must be positive");
    }
    return numChannels;
}
} // namespace shape

/**
 * @brief Calculates optimum processing order based on input/output sizes and filter support.
 *
 * The sizes of intermediate storage and time taken to compute the intermediate images
 * may depend on the order - i.e. if downscaling only one axis, it's beneficial to resample that
 * axis first, so that intermediate image is smaller.
 */
template<int ndim>
class ProcessingOrderCalculator

{
public:
    static constexpr float size_bias = 3;

    ProcessingOrderCalculator(const VecI<ndim> inSize, const VecI<ndim> outSize, const VecI<ndim> filterSupport)
        : m_inSize(inSize)
        , m_outSize(outSize)
        , m_filterSupport(filterSupport)
    {
    }

    VecI<ndim> operator()()
    {
        for (int i = 0; i < ndim; i++) cuda::GetElement(m_bestOrder, i) = i;
        m_axisVisited = {};
        m_currSize    = m_inSize;
        m_minCost     = 1e+30f;
        Run(0);
        return m_bestOrder;
    }

private:
    // recursively check every possible order in DFS fashion
    void Run(int pass, float totalCost = 0)
    {
        if (totalCost >= m_minCost)
            return; // this branch of recursion will not yield a better result - abandon it

        if (pass == ndim)
        {
            m_minCost   = totalCost;
            m_bestOrder = m_currOrder;
        }
        else
        {
            for (int a = 0; a < ndim; a++)
            {
                if (cuda::GetElement(m_axisVisited, a))
                    continue;
                cuda::GetElement(m_axisVisited, a)  = true;
                cuda::GetElement(m_currOrder, pass) = a;
                auto prevSize                       = cuda::GetElement(m_currSize, a);
                cuda::GetElement(m_currSize, a)     = cuda::GetElement(m_outSize, a);

                float passCost = PassCost(pass, a);
                Run(pass + 1, totalCost + passCost);

                cuda::GetElement(m_currSize, a)    = prevSize;
                cuda::GetElement(m_axisVisited, a) = false;
            }
        }
    }

    float PassCost(int pass, int axis)
    {
        // y-axis is likely to be the cheapest
        float axisCost        = axis == 0 ? 1.4f : axis > 1 ? 1.2f : 1.0f;
        auto  vol             = utils::Volume(m_currSize);
        float baseComputeCost = cuda::GetElement(m_filterSupport, axis) * vol;
        return axisCost * baseComputeCost + vol * size_bias;
    }

    const VecI<ndim> m_inSize, m_outSize, m_filterSupport;
    float            m_minCost;
    VecI<ndim>       m_currSize, m_bestOrder, m_currOrder, m_axisVisited;
};

template<typename IntermediateBaseT, typename Cb>
inline void RunTypedSwitch(nvcv::DataType srcDtype, nvcv::DataType dstDtype, int numChannels, const Cb &cb)
{
    using uchar = unsigned char;

#define NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(SRC_TYPE_NAME, DST_TYPE_NAME, SRC_VEC, DST_VEC) \
    ((srcDtype == nvcv::TYPE_##SRC_TYPE_NAME) && (dstDtype == nvcv::TYPE_##DST_TYPE_NAME))  \
        cb(SRC_VEC{}, IntermediateBaseT{}, DST_VEC{}, std::integral_constant<int, -1>{})

#define NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(NUM_STATIC_CHANNELS, SRC_TYPE_NAME, DST_TYPE_NAME, SRC_VEC, DST_VEC) \
    ((numChannels == NUM_STATIC_CHANNELS) && (srcDtype == nvcv::TYPE_##SRC_TYPE_NAME)                          \
     && (dstDtype == nvcv::TYPE_##DST_TYPE_NAME))                                                              \
        cb(SRC_VEC{}, IntermediateBaseT{}, DST_VEC{}, std::integral_constant<int, NUM_STATIC_CHANNELS>{})

#define NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(NUM_STATIC_CHANNELS, SRC_TYPE_NAME, DST_TYPE_NAME, SRC_VEC, DST_VEC) \
    ((numChannels == NUM_STATIC_CHANNELS)                                                                            \
     && (srcDtype == nvcv::TYPE_##SRC_TYPE_NAME || srcDtype == nvcv::TYPE_##NUM_STATIC_CHANNELS##SRC_TYPE_NAME)      \
     && (dstDtype == nvcv::TYPE_##DST_TYPE_NAME || dstDtype == nvcv::TYPE_##NUM_STATIC_CHANNELS##DST_TYPE_NAME))     \
        cb(SRC_VEC##NUM_STATIC_CHANNELS{}, Vec<IntermediateBaseT, NUM_STATIC_CHANNELS>{},                            \
           DST_VEC##NUM_STATIC_CHANNELS{}, std::integral_constant<int, NUM_STATIC_CHANNELS>{})

    // clang-format off
    if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, U8, U8, uchar, uchar);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, U8, U8, uchar, uchar);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, U8, U8, uchar, uchar);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, U8, U8, uchar, uchar);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(U8, U8, uchar, uchar);

    else if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, U8, F32, uchar, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, U8, F32, uchar, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, U8, F32, uchar, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, U8, F32, uchar, float);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(U8, F32, uchar, float);

    else if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, S16, S16, short, short);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, S16, S16, short, short);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, S16, S16, short, short);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, S16, S16, short, short);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(S16, S16, short, short);

    else if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, S16, F32, short, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, S16, F32, short, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, S16, F32, short, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, S16, F32, short, float);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(S16, F32, short, float);

    else if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, U16, U16, ushort, ushort);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, U16, U16, ushort, ushort);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, U16, U16, ushort, ushort);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, U16, U16, ushort, ushort);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(U16, U16, ushort, ushort);

    else if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, U16, F32, ushort, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, U16, F32, ushort, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, U16, F32, ushort, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, U16, F32, ushort, float);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(U16, F32, ushort, float);

    else if NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE(1, F32, F32, float, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(2, F32, F32, float, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(3, F32, F32, float, float);
    else if NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE(4, F32, F32, float, float);
    else if NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE(F32, F32, float, float);
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Unsupported input/output types. The resize operator supports the "
            "following types: uint8, int16, uint16, and float32. "
            "The output type must be same as the input type or float.");
    }
// clang-format on
#undef NVCV_RUN_DYNAMIC_CHANNELS_HQ_RESIZE
#undef NVCV_RUN_SINGLE_CHANNEL_HQ_RESIZE
#undef NVCV_RUN_MULTI_STATIC_CHANNEL_HQ_RESIZE
}

template<int _kSpatialNDim, typename IntermediateBaseT = float>
class HQResizeRun
{
public:
    static_assert(_kSpatialNDim == 2 || _kSpatialNDim == 3,
                  "Currently, the resampling operator supports only 2 or 3 spatial dimensions");

    HQResizeRun(const filter::ResamplingFiltersFactory &filtersFactory)
        : m_filtersFactory{filtersFactory}
    {
    }

    using SampleDescT = resampling::SampleDesc<_kSpatialNDim>;
    static_assert(std::is_trivially_copyable_v<SampleDescT>);
    using DynamicBatchWrapMeta = batch_wrapper::dynamic::DynamicBatchWrapMeta;

    static constexpr VecI<3> kBlockDim    = {32, 8, 1};
    static constexpr int     kSpatialNDim = _kSpatialNDim;
    // the number of buffers for intermediate results
    static constexpr int     kNumTmpBuffers = kSpatialNDim - 1;
    // use alignment suitable for maximal supported number of static channels
    static constexpr int     kIntermediateAlignment = alignof(Vec<IntermediateBaseT, 4>);

    // Computes workspace requierements for calling the operator with tensor (uniform batch) input/output
    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapeI inputShape,
                                                           const HQResizeTensorShapeI  outputShape,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation,
                                                           const bool antialias, const HQResizeRoiF *roi) const
    {
        validate::inOutNumberOfChannels(inputShape, outputShape);
        validate::sameInOutNdim(inputShape, outputShape);

        SampleDescT        sampleDesc;
        VecI<kSpatialNDim> srcShape    = shape::TensorShape<kSpatialNDim>(inputShape);
        VecI<kSpatialNDim> dstShape    = shape::TensorShape<kSpatialNDim>(outputShape);
        int                numChannels = inputShape.numChannels;
        auto [minFilter, magFilter]    = filter::GetFilterModes(minInterpolation, magInterpolation, antialias);
        SetupSampleDesc(sampleDesc, srcShape, dstShape, numChannels, roi, minFilter, magFilter);

        cvcuda::WorkspaceEstimator est;
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            // the vectorized alignment may or may not be needed, depending on the number of channels
            est.addCuda<IntermediateBaseT>(GetPassOutputVolume(sampleDesc, t) * numSamples, kIntermediateAlignment);
        }

        cvcuda::WorkspaceRequirements req{};
        req.hostMem   = est.hostMem.req;
        req.pinnedMem = est.pinnedMem.req;
        req.cudaMem   = est.cudaMem.req;

        // The allocator requries the total size of the allocation to be aligned
        cvcuda::AlignUp(req);
        return req;
    }

    // Computes workspace requirements for calling the operator with TensorBatch/ImageBatchVarShape input/output
    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapesI inputShapes,
                                                           const HQResizeTensorShapesI outputShapes,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation,
                                                           const bool antialias, const HQResizeRoisF rois) const
    {
        validate::roiBatch(numSamples, kSpatialNDim, rois);
        validate::inOutShapes(numSamples, inputShapes, outputShapes);
        auto [minFilter, magFilter] = filter::GetFilterModes(minInterpolation, magInterpolation, antialias);

        size_t intermediateSizes[kNumTmpBuffers]{};
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
        {
            const VecI<kSpatialNDim> srcShape  = shape::SampleShape<kSpatialNDim>(inputShapes, sampleIdx);
            const VecI<kSpatialNDim> dstShape  = shape::SampleShape<kSpatialNDim>(outputShapes, sampleIdx);
            const HQResizeRoiF      *sampleRoi = shape::SampleRoi(rois, sampleIdx);
            int                      numChannels
                = inputShapes.numChannels < 0 ? inputShapes.shape[sampleIdx].numChannels : inputShapes.numChannels;

            SampleDescT sampleDesc;
            SetupSampleDesc(sampleDesc, srcShape, dstShape, numChannels, sampleRoi, minFilter, magFilter);
            for (int t = 0; t < kNumTmpBuffers; t++)
            {
                intermediateSizes[t] += GetPassOutputVolume(sampleDesc, t);
            }
        }

        cvcuda::WorkspaceEstimator est;
        est.addPinned<SampleDescT>(numSamples);
        est.addCuda<SampleDescT>(numSamples);

        // reserve space for pointers and strides for intermediate wrappers
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            batch_wrapper::dynamic::AddDynamicBatchWrapMeta(est, numSamples);
        }
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            // the vectorized alignment may or may not be needed, depending on the number of channels
            est.addCuda<IntermediateBaseT>(intermediateSizes[t], kIntermediateAlignment);
        }

        cvcuda::WorkspaceRequirements req{};
        req.hostMem   = est.hostMem.req;
        req.pinnedMem = est.pinnedMem.req;
        req.cudaMem   = est.cudaMem.req;
        // The allocator requries the total size of the allocation to be aligned
        cvcuda::AlignUp(req);

        return req;
    }

    // Computes upper bound for workspace requirements, i.e. the workspace that meets the computed requirements
    // can be passed to the call with any type of input/output as long as there are no more than maxBatchSize
    // samples that do not exceed the maxShape (in the input nor in the output).
    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int maxNumSamples, const HQResizeTensorShapeI maxShape) const
    {
        validate::inOutNumberOfChannels(maxShape, maxShape);

        cvcuda::WorkspaceEstimator est;
        est.addPinned<SampleDescT>(maxNumSamples);
        est.addCuda<SampleDescT>(maxNumSamples);

        // reserve space for pointers and strides for intermediate wrappers
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            batch_wrapper::dynamic::AddDynamicBatchWrapMeta(est, maxNumSamples);
        }
        VecI<kSpatialNDim> shape = shape::TensorShape<kSpatialNDim>(maxShape);
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            size_t numElements = utils::Volume(shape) * maxNumSamples * maxShape.numChannels;
            est.addCuda<IntermediateBaseT>(numElements, kIntermediateAlignment);
        }

        cvcuda::WorkspaceRequirements req{};
        req.hostMem   = est.hostMem.req;
        req.pinnedMem = est.pinnedMem.req;
        req.cudaMem   = est.cudaMem.req;
        // The allocator requries the total size of the allocation to be aligned
        cvcuda::AlignUp(req);

        return req;
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    const bool antialias, const HQResizeRoiF *roi) const
    {
        auto tensorAccess                                                         = validate::srcDst(src, dst);
        auto &[srcAccess, dstAccess, numSamples, numChannels, srcDtype, dstDtype] = tensorAccess;

        SampleDescT        sampleDesc;
        VecI<kSpatialNDim> srcShape       = shape::TensorShape<kSpatialNDim>(src);
        VecI<kSpatialNDim> dstShape       = shape::TensorShape<kSpatialNDim>(dst);
        const auto [minFilter, magFilter] = filter::GetFilterModes(minInterpolation, magInterpolation, antialias);
        SetupSampleDesc(sampleDesc, srcShape, dstShape, numChannels, roi, minFilter, magFilter);

        cvcuda::WorkspaceAllocator allocator(ws);
        if (ws.cudaMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ws.cudaMem.ready));
        }
        IntermediateBaseT *intermediate[kNumTmpBuffers];
        // Get intermediate buffers
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            intermediate[t] = allocator.getCuda<IntermediateBaseT>(GetPassOutputVolume(sampleDesc, t) * numSamples,
                                                                   kIntermediateAlignment);
        }

        auto inMaxStride  = shape::TensorByteSize(src);
        auto outMaxStride = shape::TensorByteSize(dst);
        bool wideStride   = std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max;

        RunTypedSwitch<IntermediateBaseT>(
            srcDtype, dstDtype, numChannels,
            [&](auto dummySrcVal, auto intermediateVal, auto dummyDstVal, auto numChannelsVal)
            {
                using InT                       = decltype(dummySrcVal);
                using IntermediateT             = decltype(intermediateVal);
                using OutT                      = decltype(dummyDstVal);
                constexpr int numStaticChannels = decltype(numChannelsVal)::value;
                static_assert(numStaticChannels == -1 || numStaticChannels == cuda::NumElements<InT>);
                static_assert(cuda::NumElements<IntermediateT> == cuda::NumElements<InT>);
                static_assert(cuda::NumElements<OutT> == cuda::NumElements<IntermediateT>);

                auto &[srcAccess, dstAccess, numSamples, numChannels, srcDtype, dstDtype] = tensorAccess;
                if (wideStride)
                {
                    RunPasses<OutT, IntermediateT, InT, int64_t, numStaticChannels>(
                        sampleDesc, *dstAccess, *srcAccess, intermediate, numSamples, ws, stream);
                }
                else
                {
                    RunPasses<OutT, IntermediateT, InT, int32_t, numStaticChannels>(
                        sampleDesc, *dstAccess, *srcAccess, intermediate, numSamples, ws, stream);
                }
            });
    }

    template<typename BatchContainer>
    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const BatchContainer &src,
                    const BatchContainer &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, const bool antialias, const HQResizeRoisF rois) const
    {
        int            numSamples;
        int            uniqueNumChannels; // numChannels for ImageBatchVarShape, -1 for TensorBatch
        nvcv::DataType srcDtype, dstDtype;
        validate::srcDst(numSamples, uniqueNumChannels, srcDtype, dstDtype, src, dst);
        validate::roiBatch(numSamples, kSpatialNDim, rois);

        const auto [minFilter, magFilter] = filter::GetFilterModes(minInterpolation, magInterpolation, antialias);
        cvcuda::WorkspaceAllocator allocator(ws);
        if (ws.pinnedMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventSynchronize(ws.pinnedMem.ready));
        }
        if (ws.cudaMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ws.cudaMem.ready));
        }
        SampleDescT *sampleDescsCpu = allocator.getPinned<SampleDescT>(numSamples);
        SampleDescT *sampleDescsGpu = allocator.getCuda<SampleDescT>(numSamples);
        size_t       intermediateSizes[kNumTmpBuffers]{};

        int64_t inMaxStride  = 0;
        int64_t outMaxStride = 0;
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
        {
            const VecI<kSpatialNDim> srcShape  = shape::SampleShape<kSpatialNDim>(src, sampleIdx);
            const VecI<kSpatialNDim> dstShape  = shape::SampleShape<kSpatialNDim>(dst, sampleIdx);
            const HQResizeRoiF      *sampleRoi = shape::SampleRoi(rois, sampleIdx);
            int                      numChannels;
            if constexpr (std::is_same_v<BatchContainer, nvcv::ImageBatchVarShape>)
            {
                numChannels  = uniqueNumChannels;
                inMaxStride  = std::max(inMaxStride, shape::ImageByteSize(src[sampleIdx]));
                outMaxStride = std::max(outMaxStride, shape::ImageByteSize(dst[sampleIdx]));
            }
            else
            {
                static_assert(std::is_same_v<BatchContainer, nvcv::TensorBatch>);
                numChannels  = shape::SampleNumChannels(src, dst, sampleIdx);
                inMaxStride  = std::max(inMaxStride, shape::TensorByteSize(src[sampleIdx]));
                outMaxStride = std::max(outMaxStride, shape::TensorByteSize(dst[sampleIdx]));
            }
            SampleDescT &sampleDesc = sampleDescsCpu[sampleIdx];
            SetupSampleDesc(sampleDesc, srcShape, dstShape, numChannels, sampleRoi, minFilter, magFilter);
            for (int t = 0; t < kNumTmpBuffers; t++)
            {
                intermediateSizes[t] += GetPassOutputVolume(sampleDesc, t);
            }
        }
        bool wideStride = std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max;

        NVCV_CHECK_THROW(cudaMemcpyAsync(sampleDescsGpu, sampleDescsCpu, numSamples * sizeof(SampleDescT),
                                         cudaMemcpyHostToDevice, stream));

        // allocate space for pointers and strides for intermediate wrappers
        DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers];
        IntermediateBaseT   *intermediate[kNumTmpBuffers];
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            intermediateMeta[t]
                = batch_wrapper::dynamic::AllocateDynamicBatchWrapMeta(allocator, numSamples, wideStride);
        }
        // allocate space for intermediate data
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            intermediate[t] = allocator.getCuda<IntermediateBaseT>(intermediateSizes[t], kIntermediateAlignment);
        }

        Run(sampleDescsCpu, sampleDescsGpu, src, dst, intermediate, intermediateMeta, numSamples, srcDtype, dstDtype,
            uniqueNumChannels, wideStride, ws, stream);
    }

private:
    void Run(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu, const nvcv::ImageBatchVarShape &src,
             const nvcv::ImageBatchVarShape &dst, IntermediateBaseT *intermediate[kNumTmpBuffers],
             const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples, const nvcv::DataType srcDtype,
             const nvcv::DataType dstDtype, int uniqueNumChannels, bool wideStride, const cvcuda::Workspace &ws,
             cudaStream_t stream) const
    {
        static_assert(kSpatialNDim == 2, "ImageBatchVarShape does not support 3D spatial resampling");

        auto srcData = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
        auto dstData = dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
        if (!srcData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input must be cuda-accessible, varshape pitch-linear image batch");
        }

        if (!dstData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output must be cuda-accessible, varshape pitch-linear image batch");
        }

        RunTypedSwitch<IntermediateBaseT>(
            srcDtype, dstDtype, uniqueNumChannels,
            [&](auto dummySrcVal, auto intermediateVal, auto dummyDstVal, auto numChannelsVal)
            {
                using InT                       = decltype(dummySrcVal);
                using IntermediateT             = decltype(intermediateVal);
                using OutT                      = decltype(dummyDstVal);
                constexpr int numStaticChannels = decltype(numChannelsVal)::value;
                if constexpr (numStaticChannels == -1)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "Unsupported number of channels for ImageBatchVarShape input.");
                }
                else if constexpr (numStaticChannels != -1)
                {
                    static_assert(numStaticChannels == cuda::NumElements<InT>);
                    static_assert(cuda::NumElements<IntermediateT> == cuda::NumElements<InT>);
                    static_assert(cuda::NumElements<OutT> == cuda::NumElements<IntermediateT>);
                    if (wideStride)
                    {
                        RunPasses<OutT, IntermediateT, InT, int64_t, numStaticChannels>(
                            sampleDescsCpu, sampleDescsGpu, *dstData, *srcData, intermediate, intermediateMeta,
                            numSamples, ws, stream);
                    }
                    else
                    {
                        RunPasses<OutT, IntermediateT, InT, int32_t, numStaticChannels>(
                            sampleDescsCpu, sampleDescsGpu, *dstData, *srcData, intermediate, intermediateMeta,
                            numSamples, ws, stream);
                    }
                }
            });
    }

    void Run(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu, const nvcv::TensorBatch &src,
             const nvcv::TensorBatch &dst, IntermediateBaseT *intermediate[kNumTmpBuffers],
             const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples, const nvcv::DataType srcDtype,
             const nvcv::DataType dstDtype, int uniqueNumChannels, bool wideStride, const cvcuda::Workspace &ws,
             cudaStream_t stream) const
    {
        // Other cointainer allow exporting data with const qualifiers
        const auto srcData
            = const_cast<nvcv::TensorBatch &>(src).exportData(stream).cast<nvcv::TensorBatchDataStridedCuda>();
        const auto dstData
            = const_cast<nvcv::TensorBatch &>(dst).exportData(stream).cast<nvcv::TensorBatchDataStridedCuda>();

        if (!srcData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input must be cuda-accessible, varshape pitch-linear image batch");
        }

        if (!dstData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output must be cuda-accessible, varshape pitch-linear image batch");
        }

        uniqueNumChannels = -1;
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
        {
            if (sampleIdx == 0)
            {
                uniqueNumChannels = sampleDescsCpu[sampleIdx].channels;
            }
            else if (uniqueNumChannels != sampleDescsCpu[sampleIdx].channels)
            {
                uniqueNumChannels = -1;
                break;
            }
        }

        RunTypedSwitch<IntermediateBaseT>(
            srcDtype, dstDtype, uniqueNumChannels,
            [&](auto dummySrcVal, auto intermediateVal, auto dummyDstVal, auto numChannelsVal)
            {
                using InT                       = decltype(dummySrcVal);
                using IntermediateT             = decltype(intermediateVal);
                using OutT                      = decltype(dummyDstVal);
                constexpr int numStaticChannels = decltype(numChannelsVal)::value;
                static_assert(numStaticChannels == -1 || numStaticChannels == cuda::NumElements<InT>);
                static_assert(cuda::NumElements<IntermediateT> == cuda::NumElements<InT>);
                static_assert(cuda::NumElements<OutT> == cuda::NumElements<IntermediateT>);

                if (wideStride)
                {
                    RunPasses<OutT, IntermediateT, InT, int64_t, numStaticChannels>(
                        sampleDescsCpu, sampleDescsGpu, *dstData, *srcData, intermediate, intermediateMeta, numSamples,
                        ws, stream);
                }
                else
                {
                    RunPasses<OutT, IntermediateT, InT, int32_t, numStaticChannels>(
                        sampleDescsCpu, sampleDescsGpu, *dstData, *srcData, intermediate, intermediateMeta, numSamples,
                        ws, stream);
                }
            });
    }

    template<typename OutT, typename IntermediateT, typename InT, typename StrideT, int kNumStaticChannels,
             int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2> RunPasses(const SampleDescT                              &sampleDesc,
                                          const nvcv::TensorDataAccessStridedImagePlanar &dstAccess,
                                          const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                          IntermediateBaseT *intermediate[kNumTmpBuffers], int numSamples,
                                          const cvcuda::Workspace &ws, cudaStream_t stream) const
    {
        static_assert(kSpatialNDim == 2);
        constexpr bool kHasDynamicChannels = kNumStaticChannels == -1;
        // sample extent, spatial extents, optional dynamic channel extent
        constexpr int  kWrapNDim = 1 + kSpatialNDim + kHasDynamicChannels;
        using OutWrap            = cuda::TensorNDWrap<OutT, kWrapNDim, StrideT>;
        using InWrap             = cuda::TensorNDWrap<const InT, kWrapNDim, StrideT>;
        using InterWrap          = cuda::TensorNDWrap<IntermediateT, kWrapNDim, StrideT>;
        static_assert(std::is_trivially_copyable_v<OutWrap>);
        static_assert(std::is_trivially_copyable_v<InWrap>);
        static_assert(std::is_trivially_copyable_v<InterWrap>);
        const OutWrap outWrap
            = batch_wrapper::tensor::WrapTensor<kHasDynamicChannels, kSpatialNDim, OutT, StrideT>(dstAccess);
        const InWrap inWrap = batch_wrapper::tensor::WrapTensor<kHasDynamicChannels, kSpatialNDim, InT, StrideT>(
            srcAccess, sampleDesc.inRoiOffset);
        const InterWrap interWrap = batch_wrapper::tensor::CreateDenseWrap<kHasDynamicChannels, IntermediateT, StrideT>(
            intermediate[0], sampleDesc.channels, sampleDesc.shapes[1]);
        RunPass<kNumStaticChannels, 0>(sampleDesc, interWrap, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(sampleDesc, outWrap, interWrap, numSamples, stream);
        if (ws.cudaMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
        }
    }

    template<typename OutT, typename IntermediateT, typename InT, typename StrideT, int kNumStaticChannels,
             int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 3> RunPasses(const SampleDescT                              &sampleDesc,
                                          const nvcv::TensorDataAccessStridedImagePlanar &dstAccess,
                                          const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                          IntermediateBaseT *intermediate[kNumTmpBuffers], int numSamples,
                                          const cvcuda::Workspace &ws, cudaStream_t stream) const
    {
        static_assert(kSpatialNDim == 3);
        constexpr bool kHasDynamicChannels = kNumStaticChannels == -1;
        // sample extent, spatial extents, optional dynamic channel extent
        constexpr int  kWrapNDim = 1 + kSpatialNDim + kHasDynamicChannels;
        using OutWrap            = cuda::TensorNDWrap<OutT, kWrapNDim, StrideT>;
        using InWrap             = cuda::TensorNDWrap<const InT, kWrapNDim, StrideT>;
        using InterWrap          = cuda::TensorNDWrap<IntermediateT, kWrapNDim, StrideT>;
        static_assert(std::is_trivially_copyable_v<OutWrap>);
        static_assert(std::is_trivially_copyable_v<InWrap>);
        static_assert(std::is_trivially_copyable_v<InterWrap>);
        const OutWrap outWrap
            = batch_wrapper::tensor::WrapTensor<kHasDynamicChannels, kSpatialNDim, OutT, StrideT>(dstAccess);
        const InWrap inWrap = batch_wrapper::tensor::WrapTensor<kHasDynamicChannels, kSpatialNDim, InT, StrideT>(
            srcAccess, sampleDesc.inRoiOffset);
        const InterWrap interWrap0
            = batch_wrapper::tensor::CreateDenseWrap<kHasDynamicChannels, IntermediateT, StrideT>(
                intermediate[0], sampleDesc.channels, sampleDesc.shapes[1]);
        const InterWrap interWrap1
            = batch_wrapper::tensor::CreateDenseWrap<kHasDynamicChannels, IntermediateT, StrideT>(
                intermediate[1], sampleDesc.channels, sampleDesc.shapes[2]);
        RunPass<kNumStaticChannels, 0>(sampleDesc, interWrap0, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(sampleDesc, interWrap1, interWrap0, numSamples, stream);
        RunPass<kNumStaticChannels, 2>(sampleDesc, outWrap, interWrap1, numSamples, stream);
        if (ws.cudaMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
        }
    }

    template<typename OutT, typename IntermediateT, typename InT, typename StrideT, int kNumStaticChannels,
             typename BatchDataStridedCuda, int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2> RunPasses(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu,
                                          const BatchDataStridedCuda &dstData, const BatchDataStridedCuda &srcData,
                                          IntermediateBaseT         *intermediate[kNumTmpBuffers],
                                          const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples,
                                          const cvcuda::Workspace &ws, cudaStream_t stream) const
    {
        static_assert(kSpatialNDim == 2);
        constexpr bool kHasDynamicChannels = kNumStaticChannels == -1;
        // sample extent, spatial extents, optional dynamic channel extent
        constexpr int  kWrapNDim = 1 + kSpatialNDim + kHasDynamicChannels;
        using BatchWrapOutT
            = std::conditional_t<std::is_same_v<BatchDataStridedCuda, nvcv::ImageBatchVarShapeDataStridedCuda>,
                                 batch_wrapper::ImageBatchVarShapeWrapAdapter<OutT, StrideT>,
                                 batch_wrapper::TensorBatchWrapAdapter<OutT, kWrapNDim, StrideT>>;
        using BatchWrapInT
            = std::conditional_t<std::is_same_v<BatchDataStridedCuda, nvcv::ImageBatchVarShapeDataStridedCuda>,
                                 batch_wrapper::ImageBatchVarShapeWrapAdapter<const InT, StrideT>,
                                 batch_wrapper::TensorBatchWrapAdapter<const InT, kWrapNDim, StrideT>>;
        using DynamicBatchWrap = batch_wrapper::dynamic::DynamicBatchWrap<IntermediateT, kWrapNDim, StrideT>;
        static_assert(std::is_trivially_copyable_v<BatchWrapOutT>);
        static_assert(std::is_trivially_copyable_v<BatchWrapInT>);
        static_assert(std::is_trivially_copyable_v<DynamicBatchWrap>);
        const BatchWrapOutT    outWrap(dstData);
        const BatchWrapInT     inWrap(srcData);
        const DynamicBatchWrap intermediateWrap
            = batch_wrapper::dynamic::CreateDynamicBatchWrap<kHasDynamicChannels, IntermediateT, StrideT>(
                0, intermediate[0], intermediateMeta[0], sampleDescsCpu, numSamples, stream);
        if (ws.pinnedMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(ws.pinnedMem.ready, stream));
        }
        RunPass<kNumStaticChannels, 0>(sampleDescsCpu, sampleDescsGpu, intermediateWrap, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(sampleDescsCpu, sampleDescsGpu, outWrap, intermediateWrap, numSamples, stream);
        if (ws.cudaMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
        }
    }

    template<typename OutT, typename IntermediateT, typename InT, typename StrideT, int kNumStaticChannels,
             int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 3> RunPasses(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu,
                                          const nvcv::TensorBatchDataStridedCuda &dstData,
                                          const nvcv::TensorBatchDataStridedCuda &srcData,
                                          IntermediateBaseT                      *intermediate[kNumTmpBuffers],
                                          const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples,
                                          const cvcuda::Workspace &ws, cudaStream_t stream) const
    {
        static_assert(kSpatialNDim == 3);
        constexpr bool kHasDynamicChannels = kNumStaticChannels == -1;
        // sample extent, spatial extents, optional dynamic channel extent
        constexpr int  kWrapNDim  = 1 + kSpatialNDim + kHasDynamicChannels;
        using TensorBatchWrapOutT = batch_wrapper::TensorBatchWrapAdapter<OutT, kWrapNDim, StrideT>;
        using TensorBatchWrapInT  = batch_wrapper::TensorBatchWrapAdapter<const InT, kWrapNDim, StrideT>;
        using DynamicBatchWrap    = batch_wrapper::dynamic::DynamicBatchWrap<IntermediateT, kWrapNDim, StrideT>;
        static_assert(std::is_trivially_copyable_v<TensorBatchWrapOutT>);
        static_assert(std::is_trivially_copyable_v<TensorBatchWrapInT>);
        static_assert(std::is_trivially_copyable_v<DynamicBatchWrap>);
        const TensorBatchWrapOutT outWrap(dstData);
        const TensorBatchWrapInT  inWrap(srcData);
        const DynamicBatchWrap    intermediateWrap0
            = batch_wrapper::dynamic::CreateDynamicBatchWrap<kHasDynamicChannels, IntermediateT, StrideT>(
                0, intermediate[0], intermediateMeta[0], sampleDescsCpu, numSamples, stream);
        const DynamicBatchWrap intermediateWrap1
            = batch_wrapper::dynamic::CreateDynamicBatchWrap<kHasDynamicChannels, IntermediateT, StrideT>(
                1, intermediate[1], intermediateMeta[1], sampleDescsCpu, numSamples, stream);
        if (ws.pinnedMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(ws.pinnedMem.ready, stream));
        }
        RunPass<kNumStaticChannels, 0>(sampleDescsCpu, sampleDescsGpu, intermediateWrap0, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(sampleDescsCpu, sampleDescsGpu, intermediateWrap1, intermediateWrap0, numSamples,
                                       stream);
        RunPass<kNumStaticChannels, 2>(sampleDescsCpu, sampleDescsGpu, outWrap, intermediateWrap1, numSamples, stream);
        if (ws.cudaMem.ready != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
        }
    }

    template<int kNumStaticChannels, int kWhichPass, typename PassOutWrap, typename PassInWrap>
    void RunPass(const SampleDescT &sampleDesc, const PassOutWrap &outWrap, const PassInWrap &inWrap, int numSamples,
                 cudaStream_t stream) const
    {
        using GridHelperT = resampling::GridHelper<kSpatialNDim>;

        VecI<kSpatialNDim> numBlocks;
        {
            VecI<kSpatialNDim> outputShape = sampleDesc.shapes[kWhichPass + 1];
            VecI<kSpatialNDim> blockShape  = sampleDesc.blockShape[kWhichPass];
            numBlocks                      = utils::DivCeil(outputShape, blockShape);
            if (utils::Volume(numBlocks) == 0)
            {
                return;
            }
        }

        GridHelperT gridHelper{numBlocks, numSamples};
        dim3        block(kBlockDim.x, kBlockDim.y, kBlockDim.z);
        dim3        grid          = gridHelper.GetKernelGrid();
        const auto  devGridHelper = gridHelper.GetDeviceGridHelper();

        int sharedMemSize = RequiredSharedMemorySize(sampleDesc, kWhichPass);
        resampling::SeparableResamplingKernel<kNumStaticChannels, kWhichPass>
            <<<grid, block, sharedMemSize, stream>>>(sampleDesc, outWrap, inWrap, devGridHelper);
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    template<int kNumStaticChannels, int kWhichPass, typename PassOutWrap, typename PassInWrap>
    void RunPass(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu, const PassOutWrap &outWrap,
                 const PassInWrap &inWrap, int numSamples, cudaStream_t stream) const
    {
        using GridHelperT = resampling::GridHelper<kSpatialNDim>;

        int                maxSharedMemSize = 0;
        VecI<kSpatialNDim> maxNumBlocks{};
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
        {
            const SampleDescT &sampleDesc    = sampleDescsCpu[sampleIdx];
            int                sharedMemSize = RequiredSharedMemorySize(sampleDesc, kWhichPass);
            maxSharedMemSize                 = std::max(maxSharedMemSize, sharedMemSize);

            VecI<kSpatialNDim> outputShape = sampleDesc.shapes[kWhichPass + 1];
            VecI<kSpatialNDim> blockShape  = sampleDesc.blockShape[kWhichPass];
            VecI<kSpatialNDim> numBlocks   = utils::DivCeil(outputShape, blockShape);
            maxNumBlocks                   = cuda::max(maxNumBlocks, numBlocks);
        }

        if (utils::Volume(maxNumBlocks) == 0)
        {
            return;
        }

        GridHelperT gridHelper{maxNumBlocks, numSamples};
        dim3        block(kBlockDim.x, kBlockDim.y, kBlockDim.z);
        dim3        grid          = gridHelper.GetKernelGrid();
        const auto  devGridHelper = gridHelper.GetDeviceGridHelper();

        resampling::SeparableResamplingKernel<kNumStaticChannels, kWhichPass>
            <<<grid, block, maxSharedMemSize, stream>>>(sampleDescsGpu, outWrap, inWrap, devGridHelper);
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    int RequiredSharedMemorySize(const SampleDescT &sampleDesc, int whichPass) const
    {
        using resampling::interpolate::filter_support::RequiredSharedMemoryElements;
        if (sampleDesc.filterKind[whichPass] != filter::FilterTypeKind::ShmFilter)
        {
            return 0;
        }
        int support = sampleDesc.filter[whichPass].support();
        int axis    = cuda::GetElement(sampleDesc.processingOrder, whichPass);
        // for depth resampling y is used as well
        int resamplingAxisBlockSize = axis == 0 ? kBlockDim.x : kBlockDim.y;
        return sizeof(IntermediateBaseT) * RequiredSharedMemoryElements(support, resamplingAxisBlockSize);
    }

    void SetupSampleDesc(SampleDescT &sampleDesc, const VecI<kSpatialNDim> &srcShape,
                         const VecI<kSpatialNDim> &dstShape, int numChannels, const HQResizeRoiF *roi,
                         const filter::FilterMode &minFilter, const filter::FilterMode &magFilter) const
    {
        SetupSampleDescFilterShapeScale(sampleDesc, srcShape, dstShape, numChannels, minFilter, magFilter, roi);
        SetupBlockLayout(sampleDesc);
    }

    void SetupSampleDescFilterShapeScale(SampleDescT &sampleDesc, const VecI<kSpatialNDim> &inShape,
                                         const VecI<kSpatialNDim> &outShape, int numChannels,
                                         const filter::FilterMode &minFilter, const filter::FilterMode &magFilter,
                                         const HQResizeRoiF *roi) const
    {
        // get user provided roi
        const shape::Roi<float, kSpatialNDim> parsedRoi = ParseROI(roi, inShape);
        // setup filter based on user provided filter types and the input/output size
        filter::FilterTypeKind                filterKinds[kSpatialNDim];
        filter::ResamplingFilter              filters[kSpatialNDim];
        SetupFilters(filterKinds, filters, parsedRoi.Size(), outShape, minFilter, magFilter);
        // get the ROI that is normalized (so that roiLo <= roiHi), adjusted for filter's "halo",
        // and clampped to input shape
        const shape::Roi<int, kSpatialNDim> adjustedRoi     = AdjustRoiForFilter(parsedRoi, inShape, filters);
        VecI<kSpatialNDim>                  adjustedRoiSize = adjustedRoi.Size();
        // the processing order is permutation that maps pass number to axis resampled during given pass
        sampleDesc.processingOrder = SetupProcessingOrder(adjustedRoiSize, outShape, filters);
        // now, use filters, roi and processingOrder to populate sample descriptor
        sampleDesc.channels  = numChannels;
        sampleDesc.shapes[0] = inShape;
        // set output shapes, scaling, roi, and relevant filters for each pass
        // according to the best processingOrder of axes
        {
            VecI<kSpatialNDim> intermediateShape = adjustedRoiSize;
            for (int pass = 0; pass < kSpatialNDim; pass++)
            {
                const int   axis         = cuda::GetElement(sampleDesc.processingOrder, pass);
                const int   axisOutShape = cuda::GetElement(outShape, axis);
                const float roiStart     = cuda::GetElement(parsedRoi.lo, axis);
                const float roiEnd       = cuda::GetElement(parsedRoi.hi, axis);

                cuda::GetElement(intermediateShape, axis) = axisOutShape;
                sampleDesc.filterKind[pass]               = filterKinds[axis];
                sampleDesc.filter[pass]                   = filters[axis];
                sampleDesc.shapes[pass + 1]               = intermediateShape;

                cuda::GetElement(sampleDesc.origin, pass) = roiStart;
                cuda::GetElement(sampleDesc.scale, pass)  = (roiEnd - roiStart) / axisOutShape;

                // "Clamp" the axes processed in later passes to the input ROI
                if (pass == 0)
                {
                    // the first processed axis roi is handled simply with the `origin`
                    cuda::GetElement(sampleDesc.inRoiOffset, axis) = 0;
                }
                else
                {
                    // for the axes not resampled in the first pass, we can just use offset when accesing data
                    // (adjustedRoi.lo) and pretend the input shape is the adjustedRoi.Size()
                    cuda::GetElement(sampleDesc.shapes[0], axis)   = cuda::GetElement(adjustedRoiSize, axis);
                    cuda::GetElement(sampleDesc.inRoiOffset, axis) = cuda::GetElement(adjustedRoi.lo, axis);
                    cuda::GetElement(sampleDesc.origin, pass)
                        -= cuda::GetElement(adjustedRoi.lo, axis); // parsedRoi.lo - adjustedRoi.lo
                }
            }
        }
    }

    /**
     * @brief If user specified the roi, it's returned with reversed dims oreder ((d)hw -> wh(d)),
     * otherwise the input shape is used to create whole-plane roi.
     * Note, that in the first case, some lo and hi may be flipped (i.e. lo[d] > hi[d]).
     */
    shape::Roi<float, kSpatialNDim> ParseROI(const HQResizeRoiF *roi, VecI<kSpatialNDim> inShape) const
    {
        shape::Roi<float, kSpatialNDim> retRoi;
        for (int dim = 0; dim < kSpatialNDim; dim++)
        {
            int   axis     = kSpatialNDim - 1 - dim;
            auto  axisSize = cuda::GetElement(inShape, axis);
            float roiStart, roiEnd;
            if (roi != nullptr)
            {
                roiStart = roi->lo[dim];
                roiEnd   = roi->hi[dim];
            }
            else
            {
                roiStart = 0;
                roiEnd   = axisSize;
            }
            cuda::GetElement(retRoi.lo, axis) = roiStart;
            cuda::GetElement(retRoi.hi, axis) = roiEnd;
        }
        return retRoi;
    }

    void SetupFilters(filter::FilterTypeKind filterKind[kSpatialNDim], filter::ResamplingFilter filters[kSpatialNDim],
                      VecF<kSpatialNDim> roiShape, const VecI<kSpatialNDim> &outShape,
                      const filter::FilterMode &minFilter, const filter::FilterMode &magFilter) const
    {
        using resampling::interpolate::filter_support::kMaxGPUFilterSupport;
        static_assert(kSpatialNDim == 2 || kSpatialNDim == 3,
                      "Currently, the resampling operator supports only 2 or 3 spatial dimensions");

        for (int axis = 0; axis < kSpatialNDim; axis++)
        {
            float      inSize     = std::abs(cuda::GetElement(roiShape, axis));
            float      outSize    = cuda::GetElement(outShape, axis);
            const auto filterMode = outSize < inSize ? minFilter : magFilter;
            filterKind[axis]      = filter::GetFilterTypeKind(filterMode.filterType);
            auto &filter          = filters[axis];
            filter                = filter::GetResamplingFilter(m_filtersFactory, filterMode, inSize, outSize);

            // for very small outputs, the required support may be too big for avialable shm
            if (filter.support() > kMaxGPUFilterSupport)
            {
                filter.rescale(kMaxGPUFilterSupport);
            }
        }
    }

    /**
     * @brief Computes normalized ROI (i.e. so that roiLo <= roiHow), which is adjusted for filter's halo,
     * converted to int and clamped to the input shape
     */
    shape::Roi<int, kSpatialNDim> AdjustRoiForFilter(const shape::Roi<float, kSpatialNDim> &roi,
                                                     const VecI<kSpatialNDim>              &inShape,
                                                     const filter::ResamplingFilter         filters[kSpatialNDim]) const
    {
        shape::Roi<int, kSpatialNDim> ajustedRoi;
        for (int axis = 0; axis < kSpatialNDim; axis++)
        {
            const float &axisLo  = cuda::GetElement(roi.lo, axis);
            const float &axisHi  = cuda::GetElement(roi.hi, axis);
            const auto  &filter  = filters[axis];
            int          support = filter.numCoeffs ? filter.support() : 1;
            float        adjustedAxisLo, adjustedAxisHi;
            if (axisLo <= axisHi)
            {
                adjustedAxisLo = axisLo - filter.anchor;
                adjustedAxisHi = axisHi - filter.anchor + support;
            }
            else
            { // flipped
                adjustedAxisLo = axisHi - filter.anchor;
                adjustedAxisHi = axisLo - filter.anchor + support;
            }
            const int axisSize = cuda::GetElement(inShape, axis);
            cuda::GetElement(ajustedRoi.lo, axis)
                = std::max<int>(0, std::min<int>(axisSize, std::floor(adjustedAxisLo)));
            cuda::GetElement(ajustedRoi.hi, axis)
                = std::max<int>(0, std::min<int>(axisSize, std::ceil(adjustedAxisHi)));
        }
        return ajustedRoi;
    }

    VecI<kSpatialNDim> SetupProcessingOrder(const VecI<kSpatialNDim> &inRoiSize, const VecI<kSpatialNDim> &outSize,
                                            const filter::ResamplingFilter filters[kSpatialNDim]) const
    {
        VecI<kSpatialNDim> filterSupport;
        for (int i = 0; i < kSpatialNDim; i++)
        {
            int support = filters[i].support();
            // NN filter has support -1, so we need the max() below
            cuda::GetElement(filterSupport, i) = std::max(1, support);
        }

        return ProcessingOrderCalculator<kSpatialNDim>(inRoiSize, outSize, filterSupport)();
    }

    int64_t GetPassOutputVolume(SampleDescT sampleDesc, int pass) const
    {
        return utils::Volume(sampleDesc.shapes[pass + 1]) * sampleDesc.channels;
    }

    /**
     * @brief Calculates block layout for a 2D sample
     *
     */
    template<int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2> SetupBlockLayout(SampleDescT &sampleDesc) const
    {
        static_assert(kSpatialNDim == 2);
        int lanes = resampling::GetResizeBlockLanes();
        for (int pass = 0; pass < kSpatialNDim; pass++)
        {
            int     resamplingAxis = cuda::GetElement(sampleDesc.processingOrder, pass);
            // The threadblock is (kBlockDim.x, kBlockDim.y) for all passes.
            // In horizontal pass (resamplingAxis == 0), a single block will
            // process output slice of (kBlockDim.x, lanes * kBlockDim.y).
            // In vertical pass (resamplingAxis == 1), each block will handle
            // output slice of (kBlockDim.x * lanes, kBlockDim.y).
            VecI<2> blockShape{kBlockDim.x, kBlockDim.y};
            cuda::GetElement(blockShape, 1 - resamplingAxis) *= lanes;
            auto outputShape            = sampleDesc.shapes[pass + 1];
            sampleDesc.blockShape[pass] = cuda::clamp(blockShape, VecI<2>{1, 1}, outputShape);
        }
    }

    /**
    * @brief Calculates block layout for a 3D sample
    */
    template<int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 3> SetupBlockLayout(SampleDescT &sampleDesc) const
    {
        static_assert(kSpatialNDim == 3);
        int lanes = resampling::GetResizeBlockLanes();
        for (int pass = 0; pass < kSpatialNDim; pass++)
        {
            auto outputShape    = sampleDesc.shapes[pass + 1];
            int  resamplingAxis = cuda::GetElement(sampleDesc.processingOrder, pass);
            if (resamplingAxis < 2)
            {
                VecI<3> blockShape{kBlockDim.x, kBlockDim.y, kBlockDim.z * lanes};
                sampleDesc.blockShape[pass] = cuda::clamp(blockShape, VecI<3>{1, 1, 1}, outputShape);
            }
            else
            {
                assert(resamplingAxis == 2);
                VecI<3> blockShape{kBlockDim.x, kBlockDim.z * lanes, kBlockDim.y};
                sampleDesc.blockShape[pass] = cuda::clamp(blockShape, VecI<3>{1, 1, 1}, outputShape);
            }
        }
    }

    const filter::ResamplingFiltersFactory &m_filtersFactory;
};
} // namespace

namespace cvcuda::priv {
namespace hq_resize {

// Implements the IHQResizeImpl interface and keeps the filters fatory with initilized
// supports. The actual implementation is in a stateless HQResizeRun that is parametrized
// with the number of resampled dimensions.
class HQResizeImpl final : public IHQResizeImpl
{
public:
    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapeI inputShape,
                                                           const HQResizeTensorShapeI  outputShape,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoiF *roi) const override
    {
        if (inputShape.ndim == 2)
        {
            HQResizeRun<2> resize(m_filtersFactory);
            return resize.getWorkspaceRequirements(numSamples, inputShape, outputShape, minInterpolation,
                                                   magInterpolation, antialias, roi);
        }
        else if (inputShape.ndim == 3)
        {
            HQResizeRun<3> resize(m_filtersFactory);
            return resize.getWorkspaceRequirements(numSamples, inputShape, outputShape, minInterpolation,
                                                   magInterpolation, antialias, roi);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Only 2D or 3D resize is supported. Got unexpected number of extents to resize.");
        }
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int numSamples, const HQResizeTensorShapesI inputShapes,
                                                           const HQResizeTensorShapesI outputShapes,
                                                           const NVCVInterpolationType minInterpolation,
                                                           const NVCVInterpolationType magInterpolation, bool antialias,
                                                           const HQResizeRoisF rois) const override
    {
        if (inputShapes.ndim == 2)
        {
            HQResizeRun<2> resize(m_filtersFactory);
            return resize.getWorkspaceRequirements(numSamples, inputShapes, outputShapes, minInterpolation,
                                                   magInterpolation, antialias, rois);
        }
        else if (inputShapes.ndim == 3)
        {
            HQResizeRun<3> resize(m_filtersFactory);
            return resize.getWorkspaceRequirements(numSamples, inputShapes, outputShapes, minInterpolation,
                                                   magInterpolation, antialias, rois);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Only 2D or 3D resize is supported. Got unexpected number of extents to resize.");
        }
    }

    cvcuda::WorkspaceRequirements getWorkspaceRequirements(int                        maxBatchSize,
                                                           const HQResizeTensorShapeI maxShape) const override
    {
        if (maxShape.ndim == 2)
        {
            HQResizeRun<2> resize(m_filtersFactory);
            return resize.getWorkspaceRequirements(maxBatchSize, maxShape);
        }
        else if (maxShape.ndim == 3)
        {
            HQResizeRun<3> resize(m_filtersFactory);
            return resize.getWorkspaceRequirements(maxBatchSize, maxShape);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Only 2D or 3D resize is supported. Got unexpected number of extents to resize.");
        }
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                    const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                    bool antialias, const HQResizeRoiF *roi) override
    {
        if (src.layout().find('D') < 0)
        {
            HQResizeRun<2> resize(m_filtersFactory);
            resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, roi);
        }
        else
        {
            HQResizeRun<3> resize(m_filtersFactory);
            resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, roi);
        }
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::ImageBatchVarShape &src,
                    const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) override
    {
        HQResizeRun<2> resize(m_filtersFactory);
        resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
    }

    void operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::TensorBatch &src,
                    const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                    const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) override
    {
        if (src.layout().find('D') < 0)
        {
            HQResizeRun<2> resize(m_filtersFactory);
            resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
        }
        else
        {
            HQResizeRun<3> resize(m_filtersFactory);
            resize(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
        }
    }

private:
    filter::ResamplingFiltersFactory m_filtersFactory;
};

} // namespace hq_resize

// Constructor -----------------------------------------------------------------

HQResize::HQResize()

{
    m_impl = std::make_unique<hq_resize::HQResizeImpl>();
}

// Operator --------------------------------------------------------------------

// Workspace esitmation for Tensor input
cvcuda::WorkspaceRequirements HQResize::getWorkspaceRequirements(int batchSize, const HQResizeTensorShapeI inputShape,
                                                                 const HQResizeTensorShapeI  outputShape,
                                                                 const NVCVInterpolationType minInterpolation,
                                                                 const NVCVInterpolationType magInterpolation,
                                                                 bool antialias, const HQResizeRoiF *roi) const
{
    return m_impl->getWorkspaceRequirements(batchSize, inputShape, outputShape, minInterpolation, magInterpolation,
                                            antialias, roi);
}

// Workspace esitmation for ImageBatch and TensorBatch input
cvcuda::WorkspaceRequirements HQResize::getWorkspaceRequirements(int batchSize, const HQResizeTensorShapesI inputShapes,
                                                                 const HQResizeTensorShapesI outputShapes,
                                                                 const NVCVInterpolationType minInterpolation,
                                                                 const NVCVInterpolationType magInterpolation,
                                                                 bool antialias, const HQResizeRoisF rois) const
{
    return m_impl->getWorkspaceRequirements(batchSize, inputShapes, outputShapes, minInterpolation, magInterpolation,
                                            antialias, rois);
}

cvcuda::WorkspaceRequirements HQResize::getWorkspaceRequirements(int                        maxBatchSize,
                                                                 const HQResizeTensorShapeI maxShape) const
{
    return m_impl->getWorkspaceRequirements(maxBatchSize, maxShape);
}

// Tensor variant
void HQResize::operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::Tensor &src,
                          const nvcv::Tensor &dst, const NVCVInterpolationType minInterpolation,
                          const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi) const
{
    assert(m_impl);
    m_impl->operator()(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, roi);
}

// ImageBatchVarShape variant
void HQResize::operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::ImageBatchVarShape &src,
                          const nvcv::ImageBatchVarShape &dst, const NVCVInterpolationType minInterpolation,
                          const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) const
{
    assert(m_impl);
    m_impl->operator()(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
}

// TensorBatch variant
void HQResize::operator()(cudaStream_t stream, const cvcuda::Workspace &ws, const nvcv::TensorBatch &src,
                          const nvcv::TensorBatch &dst, const NVCVInterpolationType minInterpolation,
                          const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF rois) const
{
    assert(m_impl);
    m_impl->operator()(stream, ws, src, dst, minInterpolation, magInterpolation, antialias, rois);
}

} // namespace cvcuda::priv
