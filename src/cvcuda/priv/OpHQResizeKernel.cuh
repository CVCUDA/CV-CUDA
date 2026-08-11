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
#ifndef CVCUDA_PRIV_HQ_RESIZE_KERNEL_CUH
#define CVCUDA_PRIV_HQ_RESIZE_KERNEL_CUH

#include "OpHQResize.hpp"
#include "OpHQResizePolicy.hpp"
#include "WorkspaceUtil.hpp"
#include "cvcuda/Workspace.hpp"

#include "OpHQResizeBatchWrap.cuh"
#include "OpHQResizeFilter.cuh"
#include "OpHQResizePlanar.cuh"

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

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace cvcuda::priv::hq_resize::kernel {

namespace cuda          = nvcv::cuda;
namespace filter        = cvcuda::priv::hq_resize::filter;
namespace batch_wrapper = cvcuda::priv::hq_resize::batch_wrapper;

template<typename T, int N>
using Vec = typename cuda::MakeType<T, N>;

template<int N>
using VecI = Vec<int, N>;

template<int N>
using VecF = Vec<float, N>;

class TensorShapeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

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

/**
 * @brief Base (per-channel) type of a wrap's value type, as every dispatch predicate
 * and lane policy needs it.
 */
template<typename WrapT>
using WrapBaseT = cuda::BaseType<std::remove_const_t<typename WrapT::ValueType>>;

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

inline constexpr int GetF32CubicMagnificationBlockLanes(int computeCapability)
{
    // The wider tile wins on the measured SM80/SM86/SM90 devices, but regresses SM89.
    // Keep the current behavior everywhere else to avoid changing unmeasured architectures.
    return computeCapability == 89 ? 8 : 16;
}

static_assert(GetF32CubicMagnificationBlockLanes(80) == 16);
static_assert(GetF32CubicMagnificationBlockLanes(86) == 16);
static_assert(GetF32CubicMagnificationBlockLanes(89) == 8);
static_assert(GetF32CubicMagnificationBlockLanes(90) == 16);
static_assert(GetF32CubicMagnificationBlockLanes(100) == 16);

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
/**
 * @brief acc[c] += src[c] * w for every channel, with the fmaf sequence every fused
 * kernel must share so their rounding matches the separable passes.
 */
template<typename FloatT, typename SrcT>
void __forceinline__ __device__ FmaPerChannel(FloatT &acc, const SrcT &src, const float w)
{
    static_assert(cuda::NumElements<FloatT> == cuda::NumElements<SrcT>);
#pragma unroll
    for (int c = 0; c < cuda::NumElements<FloatT>; c++)
    {
        cuda::GetElement(acc, c) = fmaf(cuda::GetElement(src, c), w, cuda::GetElement(acc, c));
    }
}

template<typename FloatT>
void __forceinline__ __device__ ScalePerChannel(FloatT &v, const float s)
{
#pragma unroll
    for (int c = 0; c < cuda::NumElements<FloatT>; c++)
    {
        cuda::GetElement(v, c) *= s;
    }
}

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

constexpr int kVertXVec           = 4;
constexpr int kVertXVecMaxSupport = 16;
constexpr int kVertXVecMaxScale   = 4;

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
template<int  kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT,
         bool kEnableVertXVec = false>
void __forceinline__ __device__ ResampleVert(const PassOutWrap outWrap, const PassInWrap inWrap,
                                             const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, float srcY0,
                                             const float scale, const VecI<kSpatialNDim> inShape,
                                             const NumChannelsT numChannels)
{
    using OutT                  = typename PassOutWrap::ValueType;
    using InT                   = std::remove_const_t<typename PassInWrap::ValueType>;
    using InBT                  = cuda::BaseType<InT>;
    using OutBT                 = cuda::BaseType<OutT>;
    constexpr bool kUseVertXVec = kEnableVertXVec && kSpatialNDim == 2 && NumChannelsT::kHasStaticChannels
                               && NumChannelsT::kStaticChannels == 1
                               && std::is_same_v<InBT, unsigned char> && std::is_same_v<OutBT, float>;

    srcY0 += 0.5f * scale - 0.5f;

    if constexpr (kUseVertXVec)
    {
        if (scale >= 1.0f && scale <= kVertXVecMaxScale)
        {
            using InVec    = Vec<InBT, kVertXVec>;
            using OutVec   = Vec<OutBT, kVertXVec>;
            using AccumVec = Vec<float, kVertXVec>;

            for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y)
            {
                const float sy0f = y * scale + srcY0;
                const int   sy0i = cuda::round<cuda::RoundMode::DOWN, int>(sy0f);
                const float q    = sy0f - sy0i;
                const int   sy0  = cuda::clamp(sy0i, 0, inShape.y - 1);
                const int   sy1  = cuda::clamp(sy0i + 1, 0, inShape.y - 1);

                for (int x0 = lo.x + static_cast<int>(threadIdx.x) * kVertXVec; x0 < hi.x;
                     x0 += static_cast<int>(blockDim.x) * kVertXVec)
                {
                    const bool full    = x0 + kVertXVec <= hi.x;
                    const bool aligned = full
                                      && ((reinterpret_cast<std::uintptr_t>(GetWrapPtr(outWrap, VecI<2>{x0, y}))
                                           & (alignof(OutVec) - 1))
                                          == 0)
                                      && ((reinterpret_cast<std::uintptr_t>(GetWrapPtr(inWrap, VecI<2>{x0, sy0}))
                                           & (alignof(InVec) - 1))
                                          == 0)
                                      && ((reinterpret_cast<std::uintptr_t>(GetWrapPtr(inWrap, VecI<2>{x0, sy1}))
                                           & (alignof(InVec) - 1))
                                          == 0);

                    if (aligned)
                    {
                        const InVec px0 = *reinterpret_cast<const InVec *>(GetWrapPtr(inWrap, VecI<2>{x0, sy0}));
                        const InVec px1 = *reinterpret_cast<const InVec *>(GetWrapPtr(inWrap, VecI<2>{x0, sy1}));
                        AccumVec    a   = cuda::StaticCast<float>(px0);
                        AccumVec    tmp = cuda::StaticCast<float>(px1) - a;
#pragma unroll
                        for (int lane = 0; lane < kVertXVec; lane++)
                        {
                            cuda::GetElement(tmp, lane)
                                = fmaf(cuda::GetElement(tmp, lane), q, cuda::GetElement(a, lane));
                        }
                        *reinterpret_cast<OutVec *>(GetWrapPtr(outWrap, VecI<2>{x0, y}))
                            = cuda::SaturateCast<OutVec>(tmp);
                    }
                    else
                    {
                        for (int lane = 0; lane < kVertXVec && x0 + lane < hi.x; lane++)
                        {
                            const int x   = x0 + lane;
                            const InT a   = LoadPixelLdg(inWrap, numChannels, VecI<2>{x, sy0});
                            const InT b   = LoadPixelLdg(inWrap, numChannels, VecI<2>{x, sy1});
                            float     tmp = b - a;
                            tmp           = fmaf(tmp, q, static_cast<float>(a));

                            OutT &out = *GetWrapPtr(outWrap, VecI<2>{x, y});
                            out       = cuda::SaturateCast<OutT>(tmp);
                        }
                    }
                }
            }
            return;
        }
    }

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

        ForAllOrthogonalToHorz(lo, hi,
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
                                           FmaPerChannel(tmp, px, flt);
                                       }

                                       OutT &out = *GetWrapPtr(outWrap, outIdx);
                                       out       = cuda::SaturateCast<OutT>(tmp * norm);
                                   }
                                   else if constexpr (!NumChannelsT::kHasStaticChannels)
                                   {
                                       for (int c = 0; c < numChannels(); c++)
                                       {
                                           float tmp = 0;

                                           for (int k = 0, coeffIdx = coeffBase; k < support;
                                                k++, coeffIdx += coeffStride)
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
template<int  kSpatialNDim, typename PassOutWrap, typename PassInWrap, typename NumChannelsT,
         bool kEnableVertXVec = false>
void __forceinline__ __device__ ResampleVert(const PassOutWrap outWrap, const PassInWrap inWrap,
                                             const VecI<kSpatialNDim> lo, const VecI<kSpatialNDim> hi, float srcY0,
                                             const float scale, const VecI<kSpatialNDim> inShape,
                                             const filter::ResamplingFilter filter, const NumChannelsT numChannels)
{
    extern __shared__ float coeffs[];

    using OutT  = typename PassOutWrap::ValueType;
    using InT   = std::remove_const_t<typename PassInWrap::ValueType>;
    using InBT  = cuda::BaseType<InT>;
    using OutBT = cuda::BaseType<OutT>;
    // spatial extents and optional channels extent
    constexpr int  kNDim        = kSpatialNDim + !NumChannelsT::kHasStaticChannels;
    constexpr bool kUseVertXVec = kEnableVertXVec && kSpatialNDim == 2 && NumChannelsT::kHasStaticChannels
                               && NumChannelsT::kStaticChannels == 1
                               && std::is_same_v<InBT, unsigned char> && std::is_same_v<OutBT, float>;

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

        if constexpr (kUseVertXVec)
        {
            const bool canUseXVec
                = !hugeSupport && scale >= 1.0f && scale <= kVertXVecMaxScale && support <= kVertXVecMaxSupport;
            if (canUseXVec)
            {
                using InVec    = Vec<InBT, kVertXVec>;
                using OutVec   = Vec<OutBT, kVertXVec>;
                using AccumVec = Vec<float, kVertXVec>;

                for (int x0 = lo.x + static_cast<int>(threadIdx.x) * kVertXVec; x0 < hi.x;
                     x0 += static_cast<int>(blockDim.x) * kVertXVec)
                {
                    const bool full    = x0 + kVertXVec <= hi.x;
                    bool       aligned = full
                                && ((reinterpret_cast<std::uintptr_t>(GetWrapPtr(outWrap, VecI<2>{x0, y}))
                                     & (alignof(OutVec) - 1))
                                    == 0);
                    if (aligned)
                    {
                        for (int k = 0; k < support; k++)
                        {
                            const int inY = cuda::clamp(sy0 + k, 0, inShape.y - 1);
                            aligned       = aligned
                                   && ((reinterpret_cast<std::uintptr_t>(GetWrapPtr(inWrap, VecI<2>{x0, inY}))
                                        & (alignof(InVec) - 1))
                                       == 0);
                        }
                    }

                    if (aligned)
                    {
                        AccumVec tmp = cuda::SetAll<AccumVec>(0.f);
                        for (int k = 0; k < support; k++)
                        {
                            const int   inY = cuda::clamp(sy0 + k, 0, inShape.y - 1);
                            const float flt = coeffs[coeffBase + k];
                            const InVec px  = *reinterpret_cast<const InVec *>(GetWrapPtr(inWrap, VecI<2>{x0, inY}));
                            tmp             = tmp + flt * cuda::StaticCast<float>(px);
                        }

                        *reinterpret_cast<OutVec *>(GetWrapPtr(outWrap, VecI<2>{x0, y}))
                            = cuda::SaturateCast<OutVec>(tmp * norm);
                    }
                    else
                    {
                        for (int lane = 0; lane < kVertXVec && x0 + lane < hi.x; lane++)
                        {
                            const int x = x0 + lane;
                            float     tmp{};
                            for (int k = 0; k < support; k++)
                            {
                                const int   inY = cuda::clamp(sy0 + k, 0, inShape.y - 1);
                                const float flt = coeffs[coeffBase + k];
                                const InT   px  = LoadPixelLdg(inWrap, numChannels, VecI<2>{x, inY});
                                tmp             = fmaf(px, flt, tmp);
                            }

                            OutT &out = *GetWrapPtr(outWrap, VecI<2>{x, y});
                            out       = cuda::SaturateCast<OutT>(tmp * norm);
                        }
                    }
                }
                continue;
            }
        }

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
                                           FmaPerChannel(tmp, px, flt);
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
                        FmaPerChannel(tmp, px, flt);
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

template<int kSpatialNDim, bool kEnableVertXVec = false, typename PassOutWrap, typename PassInWrap,
         typename NumChannelsT>
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
        linear::ResampleVert<kSpatialNDim, PassOutWrap, PassInWrap, NumChannelsT, kEnableVertXVec>(
            outWrap, inWrap, lo, hi, origin, scale, inShape, numChannels);
    }
    else if (axis == 2)
    {
        if constexpr (kSpatialNDim == 3)
        {
            linear::ResampleDepth(outWrap, inWrap, lo, hi, origin, scale, inShape, numChannels);
        }
    }
}

template<int kSpatialNDim, bool kEnableVertXVec = false, typename PassOutWrap, typename PassInWrap,
         typename NumChannelsT>
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
        filter_support::ResampleVert<kSpatialNDim, PassOutWrap, PassInWrap, NumChannelsT, kEnableVertXVec>(
            outWrap, inWrap, lo, hi, origin, scale, inShape, filter, numChannels);
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

template<int kWhichPass, bool kEnableVertXVec, typename PassOutWrap, typename PassInWrap, int kSpatialNDim,
         typename NumChannelsT>
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
        interpolate::RunLinear<kSpatialNDim, kEnableVertXVec>(outWrap, inWrap, lo, hi, axis, inShape, origin, scale,
                                                              numChannels);
        break;
    default:
        interpolate::RunFilter<kSpatialNDim, kEnableVertXVec>(outWrap, inWrap, lo, hi, axis, inShape, origin, scale,
                                                              sampleDesc.filter[kWhichPass], numChannels);
        break;
    }
}

// Tensor variant (unfirom batch)
template<int kNumStaticChannels, int kWhichPass, bool kEnableVertXVec = false, typename PassOutWrap,
         typename PassInWrap, int kSpatialNDim>
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
    WithChannels<kNumStaticChannels>(sampleDesc.channels,
                                     [=](const NumChannels<kNumStaticChannels> numChannels) {
                                         RunResamplingPass<kWhichPass, kEnableVertXVec>(
                                             sampleDesc, outSampleView, inSampleView, lo, hi, numChannels);
                                     });
}

// Batch variant (ImageBatchVarShape, TensorBatch)
template<int kNumStaticChannels, int kWhichPass, bool kEnableVertXVec = false, typename PassOutWrap,
         typename PassInWrap, int kSpatialNDim>
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
    WithChannels<kNumStaticChannels>(sampleDesc.channels,
                                     [=](const NumChannels<kNumStaticChannels> numChannels)
                                     {
                                         if constexpr (kWhichPass == 0)
                                         {
                                             const auto inSampleView
                                                 = inWrap.GetSampleView(sampleIdx, sampleDesc.inRoiOffset);
                                             RunResamplingPass<kWhichPass, kEnableVertXVec>(
                                                 sampleDesc, outSampleView, inSampleView, lo, hi, numChannels);
                                         }
                                         else if constexpr (kWhichPass != 0)
                                         {
                                             const auto inSampleView = inWrap.GetSampleView(sampleIdx);
                                             RunResamplingPass<kWhichPass, kEnableVertXVec>(
                                                 sampleDesc, outSampleView, inSampleView, lo, hi, numChannels);
                                         }
                                     });
}

inline __host__ __device__ void DirectLinearOriginScale(const SampleDesc<2> &sampleDesc, VecF<2> &origin,
                                                        VecF<2> &scale)
{
    for (int pass = 0; pass < 2; pass++)
    {
        const int axis                 = cuda::GetElement(sampleDesc.processingOrder, pass);
        cuda::GetElement(origin, axis) = cuda::GetElement(sampleDesc.origin, pass);
        cuda::GetElement(scale, axis)  = cuda::GetElement(sampleDesc.scale, pass);
    }
}

/**
 * @brief Stores kN consecutive single-channel outputs as kVec-wide vectors when the
 * destination is vector-aligned, or element-wise otherwise (user tensors may have row
 * strides that are not a multiple of the vector size). All kN elements must be
 * in-bounds.
 */
template<int kVec, typename OutT, int kN>
void __forceinline__ __device__ StoreRowVectorized(OutT *ptr, const OutT (&vals)[kN])
{
    static_assert(cuda::NumElements<OutT> == 1);
    static_assert(kN % kVec == 0);
    using OutVec = Vec<cuda::BaseType<OutT>, kVec>;
    if ((reinterpret_cast<std::uintptr_t>(ptr) & (alignof(OutVec) - 1)) == 0)
    {
#pragma unroll
        for (int v = 0; v < kN / kVec; v++)
        {
            OutVec vec;
#pragma unroll
            for (int e = 0; e < kVec; e++)
            {
                cuda::GetElement(vec, e) = vals[kVec * v + e];
            }
            reinterpret_cast<OutVec *>(ptr)[v] = vec;
        }
    }
    else
    {
#pragma unroll
        for (int e = 0; e < kN; e++)
        {
            ptr[e] = vals[e];
        }
    }
}

/**
 * @brief Number of consecutive output columns each DirectLinear2DKernel thread produces.
 *
 * The u8 kernel is issue-bound on per-pixel coordinate math and single-byte stores,
 * so each thread produces several output pixels and merges them into one vector store.
 */
template<int kNumStaticChannels, typename InBT, typename OutBT>
inline constexpr int kDirectLinear2DLanes
    = kNumStaticChannels == 1 && std::is_same_v<InBT, unsigned char> &&std::is_same_v<OutBT, unsigned char> ? 4 : 1;

inline __device__ float DirectLinearLerp(float a, float b, float q)
{
    return fmaf(b - a, q, a);
}

template<typename FloatT, typename InSampleView, typename NumChannelsT>
FloatT __forceinline__ __device__ DirectLinearPixel(const InSampleView inSampleView, const NumChannelsT numChannels,
                                                    const int x, const int y, const VecF<2> origin, const VecF<2> scale,
                                                    const int2 inShape)
{
    constexpr int kNumStaticChannels = NumChannelsT::kStaticChannels;

    const float sx0f = static_cast<float>(x) * scale.x + origin.x + 0.5f * scale.x - 0.5f;
    const float sy0f = static_cast<float>(y) * scale.y + origin.y + 0.5f * scale.y - 0.5f;
    int         sx0  = cuda::round<cuda::RoundMode::DOWN, int>(sx0f);
    int         sy0  = cuda::round<cuda::RoundMode::DOWN, int>(sy0f);
    const float qx   = sx0f - static_cast<float>(sx0);
    const float qy   = sy0f - static_cast<float>(sy0);
    const int   sx1  = cuda::clamp(sx0 + 1, 0, inShape.x - 1);
    const int   sy1  = cuda::clamp(sy0 + 1, 0, inShape.y - 1);
    sx0              = cuda::clamp(sx0, 0, inShape.x - 1);
    sy0              = cuda::clamp(sy0, 0, inShape.y - 1);

    FloatT row0 = cuda::StaticCast<float>(interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{sx0, sy0}));
    FloatT row1 = cuda::StaticCast<float>(interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{sx0, sy1}));
    const FloatT row0b
        = cuda::StaticCast<float>(interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{sx1, sy0}));
    const FloatT row1b
        = cuda::StaticCast<float>(interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{sx1, sy1}));
#pragma unroll
    for (int c = 0; c < kNumStaticChannels; c++)
    {
        cuda::GetElement(row0, c) = DirectLinearLerp(cuda::GetElement(row0, c), cuda::GetElement(row0b, c), qx);
        cuda::GetElement(row1, c) = DirectLinearLerp(cuda::GetElement(row1, c), cuda::GetElement(row1b, c), qx);
        cuda::GetElement(row0, c) = DirectLinearLerp(cuda::GetElement(row0, c), cuda::GetElement(row1, c), qy);
    }
    return row0;
}

template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
__global__ void DirectLinear2DKernel(const SampleDesc<2> sampleDesc, const PassOutWrap outWrap, const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels > 0);
    static_assert(PassInWrap::kNumDimensions == 3);
    static_assert(PassOutWrap::kNumDimensions == 3);

    using OutT   = typename PassOutWrap::ValueType;
    using InT    = std::remove_const_t<typename PassInWrap::ValueType>;
    using OutBT  = cuda::BaseType<OutT>;
    using InBT   = cuda::BaseType<InT>;
    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;

    static_assert(kNumStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);

    constexpr int kLanes = kDirectLinear2DLanes<kNumStaticChannels, InBT, OutBT>;

    const int  sampleIdx = blockIdx.z;
    const int  x0        = (blockIdx.x * blockDim.x + threadIdx.x) * kLanes;
    const int  y         = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape  = sampleDesc.shapes[2];
    if (x0 >= outShape.x || y >= outShape.y)
    {
        return;
    }

    VecF<2> origin{};
    VecF<2> scale{};
    DirectLinearOriginScale(sampleDesc, origin, scale);

    const int2 inShape{sampleDesc.shapes[0].x, sampleDesc.shapes[0].y};

    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    const auto numChannels   = NumChannels<kNumStaticChannels>{};

    if constexpr (kLanes == 1)
    {
        const FloatT px = DirectLinearPixel<FloatT>(inSampleView, numChannels, x0, y, origin, scale, inShape);

        OutT &out = *interpolate::GetWrapPtr(outSampleView, VecI<2>{x0, y});
        out       = cuda::SaturateCast<OutT>(px);
    }
    else if constexpr (kLanes > 1)
    {
        OutT *outPtr = interpolate::GetWrapPtr(outSampleView, VecI<2>{x0, y});
        if (x0 + kLanes <= outShape.x)
        {
            OutT vals[kLanes];
#pragma unroll
            for (int l = 0; l < kLanes; l++)
            {
                const FloatT px
                    = DirectLinearPixel<FloatT>(inSampleView, numChannels, x0 + l, y, origin, scale, inShape);
                vals[l] = cuda::SaturateCast<OutT>(px);
            }
            StoreRowVectorized<kLanes>(outPtr, vals);
        }
        else
        {
            for (int l = 0; l < kLanes && x0 + l < outShape.x; l++)
            {
                const FloatT px
                    = DirectLinearPixel<FloatT>(inSampleView, numChannels, x0 + l, y, origin, scale, inShape);
                outPtr[l] = cuda::SaturateCast<OutT>(px);
            }
        }
    }
}

/**
 * @brief Exact per-axis filter support required by the fused direct-filter kernels
 * (DirectFilter2D/Batch2D, Contract2x, 2x2).
 *
 * In practice this admits non-antialiased cubic plus the antialiased contractions
 * whose scaled support rounds to 4; smaller supports (non-antialiased gaussian or
 * triangular) and larger ones (stronger antialiased contractions, lanczos) keep the
 * separable path, whose shared-memory coefficient reuse pays off there.
 */
constexpr int kMaxDirectFilterSupport = 4;

/**
 * @brief Per-axis resampling window of a fused support-based pass: the first source
 * index, the filter coefficients, and the reciprocal of their sum. Initialization
 * replicates the separable kernel's per-output coefficient computation.
 */
template<int kSupport>
struct DirectFilterAxis
{
    int   s;
    float coeffs[kSupport];
    float norm;

    void __forceinline__ __device__ Init(const int coord, const float scale, const float originAdj,
                                         const filter::ResamplingFilter flt)
    {
        const float sf    = coord * scale + originAdj;
        s                 = cuda::round<cuda::RoundMode::UP, int>(sf);
        const float f     = (s - sf) * flt.scale;
        float       total = 0;
#pragma unroll
        for (int k = 0; k < kSupport; k++)
        {
            coeffs[k] = flt(f + k * flt.scale);
            total += coeffs[k];
        }
        norm = 1.0f / total;
    }
};

/**
 * @brief Origin adjustment shared by every output coordinate of a fused pass axis.
 */
float __forceinline__ __device__ DirectFilterOriginAdj(const float origin, const float scale,
                                                       const filter::ResamplingFilter flt)
{
    return origin + 0.5f * scale - 0.5f - flt.anchor;
}

/**
 * @brief Evaluates both fused resampling passes for one output pixel, with clamped
 * source indexing. axis0 is the axis resampled by pass0 (0 = X, 1 = Y).
 */
template<typename FloatT, int kSupport, typename InSampleView, typename NumChannelsT>
FloatT __forceinline__ __device__ DirectFilterSample(const InSampleView inSampleView, const NumChannelsT numChannels,
                                                     const DirectFilterAxis<kSupport> &a0,
                                                     const DirectFilterAxis<kSupport> &a1, const int axis0,
                                                     const int inSize0, const int inSize1)
{
    constexpr int kNumStaticChannels = NumChannelsT::kStaticChannels;
    using InT                        = typename InSampleView::ValueType;
    using InBT                       = cuda::BaseType<std::remove_const_t<InT>>;

    // u8 with X resampled by pass1 (the contraction pass order): each pass0 tap row's
    // kSupport * C window bytes are contiguous, so they load as aligned 32-bit words,
    // byte-aligned with funnel shifts so all register indexing stays compile-time. The
    // per-k1 fmaf chains accumulate in the same order as the generic loop below, so
    // the results are bit-identical. Needs one word of slack before the row end and,
    // because the first word is aligned down, up to three bytes of slack before the
    // window start so an unaligned sample base is never read below its first byte.
    if constexpr (std::is_same_v<InBT, unsigned char>)
    {
        constexpr int kWinBytes = kSupport * kNumStaticChannels;
        constexpr int kWinWords = (kWinBytes + 3) / 4 + 1;
        constexpr int kMinS1    = (3 + kNumStaticChannels - 1) / kNumStaticChannels;
        const int     maxS1     = inSize1 - (kWinWords * 4 + kNumStaticChannels - 1) / kNumStaticChannels;
        if (axis0 == 1 && a1.s >= kMinS1 && a1.s <= maxS1)
        {
            FloatT passSum[kSupport]{};
#pragma unroll
            for (int k0 = 0; k0 < kSupport; k0++)
            {
                const int   row = cuda::clamp(a0.s + k0, 0, inSize0 - 1);
                const auto *pix = reinterpret_cast<const unsigned char *>(
                    interpolate::GetWrapPtr(inSampleView, VecI<2>{a1.s, row}));
                const auto      addr  = reinterpret_cast<std::uintptr_t>(pix);
                const unsigned  shift = static_cast<unsigned>(addr & 3u) * 8u;
                const unsigned *word  = reinterpret_cast<const unsigned *>(addr & ~std::uintptr_t{3});
                unsigned        w[kWinWords];
#pragma unroll
                for (int i = 0; i < kWinWords; i++)
                {
                    w[i] = __ldg(word + i);
                }
                unsigned v[kWinWords - 1];
#pragma unroll
                for (int i = 0; i < kWinWords - 1; i++)
                {
                    v[i] = __funnelshift_r(w[i], w[i + 1], shift);
                }
                const float flt = a0.coeffs[k0];
#pragma unroll
                for (int k1 = 0; k1 < kSupport; k1++)
                {
#pragma unroll
                    for (int c = 0; c < kNumStaticChannels; c++)
                    {
                        const int   j                    = k1 * kNumStaticChannels + c;
                        const float b                    = static_cast<float>((v[j / 4] >> ((j % 4) * 8)) & 0xFFu);
                        cuda::GetElement(passSum[k1], c) = fmaf(b, flt, cuda::GetElement(passSum[k1], c));
                    }
                }
            }
            FloatT acc{};
#pragma unroll
            for (int k1 = 0; k1 < kSupport; k1++)
            {
                ScalePerChannel(passSum[k1], a0.norm);
                FmaPerChannel(acc, passSum[k1], a1.coeffs[k1]);
            }
            ScalePerChannel(acc, a1.norm);
            return acc;
        }
    }

    FloatT acc{};
#pragma unroll
    for (int k1 = 0; k1 < kSupport; k1++)
    {
        const int idx1 = cuda::clamp(a1.s + k1, 0, inSize1 - 1);
        FloatT    passSum{};
#pragma unroll
        for (int k0 = 0; k0 < kSupport; k0++)
        {
            const int     idx0  = cuda::clamp(a0.s + k0, 0, inSize0 - 1);
            const VecI<2> inIdx = axis0 == 0 ? VecI<2>{idx0, idx1} : VecI<2>{idx1, idx0};
            const InT     px    = interpolate::LoadPixelLdg(inSampleView, numChannels, inIdx);
            FmaPerChannel(passSum, px, a0.coeffs[k0]);
        }
        ScalePerChannel(passSum, a0.norm);
        FmaPerChannel(acc, passSum, a1.coeffs[k1]);
    }
    ScalePerChannel(acc, a1.norm);
    return acc;
}

/**
 * @brief Evaluates one fused-contraction output pixel from its sample descriptor and
 * stores it: the shared body of the uniform-tensor and batch direct-filter kernels.
 */
template<int kNumStaticChannels, int kSupport, typename OutSampleView, typename InSampleView>
void __forceinline__ __device__ DirectFilterOutput(const SampleDesc<2> &sampleDesc, const OutSampleView outSampleView,
                                                   const InSampleView inSampleView, const int x, const int y)
{
    using OutT   = typename OutSampleView::ValueType;
    using InT    = std::remove_const_t<typename InSampleView::ValueType>;
    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;

    // pass0 resamples axis0 (0 = X, 1 = Y), pass1 the other axis
    const int  axis0  = sampleDesc.processingOrder.x;
    const int  coord0 = axis0 == 0 ? x : y;
    const int  coord1 = axis0 == 0 ? y : x;
    const int2 inShape{sampleDesc.shapes[0].x, sampleDesc.shapes[0].y};
    const int  inSize0 = axis0 == 0 ? inShape.x : inShape.y;
    const int  inSize1 = axis0 == 0 ? inShape.y : inShape.x;

    const filter::ResamplingFilter flt0 = sampleDesc.filter[0];
    const filter::ResamplingFilter flt1 = sampleDesc.filter[1];

    DirectFilterAxis<kSupport> a0, a1;
    a0.Init(coord0, sampleDesc.scale.x, DirectFilterOriginAdj(sampleDesc.origin.x, sampleDesc.scale.x, flt0), flt0);
    a1.Init(coord1, sampleDesc.scale.y, DirectFilterOriginAdj(sampleDesc.origin.y, sampleDesc.scale.y, flt1), flt1);

    const auto numChannels = NumChannels<kNumStaticChannels>{};

    const FloatT acc = DirectFilterSample<FloatT>(inSampleView, numChannels, a0, a1, axis0, inSize0, inSize1);

    OutT &out = *interpolate::GetWrapPtr(outSampleView, VecI<2>{x, y});
    out       = cuda::SaturateCast<OutT>(acc);
}

/**
 * @brief Fused two-pass support-based resampling: one thread per output pixel.
 *
 * Both resampling passes of the separable kernel are evaluated in registers, in the
 * same pass order with the same coefficient, normalization, and FMA sequences, so the
 * result matches the separable path while the float intermediate never reaches memory.
 * Only worthwhile for contractions, where the per-output coefficient recomputation is
 * amortized by the reduced output count.
 */
template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilter2DKernel(const SampleDesc<2> sampleDesc, const PassOutWrap outWrap, const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels > 0);
    static_assert(kSupport <= kMaxDirectFilterSupport);
    static_assert(PassInWrap::kNumDimensions == 3);
    static_assert(PassOutWrap::kNumDimensions == 3);

    using OutT = typename PassOutWrap::ValueType;
    using InT  = std::remove_const_t<typename PassInWrap::ValueType>;

    static_assert(kNumStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);

    const int  sampleIdx = blockIdx.z;
    const int  x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int  y         = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape  = sampleDesc.shapes[2];
    if (x >= outShape.x || y >= outShape.y)
    {
        return;
    }

    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    DirectFilterOutput<kNumStaticChannels, kSupport>(sampleDesc, outSampleView, inSampleView, x, y);
}

/**
 * @brief Number of consecutive output columns each DirectFilterContract2xKernel thread
 * produces; lanes share the column-sum window and merge into vectorized stores.
 */
template<typename InBT>
inline constexpr int kDirectFilterContract2xLanes = std::is_same_v<InBT, unsigned char> ? 4 : 2;

/**
 * @brief Fused two-pass support-based exact-2x contraction: one thread per kLanes
 * consecutive output columns.
 *
 * With scale exactly 2 on both axes the coefficient phase is the same for every output
 * coordinate, so each thread evaluates one coefficient set per axis and reuses it for
 * all its outputs, and adjacent lanes share the per-column pass0 sums. The separable
 * passes' coefficient, normalization, and FMA sequences are replicated in pass order
 * (Y first, as dispatched), so the result matches the separable path while the float
 * intermediate never reaches memory.
 */
template<int kNumStaticChannels, int kSupport, typename OutSampleView, typename InSampleView>
void __forceinline__ __device__ DirectFilterContract2xThread(const SampleDesc<2> &sampleDesc,
                                                             const OutSampleView  outSampleView,
                                                             const InSampleView inSampleView, const int x0, const int y)
{
    using OutT   = typename OutSampleView::ValueType;
    using InT    = std::remove_const_t<typename InSampleView::ValueType>;
    using OutBT  = cuda::BaseType<OutT>;
    using InBT   = cuda::BaseType<InT>;
    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;

    constexpr int kLanes = kDirectFilterContract2xLanes<InBT>;

    const auto outShape = sampleDesc.shapes[2];
    const int  inW      = sampleDesc.shapes[0].x;
    const int  inH      = sampleDesc.shapes[0].y;

    // pass0 resamples Y (dispatch requires processingOrder == {1, 0})
    const filter::ResamplingFilter fltY = sampleDesc.filter[0];
    const filter::ResamplingFilter fltX = sampleDesc.filter[1];

    DirectFilterAxis<kSupport> ay, ax;
    ay.Init(y, sampleDesc.scale.x, DirectFilterOriginAdj(sampleDesc.origin.x, sampleDesc.scale.x, fltY), fltY);
    ax.Init(x0, sampleDesc.scale.y, DirectFilterOriginAdj(sampleDesc.origin.y, sampleDesc.scale.y, fltX), fltX);

    const auto numChannels = NumChannels<kNumStaticChannels>{};

    // The source column index advances by exactly 2 per output column.
    constexpr int kCols = 2 * kLanes + kSupport - 2;
    const bool    interior
        = ax.s >= 0 && ax.s + kCols - 1 < inW && ay.s >= 0 && ay.s + kSupport - 1 < inH && x0 + kLanes <= outShape.x;

    if (interior)
    {
        FloatT colSum[kCols]{};
#pragma unroll
        for (int j = 0; j < kSupport; j++)
        {
            const int row = ay.s + j;
#pragma unroll
            for (int i = 0; i < kCols; i++)
            {
                const InT px = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{ax.s + i, row});
                FmaPerChannel(colSum[i], px, ay.coeffs[j]);
            }
        }
#pragma unroll
        for (int i = 0; i < kCols; i++)
        {
            ScalePerChannel(colSum[i], ay.norm);
        }

        OutT vals[kLanes];
#pragma unroll
        for (int l = 0; l < kLanes; l++)
        {
            FloatT acc{};
#pragma unroll
            for (int k = 0; k < kSupport; k++)
            {
                FmaPerChannel(acc, colSum[2 * l + k], ax.coeffs[k]);
            }
            ScalePerChannel(acc, ax.norm);
            vals[l] = cuda::SaturateCast<OutT>(acc);
        }

        OutT *outPtr = interpolate::GetWrapPtr(outSampleView, VecI<2>{x0, y});
        StoreRowVectorized<kLanes>(outPtr, vals);
    }
    else
    {
        for (int l = 0; l < kLanes && x0 + l < outShape.x; l++)
        {
            DirectFilterOutput<kNumStaticChannels, kSupport>(sampleDesc, outSampleView, inSampleView, x0 + l, y);
        }
    }
}

template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilterContract2xKernel(const SampleDesc<2> sampleDesc, const PassOutWrap outWrap,
                                             const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels > 0);
    static_assert(kSupport <= kMaxDirectFilterSupport);
    static_assert(PassInWrap::kNumDimensions == 3);
    static_assert(PassOutWrap::kNumDimensions == 3);

    constexpr int kLanes = kDirectFilterContract2xLanes<WrapBaseT<PassInWrap>>;

    const int  x0       = (blockIdx.x * blockDim.x + threadIdx.x) * kLanes;
    const int  y        = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape = sampleDesc.shapes[2];
    if (x0 >= outShape.x || y >= outShape.y)
    {
        return;
    }

    const int  sampleIdx     = blockIdx.z;
    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    DirectFilterContract2xThread<kNumStaticChannels, kSupport>(sampleDesc, outSampleView, inSampleView, x0, y);
}

/**
 * @brief Batch (ImageBatchVarShape / TensorBatch) variant of DirectFilterContract2xKernel.
 */
template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilterContract2xBatchKernel(const SampleDesc<2> *__restrict__ samples, const PassOutWrap outWrap,
                                                  const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels == 1);
    static_assert(kSupport <= kMaxDirectFilterSupport);

    constexpr int kLanes = kDirectFilterContract2xLanes<WrapBaseT<PassInWrap>>;

    const int  sampleIdx  = blockIdx.z;
    const auto sampleDesc = samples[sampleIdx];
    const int  x0         = (blockIdx.x * blockDim.x + threadIdx.x) * kLanes;
    const int  y          = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape   = sampleDesc.shapes[2];
    if (x0 >= outShape.x || y >= outShape.y)
    {
        return;
    }

    const auto inSampleView  = inWrap.GetSampleView(sampleIdx);
    const auto outSampleView = outWrap.GetSampleView(sampleIdx);
    DirectFilterContract2xThread<kNumStaticChannels, kSupport>(sampleDesc, outSampleView, inSampleView, x0, y);
}

/**
 * @brief Maximum per-axis phase count of the fused magnification kernel; bounds the
 * shared-memory coefficient tables.
 */
constexpr int kMaxDirectFilterPhases = 64;

/**
 * @brief Cooperatively fills the per-phase coefficient/offset tables consumed by
 * the
 * phased kernels. Layout: bx*(kSupport+1) floats for X, by*(kSupport+1)
 * floats for Y, then bx + by ints of source starts.
 */
template<int kSupport>
void __forceinline__ __device__ DirectFilterPhaseTableInitShm(const SampleDesc<2> &sampleDesc, const VecI<2> phaseCount,
                                                              float *shm)
{
    constexpr int kRow = kSupport + 1;
    const int     bx   = phaseCount.x;
    const int     by   = phaseCount.y;
    float        *cx   = shm;
    float        *cy   = cx + bx * kRow;
    int          *sbx  = reinterpret_cast<int *>(cy + by * kRow);
    int          *sby  = sbx + bx;

    const filter::ResamplingFilter fltX = sampleDesc.filter[0];
    const filter::ResamplingFilter fltY = sampleDesc.filter[1];
    const float                    adjX = DirectFilterOriginAdj(sampleDesc.origin.x, sampleDesc.scale.x, fltX);
    const float                    adjY = DirectFilterOriginAdj(sampleDesc.origin.y, sampleDesc.scale.y, fltY);

    const int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockSize = blockDim.x * blockDim.y;
    for (int p = tid; p < bx + by; p += blockSize)
    {
        DirectFilterAxis<kSupport> a;
        if (p < bx)
        {
            a.Init(p, sampleDesc.scale.x, adjX, fltX);
        }
        else
        {
            a.Init(p - bx, sampleDesc.scale.y, adjY, fltY);
        }
        float *row = (p < bx ? cx + p * kRow : cy + (p - bx) * kRow);
#pragma unroll
        for (int k = 0; k < kSupport; k++)
        {
            row[k] = a.coeffs[k];
        }
        row[kSupport]                   = a.norm;
        (p < bx ? sbx[p] : sby[p - bx]) = a.s;
    }
    __syncthreads();
}

/**
 * @brief Fused contraction with per-phase coefficient tables: when both pass axes have
 * small rational phase counts, the block builds the same shared tables as the
 * magnification kernel (pass-ordered) and each thread reconstructs its two axis
 * windows by phase lookup instead of evaluating the filter table per output pixel.
 * The reconstructed windows feed the unmodified DirectFilterSample, so results are
 * bit-identical to the per-pixel kernel.
 */
/**
 * @brief Phase-lookup evaluation of one fused-contraction output pixel: reconstructs
 * both pass windows from the shared tables and feeds the unmodified
 * DirectFilterSample. Shared by the uniform-tensor and batch phased kernels.
 */
template<int kNumStaticChannels, int kSupport, typename OutSampleView, typename InSampleView>
void __forceinline__ __device__ DirectFilterPhasedOutput(const SampleDesc<2> &sampleDesc, const VecI<2> phaseCount,
                                                         const VecI<2> phaseStep, const float *shm,
                                                         const OutSampleView outSampleView,
                                                         const InSampleView inSampleView, const int x, const int y)
{
    using OutT   = typename OutSampleView::ValueType;
    using InT    = std::remove_const_t<typename InSampleView::ValueType>;
    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;

    constexpr int kRow = kSupport + 1;
    const int     b0   = phaseCount.x;
    const int     b1   = phaseCount.y;
    const float  *c0   = shm;
    const float  *c1   = c0 + b0 * kRow;
    const int    *sb0  = reinterpret_cast<const int *>(c1 + b1 * kRow);
    const int    *sb1  = sb0 + b0;

    // pass0 resamples axis0 (0 = X, 1 = Y); tables and coords are pass-ordered
    const int  axis0  = sampleDesc.processingOrder.x;
    const int  coord0 = axis0 == 0 ? x : y;
    const int  coord1 = axis0 == 0 ? y : x;
    const int2 inShape{sampleDesc.shapes[0].x, sampleDesc.shapes[0].y};
    const int  inSize0 = axis0 == 0 ? inShape.x : inShape.y;
    const int  inSize1 = axis0 == 0 ? inShape.y : inShape.x;

    DirectFilterAxis<kSupport> a0, a1;
    {
        const int    p0   = coord0 % b0;
        const int    p1   = coord1 % b1;
        const float *row0 = c0 + p0 * kRow;
        const float *row1 = c1 + p1 * kRow;
#pragma unroll
        for (int k = 0; k < kSupport; k++)
        {
            a0.coeffs[k] = row0[k];
            a1.coeffs[k] = row1[k];
        }
        a0.norm = row0[kSupport];
        a1.norm = row1[kSupport];
        a0.s    = sb0[p0] + (coord0 / b0) * phaseStep.x;
        a1.s    = sb1[p1] + (coord1 / b1) * phaseStep.y;
    }

    const auto numChannels = NumChannels<kNumStaticChannels>{};

    const FloatT acc = DirectFilterSample<FloatT>(inSampleView, numChannels, a0, a1, axis0, inSize0, inSize1);

    OutT &out = *interpolate::GetWrapPtr(outSampleView, VecI<2>{x, y});
    out       = cuda::SaturateCast<OutT>(acc);
}

/**
 * @brief Fused contraction with per-phase coefficient tables: when both pass axes have
 * small rational phase counts, the block builds the same shared tables as the
 * magnification kernel (pass-ordered) and each thread reconstructs its two axis
 * windows by phase lookup instead of evaluating the filter table per output pixel.
 * Results are bit-identical to the per-pixel kernel.
 */
template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilterPhased2DKernel(const SampleDesc<2> sampleDesc, const VecI<2> phaseCount,
                                           const VecI<2> phaseStep, const PassOutWrap outWrap, const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels > 0);
    static_assert(kSupport <= kMaxDirectFilterSupport);
    static_assert(PassInWrap::kNumDimensions == 3);
    static_assert(PassOutWrap::kNumDimensions == 3);

    extern __shared__ float shm[];
    DirectFilterPhaseTableInitShm<kSupport>(sampleDesc, phaseCount, shm);

    const int  sampleIdx = blockIdx.z;
    const int  x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int  y         = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape  = sampleDesc.shapes[2];
    if (x >= outShape.x || y >= outShape.y)
    {
        return;
    }

    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    DirectFilterPhasedOutput<kNumStaticChannels, kSupport>(sampleDesc, phaseCount, phaseStep, shm, outSampleView,
                                                           inSampleView, x, y);
}

/**
 * @brief Batch (ImageBatchVarShape / TensorBatch) variant of DirectFilterPhased2DKernel;
 * dispatch requires every sample to share one shape, so one phase table serves all.
 */
template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilterPhasedBatch2DKernel(const SampleDesc<2> *__restrict__ samples, const VecI<2> phaseCount,
                                                const VecI<2> phaseStep, const PassOutWrap outWrap,
                                                const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels > 0);
    static_assert(kSupport <= kMaxDirectFilterSupport);

    extern __shared__ float shm[];
    const int               sampleIdx  = blockIdx.z;
    const auto              sampleDesc = samples[sampleIdx];
    DirectFilterPhaseTableInitShm<kSupport>(sampleDesc, phaseCount, shm);

    const int  x        = blockIdx.x * blockDim.x + threadIdx.x;
    const int  y        = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape = sampleDesc.shapes[2];
    if (x >= outShape.x || y >= outShape.y)
    {
        return;
    }

    const auto inSampleView  = inWrap.GetSampleView(sampleIdx);
    const auto outSampleView = outWrap.GetSampleView(sampleIdx);
    DirectFilterPhasedOutput<kNumStaticChannels, kSupport>(sampleDesc, phaseCount, phaseStep, shm, outSampleView,
                                                           inSampleView, x, y);
}

/**
 * @brief Batch (ImageBatchVarShape / TensorBatch) variant of DirectFilter2DKernel:
 * per-sample descriptors, grid sized for the largest sample with per-block early exit.
 */
template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilterBatch2DKernel(const SampleDesc<2> *__restrict__ samples, const PassOutWrap outWrap,
                                          const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels > 0);
    static_assert(kSupport <= kMaxDirectFilterSupport);

    const int  sampleIdx  = blockIdx.z;
    const auto sampleDesc = samples[sampleIdx];
    const int  x          = blockIdx.x * blockDim.x + threadIdx.x;
    const int  y          = blockIdx.y * blockDim.y + threadIdx.y;
    const auto outShape   = sampleDesc.shapes[2];
    if (x >= outShape.x || y >= outShape.y)
    {
        return;
    }

    const auto inSampleView  = inWrap.GetSampleView(sampleIdx);
    const auto outSampleView = outWrap.GetSampleView(sampleIdx);
    DirectFilterOutput<kNumStaticChannels, kSupport>(sampleDesc, outSampleView, inSampleView, x, y);
}

/**
 * @brief Number of source columns each DirectFilter2x2Kernel thread expands.
 *
 * The single-channel kernels are load-transaction-bound on the (kSupport+1)^2 source
 * neighborhood, so several source columns per thread let adjacent lanes share the
 * column loads and the outputs merge into vectorized stores.
 */
template<typename InBT, typename OutBT>
inline constexpr int kDirectFilter2x2Lanes = std::is_same_v<InBT, unsigned char> ? 4 : 2;

/**
 * @brief Fused two-pass support-based 2x magnification: one thread per kLanes source
 * columns, producing the corresponding 2*kLanes x 2 output block.
 *
 * With scale exactly 0.5 on both axes, only two resampling phases exist per axis
 * (even/odd output parity), so each thread computes the four coefficient sets once and
 * reuses them for all its outputs. The coefficient, normalization, and FMA sequences
 * replicate the separable passes in pass order (X first, as dispatched), so the result
 * matches the separable path while the float intermediate never reaches memory.
 */
template<int kNumStaticChannels, int kSupport, typename PassOutWrap, typename PassInWrap>
__global__ void DirectFilter2x2Kernel(const SampleDesc<2> sampleDesc, const PassOutWrap outWrap,
                                      const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels == 1);
    static_assert(kSupport <= kMaxDirectFilterSupport);
    static_assert(PassInWrap::kNumDimensions == 3);
    static_assert(PassOutWrap::kNumDimensions == 3);

    using OutT   = typename PassOutWrap::ValueType;
    using InT    = std::remove_const_t<typename PassInWrap::ValueType>;
    using OutBT  = cuda::BaseType<OutT>;
    using InBT   = cuda::BaseType<InT>;
    using FloatT = cuda::ConvertBaseTypeTo<float, InT>;

    static_assert(kNumStaticChannels == cuda::NumElements<InT>);
    static_assert(cuda::NumElements<OutT> == cuda::NumElements<InT>);

    constexpr int kLanes = kDirectFilter2x2Lanes<InBT, OutBT>;

    const int tX0 = (blockIdx.x * blockDim.x + threadIdx.x) * kLanes;
    const int tY  = blockIdx.y * blockDim.y + threadIdx.y;
    const int inW = sampleDesc.shapes[0].x;
    const int inH = sampleDesc.shapes[0].y;
    if (tX0 >= inW || tY >= inH)
    {
        return;
    }

    const int sampleIdx = blockIdx.z;

    const filter::ResamplingFilter fltX = sampleDesc.filter[0];
    const filter::ResamplingFilter fltY = sampleDesc.filter[1];
    const float                    adjX = DirectFilterOriginAdj(sampleDesc.origin.x, sampleDesc.scale.x, fltX);
    const float                    adjY = DirectFilterOriginAdj(sampleDesc.origin.y, sampleDesc.scale.y, fltY);

    // Per-parity windows at this thread's first lane; lane l shifts the source index
    // by exactly l (the source position advances by 1 per output-parity pair).
    DirectFilterAxis<kSupport> ax[2], ay[2];
#pragma unroll
    for (int p = 0; p < 2; p++)
    {
        ax[p].Init(2 * tX0 + p, sampleDesc.scale.x, adjX, fltX);
        ay[p].Init(2 * tY + p, sampleDesc.scale.y, adjY, fltY);
    }

    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    const auto numChannels   = NumChannels<kNumStaticChannels>{};

    // The support-4 table filters anchor at support/2, so the odd output phase starts
    // exactly one source pixel after the even one; anything else takes the edge path.
    const bool interior = ax[1].s - ax[0].s == 1 && ay[1].s - ay[0].s == 1 && ax[0].s >= 0
                       && ax[0].s + kLanes + kMaxDirectFilterSupport - 1 < inW && ay[0].s >= 0
                       && ay[1].s + kSupport - 1 < inH && tX0 + kLanes <= inW;

    if (interior)
    {
        constexpr int kCols = kLanes + kMaxDirectFilterSupport;
        constexpr int kOut  = 2 * kLanes;
        static_assert(kOut % 4 == 0);

        FloatT acc[2][kOut]{};
        // kSupport + 1 rows: the even output row uses rows [0, kSupport), the odd one
        // [1, kSupport + 1). All register-array indices below are compile-time.
#pragma unroll
        for (int j = 0; j < kSupport + 1; j++)
        {
            const int row = ay[0].s + j;
            FloatT    cols[kCols];
#pragma unroll
            for (int i = 0; i < kCols; i++)
            {
                cols[i] = cuda::StaticCast<float>(
                    interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{ax[0].s + i, row}));
            }
            FloatT xv[2][kLanes];
#pragma unroll
            for (int p = 0; p < 2; p++)
            {
#pragma unroll
                for (int l = 0; l < kLanes; l++)
                {
                    FloatT passSum{};
#pragma unroll
                    for (int k = 0; k < kSupport; k++)
                    {
                        FmaPerChannel(passSum, cols[p + l + k], ax[p].coeffs[k]);
                    }
                    ScalePerChannel(passSum, ax[p].norm);
                    xv[p][l] = passSum;
                }
            }
#pragma unroll
            for (int q = 0; q < 2; q++)
            {
                const int kq = j - q;
                if (kq < 0 || kq >= kSupport)
                {
                    continue;
                }
#pragma unroll
                for (int p = 0; p < 2; p++)
                {
#pragma unroll
                    for (int l = 0; l < kLanes; l++)
                    {
                        FmaPerChannel(acc[q][2 * l + p], xv[p][l], ay[q].coeffs[kq]);
                    }
                }
            }
        }

#pragma unroll
        for (int q = 0; q < 2; q++)
        {
            OutT vals[kOut];
#pragma unroll
            for (int e = 0; e < kOut; e++)
            {
                ScalePerChannel(acc[q][e], ay[q].norm);
                vals[e] = cuda::SaturateCast<OutT>(acc[q][e]);
            }
            OutT *rowPtr = interpolate::GetWrapPtr(outSampleView, VecI<2>{2 * tX0, 2 * tY + q});
            StoreRowVectorized<4>(rowPtr, vals);
        }
    }
    else
    {
        for (int l = 0; l < kLanes && tX0 + l < inW; l++)
        {
            DirectFilterAxis<kSupport> axl[2];
#pragma unroll
            for (int p = 0; p < 2; p++)
            {
                axl[p].Init(2 * (tX0 + l) + p, sampleDesc.scale.x, adjX, fltX);
            }
#pragma unroll
            for (int q = 0; q < 2; q++)
            {
#pragma unroll
                for (int p = 0; p < 2; p++)
                {
                    const FloatT acc
                        = DirectFilterSample<FloatT>(inSampleView, numChannels, axl[p], ay[q], 0, inW, inH);
                    OutT &out = *interpolate::GetWrapPtr(outSampleView, VecI<2>{2 * (tX0 + l) + p, 2 * tY + q});
                    out       = cuda::SaturateCast<OutT>(acc);
                }
            }
        }
    }
}

/**
 * @brief Number of source columns each DirectLinear2x2Kernel thread expands.
 *
 * The u8 kernel is L1-transaction-bound on the 3x3 single-byte neighborhood loads,
 * so widening a thread to several source columns lets adjacent lanes share the
 * column loads and the 2x2 outputs merge into vectorized stores.
 */
template<typename InBT, typename OutBT>
inline constexpr int kDirectLinear2x2Lanes
    = std::is_same_v<InBT, unsigned char> &&std::is_same_v<OutBT, unsigned char> ? 4 : 1;

template<typename OutSampleView, typename InSampleView, typename NumChannelsT>
void __forceinline__ __device__ DirectLinear2x2Pixel(const OutSampleView outSampleView, const InSampleView inSampleView,
                                                     const NumChannelsT numChannels, const int srcX, const int y0,
                                                     const int y1, const int y2, const int inW, const int outY)
{
    using OutT = typename OutSampleView::ValueType;

    const int x0 = cuda::max(srcX - 1, 0);
    const int x1 = srcX;
    const int x2 = cuda::min(srcX + 1, inW - 1);

    const float p00 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x0, y0});
    const float p10 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x1, y0});
    const float p20 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x2, y0});
    const float p01 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x0, y1});
    const float p11 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x1, y1});
    const float p21 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x2, y1});
    const float p02 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x0, y2});
    const float p12 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x1, y2});
    const float p22 = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x2, y2});

    const int outX = srcX * 2;

    const float evenTop = DirectLinearLerp(p00, p10, 0.75f);
    const float evenMid = DirectLinearLerp(p01, p11, 0.75f);
    const float evenBot = DirectLinearLerp(p02, p12, 0.75f);
    const float oddTop  = DirectLinearLerp(p10, p20, 0.25f);
    const float oddMid  = DirectLinearLerp(p11, p21, 0.25f);
    const float oddBot  = DirectLinearLerp(p12, p22, 0.25f);

    *interpolate::GetWrapPtr(outSampleView, VecI<2>{outX, outY})
        = cuda::SaturateCast<OutT>(DirectLinearLerp(evenTop, evenMid, 0.75f));
    *interpolate::GetWrapPtr(outSampleView, VecI<2>{outX + 1, outY})
        = cuda::SaturateCast<OutT>(DirectLinearLerp(oddTop, oddMid, 0.75f));
    *interpolate::GetWrapPtr(outSampleView, VecI<2>{outX, outY + 1})
        = cuda::SaturateCast<OutT>(DirectLinearLerp(evenMid, evenBot, 0.25f));
    *interpolate::GetWrapPtr(outSampleView, VecI<2>{outX + 1, outY + 1})
        = cuda::SaturateCast<OutT>(DirectLinearLerp(oddMid, oddBot, 0.25f));
}

template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
__global__ void DirectLinear2x2Kernel(const SampleDesc<2> sampleDesc, const PassOutWrap outWrap,
                                      const PassInWrap inWrap)
{
    static_assert(kNumStaticChannels == 1);
    static_assert(PassInWrap::kNumDimensions == 3);
    static_assert(PassOutWrap::kNumDimensions == 3);

    using OutT  = typename PassOutWrap::ValueType;
    using InT   = std::remove_const_t<typename PassInWrap::ValueType>;
    using OutBT = cuda::BaseType<OutT>;
    using InBT  = cuda::BaseType<InT>;

    static_assert(cuda::NumElements<OutT> == 1);
    static_assert(cuda::NumElements<InT> == 1);

    constexpr int kLanes = kDirectLinear2x2Lanes<InBT, OutBT>;

    const int srcX0     = (blockIdx.x * blockDim.x + threadIdx.x) * kLanes;
    const int srcY      = blockIdx.y * blockDim.y + threadIdx.y;
    const int sampleIdx = blockIdx.z;
    const int inW       = sampleDesc.shapes[0].x;
    const int inH       = sampleDesc.shapes[0].y;
    if (srcX0 >= inW || srcY >= inH)
    {
        return;
    }

    const int y0 = cuda::max(srcY - 1, 0);
    const int y1 = srcY;
    const int y2 = cuda::min(srcY + 1, inH - 1);

    const auto inSampleView  = batch_wrapper::tensor::GetSampleView(inWrap, sampleIdx);
    const auto outSampleView = batch_wrapper::tensor::GetSampleView(outWrap, sampleIdx);
    const auto numChannels   = NumChannels<kNumStaticChannels>{};

    if constexpr (kLanes == 1)
    {
        DirectLinear2x2Pixel(outSampleView, inSampleView, numChannels, srcX0, y0, y1, y2, inW, srcY * 2);
    }
    else if constexpr (kLanes > 1)
    {
        const int outY = srcY * 2;
        // The vector body needs the whole [srcX0-1, srcX0+kLanes] column window unclamped.
        if (srcX0 - 1 >= 0 && srcX0 + kLanes <= inW - 1)
        {
            float cols0[kLanes + 2], cols1[kLanes + 2], cols2[kLanes + 2];
#pragma unroll
            for (int i = 0; i < kLanes + 2; i++)
            {
                const int x = srcX0 - 1 + i;
                cols0[i]    = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x, y0});
                cols1[i]    = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x, y1});
                cols2[i]    = interpolate::LoadPixelLdg(inSampleView, numChannels, VecI<2>{x, y2});
            }

            OutBT top[2 * kLanes], bot[2 * kLanes];
#pragma unroll
            for (int l = 0; l < kLanes; l++)
            {
                const float evenTop = DirectLinearLerp(cols0[l], cols0[l + 1], 0.75f);
                const float evenMid = DirectLinearLerp(cols1[l], cols1[l + 1], 0.75f);
                const float evenBot = DirectLinearLerp(cols2[l], cols2[l + 1], 0.75f);
                const float oddTop  = DirectLinearLerp(cols0[l + 1], cols0[l + 2], 0.25f);
                const float oddMid  = DirectLinearLerp(cols1[l + 1], cols1[l + 2], 0.25f);
                const float oddBot  = DirectLinearLerp(cols2[l + 1], cols2[l + 2], 0.25f);

                top[2 * l]     = cuda::SaturateCast<OutBT>(DirectLinearLerp(evenTop, evenMid, 0.75f));
                top[2 * l + 1] = cuda::SaturateCast<OutBT>(DirectLinearLerp(oddTop, oddMid, 0.75f));
                bot[2 * l]     = cuda::SaturateCast<OutBT>(DirectLinearLerp(evenMid, evenBot, 0.25f));
                bot[2 * l + 1] = cuda::SaturateCast<OutBT>(DirectLinearLerp(oddMid, oddBot, 0.25f));
            }

            static_assert(2 * kLanes % 4 == 0);
            // All 2*kLanes outputs are in-bounds (outW == 2*inW); alignment alone picks
            // each row's store path.
            StoreRowVectorized<4>(interpolate::GetWrapPtr(outSampleView, VecI<2>{srcX0 * 2, outY}), top);
            StoreRowVectorized<4>(interpolate::GetWrapPtr(outSampleView, VecI<2>{srcX0 * 2, outY + 1}), bot);
        }
        else
        {
            for (int l = 0; l < kLanes && srcX0 + l < inW; l++)
            {
                DirectLinear2x2Pixel(outSampleView, inSampleView, numChannels, srcX0 + l, y0, y1, y2, inW, outY);
            }
        }
    }
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

    // Channel-first 2D tensors (NCHW/CHW) report numPlanes == numChannels; the operator views each
    // channel plane as a single-channel image. Other multi-plane layouts (e.g. 3D channel-first) are
    // not supported.
    const bool isPlanar2D = srcData->layout() == nvcv::TENSOR_NCHW || srcData->layout() == nvcv::TENSOR_CHW;
    if (numPlanes > 1 && !isPlanar2D)
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

    // Multi-plane (planar, e.g. RGB8p) images are supported: each plane is processed as an
    // independent single-channel image. 2-plane planar has no defined format and is rejected.
    if (numPlanes == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-plane planar images are not supported");
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

inline const HQResizeRoiF *SampleRoi(const HQResizeRoisF &rois, int sampleIdx)
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
            throw TensorShapeError(
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

        // Planar (NCHW/CHW) input: view each (sample, channel) plane as a single-channel image and
        // run the interleaved single-channel path over N*C samples. HQResize is channel-independent,
        // so the result is bit-identical to the interleaved path and no new kernels are needed.
        // Workspace invariance: the caller sized the workspace from the public shape, and per-pass
        // volume Volume(pass)*C*N == Volume(pass)*1*(N*C), so the expanded run consumes exactly the
        // allocated buffer. The views must outlive RunPasses, hence the scope-level Optionals.
        nvcv::Optional<nvcv::TensorDataStridedCuda> srcPlanarView, dstPlanarView;
        const bool isPlanar = (src.layout() == nvcv::TENSOR_NCHW || src.layout() == nvcv::TENSOR_CHW);
        if (isPlanar)
        {
            auto srcData                  = src.exportData<nvcv::TensorDataStridedCuda>();
            auto dstData                  = dst.exportData<nvcv::TensorDataStridedCuda>();
            srcPlanarView                 = planar::PlanarAsSingleChannelView(*srcData, *srcAccess);
            dstPlanarView                 = planar::PlanarAsSingleChannelView(*dstData, *dstAccess);
            srcAccess                     = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPlanarView);
            dstAccess                     = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPlanarView);
            const int64_t expandedSamples = static_cast<int64_t>(numSamples) * numChannels;
            if (expandedSamples > 65535)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Planar HQResize launch exceeds the CUDA grid limit: N*C must not exceed 65535");
            }
            numSamples  = static_cast<int>(expandedSamples);
            numChannels = 1;
        }

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
            [this, &tensorAccess, &wideStride, &sampleDesc, &intermediate, &ws, &stream](
                auto dummySrcVal, auto intermediateVal, auto dummyDstVal, auto numChannelsVal)
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

        // Planar (multi-plane, e.g. RGB8p) var-shape input: process each of the numImages*channels
        // planes as an independent single-channel image. The kernel runs over the expanded sample
        // count with channels=1 and the var-shape adapter decodes each expanded index to its
        // {image, plane}. The caller must size the workspace for the expanded sample count -- the
        // bindings/tests build the workspace shapes as numImages*channels single-channel samples, so
        // the per-sample metadata (SampleDesc, wrap meta) and intermediate volume match.
        int planarChannels = 1;
        if constexpr (std::is_same_v<BatchContainer, nvcv::ImageBatchVarShape>)
        {
            const auto &fmt = src.uniqueFormat();
            if (fmt && fmt.numPlanes() > 1)
            {
                planarChannels = fmt.numPlanes();
            }
        }
        const bool    isPlanar           = planarChannels > 1;
        const int     kernelChannels     = isPlanar ? 1 : uniqueNumChannels;
        const int64_t numKernelSamples64 = static_cast<int64_t>(numSamples) * planarChannels;
        if (isPlanar && numKernelSamples64 > 65535)
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Planar HQResize launch exceeds the CUDA grid limit: numImages*channels must not exceed 65535");
        }
        const int numKernelSamples = static_cast<int>(numKernelSamples64);

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
        SampleDescT *sampleDescsCpu = allocator.getPinned<SampleDescT>(numKernelSamples);
        SampleDescT *sampleDescsGpu = allocator.getCuda<SampleDescT>(numKernelSamples);
        size_t       intermediateSizes[kNumTmpBuffers]{};

        int64_t inMaxStride  = 0;
        int64_t outMaxStride = 0;
        for (int sampleIdx = 0; sampleIdx < numKernelSamples; sampleIdx++)
        {
            // For planar, every channel plane of an image shares the image's spatial shape/ROI.
            const int                imageIdx  = isPlanar ? sampleIdx / planarChannels : sampleIdx;
            const VecI<kSpatialNDim> srcShape  = shape::SampleShape<kSpatialNDim>(src, imageIdx);
            const VecI<kSpatialNDim> dstShape  = shape::SampleShape<kSpatialNDim>(dst, imageIdx);
            const HQResizeRoiF      *sampleRoi = shape::SampleRoi(rois, imageIdx);
            int                      numChannels;
            if constexpr (std::is_same_v<BatchContainer, nvcv::ImageBatchVarShape>)
            {
                numChannels  = kernelChannels;
                inMaxStride  = std::max(inMaxStride, shape::ImageByteSize(src[imageIdx]));
                outMaxStride = std::max(outMaxStride, shape::ImageByteSize(dst[imageIdx]));
            }
            else
            {
                static_assert(std::is_same_v<BatchContainer, nvcv::TensorBatch>);
                numChannels  = shape::SampleNumChannels(src, dst, imageIdx);
                inMaxStride  = std::max(inMaxStride, shape::TensorByteSize(src[imageIdx]));
                outMaxStride = std::max(outMaxStride, shape::TensorByteSize(dst[imageIdx]));
            }
            SampleDescT &sampleDesc = sampleDescsCpu[sampleIdx];
            SetupSampleDesc(sampleDesc, srcShape, dstShape, numChannels, sampleRoi, minFilter, magFilter);
            for (int t = 0; t < kNumTmpBuffers; t++)
            {
                intermediateSizes[t] += GetPassOutputVolume(sampleDesc, t);
            }
        }
        bool wideStride = std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max;

        NVCV_CHECK_THROW(cudaMemcpyAsync(sampleDescsGpu, sampleDescsCpu, numKernelSamples * sizeof(SampleDescT),
                                         cudaMemcpyHostToDevice, stream));

        // allocate space for pointers and strides for intermediate wrappers
        DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers];
        IntermediateBaseT   *intermediate[kNumTmpBuffers];
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            intermediateMeta[t]
                = batch_wrapper::dynamic::AllocateDynamicBatchWrapMeta(allocator, numKernelSamples, wideStride);
        }
        // allocate space for intermediate data
        for (int t = 0; t < kNumTmpBuffers; t++)
        {
            intermediate[t] = allocator.getCuda<IntermediateBaseT>(intermediateSizes[t], kIntermediateAlignment);
        }

        Run(sampleDescsCpu, sampleDescsGpu, src, dst, intermediate, intermediateMeta, numKernelSamples, srcDtype,
            dstDtype, kernelChannels, wideStride, ws, stream, planarChannels);
    }

private:
    void Run(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu, const nvcv::ImageBatchVarShape &src,
             const nvcv::ImageBatchVarShape &dst, IntermediateBaseT *intermediate[kNumTmpBuffers],
             const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples, const nvcv::DataType srcDtype,
             const nvcv::DataType dstDtype, int uniqueNumChannels, bool wideStride, const cvcuda::Workspace &ws,
             cudaStream_t stream, int planarChannels = 1) const
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
            [this, &wideStride, &sampleDescsCpu, &sampleDescsGpu, &dstData, &srcData, &intermediate, &intermediateMeta,
             &numSamples, &ws, &stream,
             &planarChannels](auto dummySrcVal, auto intermediateVal, auto dummyDstVal, auto numChannelsVal)
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
                            numSamples, ws, stream, planarChannels);
                    }
                    else
                    {
                        RunPasses<OutT, IntermediateT, InT, int32_t, numStaticChannels>(
                            sampleDescsCpu, sampleDescsGpu, *dstData, *srcData, intermediate, intermediateMeta,
                            numSamples, ws, stream, planarChannels);
                    }
                }
            });
    }

    void Run(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu, const nvcv::TensorBatch &src,
             const nvcv::TensorBatch &dst, IntermediateBaseT *intermediate[kNumTmpBuffers],
             const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples, const nvcv::DataType srcDtype,
             const nvcv::DataType dstDtype, int uniqueNumChannels, bool wideStride, const cvcuda::Workspace &ws,
             cudaStream_t stream, int planarChannels = 1) const
    {
        // Planar TensorBatch is not yet wired (the var-shape adapter handles plane decode; the
        // tensor-batch adapter does not), so the operator only ever passes planarChannels == 1 here.
        NVCV_ASSERT(planarChannels == 1);

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
            [this, &wideStride, &sampleDescsCpu, &sampleDescsGpu, &dstData, &srcData, &intermediate, &intermediateMeta,
             &numSamples, &ws, &stream](auto dummySrcVal, auto intermediateVal, auto dummyDstVal, auto numChannelsVal)
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

    static void RecordReady(cudaEvent_t event, cudaStream_t stream)
    {
        if (event != nullptr)
        {
            NVCV_CHECK_THROW(cudaEventRecord(event, stream));
        }
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
        const bool useDirect
            = ShouldUseDirectTensorPath(sampleDesc, std::is_same_v<WrapBaseT<InWrap>, float>, kNumStaticChannels);
        if (numSamples > 0 && useDirect
            && (TryRunDirectLinear<DirectLinearKind::kScale2x2, kNumStaticChannels>(sampleDesc, outWrap, inWrap,
                                                                                    numSamples, stream)
                || TryRunDirectLinear<DirectLinearKind::kGeneric, kNumStaticChannels>(sampleDesc, outWrap, inWrap,
                                                                                      numSamples, stream)
                || TryRunDirectFilter2x2<kNumStaticChannels>(sampleDesc, outWrap, inWrap, numSamples, stream)
                || TryRunDirectFilterContract2x<kNumStaticChannels>(sampleDesc, outWrap, inWrap, numSamples, stream)
                || TryRunDirectFilter<kNumStaticChannels>(sampleDesc, outWrap, inWrap, numSamples, stream)))
        {
            RecordReady(ws.cudaMem.ready, stream);
            return;
        }
        const InterWrap interWrap = batch_wrapper::tensor::CreateDenseWrap<kHasDynamicChannels, IntermediateT, StrideT>(
            intermediate[0], sampleDesc.channels, sampleDesc.shapes[1]);

        SampleDescT runDesc = sampleDesc;
        SetupTensorBlockLayout<OutT, InT, kNumStaticChannels>(runDesc);
        RunPass<kNumStaticChannels, 0>(runDesc, interWrap, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(runDesc, outWrap, interWrap, numSamples, stream);
        RecordReady(ws.cudaMem.ready, stream);
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
        RecordReady(ws.cudaMem.ready, stream);
    }

    template<typename OutT, typename IntermediateT, typename InT, typename StrideT, int kNumStaticChannels,
             typename BatchDataStridedCuda, int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2> RunPasses(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu,
                                          const BatchDataStridedCuda &dstData, const BatchDataStridedCuda &srcData,
                                          IntermediateBaseT         *intermediate[kNumTmpBuffers],
                                          const DynamicBatchWrapMeta intermediateMeta[kNumTmpBuffers], int numSamples,
                                          const cvcuda::Workspace &ws, cudaStream_t stream,
                                          int planarChannels = 1) const
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
        // planarChannels > 1 only for the ImageBatchVarShape planar path; the adapter then decodes
        // each expanded sample index to its {image, plane}. The TensorBatch adapter ignores it.
        const BatchWrapOutT outWrap(dstData, planarChannels);
        const BatchWrapInT  inWrap(srcData, planarChannels);
        if (numSamples > 0
            && TryRunDirectFilterBatch<kNumStaticChannels>(sampleDescsCpu, sampleDescsGpu, outWrap, inWrap, numSamples,
                                                           stream))
        {
            RecordReady(ws.pinnedMem.ready, stream);
            RecordReady(ws.cudaMem.ready, stream);
            return;
        }
        const DynamicBatchWrap intermediateWrap
            = batch_wrapper::dynamic::CreateDynamicBatchWrap<kHasDynamicChannels, IntermediateT, StrideT>(
                0, intermediate[0], intermediateMeta[0], sampleDescsCpu, numSamples, stream);
        RecordReady(ws.pinnedMem.ready, stream);
        RunPass<kNumStaticChannels, 0>(sampleDescsCpu, sampleDescsGpu, intermediateWrap, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(sampleDescsCpu, sampleDescsGpu, outWrap, intermediateWrap, numSamples, stream);
        RecordReady(ws.cudaMem.ready, stream);
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
        RecordReady(ws.pinnedMem.ready, stream);
        RunPass<kNumStaticChannels, 0>(sampleDescsCpu, sampleDescsGpu, intermediateWrap0, inWrap, numSamples, stream);
        RunPass<kNumStaticChannels, 1>(sampleDescsCpu, sampleDescsGpu, intermediateWrap1, intermediateWrap0, numSamples,
                                       stream);
        RunPass<kNumStaticChannels, 2>(sampleDescsCpu, sampleDescsGpu, outWrap, intermediateWrap1, numSamples, stream);
        RecordReady(ws.cudaMem.ready, stream);
    }

    enum class DirectLinearKind
    {
        kScale2x2,
        kGeneric
    };

    template<DirectLinearKind kKind, int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    bool TryRunDirectLinear(const SampleDescT &sampleDesc, const PassOutWrap &outWrap, const PassInWrap &inWrap,
                            int numSamples, cudaStream_t stream) const
    {
        if constexpr (IsDirectLinearSupported<kNumStaticChannels, PassOutWrap, PassInWrap>()
                      && !(kKind == DirectLinearKind::kScale2x2 && kNumStaticChannels != 1))
        {
            if (!IsDirectLinearNoRoi(sampleDesc))
            {
                return false;
            }

            // 3-channel u8 fuses only contractions: the magnification case loses on
            // the reference SKUs, where the separable passes are bandwidth-fed.
            if constexpr (kNumStaticChannels == 3 && std::is_same_v<WrapBaseT<PassInWrap>, unsigned char>)
            {
                if (HasMagnifyingAxis(sampleDesc))
                {
                    return false;
                }
            }

            const VecI<2> outShape = sampleDesc.shapes[2];
            if (outShape.x <= 0 || outShape.y <= 0)
            {
                return false;
            }

            dim3 block(kBlockDim.x, kBlockDim.y, 1);
            if constexpr (kKind == DirectLinearKind::kScale2x2)
            {
                VecF<2> origin{};
                VecF<2> scale{};
                resampling::DirectLinearOriginScale(sampleDesc, origin, scale);

                const VecI<2> inShape = sampleDesc.shapes[0];
                if (origin.x != 0.f || origin.y != 0.f || scale.x != 0.5f || scale.y != 0.5f
                    || outShape.x != 2 * inShape.x || outShape.y != 2 * inShape.y)
                {
                    return false;
                }

                constexpr int kLanes = resampling::kDirectLinear2x2Lanes<WrapBaseT<PassInWrap>, WrapBaseT<PassOutWrap>>;
                dim3          grid(utils::DivCeil(inShape.x, static_cast<int>(block.x) * kLanes),
                                   utils::DivCeil(inShape.y, static_cast<int>(block.y)), numSamples);
                resampling::DirectLinear2x2Kernel<kNumStaticChannels>
                    <<<grid, block, 0, stream>>>(sampleDesc, outWrap, inWrap);
            }
            else
            {
                constexpr int kLanes = resampling::kDirectLinear2DLanes<kNumStaticChannels, WrapBaseT<PassInWrap>,
                                                                        WrapBaseT<PassOutWrap>>;
                dim3          grid(utils::DivCeil(outShape.x, static_cast<int>(block.x) * kLanes),
                                   utils::DivCeil(outShape.y, static_cast<int>(block.y)), numSamples);
                resampling::DirectLinear2DKernel<kNumStaticChannels>
                    <<<grid, block, 0, stream>>>(sampleDesc, outWrap, inWrap);
            }
            NVCV_CHECK_THROW(cudaGetLastError());
            return true;
        }
        return false;
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    static constexpr bool IsDirectLinearSupported()
    {
        using OutBT = WrapBaseT<PassOutWrap>;
        using InBT  = WrapBaseT<PassInWrap>;
        return kSpatialNDim == 2 && (kNumStaticChannels == 1 || kNumStaticChannels == 3)
            && ((std::is_same_v<InBT, unsigned char> && std::is_same_v<OutBT, unsigned char>)
                || (std::is_same_v<InBT, float> && std::is_same_v<OutBT, float>));
    }

    static bool IsDirectLinearNoRoi(const SampleDescT &sampleDesc)
    {
        return sampleDesc.filterKind[0] == filter::FilterTypeKind::Linear
            && sampleDesc.filterKind[1] == filter::FilterTypeKind::Linear && sampleDesc.inRoiOffset.x == 0
            && sampleDesc.inRoiOffset.y == 0;
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    static constexpr bool IsDirectFilterSupported()
    {
        // Same type/channel set as the direct-linear kernels, except single-channel
        // u8: the separable vectorized vertical pass keeps the reference-SKU edge over
        // every fused u8 C1 variant, including the phased word-load kernel.
        return IsDirectLinearSupported<kNumStaticChannels, PassOutWrap, PassInWrap>()
            && !(kNumStaticChannels == 1 && std::is_same_v<WrapBaseT<PassInWrap>, unsigned char>);
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    static constexpr bool IsDirectFilter2x2Supported()
    {
        // Single-channel only (its lane structure amortizes the byte loads). 3-channel
        // stays separable: u8 C3 is L1-bound on the per-quad byte loads, and f32 C3
        // loses on the bandwidth-rich reference SKUs.
        return kNumStaticChannels == 1 && IsDirectLinearSupported<kNumStaticChannels, PassOutWrap, PassInWrap>();
    }

    /**
     * @brief Derives pass-ordered phase counts/steps for any rational resampling of a
     * full-plane sample (source extent / gcd per pass axis), or returns false when a
     * period exceeds the shared-table cap.
     */
    static bool ComputePassPhases(const SampleDescT &sampleDesc, VecI<2> &phaseCount, VecI<2> &phaseStep)
    {
        if constexpr (kSpatialNDim != 2)
        {
            return false;
        }
        else
        {
            const VecI<2> inShape  = sampleDesc.shapes[0];
            const VecI<2> outShape = sampleDesc.shapes[2];
            if (inShape.x <= 0 || inShape.y <= 0 || outShape.x <= 0 || outShape.y <= 0)
            {
                return false;
            }
            const int axis0 = sampleDesc.processingOrder.x;
            const int in0   = axis0 == 0 ? inShape.x : inShape.y;
            const int in1   = axis0 == 0 ? inShape.y : inShape.x;
            const int out0  = axis0 == 0 ? outShape.x : outShape.y;
            const int out1  = axis0 == 0 ? outShape.y : outShape.x;
            // The integer-derived phases describe exactly the full-plane mapping; any
            // ROI that shifts the origin or changes the scale must stay per-pixel.
            if (sampleDesc.origin.x != 0.f || sampleDesc.origin.y != 0.f
                || sampleDesc.scale.x != static_cast<float>(in0) / static_cast<float>(out0)
                || sampleDesc.scale.y != static_cast<float>(in1) / static_cast<float>(out1))
            {
                return false;
            }
            const int g0 = std::gcd(in0, out0);
            const int g1 = std::gcd(in1, out1);
            phaseCount   = {out0 / g0, out1 / g1};
            phaseStep    = {in0 / g0, in1 / g1};
            return phaseCount.x <= resampling::kMaxDirectFilterPhases
                && phaseCount.y <= resampling::kMaxDirectFilterPhases;
        }
    }

    static int PhaseTableShmSize(const VecI<2> &phaseCount, const int support = resampling::kMaxDirectFilterSupport)
    {
        const int phases = phaseCount.x + phaseCount.y;
        return phases * (support + 1) * sizeof(float) + phases * sizeof(int);
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    bool TryRunDirectFilter2x2(const SampleDescT &sampleDesc, const PassOutWrap &outWrap, const PassInWrap &inWrap,
                               int numSamples, cudaStream_t stream) const
    {
        if constexpr (IsDirectFilter2x2Supported<kNumStaticChannels, PassOutWrap, PassInWrap>())
        {
            if (!IsDirectFilterEligible(sampleDesc))
            {
                return false;
            }
            // Exact-2x magnification with X resampled first: two phases per axis.
            const VecI<2> inShape  = sampleDesc.shapes[0];
            const VecI<2> outShape = sampleDesc.shapes[2];
            if (sampleDesc.processingOrder.x != 0 || sampleDesc.origin.x != 0.f || sampleDesc.origin.y != 0.f
                || sampleDesc.scale.x != 0.5f || sampleDesc.scale.y != 0.5f || outShape.x != 2 * inShape.x
                || outShape.y != 2 * inShape.y)
            {
                return false;
            }

            constexpr int kLanes = resampling::kDirectFilter2x2Lanes<WrapBaseT<PassInWrap>, WrapBaseT<PassOutWrap>>;
            dim3          block(kBlockDim.x, kBlockDim.y, 1);
            dim3          grid(utils::DivCeil(inShape.x, static_cast<int>(block.x) * kLanes),
                               utils::DivCeil(inShape.y, static_cast<int>(block.y)), numSamples);
            resampling::DirectFilter2x2Kernel<kNumStaticChannels, resampling::kMaxDirectFilterSupport>
                <<<grid, block, 0, stream>>>(sampleDesc, outWrap, inWrap);
            NVCV_CHECK_THROW(cudaGetLastError());
            return true;
        }
        return false;
    }

    // Common gate of every fused support-based kernel: no ROI offset and support-4
    // table filters on both axes.
    static bool IsDirectFilterEligible(const SampleDescT &sampleDesc)
    {
        return sampleDesc.inRoiOffset.x == 0 && sampleDesc.inRoiOffset.y == 0
            && sampleDesc.filterKind[0] == filter::FilterTypeKind::ShmFilter
            && sampleDesc.filterKind[1] == filter::FilterTypeKind::ShmFilter
            && sampleDesc.filter[0].support() == resampling::kMaxDirectFilterSupport
            && sampleDesc.filter[1].support() == resampling::kMaxDirectFilterSupport;
    }

    static bool IsCubicFilter(const SampleDescT &sampleDesc)
    {
        return IsDirectFilterEligible(sampleDesc)
            && sampleDesc.filter[0].numCoeffs == filter::ResamplingFiltersFactory::kCubicSize
            && sampleDesc.filter[1].numCoeffs == filter::ResamplingFiltersFactory::kCubicSize;
    }

    bool ShouldUseDirectTensorPath(const SampleDescT &sampleDesc, bool isFloat, int numChannels) const
    {
        DirectTensorPathDesc pathDesc{};
        pathDesc.linear = IsDirectLinearNoRoi(sampleDesc);
        pathDesc.cubic  = IsCubicFilter(sampleDesc);
        pathDesc.xFirst = sampleDesc.processingOrder.x == 0;
        if (pathDesc.linear)
        {
            VecF<2> origin{};
            VecF<2> scale{};
            resampling::DirectLinearOriginScale(sampleDesc, origin, scale);
            pathDesc.originX = origin.x;
            pathDesc.originY = origin.y;
            pathDesc.scaleX  = scale.x;
            pathDesc.scaleY  = scale.y;
        }
        else
        {
            pathDesc.originX = sampleDesc.origin.x;
            pathDesc.originY = sampleDesc.origin.y;
            pathDesc.scaleX  = sampleDesc.scale.x;
            pathDesc.scaleY  = sampleDesc.scale.y;
        }
        pathDesc.inWidth   = sampleDesc.shapes[0].x;
        pathDesc.inHeight  = sampleDesc.shapes[0].y;
        pathDesc.outWidth  = sampleDesc.shapes[2].x;
        pathDesc.outHeight = sampleDesc.shapes[2].y;

        return ShouldUseDirectTensorPathForSM(m_filtersFactory.GetDeviceComputeCapability(), pathDesc, isFloat,
                                              numChannels);
    }

    static bool HasMagnifyingAxis(const SampleDescT &sampleDesc)
    {
        return std::abs(sampleDesc.scale.x) < 1.f || std::abs(sampleDesc.scale.y) < 1.f;
    }

    // Minimum per-sample output pixels for the u8 uniform-tensor Contract2x kernel
    // (4k-class planes; smaller planes stay on the separable VertXVec path, which
    // wins there on the bandwidth-rich reference SKUs).
    static constexpr int64_t kDirectFilterContract2xMinU8OutPixels = 2000000;

    static bool IsDirectFilterContract2x(const SampleDescT &sampleDesc)
    {
        // Exact-2x contraction with Y resampled first: one coefficient phase per axis.
        return IsDirectFilterEligible(sampleDesc) && sampleDesc.processingOrder.x == 1 && sampleDesc.scale.x == 2.0f
            && sampleDesc.scale.y == 2.0f;
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    static constexpr bool IsDirectFilterContract2xSupported()
    {
        // Single-channel only: multi-channel already amortizes the coefficient
        // evaluations over channels in the generic fused kernel, and the phase-shared
        // kernel measures slightly worse there.
        return kSpatialNDim == 2 && kNumStaticChannels == 1
            && IsDirectLinearSupported<kNumStaticChannels, PassOutWrap, PassInWrap>();
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    bool TryRunDirectFilterContract2x(const SampleDescT &sampleDesc, const PassOutWrap &outWrap,
                                      const PassInWrap &inWrap, int numSamples, cudaStream_t stream) const
    {
        if constexpr (IsDirectFilterContract2xSupported<kNumStaticChannels, PassOutWrap, PassInWrap>())
        {
            if (!IsDirectFilterContract2x(sampleDesc))
            {
                return false;
            }
            const VecI<2> outShape = sampleDesc.shapes[2];
            if (outShape.x <= 0 || outShape.y <= 0)
            {
                return false;
            }

            using InBT = WrapBaseT<PassInWrap>;
            // The issue-bound u8 kernel only beats the separable VertXVec path on
            // large planes: 4k-class outputs win on every measured SKU, smaller ones
            // lose on the reference SKUs. f32 wins at every measured size.
            if constexpr (std::is_same_v<InBT, unsigned char>)
            {
                if (static_cast<int64_t>(outShape.x) * outShape.y < kDirectFilterContract2xMinU8OutPixels)
                {
                    return false;
                }
            }
            constexpr int kLanes = resampling::kDirectFilterContract2xLanes<InBT>;
            dim3          block(kBlockDim.x, kBlockDim.y, 1);
            dim3          grid(utils::DivCeil(outShape.x, static_cast<int>(block.x) * kLanes),
                               utils::DivCeil(outShape.y, static_cast<int>(block.y)), numSamples);
            resampling::DirectFilterContract2xKernel<kNumStaticChannels, resampling::kMaxDirectFilterSupport>
                <<<grid, block, 0, stream>>>(sampleDesc, outWrap, inWrap);
            NVCV_CHECK_THROW(cudaGetLastError());
            return true;
        }
        return false;
    }

    template<int kNumStaticChannels, typename BatchOutWrap, typename BatchInWrap>
    bool TryRunDirectFilterBatch(const SampleDescT *sampleDescsCpu, const SampleDescT *sampleDescsGpu,
                                 const BatchOutWrap &outWrap, const BatchInWrap &inWrap, int numSamples,
                                 cudaStream_t stream) const
    {
        // The generic fused batch kernel is f32-only: the u8 batch rows lose on the
        // reference SKUs. The Contract2x batch variant keeps u8, which wins on every
        // measured SKU against the heavier batch separable baseline.
        constexpr bool kGenericOk = IsDirectFilterSupported<kNumStaticChannels, BatchOutWrap, BatchInWrap>()
                                 && std::is_same_v<WrapBaseT<BatchInWrap>, float>;
        constexpr bool kContract2xOk
            = IsDirectFilterContract2xSupported<kNumStaticChannels, BatchOutWrap, BatchInWrap>();
        constexpr bool kPhasedOk = IsDirectFilterSupported<kNumStaticChannels, BatchOutWrap, BatchInWrap>();
        if constexpr (kSpatialNDim != 2 || !(kGenericOk || kContract2xOk || kPhasedOk))
        {
            return false;
        }
        else
        {
            if (numSamples <= 0)
            {
                return false;
            }
            // The sample descriptors live in pinned staging memory, where host reads are
            // uncached and slow enough to gap the GPU between iterations: reject on the
            // first sample alone where possible, and stage the descriptors locally before
            // the remaining per-sample checks.
            const SampleDescT desc0 = sampleDescsCpu[0];
            if (!IsDirectFilterEligible(desc0) || HasMagnifyingAxis(desc0))
            {
                return false;
            }
            VecI<2>    phaseCount, phaseStep;
            const bool phased0
                = ComputePassPhases(desc0, phaseCount, phaseStep)
               && (std::is_same_v<WrapBaseT<BatchInWrap>,
                                  float> || (kNumStaticChannels == 3 && (phaseCount.x > 1 || phaseCount.y > 1)));
            if constexpr (!kGenericOk)
            {
                if (!(kContract2xOk && IsDirectFilterContract2x(desc0)) && !(kPhasedOk && phased0))
                {
                    return false;
                }
            }
            const std::vector<SampleDescT> stagedDescs(sampleDescsCpu, sampleDescsCpu + numSamples);
            const SampleDescT             *descs = stagedDescs.data();

            // All samples must qualify; a mixed batch falls back to the separable passes.
            bool    allContract2x = true;
            bool    uniformShape  = true;
            VecI<2> maxOutShape{};
            for (int i = 0; i < numSamples; i++)
            {
                const SampleDescT &d = descs[i];
                if (!IsDirectFilterEligible(d) || HasMagnifyingAxis(d))
                {
                    return false;
                }
                allContract2x = allContract2x && IsDirectFilterContract2x(d);
                uniformShape  = uniformShape && d.shapes[0].x == descs[0].shapes[0].x
                            && d.shapes[0].y == descs[0].shapes[0].y && d.shapes[2].x == descs[0].shapes[2].x
                            && d.shapes[2].y == descs[0].shapes[2].y;
                maxOutShape = cuda::max(maxOutShape, d.shapes[2]);
            }
            if (maxOutShape.x <= 0 || maxOutShape.y <= 0)
            {
                return false;
            }

            dim3 block(kBlockDim.x, kBlockDim.y, 1);
            if constexpr (kContract2xOk)
            {
                if (allContract2x)
                {
                    constexpr int kLanes = resampling::kDirectFilterContract2xLanes<WrapBaseT<BatchInWrap>>;
                    dim3          grid(utils::DivCeil(maxOutShape.x, static_cast<int>(block.x) * kLanes),
                                       utils::DivCeil(maxOutShape.y, static_cast<int>(block.y)), numSamples);
                    resampling::DirectFilterContract2xBatchKernel<kNumStaticChannels,
                                                                  resampling::kMaxDirectFilterSupport>
                        <<<grid, block, 0, stream>>>(sampleDescsGpu, outWrap, inWrap);
                    NVCV_CHECK_THROW(cudaGetLastError());
                    return true;
                }
            }
            if constexpr (kPhasedOk)
            {
                // Uniform-shape phased contraction: one shared coefficient table serves
                // every sample. u8 fuses only 3-channel samples at non-integer scales;
                // single-channel u8 (planar planes) keeps the separable path per the
                // reference-SKU evidence.
                bool allPhased = uniformShape && phased0;
                for (int i = 1; allPhased && i < numSamples; i++)
                {
                    VecI<2> samplePhaseCount, samplePhaseStep;
                    allPhased = ComputePassPhases(descs[i], samplePhaseCount, samplePhaseStep)
                             && samplePhaseCount.x == phaseCount.x && samplePhaseCount.y == phaseCount.y
                             && samplePhaseStep.x == phaseStep.x && samplePhaseStep.y == phaseStep.y;
                }
                if (allPhased)
                {
                    dim3 grid(utils::DivCeil(maxOutShape.x, static_cast<int>(block.x)),
                              utils::DivCeil(maxOutShape.y, static_cast<int>(block.y)), numSamples);
                    resampling::DirectFilterPhasedBatch2DKernel<kNumStaticChannels, resampling::kMaxDirectFilterSupport>
                        <<<grid, block, PhaseTableShmSize(phaseCount), stream>>>(sampleDescsGpu, phaseCount, phaseStep,
                                                                                 outWrap, inWrap);
                    NVCV_CHECK_THROW(cudaGetLastError());
                    return true;
                }
            }
            if constexpr (kGenericOk)
            {
                dim3 grid(utils::DivCeil(maxOutShape.x, static_cast<int>(block.x)),
                          utils::DivCeil(maxOutShape.y, static_cast<int>(block.y)), numSamples);
                resampling::DirectFilterBatch2DKernel<kNumStaticChannels, resampling::kMaxDirectFilterSupport>
                    <<<grid, block, 0, stream>>>(sampleDescsGpu, outWrap, inWrap);
                NVCV_CHECK_THROW(cudaGetLastError());
                return true;
            }
            return false;
        }
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    bool TryRunDirectFilter(const SampleDescT &sampleDesc, const PassOutWrap &outWrap, const PassInWrap &inWrap,
                            int numSamples, cudaStream_t stream) const
    {
        if constexpr (IsDirectFilterSupported<kNumStaticChannels, PassOutWrap, PassInWrap>())
        {
            // Contraction only: for magnifications the per-output-pixel coefficient
            // recomputation outweighs the intermediate-traffic savings.
            if (HasMagnifyingAxis(sampleDesc))
            {
                return false;
            }
            const VecI<2> outShape = sampleDesc.shapes[2];
            if (outShape.x <= 0 || outShape.y <= 0)
            {
                return false;
            }

            dim3       block(kBlockDim.x, kBlockDim.y, 1);
            dim3       grid(utils::DivCeil(outShape.x, static_cast<int>(block.x)),
                            utils::DivCeil(outShape.y, static_cast<int>(block.y)), numSamples);
            VecI<2>    phaseCount, phaseStep;
            const bool phased = ComputePassPhases(sampleDesc, phaseCount, phaseStep);
            if (IsDirectFilterEligible(sampleDesc))
            {
                // Support-4 filters: cubic and the isotropic-2x antialiased linear. The
                // phased variant replaces the per-output coefficient evaluation with
                // shared-table lookups whenever the scale is rational with small
                // periods (identical math either way).
                if (phased)
                {
                    resampling::DirectFilterPhased2DKernel<kNumStaticChannels, resampling::kMaxDirectFilterSupport>
                        <<<grid, block, PhaseTableShmSize(phaseCount), stream>>>(sampleDesc, phaseCount, phaseStep,
                                                                                 outWrap, inWrap);
                }
                else
                {
                    resampling::DirectFilter2DKernel<kNumStaticChannels, resampling::kMaxDirectFilterSupport>
                        <<<grid, block, 0, stream>>>(sampleDesc, outWrap, inWrap);
                }
                NVCV_CHECK_THROW(cudaGetLastError());
                return true;
            }
            return false;
        }
        return false;
    }

    template<int kNumStaticChannels, typename PassOutWrap, typename PassInWrap>
    static constexpr bool IsVertXVecSupported()
    {
        using OutT  = typename PassOutWrap::ValueType;
        using InT   = std::remove_const_t<typename PassInWrap::ValueType>;
        using OutBT = cuda::BaseType<OutT>;
        using InBT  = cuda::BaseType<InT>;
        return kSpatialNDim == 2 && kNumStaticChannels == 1
            && std::is_same_v<OutBT, float> && std::is_same_v<InBT, unsigned char>;
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
            if (utils::Volume(blockShape) == 0)
            {
                return;
            }
            numBlocks = utils::DivCeil(outputShape, blockShape);
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
        // Keep the vectorized vertical path in its own instantiation so other tensor passes
        // retain their original register footprint.
        if constexpr (IsVertXVecSupported<kNumStaticChannels, PassOutWrap, PassInWrap>())
        {
            if (ShouldEnableVertXVec<kNumStaticChannels, kWhichPass, PassOutWrap, PassInWrap>(sampleDesc))
            {
                resampling::SeparableResamplingKernel<kNumStaticChannels, kWhichPass, true>
                    <<<grid, block, sharedMemSize, stream>>>(sampleDesc, outWrap, inWrap, devGridHelper);
            }
            else
            {
                resampling::SeparableResamplingKernel<kNumStaticChannels, kWhichPass>
                    <<<grid, block, sharedMemSize, stream>>>(sampleDesc, outWrap, inWrap, devGridHelper);
            }
        }
        else
        {
            resampling::SeparableResamplingKernel<kNumStaticChannels, kWhichPass>
                <<<grid, block, sharedMemSize, stream>>>(sampleDesc, outWrap, inWrap, devGridHelper);
        }
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
            if (utils::Volume(blockShape) == 0)
                continue;
            VecI<kSpatialNDim> numBlocks = utils::DivCeil(outputShape, blockShape);
            maxNumBlocks                 = cuda::max(maxNumBlocks, numBlocks);
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

    template<int kNumStaticChannels, int kWhichPass, typename PassOutWrap, typename PassInWrap>
    bool ShouldEnableVertXVec(const SampleDescT &sampleDesc) const
    {
        if constexpr (IsVertXVecSupported<kNumStaticChannels, PassOutWrap, PassInWrap>())
        {
            using resampling::interpolate::kVertXVecMaxScale;
            using resampling::interpolate::kVertXVecMaxSupport;
            using resampling::interpolate::filter_support::CanComputeCoefPerThread;
            int axis = cuda::GetElement(sampleDesc.processingOrder, kWhichPass);
            if (axis == 1 && sampleDesc.filterKind[kWhichPass] == filter::FilterTypeKind::Linear)
            {
                float scale = std::abs(cuda::GetElement(sampleDesc.scale, kWhichPass));
                return scale >= 1.0f && scale <= kVertXVecMaxScale;
            }
            else if (axis == 1 && sampleDesc.filterKind[kWhichPass] == filter::FilterTypeKind::ShmFilter)
            {
                int   support = sampleDesc.filter[kWhichPass].support();
                float scale   = std::abs(cuda::GetElement(sampleDesc.scale, kWhichPass));
                return scale >= 1.0f && scale <= kVertXVecMaxScale && support <= kVertXVecMaxSupport
                    && CanComputeCoefPerThread(support, kBlockDim.y);
            }
        }
        return false;
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
            SetupBlockLayoutPass(sampleDesc, pass, lanes);
        }
    }

    template<int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2> SetupBlockLayoutPass(SampleDescT &sampleDesc, int pass, int lanes) const
    {
        static_assert(kSpatialNDim == 2);
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

    template<typename OutT, typename InT, int kNumStaticChannels, int ndim = kSpatialNDim>
    std::enable_if_t<ndim == 2> SetupTensorBlockLayout(SampleDescT &sampleDesc) const
    {
        static_assert(kSpatialNDim == 2);
        using InBT  = cuda::BaseType<InT>;
        using OutBT = cuda::BaseType<OutT>;
        constexpr bool kSingleChannelU8
            = kNumStaticChannels == 1 && std::is_same_v<InBT, unsigned char> && std::is_same_v<OutBT, unsigned char>;
        constexpr bool kSingleChannelF32
            = kNumStaticChannels == 1 && std::is_same_v<InBT, float> && std::is_same_v<OutBT, float>;
        if constexpr (kSingleChannelU8 || kSingleChannelF32)
        {
            if (resampling::GetResizeBlockLanes() == 8)
            {
                bool isCubicSupport4 = true;
                for (int pass = 0; pass < kSpatialNDim; pass++)
                {
                    const auto &flt                 = sampleDesc.filter[pass];
                    const bool  passIsCubicSupport4 = sampleDesc.filterKind[pass] == filter::FilterTypeKind::ShmFilter
                                                  && flt.numCoeffs == filter::ResamplingFiltersFactory::kCubicSize
                                                  && flt.support() == 4;
                    isCubicSupport4 = isCubicSupport4 && passIsCubicSupport4;
                }

                if (isCubicSupport4)
                {
                    bool isCubicMagnification = true;
                    for (int pass = 0; pass < kSpatialNDim; pass++)
                    {
                        const float scale    = std::abs(cuda::GetElement(sampleDesc.scale, pass));
                        isCubicMagnification = isCubicMagnification && scale < 1.0f;
                    }

                    int lanes = 8;
                    if constexpr (kSingleChannelU8)
                    {
                        lanes = isCubicMagnification ? 64 : 8;
                    }
                    else if constexpr (kSingleChannelF32)
                    {
                        lanes = isCubicMagnification ? resampling::GetF32CubicMagnificationBlockLanes(
                                    m_filtersFactory.GetDeviceComputeCapability())
                                                     : 8;
                    }

                    if (lanes != 8)
                    {
                        for (int pass = 0; pass < kSpatialNDim; pass++)
                        {
                            SetupBlockLayoutPass(sampleDesc, pass, lanes);
                        }
                    }
                }
            }
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
} // namespace cvcuda::priv::hq_resize::kernel

#endif // CVCUDA_PRIV_HQ_RESIZE_KERNEL_CUH
