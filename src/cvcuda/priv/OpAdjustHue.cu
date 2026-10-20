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

#include "CudaDeviceUtils.hpp"
#include "OpAdjustHue.hpp"
#include "TensorAccessValidation.hpp"

#include "ChannelAxisCommon.cuh"
#include "ElementStreamCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
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

#include <type_traits>

namespace cuda         = nvcv::cuda;
namespace util         = nvcv::util;
namespace channel_axis = cvcuda::priv::channel_axis;
namespace elem_stream  = cvcuda::priv::elem_stream;

namespace {

// torchvision's float->uint8 scale factor (255 + 1 - eps): x in [0,1] maps to [0, 255.999], truncated.
constexpr float kU8Scale = 255.0f + 1.0f - 1e-3f;

// Correctly rounded 1/6 for MarksteinDiv; no reciprocal instruction is needed.
constexpr float kSixRcp = 0.1666666716337204f;

// Requires correctly rounded 1/b; subnormal quotients can differ from IEEE division, so floats must use `/`.
// Exhaustive testing of all 2^24 uint8 RGB triples matches `/`; hueShift follows all divisions.
inline __device__ float MarksteinDiv(float a, float b, float rcp)
{
    const float q0 = a * rcp;
    const float r  = __fmaf_rn(-b, q0, a);
    return __fmaf_rn(r, rcp, q0);
}

// Core: rotate the hue of a single normalized RGB pixel (channels in [0, 1]) by `hueShift`, using
// torchvision's RGB<->HSV operation order. Shared by the interleaved and planar kernels.
// Preserve (K + q1) - q2 association; the leading 0 + is exact because maxc - channel never yields -0.
template<bool kFastDiv>
inline __device__ void HsvHueRotate(float r, float g, float b, float hueShift, float &outR, float &outG, float &outB)
{
    const float maxc = fmaxf(fmaxf(r, g), b);
    const float minc = fminf(fminf(r, g), b);
    const float v    = maxc;
    const float cr   = maxc - minc;
    const bool  eqc  = (maxc == minc);

    const float s   = cr / (eqc ? 1.0f : maxc);
    const float crd = eqc ? 1.0f : cr;

    const float dr  = maxc - r;
    const float dg  = maxc - g;
    const float db  = maxc - b;
    const bool  isR = (maxc == r);
    const bool  isG = (maxc == g);

    const float n1 = isR ? db : (isG ? dr : dg);
    const float n2 = isR ? dg : (isG ? db : dr);
    const float ko = isR ? 0.0f : (isG ? 2.0f : 4.0f);

    float hh;
    float h;
    if constexpr (kFastDiv)
    {
        const float ircd = __frcp_rn(crd);
        hh               = (ko + MarksteinDiv(n1, crd, ircd)) - MarksteinDiv(n2, crd, ircd);
        h                = MarksteinDiv(hh, 6.0f, kSixRcp) + 1.0f;
    }
    else
    {
        hh = (ko + n1 / crd) - n2 / crd;
        h  = hh / 6.0f + 1.0f;
    }
    h = h - floorf(h); // torch fmod(.,1) — arg is positive here, so == frac

    h = h + hueShift;
    h = h - floorf(h); // torch.remainder(.,1): result in [0,1)

    // HSV -> RGB
    const float h6 = h * 6.0f;
    const float ii = floorf(h6);
    const float f  = h6 - ii;
    const int   i  = static_cast<int>(ii) % 6;

    const float sxf       = s * f;
    const float oneMinusS = 1.0f - s;
    const float p         = fminf(fmaxf(oneMinusS * v, 0.0f), 1.0f);
    const float q         = fminf(fmaxf((1.0f - sxf) * v, 0.0f), 1.0f);
    const float t         = fminf(fmaxf((sxf + oneMinusS) * v, 0.0f), 1.0f);

    // Use predicated selects: a runtime-indexed sector table would spill to local memory.
    const bool i0 = (i == 0), i1 = (i == 1), i2 = (i == 2), i3 = (i == 3), i4 = (i == 4), i5 = (i == 5);
    outR = (i0 || i5) ? v : (i1 ? q : (i4 ? t : p));
    outG = (i1 || i2) ? v : (i3 ? q : (i0 ? t : p));
    outB = (i3 || i4) ? v : (i5 ? q : (i2 ? t : p));
}

// uint8 input is scaled to [0,1] before the round-trip; float input is used directly (matching
// torchvision's to_dtype(scale=True), which leaves float32 unchanged). __half is a float type in
// value semantics but is not classified by std::is_floating_point, so it is routed explicitly —
// it must not take the U8 1/255 scale path.
template<typename BT>
inline __host__ __device__ float NormalizeIn(BT v)
{
    if constexpr (cuda::detail::IsFloatingPointV<BT>)
    {
        return static_cast<float>(v);
    }
    else
    {
        return static_cast<float>(v) * (1.0f / 255.0f);
    }
}

// Store the rotated channel: uint8 scales back and truncates toward zero (torchvision's cast); float
// is written as-is so value-channel outputs outside [0,1] retain torchvision's behavior. __half
// takes the float branch (no 255.999 scale, no truncation); static_cast rounds the float result to
// the nearest half.
template<typename BT>
inline __host__ __device__ BT StoreOut(float x)
{
    if constexpr (cuda::detail::IsFloatingPointV<BT>)
    {
        return static_cast<BT>(x);
    }
    else
    {
        return cuda::SaturateCast<BT>(cuda::round<cuda::RoundMode::ZERO>(x * kU8Scale));
    }
}

template<typename BT>
inline constexpr bool kUseFastDiv = std::is_same_v<BT, unsigned char>;

template<typename BT>
inline __device__ void AdjustHueChannels(BT r, BT g, BT b, float hueShift, BT &oR, BT &oG, BT &oB)
{
    float fr;
    float fg;
    float fb;
    HsvHueRotate<kUseFastDiv<BT>>(NormalizeIn<BT>(r), NormalizeIn<BT>(g), NormalizeIn<BT>(b), hueShift, fr, fg, fb);
    oR = StoreOut<BT>(fr);
    oG = StoreOut<BT>(fg);
    oB = StoreOut<BT>(fb);
}

// Core per-pixel hue rotation, shared by the interleaved and planar paths. 1-channel pixels are
// returned unchanged (torchvision returns the image as-is).
template<typename T>
inline __device__ T AdjustHuePixel(T pixel, float hueShift)
{
    using BT               = cuda::BaseType<T>;
    constexpr int numChans = cuda::NumElements<T>;

    T out{};
    if constexpr (numChans == 1)
    {
        out = pixel;
    }
    else
    {
        static_assert(numChans == 3, "AdjustHue supports 1- or 3-channel pixels");
        AdjustHueChannels<BT>(cuda::GetElement(pixel, 0), cuda::GetElement(pixel, 1), cuda::GetElement(pixel, 2),
                              hueShift, cuda::GetElement(out, 0), cuda::GetElement(out, 1), cuda::GetElement(out, 2));
    }
    return out;
}

struct HueGroupOp
{
    // Forcing more resident blocks spills the long HSV round trip.
    static constexpr int kMinBlocksPerMP = 1;

    using Param = float;

    template<typename BT>
    inline static __device__ void Channels(BT r, BT g, BT b, const Param &hueShift, BT &oR, BT &oG, BT &oB)
    {
        AdjustHueChannels<BT>(r, g, b, hueShift, oR, oG, oB);
    }
};

// U8 uses 8-byte vectors because 16-byte groups halve occupancy through HSV register pressure.
template<typename BT>
inline constexpr int kHueGroupBytes = sizeof(BT) == 1 ? 8 : 16;

// Interleaved ((N)HWC) kernels ----------------------------------------------------------

template<typename T, class SrcWrapper, class DstWrapper>
__global__ void AdjustHueInterleaved(SrcWrapper src, DstWrapper dst, int2 size, float hueShift)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }
    dst[coord] = AdjustHuePixel<T>(src[coord], hueShift);
}

template<typename T, class SrcWrapper, class DstWrapper>
__global__ void AdjustHueInterleavedVarShape(SrcWrapper src, DstWrapper dst, float hueShift)
{
    const int  z = blockIdx.z;
    const int2 size{dst.width(z), dst.height(z)};
    int3       coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }
    dst[coord] = AdjustHuePixel<T>(src[coord], hueShift);
}

// Planar ((N)CHW) kernels ---------------------------------------------------------------

template<typename BT, int numChannels, class SrcWrapper, class DstWrapper>
inline __device__ void DoAdjustHuePlanar(SrcWrapper src, DstWrapper dst, int2 size, int z, float hueShift)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }
    if constexpr (numChannels == 1)
    {
        *dst.ptr(z, 0, coord.y, coord.x) = *src.ptr(z, 0, coord.y, coord.x);
    }
    else
    {
        using Vec = cuda::MakeType<BT, 3>;
        Vec pixel;
        cuda::GetElement(pixel, 0) = *src.ptr(z, 0, coord.y, coord.x);
        cuda::GetElement(pixel, 1) = *src.ptr(z, 1, coord.y, coord.x);
        cuda::GetElement(pixel, 2) = *src.ptr(z, 2, coord.y, coord.x);

        const Vec out = AdjustHuePixel<Vec>(pixel, hueShift);

        *dst.ptr(z, 0, coord.y, coord.x) = cuda::GetElement(out, 0);
        *dst.ptr(z, 1, coord.y, coord.x) = cuda::GetElement(out, 1);
        *dst.ptr(z, 2, coord.y, coord.x) = cuda::GetElement(out, 2);
    }
}

template<typename BT, int numChannels, class SrcWrapper, class DstWrapper>
__global__ void AdjustHuePlanar(SrcWrapper src, DstWrapper dst, int2 size, float hueShift)
{
    DoAdjustHuePlanar<BT, numChannels>(src, dst, size, blockIdx.z, hueShift);
}

template<typename BT, int numChannels, class SrcWrapper, class DstWrapper>
__global__ void AdjustHuePlanarVarShape(SrcWrapper src, DstWrapper dst, float hueShift)
{
    const int  z = blockIdx.z;
    const int2 size{dst.width(z), dst.height(z)};
    DoAdjustHuePlanar<BT, numChannels>(src, dst, size, z, hueShift);
}

// Launchers -----------------------------------------------------------------------------

constexpr int kGridZLimit = 65535;

inline void CheckBatchLimit(int numSamples)
{
    if (numSamples > kGridZLimit)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }
}

struct HueDeviceProperties
{
    int     sm            = 0;
    int     l2Bytes       = 0;
    int64_t residentBytes = 0;
};

inline const HueDeviceProperties &GetHueDeviceProperties()
{
    int device = 0;
    NVCV_CHECK_THROW(cudaGetDevice(&device));
    static thread_local int                 cachedDevice = -1;
    static thread_local HueDeviceProperties cachedProperties;
    if (cachedDevice != device)
    {
        HueDeviceProperties properties;
        NVCV_CHECK_THROW(cvcuda::priv::GetDeviceSM(device, properties.sm));
        int multiprocessors = 0;
        int maxThreads      = 0;
        if (properties.sm != 75)
        {
            NVCV_CHECK_THROW(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
            NVCV_CHECK_THROW(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
            NVCV_CHECK_THROW(cudaDeviceGetAttribute(&properties.l2Bytes, cudaDevAttrL2CacheSize, device));
            constexpr int64_t kVectorWorkingSetPerThread = 2 * 3 * 16;
            properties.residentBytes = static_cast<int64_t>(multiprocessors) * maxThreads * kVectorWorkingSetPerThread;
        }
        cachedProperties = properties;
        cachedDevice     = device;
    }
    return cachedProperties;
}

template<typename T, int numChannels>
inline void RunTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                      const nvcv::TensorDataStridedCuda &dstData, bool isPlanar, float hueShift)
{
    using BT = cuda::BaseType<T>;

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

    if constexpr (numChannels == 3)
    {
        const bool ranGroup = [&]
        {
            if constexpr (std::is_same_v<BT, float>)
            {
                if (!isPlanar)
                {
                    const auto &properties = GetHueDeviceProperties();
                    if (properties.sm == 75)
                    {
                        return elem_stream::TryRunPixelGroup<8, BT, false, HueGroupOp>(
                            stream, srcData, dstData, *srcAccess, *dstAccess, hueShift);
                    }
                    const int64_t tensorBytes
                        = static_cast<int64_t>(size.x) * size.y * srcAccess->numSamples() * 3 * sizeof(float) * 2;
                    if (properties.l2Bytes > 0 && std::min(tensorBytes, properties.residentBytes) > properties.l2Bytes)
                    {
                        return elem_stream::TryRunPixelGroup<4, BT, false, HueGroupOp>(
                            stream, srcData, dstData, *srcAccess, *dstAccess, hueShift);
                    }
                }
            }
            return isPlanar ? elem_stream::TryRunPixelGroup<kHueGroupBytes<BT>, BT, true, HueGroupOp>(
                       stream, srcData, dstData, *srcAccess, *dstAccess, hueShift)
                            : elem_stream::TryRunPixelGroup<kHueGroupBytes<BT>, BT, false, HueGroupOp>(
                                stream, srcData, dstData, *srcAccess, *dstAccess, hueShift);
        }();
        if (ranGroup)
        {
            return;
        }
    }

    const dim3 block(32, 4, 1);
    const dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y),
                    static_cast<unsigned int>(srcAccess->numSamples()));

    if (isPlanar)
    {
        auto src = cuda::CreateTensorWrapNCHW<const BT, int32_t>(srcData);
        auto dst = cuda::CreateTensorWrapNCHW<BT, int32_t>(dstData);
        AdjustHuePlanar<BT, numChannels><<<grid, block, 0, stream>>>(src, dst, size, hueShift);
    }
    else
    {
        auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(srcData);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(dstData);
        AdjustHueInterleaved<T><<<grid, block, 0, stream>>>(src, dst, size, hueShift);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, int numChannels>
inline void RunVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, bool isPlanar,
                        const elem_stream::GroupBatchInfo &groupInfo, float hueShift)
{
    using BT = cuda::BaseType<T>;

    CheckBatchLimit(dstData.numImages());

    if constexpr (numChannels == 3)
    {
        const bool ranGroup = isPlanar
                                ? elem_stream::TryRunPixelGroupVarShape<kHueGroupBytes<BT>, BT, true, HueGroupOp>(
                                    stream, srcData, dstData, groupInfo, hueShift)
                                : elem_stream::TryRunPixelGroupVarShape<kHueGroupBytes<BT>, BT, false, HueGroupOp>(
                                    stream, srcData, dstData, groupInfo, hueShift);
        if (ranGroup)
        {
            return;
        }
    }

    const int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
    const dim3 block(32, 4, 1);
    const dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y),
                    static_cast<unsigned int>(dstMaxSize.z));

    if (isPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const BT> src(srcData);
        cuda::ImageBatchVarShapeWrap<BT>       dst(dstData);
        AdjustHuePlanarVarShape<BT, numChannels><<<grid, block, 0, stream>>>(src, dst, hueShift);
    }
    else
    {
        cuda::ImageBatchVarShapeWrap<const T> src(srcData);
        cuda::ImageBatchVarShapeWrap<T>       dst(dstData);
        AdjustHueInterleavedVarShape<T><<<grid, block, 0, stream>>>(src, dst, hueShift);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Dispatch over base dtype (u8 / f16 / f32) and channel count (1 / 3) -------------------

template<typename Cb>
inline void RunChannelSwitch(int numChannels, nvcv::DataType dtype, const Cb &cb)
{
    const bool isU8  = (dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_3U8);
    const bool isF16 = (dtype == nvcv::TYPE_F16 || dtype == nvcv::TYPE_3F16);
    const bool isF32 = (dtype == nvcv::TYPE_F32 || dtype == nvcv::TYPE_3F32);

    if (!isU8 && !isF16 && !isF32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: AdjustHue supports 8-bit unsigned, 16-bit float and 32-bit float");
    }

    if (numChannels == 1)
    {
        if (isU8)
        {
            cb(uchar1{}, std::integral_constant<int, 1>{});
        }
        else if (isF16)
        {
            cb(half1{}, std::integral_constant<int, 1>{});
        }
        else
        {
            cb(float1{}, std::integral_constant<int, 1>{});
        }
    }
    else if (numChannels == 3)
    {
        if (isU8)
        {
            cb(uchar3{}, std::integral_constant<int, 3>{});
        }
        else if (isF16)
        {
            cb(half3{}, std::integral_constant<int, 3>{});
        }
        else
        {
            cb(float3{}, std::integral_constant<int, 3>{});
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels: AdjustHue supports 1 or 3 channels");
    }
}

// Validation ----------------------------------------------------------------------------

inline void ValidateHue(double hue)
{
    if (!(hue >= -0.5 && hue <= 0.5))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The hue factor must be in the range [-0.5, 0.5]");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

AdjustHue::AdjustHue() {}

// Tensor input variant
void AdjustHue::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, double hue) const
{
    ValidateHue(hue);

    bool           isEmpty;
    int            numChannels;
    int            numSamples;
    nvcv::DataType dtype;
    auto           srcData = cvcuda::priv::RequireCudaTensor(src, "Input");
    auto           dstData = cvcuda::priv::RequireCudaTensor(dst, "Output");
    const bool     isPlanar
        = channel_axis::ValidateSrcDstTensors(isEmpty, numChannels, dtype, numSamples, srcData, dstData);
    if (isEmpty)
    {
        return;
    }

    const float hueShift = static_cast<float>(hue);

    RunChannelSwitch(numChannels, dtype,
                     [&](auto dummy, auto channelsIC)
                     {
                         using T                   = decltype(dummy);
                         constexpr int numChannels = decltype(channelsIC)::value;
                         RunTensor<T, numChannels>(stream, *srcData, *dstData, isPlanar, hueShift);
                     });
}

// VarShape input variant
void AdjustHue::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                           const nvcv::ImageBatchVarShape &dst, double hue) const
{
    ValidateHue(hue);

    bool                        isEmpty;
    int                         numChannels;
    nvcv::DataType              dtype;
    elem_stream::GroupBatchInfo groupInfo;
    auto                        srcData = cvcuda::priv::RequireCudaVarShape(src, stream, "Input");
    auto                        dstData = cvcuda::priv::RequireCudaVarShape(dst, stream, "Output");
    const bool                  isPlanar
        = channel_axis::ValidateSrcDstVarBatch(isEmpty, numChannels, dtype, src, dst, srcData, dstData, &groupInfo);
    if (isEmpty)
    {
        return;
    }

    const float hueShift = static_cast<float>(hue);

    RunChannelSwitch(numChannels, dtype,
                     [&](auto dummy, auto channelsIC)
                     {
                         using T                   = decltype(dummy);
                         constexpr int numChannels = decltype(channelsIC)::value;
                         RunVarShape<T, numChannels>(stream, *srcData, *dstData, isPlanar, groupInfo, hueShift);
                     });
}

} // namespace cvcuda::priv
