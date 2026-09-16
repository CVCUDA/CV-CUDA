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

#include "OpAdjustHue.hpp"

#include "ChannelAxisCommon.cuh"

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

namespace {

// torchvision's float->uint8 scale factor (255 + 1 - eps): x in [0,1] maps to [0, 255.999], truncated.
constexpr float kU8Scale = 255.0f + 1.0f - 1e-3f;

// Core: rotate the hue of a single normalized RGB pixel (channels in [0, 1]) by `hueShift`, using
// torchvision's RGB<->HSV operation order. Shared by the interleaved and planar kernels.
inline __host__ __device__ void HsvHueRotate(float r, float g, float b, float hueShift, float &outR, float &outG,
                                             float &outB)
{
    const float maxc = fmaxf(fmaxf(r, g), b);
    const float minc = fminf(fminf(r, g), b);
    const float v    = maxc;
    const float cr   = maxc - minc;
    const bool  eqc  = (maxc == minc);

    const float s   = cr / (eqc ? 1.0f : maxc);
    const float crd = eqc ? 1.0f : cr;
    const float rc  = (maxc - r) / crd;
    const float gc  = (maxc - g) / crd;
    const float bc  = (maxc - b) / crd;

    // Hue sector (mutually-exclusive branches == torchvision's masked hr+hg+hb sum).
    float hh;
    if (maxc == r)
    {
        hh = bc - gc;
    }
    else if (maxc == g)
    {
        hh = 2.0f + rc - bc;
    }
    else
    {
        hh = 4.0f + gc - rc;
    }

    float h = hh / 6.0f + 1.0f;
    h       = h - floorf(h); // torch fmod(.,1) — arg is positive here, so == frac

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

    switch (i)
    {
    case 0:
        outR = v, outG = t, outB = p;
        break;
    case 1:
        outR = q, outG = v, outB = p;
        break;
    case 2:
        outR = p, outG = v, outB = t;
        break;
    case 3:
        outR = p, outG = q, outB = v;
        break;
    case 4:
        outR = t, outG = p, outB = v;
        break;
    default: // 5
        outR = v, outG = p, outB = q;
        break;
    }
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

// Core per-pixel hue rotation, shared by the interleaved and planar paths. 1-channel pixels are
// returned unchanged (torchvision returns the image as-is).
template<typename T>
inline __host__ __device__ T AdjustHuePixel(T pixel, float hueShift)
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
        float outR;
        float outG;
        float outB;
        HsvHueRotate(NormalizeIn<BT>(cuda::GetElement(pixel, 0)), NormalizeIn<BT>(cuda::GetElement(pixel, 1)),
                     NormalizeIn<BT>(cuda::GetElement(pixel, 2)), hueShift, outR, outG, outB);
        cuda::GetElement(out, 0) = StoreOut<BT>(outR);
        cuda::GetElement(out, 1) = StoreOut<BT>(outG);
        cuda::GetElement(out, 2) = StoreOut<BT>(outB);
    }
    return out;
}

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
                        const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, bool isPlanar, float hueShift)
{
    using BT = cuda::BaseType<T>;

    CheckBatchLimit(dstData.numImages());

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
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
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

    bool           isEmpty;
    int            numChannels;
    nvcv::DataType dtype;
    auto           srcData = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    auto           dstData = dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    const bool isPlanar = channel_axis::ValidateSrcDstVarBatch(isEmpty, numChannels, dtype, src, dst, srcData, dstData);
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
                         RunVarShape<T, numChannels>(stream, *srcData, *dstData, isPlanar, hueShift);
                     });
}

} // namespace cvcuda::priv
