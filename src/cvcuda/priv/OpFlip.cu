/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpFlip.hpp"
#include "PlanarTensorView.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Optional.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// Flip is a pure pixel remap, so it is memory/latency-bound: one element per thread leaves too few
// memory requests in flight (ncu on the single-channel uchar kernel: DRAM 45%, "latency issues").
// Each thread instead processes NIX columns strided by the total x-thread count, so NIX independent
// loads/stores are issued together (memory-level parallelism) while consecutive threads still touch
// consecutive columns (coalesced). NIX>1 needs the grid sized to divUp(width, NIX) x-threads. The
// remap is unchanged, so every output is bit-exact with the one-element-per-thread version.
constexpr int kNIX       = 4;
constexpr int kScalarNIX = 1;

// Default width of the single-channel vectorized tensor access, in elements. The planar float32
// path narrows it on one architecture; see RunFlipPlanarTensor.
constexpr int kVec = 4;

// Columns per group in the var-shape U8 group copy; each thread handles kNIX such groups.
// Independent of kVec, and not retunable on its own: FlipU8Group moves the group as one uchar4.
constexpr int kU8GroupWidth = 4;

constexpr dim3 kBlock{32, 8, 1};

// Compute capability of the current device as major * 10 + minor, or 0 when it cannot be queried —
// which simply leaves the per-architecture scheduling exceptions below unselected.
inline int CurrentDeviceSMOrZero()
{
    int sm = 0;
    return cvcuda::priv::GetCurrentDeviceSM(sm) == cudaSuccess ? sm : 0;
}

// Geometry providers -------------------------------------------------------------------------
//
// Every flip direction is the same remap with two independent mirror flags. Collapsing the three
// directions into flags is what lets one kernel serve both submission variants; only *where the
// flags and the size come from* differs, and that is the whole of the difference between Tensor
// and VarShape below.

struct FlipGeometry
{
    int2 size;
    bool mirrorX;
    bool mirrorY;
};

// Tensor: one size and one direction for the whole batch. The direction is known on the host, so it
// rides in the type rather than in a field: the kernel's mirror branches then fold away at compile
// time, which is what the legacy one-kernel-per-direction split bought and what a runtime flag would
// give back (measured: ~4% on the uchar4 NHWC tensor row).
template<bool MirrorX, bool MirrorY>
struct TensorGeometry
{
    int2 size;

    template<class DstWrapper>
    inline __device__ FlipGeometry operator()(int, const DstWrapper &) const
    {
        return {size, MirrorX, MirrorY};
    }
};

// VarShape: per-image size and per-image flip code. Unlike the tensor entry point, a code outside
// {1, 0, -1} means "copy the image through unchanged", so both flags stay false.
struct VarShapeGeometry
{
    cuda::Tensor1DWrap<int> flipCode;

    template<class DstWrapper>
    inline __device__ FlipGeometry operator()(int z, const DstWrapper &dst) const
    {
        const int code = flipCode[z];
        return {
            int2{dst.width(z), dst.height(z)},
            code == 1 || code == -1, code == 0 || code == -1
        };
    }
};

// Kernels ------------------------------------------------------------------------------------

// Planar channel planes are looped inside the thread rather than given their own grid.z, so the
// per-pixel coordinate math is paid once for all of them.
template<int NIX, bool IsPlanar, class Geometry, class SrcWrapper, class DstWrapper>
__global__ void FlipKernel(SrcWrapper src, DstWrapper dst, Geometry geometry, int numPlanes)
{
    const int          z = blockIdx.z;
    const FlipGeometry g = geometry(z, dst);

    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    if (dstY >= g.size.y)
    {
        return;
    }

    const int srcY = g.mirrorY ? g.size.y - 1 - dstY : dstY;
    const int x0   = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int dstX = x0 + i * step;
        if (dstX >= g.size.x)
        {
            continue;
        }

        const int srcX = g.mirrorX ? g.size.x - 1 - dstX : dstX;

        if constexpr (IsPlanar)
        {
            for (int p = 0; p < numPlanes; ++p)
            {
                *dst.ptr(z, p, dstY, dstX) = *src.ptr(z, p, srcY, srcX);
            }
        }
        else
        {
            *dst.ptr(z, dstY, dstX) = *src.ptr(z, srcY, srcX);
        }
    }
}

template<int VEC, typename WideT>
inline __device__ WideT ReverseLanes(WideT v)
{
    WideT out;
#pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
        cuda::GetElement(out, k) = cuda::GetElement(v, VEC - 1 - k);
    }
    return out;
}

// Wide single-channel tensor flip: a run of VEC contiguous columns is loaded/stored as one
// MakeType<T,VEC> vector. A single-channel uchar plane moves only 1 byte/thread, so even with NIX
// MLP the kernel pays the per-element index overhead VEC times more than it must; coalescing VEC
// columns into one wide access cuts that overhead ~VEC x and raises bytes/thread. A horizontal flip
// maps destination block j to the mirrored source block (W/VEC-1-j) with its VEC lanes reversed in
// registers; this is exact only when the column count is a multiple of VEC and both buffers are
// vector-aligned, which the caller checks. The result is bit-identical to the scalar flip.
template<int NIX, int VEC, class Geometry, class SrcWrapper, class DstWrapper>
__global__ void FlipWideTensorKernel(SrcWrapper src, DstWrapper dst, Geometry geometry)
{
    const int          z = blockIdx.z;
    const FlipGeometry g = geometry(z, dst);

    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    if (dstY >= g.size.y)
    {
        return;
    }

    const int srcY = g.mirrorY ? g.size.y - 1 - dstY : dstY;
    const int x0   = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int dstX = x0 + i * step;
        if (dstX >= g.size.x)
        {
            continue;
        }

        if (g.mirrorX)
        {
            *dst.ptr(z, dstY, dstX) = ReverseLanes<VEC>(*src.ptr(z, srcY, g.size.x - 1 - dstX));
        }
        else
        {
            *dst.ptr(z, dstY, dstX) = *src.ptr(z, srcY, dstX);
        }
    }
}

// Move one group of a var-shape row. The tensor path can hoist its alignment test to the host
// because every sample shares one stride; a var-shape batch cannot, so eligibility is decided per
// group here and unaligned or partial groups take the bit-exact scalar tail.
inline __device__ void FlipU8Group(const unsigned char *srcRow, unsigned char *dstRow, int width, int dstX,
                                   bool mirrorX)
{
    const int            srcX    = mirrorX ? width - kU8GroupWidth - dstX : dstX;
    unsigned char       *dstPtr  = dstRow + dstX;
    const unsigned char *srcPtr  = srcX >= 0 ? srcRow + srcX : nullptr;
    const bool           full    = dstX + kU8GroupWidth <= width;
    const bool           aligned = full && srcPtr != nullptr
                      && ((reinterpret_cast<std::uintptr_t>(srcPtr) | reinterpret_cast<std::uintptr_t>(dstPtr))
                          & (alignof(uchar4) - 1))
                             == 0;

    if (aligned)
    {
        const uchar4 value                  = *reinterpret_cast<const uchar4 *>(srcPtr);
        *reinterpret_cast<uchar4 *>(dstPtr) = mirrorX ? ReverseLanes<kU8GroupWidth>(value) : value;
        return;
    }

#pragma unroll
    for (int k = 0; k < kU8GroupWidth; ++k)
    {
        const int x = dstX + k;
        if (x >= width)
        {
            break;
        }
        dstRow[x] = srcRow[mirrorX ? width - 1 - x : x];
    }
}

// Single-channel U8 VarShape (interleaved C1 and every plane of a planar batch) uses four adjacent
// columns per thread. Consecutive threads therefore move a full 128-byte warp segment instead of
// only 32 bytes, while NIX retains the independent requests that hide the ImageBatchVarShape
// pointer-lookup latency.
template<int NIX, bool IsPlanar>
__global__ void FlipWideU8VarShapeKernel(cuda::ImageBatchVarShapeWrap<const unsigned char> src,
                                         cuda::ImageBatchVarShapeWrap<unsigned char> dst, VarShapeGeometry geometry,
                                         int numPlanes)
{
    const int          z = blockIdx.z;
    const FlipGeometry g = geometry(z, dst);

    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    if (dstY >= g.size.y)
    {
        return;
    }

    const int srcY   = g.mirrorY ? g.size.y - 1 - dstY : dstY;
    const int group0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int step   = gridDim.x * blockDim.x;

    // Compile-time 1 for the interleaved instantiation, so the loop and the plane index fold away.
    const int planeCount = IsPlanar ? numPlanes : 1;

    for (int p = 0; p < planeCount; ++p)
    {
        const unsigned char *srcRow = src.ptr(z, p, srcY, 0);
        unsigned char       *dstRow = dst.ptr(z, p, dstY, 0);

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int dstX = (group0 + i * step) * kU8GroupWidth;
            if (dstX < g.size.x)
            {
                FlipU8Group(srcRow, dstRow, g.size.x, dstX, g.mirrorX);
            }
        }
    }
}

// Launchers ----------------------------------------------------------------------------------

inline dim3 FlipGrid(int width, int height, int nix, int numSamples)
{
    return dim3(util::DivUp(util::DivUp(width, nix), static_cast<int>(kBlock.x)),
                util::DivUp(height, static_cast<int>(kBlock.y)), static_cast<unsigned int>(numSamples));
}

// Resolving the flip code here turns the direction into a compile-time property of the geometry
// type for everything downstream.
template<class Fn>
void WithTensorDirection(int2 size, int32_t flipCode, const Fn &fn)
{
    if (flipCode > 0)
    {
        fn(TensorGeometry<true, false>{size});
    }
    else if (flipCode == 0)
    {
        fn(TensorGeometry<false, true>{size});
    }
    else
    {
        fn(TensorGeometry<true, true>{size});
    }
}

// Every sample of a tensor shares one set of strides, so vector eligibility is a host-side property
// of one buffer: VEC-divisible columns plus VEC-aligned base pointer and row/sample strides. Both
// buffers must qualify; the caller tests each.
template<typename T, int VEC>
bool WideSingleChannelEligible(const nvcv::TensorDataStridedCuda              &data,
                               const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    if (access.numCols() % VEC != 0)
    {
        return false;
    }

    const auto mask = static_cast<std::uintptr_t>(alignof(cuda::MakeType<T, VEC>) - 1);

    return ((reinterpret_cast<std::uintptr_t>(data.basePtr()) | static_cast<std::uintptr_t>(access.rowStride())
             | static_cast<std::uintptr_t>(access.sampleStride()))
            & mask)
        == 0;
}

template<int NIX, typename T, class Geometry>
void LaunchTensorScalar(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                        const nvcv::TensorDataStridedCuda &outData, const Geometry &geometry, int numSamples)
{
    auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    FlipKernel<NIX, false>
        <<<FlipGrid(geometry.size.x, geometry.size.y, NIX, numSamples), kBlock, 0, stream>>>(src, dst, geometry, 1);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, int VEC, class Geometry>
void LaunchTensorWide(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                      const nvcv::TensorDataStridedCuda &outData, const Geometry &geometry, int numSamples)
{
    using WideT = cuda::MakeType<T, VEC>;

    auto src = cuda::CreateTensorWrapNHW<const WideT, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<WideT, int32_t>(outData);

    // numCols is VEC-divisible (caller-checked), so the column blocks line up exactly.
    const Geometry vecGeometry{
        int2{geometry.size.x / VEC, geometry.size.y}
    };

    FlipWideTensorKernel<kNIX, VEC>
        <<<FlipGrid(vecGeometry.size.x, vecGeometry.size.y, kNIX, numSamples), kBlock, 0, stream>>>(src, dst,
                                                                                                    vecGeometry);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Single-channel tensor flip: take the vectorized kernel when the buffers allow it, otherwise the
// scalar one. Used by the interleaved C==1 path and by the planar path, where each (sample, plane)
// of the flattened view is a single-channel image.
template<typename T, class Geometry>
void RunTensorSingleChannel(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                            const nvcv::TensorDataStridedCuda &outData, const Geometry &geometry, int numSamples,
                            const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                            const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if (WideSingleChannelEligible<T, kVec>(inData, inAccess) && WideSingleChannelEligible<T, kVec>(outData, outAccess))
    {
        LaunchTensorWide<T, kVec>(stream, inData, outData, geometry, numSamples);
    }
    else
    {
        LaunchTensorScalar<kNIX, T>(stream, inData, outData, geometry, numSamples);
    }
}

template<typename T, class Geometry>
void RunFlipTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                   const nvcv::TensorDataStridedCuda &outData, const Geometry &geometry, int numSamples,
                   const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                   const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if constexpr (cuda::NumElements<T> == 1)
    {
        RunTensorSingleChannel<T>(stream, inData, outData, geometry, numSamples, inAccess, outAccess);
    }
    else if constexpr (std::is_same_v<T, float3>)
    {
        // Interleaved float3 regresses under the batched scheduling; keep it at one element/thread.
        LaunchTensorScalar<kScalarNIX, T>(stream, inData, outData, geometry, numSamples);
    }
    else
    {
        LaunchTensorScalar<kNIX, T>(stream, inData, outData, geometry, numSamples);
    }
}

// Planar tensor flip. Each (sample, plane) of the flattened view is a single-channel image, so this
// is RunTensorSingleChannel plus the two architecture exceptions that only the float32 C3 planar
// shape triggers. `flatSamples` is the flattened N*C plane count.
template<typename T, class Geometry>
void RunFlipPlanarTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inView,
                         const nvcv::TensorDataStridedCuda &outView, const Geometry &geometry, int flatSamples,
                         int channels, const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                         const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if constexpr (std::is_same_v<T, float>)
    {
        if (channels == 3)
        {
            const int computeCapability = CurrentDeviceSMOrZero();

            // GA102 / SM86 regresses on the vectorized path. Note this is the
            // one-element-per-thread variant, not the default single-channel path below.
            if (computeCapability == 86)
            {
                LaunchTensorScalar<kScalarNIX, T>(stream, inView, outView, geometry, flatSamples);
                return;
            }

            // CUDA 12 on SM80 has unstable event completion after 128-bit accesses; 64-bit accesses
            // retain vectorization without affecting CUDA 13 or other architectures. `if constexpr`
            // so CUDA 13 does not instantiate the narrowed kernels at all.
            if constexpr (CUDART_VERSION < 13000)
            {
                if (computeCapability == 80 && WideSingleChannelEligible<T, 2>(inView, inAccess)
                    && WideSingleChannelEligible<T, 2>(outView, outAccess))
                {
                    LaunchTensorWide<T, 2>(stream, inView, outView, geometry, flatSamples);
                    return;
                }
            }
        }
    }

    RunTensorSingleChannel<T>(stream, inView, outView, geometry, flatSamples, inAccess, outAccess);
}

template<int NIX, bool IsPlanar, typename T>
void LaunchVarShapeScalar(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &outData, const VarShapeGeometry &geometry,
                          int numPlanes)
{
    cuda::ImageBatchVarShapeWrap<const T> src(inData);
    cuda::ImageBatchVarShapeWrap<T>       dst(outData);

    FlipKernel<NIX, IsPlanar>
        <<<FlipGrid(inData.maxSize().w, inData.maxSize().h, NIX, outData.numImages()), kBlock, 0, stream>>>(
            src, dst, geometry, numPlanes);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<bool IsPlanar>
void LaunchVarShapeWideU8(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &outData, const VarShapeGeometry &geometry,
                          int numPlanes)
{
    cuda::ImageBatchVarShapeWrap<const unsigned char> src(inData);
    cuda::ImageBatchVarShapeWrap<unsigned char>       dst(outData);

    const int groups = util::DivUp(inData.maxSize().w, kU8GroupWidth);

    FlipWideU8VarShapeKernel<kNIX, IsPlanar>
        <<<FlipGrid(groups, inData.maxSize().h, kNIX, outData.numImages()), kBlock, 0, stream>>>(src, dst, geometry,
                                                                                                 numPlanes);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<bool IsPlanar, typename T>
void RunFlipVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                     const nvcv::ImageBatchVarShapeDataStridedCuda &outData, const VarShapeGeometry &geometry,
                     int numPlanes)
{
    if constexpr (std::is_same_v<T, unsigned char>)
    {
        LaunchVarShapeWideU8<IsPlanar>(stream, inData, outData, geometry, numPlanes);
    }
    else if constexpr (std::is_same_v<T, float3>)
    {
        LaunchVarShapeScalar<kScalarNIX, IsPlanar, T>(stream, inData, outData, geometry, numPlanes);
    }
    else
    {
        LaunchVarShapeScalar<kNIX, IsPlanar, T>(stream, inData, outData, geometry, numPlanes);
    }
}

// Dispatch and validation ----------------------------------------------------------------------

// Flip moves values without arithmetic, so a dtype only has to name a width and a signedness: F16
// reuses the 16-bit unsigned kernels bit-exactly. S16 is declared by the var-shape submission only,
// which is why the caller passes `AllowS16`. It is a template parameter rather than an argument so
// the Tensor submission does not instantiate kernels it can never dispatch to.
template<bool AllowS16, class Cb>
void RunBaseTypeSwitch(nvcv::DataType dtype, const Cb &cb)
{
    const auto bpc         = dtype.bitsPerChannel();
    const int  numChannels = dtype.numChannels();
    for (int i = 1; i < numChannels; ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have the same bit depth");
        }
    }

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
        if (bpc[0] == 8)
        {
            return cb(std::uint8_t{});
        }
        if (bpc[0] == 16)
        {
            return cb(std::uint16_t{});
        }
        break;

    case nvcv::DataKind::SIGNED:
        if constexpr (AllowS16)
        {
            if (bpc[0] == 16)
            {
                return cb(std::int16_t{});
            }
        }
        if (bpc[0] == 32)
        {
            return cb(std::int32_t{});
        }
        break;

    case nvcv::DataKind::FLOAT:
        if (bpc[0] == 16)
        {
            return cb(std::uint16_t{});
        }
        if (bpc[0] == 32)
        {
            return cb(float{});
        }
        break;

    default:
        break;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Invalid data type: Flip supports 8-bit unsigned, 16-bit unsigned, 32-bit signed, "
                          "16-bit float and 32-bit float%s",
                          AllowS16 ? ", and 16-bit signed" : "");
}

// Planar images are flipped one single-channel plane at a time, so the planar callback takes the
// base type; the interleaved one takes the packed pixel type.
template<bool AllowS16, class Cb>
void RunChannelSwitch(int numChannels, bool isPlanar, nvcv::DataType dtype, const Cb &cb)
{
    // Two-channel input has no planar counterpart and no interleaved kernel (matches Normalize and
    // Resize); anything above four channels is not an image pixel.
    if (numChannels == 2 || numChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of channels: %d", numChannels);
    }

    RunBaseTypeSwitch<AllowS16>(dtype,
                                [&](auto baseVal)
                                {
                                    using BT = decltype(baseVal);

                                    if (isPlanar)
                                    {
                                        return cb(BT{}, std::true_type{});
                                    }

                                    switch (numChannels)
                                    {
                                    case 1:
                                        // The bare base type, not MakeType<BT,1>: only the scalar
                                        // spelling composes with MakeType<BT,VEC> in the vectorized
                                        // path, which is why same_shape::DispatchChannels (whose C1
                                        // arm yields uchar1) cannot serve this switch.
                                        return cb(BT{}, std::false_type{});
                                    case 3:
                                        return cb(cuda::MakeType<BT, 3>{}, std::false_type{});
                                    default:
                                        return cb(cuda::MakeType<BT, 4>{}, std::false_type{});
                                    }
                                });
}

inline void ValidateTensorLayout(const nvcv::TensorLayout &inLayout, const nvcv::TensorLayout &outLayout)
{
    if (inLayout != outLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }
    if (!(inLayout == nvcv::TENSOR_NHWC || inLayout == nvcv::TENSOR_HWC || inLayout == nvcv::TENSOR_NCHW
          || inLayout == nvcv::TENSOR_CHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid layout, the valid layouts are: NHWC, HWC, NCHW, CHW");
    }
}

// The kernels index with 32-bit strides, so the addressed byte range has to fit int32.
inline void ValidateStrideRange(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    const int64_t inMaxStride  = inAccess.sampleStride() * inAccess.numSamples();
    const int64_t outMaxStride = outAccess.sampleStride() * outAccess.numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }
}

// Regression: OpFlip_Negative.tensor_output_shape_must_match_input
inline void ValidateSameShape(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                              const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if (inAccess.numSamples() != outAccess.numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }
    if (inAccess.numChannels() != outAccess.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (inAccess.numRows() != outAccess.numRows() || inAccess.numCols() != outAccess.numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }
}

} // namespace

namespace cvcuda::priv {

Flip::Flip(int32_t) {}

void Flip::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t flipCode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Flip::operator()[Tensor]");

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

    if (inData->dtype() != outData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }

    ValidateTensorLayout(inData->layout(), outData->layout());

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    NVCV_ASSERT(outAccess);

    ValidateStrideRange(*inAccess, *outAccess);
    ValidateSameShape(*inAccess, *outAccess);

    const int numSamples = static_cast<int>(outAccess->numSamples());
    const int channels   = inAccess->numChannels();

    // Every (sample, channel) plane of a planar tensor is a single-channel image, and flip never
    // mixes channels, so the shared helper flattens N*C into the sample axis (validating the
    // channel count and the CUDA grid-z limit) and the interleaved single-channel path runs
    // unchanged on the result. Bit-exact with flipping the equivalent NHWC data.
    const auto planarViews = PlanarSingleChannelViews(*inData, *outData);
    const bool isPlanar    = planarViews.has_value();

    // Accesses onto the flattened views, built once here rather than inside the dispatch callback,
    // which is instantiated once per direction and per pixel type.
    nvcv::Optional<nvcv::TensorDataAccessStridedImagePlanar> inViewAccess, outViewAccess;
    if (isPlanar)
    {
        inViewAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(planarViews->first);
        NVCV_ASSERT(inViewAccess);
        outViewAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(planarViews->second);
        NVCV_ASSERT(outViewAccess);
    }

    const int  flatSamples = numSamples * channels;
    const int2 size{static_cast<int>(outAccess->numCols()), static_cast<int>(outAccess->numRows())};

    // The interleaved kernel puts numSamples in grid.z (the planar path is validated inside
    // PlanarSingleChannelViews above). Guard here so a large batch fails fast with
    // INVALID_ARGUMENT rather than reaching a kernel launch that fails with a CUDA error.
    if (!isPlanar && numSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Flip requires numSamples <= 65535 (CUDA grid-z limit)");
    }

    WithTensorDirection(
        size, flipCode,
        [&](auto geometry)
        {
            RunChannelSwitch</*AllowS16*/ false>(
                channels, isPlanar, inData->dtype(),
                [&](auto pixelVal, auto planar)
                {
                    using T = decltype(pixelVal);

                    if constexpr (decltype(planar)::value)
                    {
                        RunFlipPlanarTensor<T>(stream, planarViews->first, planarViews->second, geometry, flatSamples,
                                               channels, *inViewAccess, *outViewAccess);
                    }
                    else
                    {
                        RunFlipTensor<T>(stream, *inData, *outData, geometry, numSamples, *inAccess, *outAccess);
                    }
                });
        });
}

void Flip::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                      const nvcv::Tensor &flipCode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Flip::operator()[ImageBatchVarShape]");

    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto flipCodeData = flipCode.exportData<nvcv::TensorDataStridedCuda>();
    if (flipCodeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Flip Code must be cuda-accessible, pitch-linear tensor");
    }
    if (flipCodeData->dtype() != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Flip Code must be an int32 (S32) tensor");
    }

    const nvcv::ImageFormat inFormat  = inData->uniqueFormat();
    const nvcv::ImageFormat outFormat = outData->uniqueFormat();
    if (!inFormat || !outFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images in a batch must have the same format");
    }
    if (inFormat != outFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }

    const int numPlanes = inFormat.numPlanes();
    const int channels  = inFormat.numChannels();
    if (numPlanes > 1 && numPlanes != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
    }

    const nvcv::DataType planeDataType = inFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (inFormat.planeDataType(i) != planeDataType)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    const bool isPlanar = numPlanes > 1;

    const int32_t numImages = outData->numImages();

    // The var-shape kernels launch one grid.z block per image (planes are looped inside), so the
    // image count has to fit CUDA's grid-z limit.
    if (numImages > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Flip requires numImages <= 65535 (CUDA grid-z limit)");
    }

    // Regression: OpFlip_Negative.varshape_output_size_must_match_input
    if (inData->numImages() != numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images");
    }
    for (int32_t i = 0; i < numImages; ++i)
    {
        if (in[i].size() != out[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output must have matching width and height");
        }
    }

    // Validate element count before wrapping flipCodeData as Tensor1DWrap<int>: a shorter tensor
    // would read out of bounds on the device without an explicit check here.
    if (flipCodeData->shape()[0] < numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Flip Code tensor has %d element(s) but numImages is %d", (int)flipCodeData->shape()[0],
                              numImages);
    }

    const VarShapeGeometry geometry{cuda::Tensor1DWrap<int>(*flipCodeData)};

    RunChannelSwitch</*AllowS16*/ true>(channels, isPlanar, planeDataType,
                                        [&](auto pixelVal, auto planar)
                                        {
                                            using T = decltype(pixelVal);
                                            RunFlipVarShape<decltype(planar)::value, T>(stream, *inData, *outData,
                                                                                        geometry, numPlanes);
                                        });
}

} // namespace cvcuda::priv
