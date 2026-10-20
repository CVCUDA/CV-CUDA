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

#include "AdjustSharpnessPolicy.hpp"
#include "OpAdjustSharpness.hpp"

#include "PhotometricBound.cuh"
#include "SameShapeCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <limits>
#include <type_traits>

namespace cuda        = nvcv::cuda;
namespace util        = nvcv::util;
namespace same_shape  = cvcuda::priv::same_shape;
namespace elem_stream = cvcuda::priv::elem_stream;

namespace {

// One-instruction clamp-and-narrow, exact against fminf(fmaxf(v, 0), bound) over all 2^32 float patterns.
template<typename BT>
inline __device__ BT AdjustSharpnessStore(float v, float bound)
{
    if constexpr (std::is_same_v<BT, float>)
    {
        (void)bound;
        return __saturatef(v);
    }
    else if constexpr (std::is_same_v<BT, unsigned char>)
    {
        (void)bound;
        unsigned int out;
        asm("cvt.rzi.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(v));
        return static_cast<BT>(out);
    }
    else if constexpr (std::is_same_v<BT, unsigned short>)
    {
        (void)bound;
        unsigned int out;
        asm("cvt.rzi.sat.u16.f32 %0, %1;" : "=r"(out) : "f"(v));
        return static_cast<BT>(out);
    }
    else
    {
        return static_cast<BT>(fminf(fmaxf(v, 0.0f), bound));
    }
}

// Per-channel interior compute, shared bit-exactly with the CPU gold reference (see
// TestOpAdjustSharpness.cpp). Every multiply-add is an explicit `fmaf` and the accumulation order is
// fixed, so the correctly-rounded IEEE-754 result is identical on host and device regardless of the
// compiler's FMA-contraction setting. `c11` is the center (original) pixel; the eight edge taps
// carry weight 1/13 and the center 5/13. Integer types round the smoothed value to nearest (ties to
// even) before blending, then the clamped blend is truncated toward zero on the final cast — both
// matching torchvision's `blurred.round_()` + `.to(dtype)`.
template<typename BT>
inline __device__ BT AdjustSharpnessScalar(float c00, float c01, float c02, float c10, float c11, float c12, float c20,
                                           float c21, float c22, float oneMinusFactor, float bound)
{
    constexpr float kEdge   = 1.0f / 13.0f;
    constexpr float kCenter = 5.0f / 13.0f;

    float blur = c11 * kCenter;
    blur       = fmaf(c00, kEdge, blur);
    blur       = fmaf(c01, kEdge, blur);
    blur       = fmaf(c02, kEdge, blur);
    blur       = fmaf(c10, kEdge, blur);
    blur       = fmaf(c12, kEdge, blur);
    blur       = fmaf(c20, kEdge, blur);
    blur       = fmaf(c21, kEdge, blur);
    blur       = fmaf(c22, kEdge, blur);

    // Round-to-nearest-even matches torch.round on the smoothed image for integer dtypes; __half
    // keeps the unrounded float value like the other float types (std::is_floating_point excludes
    // it), deferring precision loss to the final round-to-nearest-half store.
    if constexpr (!cuda::detail::IsFloatingPointV<BT>)
    {
        blur = rintf(blur);
    }

    // out = in + (1 - factor) * (blur - in) = factor*in + (1-factor)*blur
    const float out = fmaf(oneMinusFactor, blur - c11, c11);
    return AdjustSharpnessStore<BT>(out, bound);
}

// Interior pixels are blended; the 1-pixel border is copied unchanged (there is no border
// extension). When a dimension is < 3 every pixel is a border pixel, so the whole image is copied —
// this reproduces torchvision returning the input for images with height or width <= 2.
template<bool IsPlanar, class SrcWrapper, class DstWrapper>
inline __device__ void DoAdjustSharpness(SrcWrapper src, DstWrapper dst, int2 size, int plane, float oneMinusFactor)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    using BT                         = cuda::BaseType<SrcT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(!IsPlanar || numChannels == 1);

    const int col    = blockIdx.x * blockDim.x + threadIdx.x;
    const int row    = blockIdx.y * blockDim.y + threadIdx.y;
    const int sample = blockIdx.z;
    if (col >= size.x || row >= size.y)
    {
        return;
    }

    if (col == 0 || row == 0 || col == size.x - 1 || row == size.y - 1)
    {
        const auto coord = same_shape::GetCoordForLayout<IsPlanar>(int3{col, row, sample}, plane);
        dst[coord]       = src[coord];
        return;
    }

    const float bound = cvcuda::priv::PhotometricUpperBound<BT, float>();

    const SrcT n00 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col - 1, row - 1, sample}, plane)];
    const SrcT n01 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col, row - 1, sample}, plane)];
    const SrcT n02 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col + 1, row - 1, sample}, plane)];
    const SrcT n10 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col - 1, row, sample}, plane)];
    const SrcT n11 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col, row, sample}, plane)];
    const SrcT n12 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col + 1, row, sample}, plane)];
    const SrcT n20 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col - 1, row + 1, sample}, plane)];
    const SrcT n21 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col, row + 1, sample}, plane)];
    const SrcT n22 = src[same_shape::GetCoordForLayout<IsPlanar>(int3{col + 1, row + 1, sample}, plane)];

    DstT out{};
#pragma unroll
    for (int ch = 0; ch < numChannels; ++ch)
    {
        cuda::GetElement(out, ch) = AdjustSharpnessScalar<BT>(
            static_cast<float>(cuda::GetElement(n00, ch)), static_cast<float>(cuda::GetElement(n01, ch)),
            static_cast<float>(cuda::GetElement(n02, ch)), static_cast<float>(cuda::GetElement(n10, ch)),
            static_cast<float>(cuda::GetElement(n11, ch)), static_cast<float>(cuda::GetElement(n12, ch)),
            static_cast<float>(cuda::GetElement(n20, ch)), static_cast<float>(cuda::GetElement(n21, ch)),
            static_cast<float>(cuda::GetElement(n22, ch)), oneMinusFactor, bound);
    }
    dst[same_shape::GetCoordForLayout<IsPlanar>(int3{col, row, sample}, plane)] = out;
}

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void AdjustSharpness(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, float oneMinusFactor)
{
    same_shape::ForEachPlane<isPlanar>(numPlanes,
                                       [=](int p) { DoAdjustSharpness<isPlanar>(src, dst, size, p, oneMinusFactor); });
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void AdjustSharpness(SrcWrapper src, DstWrapper dst, int numPlanes, float oneMinusFactor)
{
    const int z = blockIdx.z;
    int2      size{dst.width(z), dst.height(z)};

    same_shape::ForEachPlane<isPlanar>(numPlanes,
                                       [=](int p) { DoAdjustSharpness<isPlanar>(src, dst, size, p, oneMinusFactor); });
}

// One thread per output element, reusing its column down `Rows` rows: `Rows + 2` row loads, not `3 * Rows`.
constexpr int kElemRows   = 12;
constexpr int kElemBlockW = 128;
constexpr int kPackedCols = 2;
constexpr int kPackedRows = 2;

struct ElemStream
{
    int32_t srcRowBytes;
    int32_t dstRowBytes;
    int32_t srcSliceBytes;
    int32_t dstSliceBytes;
    int32_t width;  // in pixels
    int32_t height; // in rows
};

struct PackedStream
{
    int32_t rowBytes;
    int32_t sliceBytes;
    int32_t width;  // in pixels
    int32_t height; // in rows
};

// A thread owns a 2xRows output tile. Its (2 + 2)x(Rows + 2) halo reduces redundant global loads
// while keeping several independent pixels live; float2 center loads make adjacent warp lanes
// cover contiguous memory transactions. This balance retains the latency hiding that is lost by a
// rolling three-row window, without the much larger live set of wider tiles.
template<int Rows>
__global__ __launch_bounds__(kElemBlockW) void AdjustSharpnessPacked2(const NVCVByte *__restrict__ srcBase,
                                                                      NVCVByte *__restrict__ dstBase, PackedStream st,
                                                                      float oneMinusFactor)
{
    constexpr int kLoadCols = kPackedCols + 2;
    constexpr int kLoadRows = Rows + 2;

    const int col = static_cast<int>(blockIdx.x * kElemBlockW + threadIdx.x) * kPackedCols;
    if (col >= st.width)
    {
        return;
    }
    const int row0 = static_cast<int>(blockIdx.y) * Rows;

    const NVCVByte *srcSlice = srcBase + static_cast<int32_t>(blockIdx.z) * st.sliceBytes;
    NVCVByte       *dstSlice = dstBase + static_cast<int32_t>(blockIdx.z) * st.sliceBytes;

    const bool interiorCols = col >= 1 && col + kPackedCols < st.width;
    const bool fullRows     = row0 >= 1 && row0 + Rows + 1 <= st.height;

    float v[kLoadRows][kLoadCols];
    if (fullRows && interiorCols)
    {
#pragma unroll
        for (int i = 0; i < kLoadRows; ++i)
        {
            const float *p      = reinterpret_cast<const float *>(srcSlice + (row0 - 1 + i) * st.rowBytes) + col;
            const float2 center = *reinterpret_cast<const float2 *>(p);
            v[i][0]             = p[-1];
            v[i][1]             = center.x;
            v[i][2]             = center.y;
            v[i][3]             = p[kPackedCols];
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kLoadRows; ++i)
        {
            const int    row = min(max(row0 - 1 + i, 0), st.height - 1);
            const float *p   = reinterpret_cast<const float *>(srcSlice + row * st.rowBytes);
#pragma unroll
            for (int j = 0; j < kLoadCols; ++j)
            {
                v[i][j] = p[min(max(col - 1 + j, 0), st.width - 1)];
            }
        }
    }

    constexpr float bound = 1.0f;
#pragma unroll
    for (int i = 0; i < Rows; ++i)
    {
        const int row = row0 + i;
        if (row >= st.height)
        {
            return;
        }

        float2 out;
#pragma unroll
        for (int j = 0; j < kPackedCols; ++j)
        {
            const int c = col + j;
            cuda::GetElement(out, j)
                = row == 0 || row == st.height - 1 || c == 0 || c == st.width - 1
                    ? v[i + 1][j + 1]
                    : AdjustSharpnessScalar<float>(v[i][j], v[i][j + 1], v[i][j + 2], v[i + 1][j], v[i + 1][j + 1],
                                                   v[i + 1][j + 2], v[i + 2][j], v[i + 2][j + 1], v[i + 2][j + 2],
                                                   oneMinusFactor, bound);
        }
        float *d                       = reinterpret_cast<float *>(dstSlice + row * st.rowBytes) + col;
        *reinterpret_cast<float2 *>(d) = out;
    }
}

// Shared body of the two element kernels: they differ only in how the slice base, row stride and
// height are found. `e` and `r0` are already range-checked by the caller.
//
// The slice pointers are deliberately NOT `__restrict__`-qualified. The tensor kernel's srcBase and
// dstBase are, and that qualification still reaches these loads and stores through inlining -- the
// tensor kernels' PTX is unchanged either way. The var-shape kernel's plane base pointers are not
// and never were, so qualifying the shared parameter would hand the var-shape kernels a no-alias
// promise they did not previously make: measured at -arch=sm_80, 12 var-shape kernels promote
// ld.global.* to ld.global.nc.* -- read-only-cache loads, at an unchanged instruction count. That
// is a new assertion about var-shape aliasing, not a restoration of an existing one.
template<typename BT, int Channels, int Rows>
inline __device__ void AdjustSharpnessElemBody(const NVCVByte *srcSlice, int32_t srcRowBytes, NVCVByte *dstSlice,
                                               int32_t dstRowBytes, int32_t height, int rowLen, int e, int r0,
                                               float oneMinusFactor)
{
    constexpr int kNumRows = Rows + 2;

    const bool interior = (e >= Channels) && (e < rowLen - Channels);
    const bool fullTile = (r0 >= 1) && (r0 + Rows + 1 <= height);

    float v[kNumRows][3];
    if (fullTile && interior)
    {
#pragma unroll
        for (int i = 0; i < kNumRows; ++i)
        {
            const BT *p = reinterpret_cast<const BT *>(srcSlice + (r0 - 1 + i) * srcRowBytes) + e;
            v[i][0]     = static_cast<float>(p[-Channels]);
            v[i][1]     = static_cast<float>(p[0]);
            v[i][2]     = static_cast<float>(p[Channels]);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kNumRows; ++i)
        {
            const int row = min(max(r0 - 1 + i, 0), height - 1);
            const BT *p   = reinterpret_cast<const BT *>(srcSlice + row * srcRowBytes) + e;
            v[i][0]       = interior ? static_cast<float>(p[-Channels]) : 0.0f;
            v[i][1]       = static_cast<float>(p[0]);
            v[i][2]       = interior ? static_cast<float>(p[Channels]) : 0.0f;
        }
    }

    const float bound = cvcuda::priv::PhotometricUpperBound<BT, float>();
#pragma unroll
    for (int k = 0; k < Rows; ++k)
    {
        const int row = r0 + k;
        if (row >= height)
        {
            return;
        }
        BT *d = reinterpret_cast<BT *>(dstSlice + row * dstRowBytes) + e;
        if (row == 0 || row == height - 1 || !interior)
        {
            *d = static_cast<BT>(v[k + 1][1]);
        }
        else
        {
            *d = AdjustSharpnessScalar<BT>(v[k][0], v[k][1], v[k][2], v[k + 1][0], v[k + 1][1], v[k + 1][2],
                                           v[k + 2][0], v[k + 2][1], v[k + 2][2], oneMinusFactor, bound);
        }
    }
}

template<typename BT, int Channels, int Rows>
__global__ __launch_bounds__(kElemBlockW) void AdjustSharpnessElem(const NVCVByte *__restrict__ srcBase,
                                                                   NVCVByte *__restrict__ dstBase, ElemStream st,
                                                                   float oneMinusFactor)
{
    const int rowLen = st.width * Channels;
    const int e      = static_cast<int>(blockIdx.x * kElemBlockW + threadIdx.x);
    if (e >= rowLen)
    {
        return;
    }
    const int r0 = static_cast<int>(blockIdx.y) * Rows;

    AdjustSharpnessElemBody<BT, Channels, Rows>(srcBase + static_cast<int32_t>(blockIdx.z) * st.srcSliceBytes,
                                                st.srcRowBytes,
                                                dstBase + static_cast<int32_t>(blockIdx.z) * st.dstSliceBytes,
                                                st.dstRowBytes, st.height, rowLen, e, r0, oneMinusFactor);
}

template<typename BT, int Channels, int Rows>
__global__ __launch_bounds__(kElemBlockW) void AdjustSharpnessElemVarShape(
    cuda::ImageBatchVarShapeWrap<const cuda::MakeType<BT, Channels>> src,
    cuda::ImageBatchVarShapeWrap<cuda::MakeType<BT, Channels>> dst, int numPlanes, float oneMinusFactor)
{
    // `blockIdx.z` is block-uniform, so this division runs once per block, not once per thread.
    const int sample = static_cast<int>(blockIdx.z) / numPlanes;
    const int plane  = static_cast<int>(blockIdx.z) - sample * numPlanes;

    const NVCVImagePlaneStrided sp = src.plane(sample, plane);
    const NVCVImagePlaneStrided dp = dst.plane(sample, plane);

    const int height = sp.height;
    const int rowLen = sp.width * Channels;
    const int e      = static_cast<int>(blockIdx.x * kElemBlockW + threadIdx.x);
    if (e >= rowLen)
    {
        return;
    }
    const int r0 = static_cast<int>(blockIdx.y) * Rows;
    // Present here and deliberately absent from the tensor kernel, which is otherwise the same
    // wrapper: grid.y there is DivUp(that tensor's height, Rows), so `blockIdx.y * Rows` is always
    // below its height. Here grid.y is sized from the batch's *bounding* height, so a shorter
    // image in the batch really does get blocks whose first row is past its end. Do not "restore
    // the symmetry" by adding the guard there; it would be dead code in every launch.
    if (r0 >= height)
    {
        return;
    }

    AdjustSharpnessElemBody<BT, Channels, Rows>(sp.basePtr, sp.rowStride, dp.basePtr, dp.rowStride, height, rowLen, e,
                                                r0, oneMinusFactor);
}

inline bool TryRunPacked2PlanarFloatTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                           const nvcv::TensorDataStridedCuda &dstData, float oneMinusFactor)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);

    constexpr int64_t kAlignment = sizeof(float2);
    auto              isAligned  = [](const void *p)
    {
        return reinterpret_cast<uintptr_t>(p) % kAlignment == 0;
    };
    auto isStrideAligned = [](int64_t stride)
    {
        return stride % kAlignment == 0;
    };

    if (srcAccess->colStride() != sizeof(float) || dstAccess->colStride() != sizeof(float)
        || srcAccess->numCols() % kPackedCols != 0 || !isAligned(srcData.basePtr()) || !isAligned(dstData.basePtr())
        || srcAccess->rowStride() != dstAccess->rowStride() || srcAccess->planeStride() != dstAccess->planeStride()
        || !isStrideAligned(srcAccess->sampleStride()) || !isStrideAligned(srcAccess->planeStride())
        || !isStrideAligned(srcAccess->rowStride()) || !isStrideAligned(dstAccess->sampleStride())
        || !isStrideAligned(dstAccess->planeStride()) || !isStrideAligned(dstAccess->rowStride()))
    {
        return false;
    }

    const int64_t numSamples = srcAccess->numSamples();
    const int64_t numPlanes  = srcAccess->numPlanes();
    if (numSamples > 1
        && (srcAccess->sampleStride() != numPlanes * srcAccess->planeStride()
            || dstAccess->sampleStride() != numPlanes * dstAccess->planeStride()))
    {
        return false;
    }

    const int64_t numSlices = numSamples * numPlanes;
    if (numSlices > elem_stream::kMaxGridYZ
        || !same_shape::FitsRowGridLimit(static_cast<int>(srcAccess->numRows()), kPackedRows))
    {
        return false;
    }

    PackedStream st;
    st.rowBytes   = static_cast<int32_t>(srcAccess->rowStride());
    st.sliceBytes = static_cast<int32_t>(srcAccess->planeStride());
    st.width      = static_cast<int32_t>(srcAccess->numCols());
    st.height     = static_cast<int32_t>(srcAccess->numRows());

    dim3 block(kElemBlockW, 1, 1);
    dim3 grid(static_cast<unsigned>(util::DivUp(st.width, kPackedCols * kElemBlockW)),
              static_cast<unsigned>(util::DivUp(st.height, kPackedRows)), static_cast<unsigned>(numSlices));
    AdjustSharpnessPacked2<kPackedRows>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const NVCVByte *>(srcData.basePtr()),
                                     reinterpret_cast<NVCVByte *>(dstData.basePtr()), st, oneMinusFactor);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<bool isPlanar, typename ValueT>
inline bool TryRunElemTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                             const nvcv::TensorDataStridedCuda &dstData, float oneMinusFactor)
{
    using BT                       = cuda::BaseType<ValueT>;
    static constexpr int kChannels = cuda::NumElements<ValueT>;

    if constexpr (!cvcuda::priv::UseElementPathForShape(isPlanar, kChannels, sizeof(BT)))
    {
        return false;
    }
    static_assert(!isPlanar || kChannels == 1);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);

    constexpr int64_t kElemSize = static_cast<int64_t>(sizeof(BT));

    if (srcAccess->colStride() != kChannels * kElemSize || dstAccess->colStride() != kChannels * kElemSize)
    {
        return false;
    }
    if constexpr (!isPlanar)
    {
        if (srcAccess->chStride() != kElemSize || dstAccess->chStride() != kElemSize)
        {
            return false;
        }
    }

    const int64_t numSamples = srcAccess->numSamples();
    int64_t       numSlices  = numSamples;
    int64_t       srcSlice   = srcAccess->sampleStride();
    int64_t       dstSlice   = dstAccess->sampleStride();
    if constexpr (isPlanar)
    {
        // A single sample never leaves itself, so its stride (0 for rank-3 CHW) need not agree.
        const int64_t numPlanes = srcAccess->numPlanes();
        if (numSamples > 1
            && (srcAccess->sampleStride() != numPlanes * srcAccess->planeStride()
                || dstAccess->sampleStride() != numPlanes * dstAccess->planeStride()))
        {
            return false;
        }
        numSlices = numSamples * numPlanes;
        srcSlice  = srcAccess->planeStride();
        dstSlice  = dstAccess->planeStride();
    }
    if (numSlices > elem_stream::kMaxGridYZ)
    {
        return false;
    }

    ElemStream st;
    st.srcRowBytes   = static_cast<int32_t>(srcAccess->rowStride());
    st.dstRowBytes   = static_cast<int32_t>(dstAccess->rowStride());
    st.srcSliceBytes = static_cast<int32_t>(srcSlice);
    st.dstSliceBytes = static_cast<int32_t>(dstSlice);
    st.width         = static_cast<int32_t>(srcAccess->numCols());
    st.height        = static_cast<int32_t>(srcAccess->numRows());

    // grid.y is 16-bit; this path folds kElemRows rows into one block, so its ceiling is higher
    // than the scalar kernel's but still finite. Decline rather than throw -- the caller's scalar
    // fallback owns the rejection.
    if (!same_shape::FitsRowGridLimit(st.height, kElemRows))
    {
        return false;
    }

    const int64_t rowLen = static_cast<int64_t>(st.width) * kChannels;
    dim3          block(kElemBlockW, 1, 1);
    dim3          grid(static_cast<unsigned>(util::DivUp(rowLen, static_cast<int64_t>(kElemBlockW))),
                       static_cast<unsigned>(util::DivUp(st.height, kElemRows)), static_cast<unsigned>(numSlices));

    AdjustSharpnessElem<BT, kChannels, kElemRows>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const NVCVByte *>(srcData.basePtr()),
                                     reinterpret_cast<NVCVByte *>(dstData.basePtr()), st, oneMinusFactor);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<bool isPlanar, typename ValueT>
inline bool TryRunElemVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, int numPlanes,
                               float oneMinusFactor)
{
    using BT                       = cuda::BaseType<ValueT>;
    static constexpr int kChannels = cuda::NumElements<ValueT>;

    if constexpr (!cvcuda::priv::UseElementPathForShape(isPlanar, kChannels, sizeof(BT)))
    {
        return false;
    }

    const int64_t numSlices = static_cast<int64_t>(dstData.numImages()) * numPlanes;
    if (numSlices > elem_stream::kMaxGridYZ)
    {
        return false;
    }

    if (!same_shape::FitsRowGridLimit(dstData.maxSize().h, kElemRows))
    {
        return false;
    }

    const int64_t rowLen = static_cast<int64_t>(dstData.maxSize().w) * kChannels;
    dim3          block(kElemBlockW, 1, 1);
    dim3          grid(static_cast<unsigned>(util::DivUp(rowLen, static_cast<int64_t>(kElemBlockW))),
                       static_cast<unsigned>(util::DivUp(dstData.maxSize().h, kElemRows)), static_cast<unsigned>(numSlices));

    cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
    cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
    AdjustSharpnessElemVarShape<BT, kChannels, kElemRows>
        <<<grid, block, 0, stream>>>(src, dst, numPlanes, oneMinusFactor);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

// Run AdjustSharpness kernel -------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunAdjustSharpness(cudaStream_t stream, const SrcData &srcData, const DstData &dstData, double factor)
{
    const float oneMinusFactor = static_cast<float>(1.0 - factor);
    dim3        block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto       srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto       dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        // Each sample maps to one grid.z block (planar channels are looped inside the kernel).
        const auto prep = same_shape::MakeTensorLaunchPrep(*srcAccess, *dstAccess);
        const int2 size = prep.size;

        using StrideType = int32_t;

        if constexpr (isPlanar && std::is_same_v<cuda::BaseType<ValueT>, float>)
        {
            if (TryRunPacked2PlanarFloatTensor(stream, srcData, dstData, oneMinusFactor))
            {
                return;
            }
        }

        if (TryRunElemTensor<isPlanar, ValueT>(stream, srcData, dstData, oneMinusFactor))
        {
            return;
        }

        if constexpr (!isPlanar)
        {
            auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
            AdjustSharpness<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, 1, oneMinusFactor);
        }
        else
        {
            const int numPlanes = srcAccess->numPlanes();
            auto [src, dst]
                = same_shape::MakePlanarWrapPair<ValueT, ValueT, StrideType>(srcData, *srcAccess, dstData, *dstAccess);
            AdjustSharpness<isPlanar>
                <<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, numPlanes, oneMinusFactor);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        const auto prep = same_shape::MakeVarShapeLaunchPrep(dstData);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        if (TryRunElemVarShape<isPlanar, ValueT>(stream, srcData, dstData, numPlanes, oneMinusFactor))
        {
            return;
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        AdjustSharpness<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, numPlanes, oneMinusFactor);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

// Validation ------------------------------------------------------------------------------

inline void ValidateSharpnessFactor(double sharpnessFactor)
{
    // Above FLT_MAX, float(1.0 - factor) is out of float range (undefined behaviour).
    if (sharpnessFactor < 0.0 || sharpnessFactor > std::numeric_limits<float>::max())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Sharpness factor must be non-negative and not greater than FLT_MAX");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

AdjustSharpness::AdjustSharpness() {}

// Tensor input variant
void AdjustSharpness::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                 double sharpnessFactor) const
{
    ValidateSharpnessFactor(sharpnessFactor);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto info    = same_shape::ValidateSrcDstTensors(srcData, dstData);

    same_shape::DispatchTypeAndChannels(info.numInterleavedChannels, info.numPlanes, info.srcDtype, "AdjustSharpness",
                                        [&stream, &srcData, &dstData, sharpnessFactor](auto dummyVal, auto isPlanar)
                                        {
                                            using ValueT   = decltype(dummyVal);
                                            using IsPlanar = decltype(isPlanar);
                                            RunAdjustSharpness<IsPlanar::value, ValueT>(stream, *srcData, *dstData,
                                                                                        sharpnessFactor);
                                        });
}

// VarShape input variant
void AdjustSharpness::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                 const nvcv::ImageBatchVarShape &dst, double sharpnessFactor) const
{
    ValidateSharpnessFactor(sharpnessFactor);

    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto srcDstData = same_shape::ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    same_shape::DispatchTypeAndChannels(numInterleavedChannels, numPlanes, dtype, "AdjustSharpness",
                                        [&stream, &srcDstData, sharpnessFactor](auto dummyVal, auto isPlanar)
                                        {
                                            using ValueT             = decltype(dummyVal);
                                            using IsPlanar           = decltype(isPlanar);
                                            auto &[srcData, dstData] = srcDstData;
                                            RunAdjustSharpness<IsPlanar::value, ValueT>(stream, *srcData, *dstData,
                                                                                        sharpnessFactor);
                                        });
}

} // namespace cvcuda::priv
