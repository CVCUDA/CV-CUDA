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
/**
 * @file ResizeTapCommon.cuh
 *
 * @brief The dtype-generic tap pipeline shared by the LINEAR, CUBIC and AREA resize kernels.
 *
 * Every one of those kernels is built from the same four pieces: an output pack whose width is
 * derived from the element type, a staged register window of source taps, a weighted reduction of
 * that window, and one vector store per destination row. The pieces do not depend on the element
 * type -- only a handful of *decisions* about how to evaluate the reduction do, and those are the
 * named policies below rather than `sizeof(BaseType<T>)` tests spelled out at each use site. That
 * keeps one implementation of each kernel serving both the byte and the float paths, with the
 * dtype-specific part reduced to a documented predicate.
 *
 * The reference semantics for LINEAR and CUBIC are the general kernels in OpResize.cu
 * (LinearResize / CubicResize); every specialization here is required to reproduce them.
 */
#ifndef CVCUDA_PRIV_RESIZE_TAP_COMMON_CUH
#define CVCUDA_PRIV_RESIZE_TAP_COMMON_CUH

#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <type_traits>

namespace cvcuda::priv::resize {

namespace cuda = nvcv::cuda;

// Output pack geometry --------------------------------------------------------

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

// Write a pack of N elements of type T as a different pack type DPT. Callers align v to DPT<T>.
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

// Common store epilogue: emit the thread's NIX outputs as one vector store when the whole pack is
// in range and the row is aligned, otherwise as guarded per-element stores. Both conditions are
// checked per thread, including a possible partial pack in the final block. Callers that own a
// full in-range pack pass validCount == NIX<T>, which folds the guard away.
template<typename T>
__device__ __forceinline__ void StoreOutputPack(T *dstRow, int dstX0, const T (&pack)[NIX<T>], int validCount)
{
    if (validCount == NIX<T> && CheckRowAlign(dstRow))
    {
        WritePack(dstRow[dstX0], pack);
    }
    else
    {
        T *dstPtr = dstRow + dstX0;
#pragma unroll
        for (int i = 0; i < NIX<T>; ++i)
            if (i < validCount)
                dstPtr[i] = pack[i];
    }
}

// Element-class policies ------------------------------------------------------
//
// The tap pipeline is one implementation per interpolation; these predicates are the only places
// the element type is allowed to change what it does, and each one carries the reason.

// 1-byte base types (uchar1/uchar3/uchar4): narrow elements with an exact conversion to float taps.
template<typename T>
constexpr bool kIsByteElem = sizeof(cuda::BaseType<T>) == 1;

// Scalar float. The flattened planar view of an NCHW float tensor is exactly this type, so a
// specialization gated here also covers every float3/float4 planar plane while leaving the
// interleaved float3/float4 kernels -- against which planar parity is asserted -- untouched.
template<typename T>
constexpr bool kIsScalarFloat = sizeof(cuda::BaseType<T>) == 4 && cuda::NumElements<T> == 1;

// Element types whose staged window is small enough to also block a second output row per thread:
// the union window of a vertical output pair is 5 rows deep, which is 20 registers for a scalar
// element but 80 for a float4-typed one, and the vector instantiations already sit at 94-113.
template<typename T>
constexpr bool kBlocksRowPairs = cuda::NumElements<T> == 1 && (kIsByteElem<T> || kIsScalarFloat<T>);

// Tap staging -----------------------------------------------------------------

// Stage the 2*NIX consecutive source elements of an exact-2x downscale box from both source rows.
// 2*NIX elements are exactly two DPT packs, so an aligned in-range thread reads each row with two
// vector loads instead of 2*NIX scalar ones; the tail and misaligned cases fall back to guarded
// per-element reads. The source width is exactly 2*dstSizeX, which is what bounds the guard.
//
// Shared by the exact-2x LINEAR and AREA downscales: both consume the same 2x2 box per output.
// Callers align win0 and win1 to DPT<T> for the vector loads.
template<typename T>
__device__ __forceinline__ void StageContract2xRows(const T *row0, const T *row1, int sx0, int validCount, int dstSizeX,
                                                    T (&win0)[2 * NIX<T>], T (&win1)[2 * NIX<T>])
{
    if (validCount == NIX<T> && CheckRowAlign(const_cast<T *>(row0) + sx0)
        && CheckRowAlign(const_cast<T *>(row1) + sx0))
    {
#pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            reinterpret_cast<DPT<T> *>(win0)[p] = reinterpret_cast<const DPT<T> *>(row0 + sx0)[p];
            reinterpret_cast<DPT<T> *>(win1)[p] = reinterpret_cast<const DPT<T> *>(row1 + sx0)[p];
        }
    }
    else
    {
#pragma unroll
        for (int c = 0; c < 2 * NIX<T>; ++c)
        {
            const int sx = sx0 + c;
            win0[c]      = (sx < 2 * dstSizeX) ? row0[sx] : T{};
            win1[c]      = (sx < 2 * dstSizeX) ? row1[sx] : T{};
        }
    }
}

// LINEAR reductions -----------------------------------------------------------

// LinearBlend with the four tap coefficients supplied. Each q is the value of the identical
// sub-expression -- a single rounded FADD feeding a single rounded FMUL -- and the accumulation
// order is unchanged, so the result is bit-identical to LinearBlend(p00, p10, p01, p11, w). Used
// where the coefficients are loop-invariant across several taps (the planar path shares them across
// channel planes) and would otherwise be recomputed per tap.
template<typename T>
__device__ __forceinline__ T LinearBlendW(T p00, T p10, T p01, T p11, float q00, float q10, float q01, float q11)
{
    return cuda::SaturateCast<T>(p00 * q00 + p10 * q10 + p01 * q01 + p11 * q11);
}

// CUBIC reductions ------------------------------------------------------------

inline __device__ void GetCubicCoeffs(float delta, float &w0, float &w1, float &w2, float &w3)
{
    constexpr float A = -0.75f;

    w0 = ((A * (delta + 1) - 5 * A) * (delta + 1) + 8 * A) * (delta + 1) - 4 * A;
    w1 = ((A + 2) * delta - (A + 3)) * delta * delta + 1;
    w2 = ((A + 2) * (1 - delta) - (A + 3)) * (1 - delta) * (1 - delta) + 1;
    w3 = 1.f - w0 - w1 - w2;
}

} // namespace cvcuda::priv::resize

#endif // CVCUDA_PRIV_RESIZE_TAP_COMMON_CUH
