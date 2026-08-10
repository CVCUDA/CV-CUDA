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

#include "BenchFillKernels.hpp"

#include <cstdint>
#include <type_traits>

namespace benchutils {

namespace {

// Numerical Recipes LCG — 32-bit; not cryptographic, fine for varied test data.
inline __device__ std::uint32_t lcg_step(std::uint32_t s)
{
    return s * 1664525u + 1013904223u;
}

inline __device__ std::uint32_t hash32(std::uint64_t seed, std::uint64_t idx)
{
    std::uint32_t s = static_cast<std::uint32_t>(seed ^ idx) ^ static_cast<std::uint32_t>(seed >> 32);
    s               = lcg_step(s);
    s               = lcg_step(s);
    s               = lcg_step(s);
    return s;
}

template<typename T>
inline __device__ T sample_typed(std::uint32_t s)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        // Default float range in host LcgValues is uniform_real_distribution(-1, +1).
        constexpr T kInv = static_cast<T>(2.0) / static_cast<T>(4294967295.0);
        return static_cast<T>(-1) + static_cast<T>(s) * kInv;
    }
    else
    {
        // For integral T, truncating the 32-bit word produces uniform values
        // across the full T range (matches uniform_int_distribution<T>::min..max).
        return static_cast<T>(s);
    }
}

template<typename T>
__global__ void randomFillTypedKernel(T *data, size_t n_elements, std::uint64_t seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n_elements)
    {
        return;
    }
    data[idx] = sample_typed<T>(hash32(seed, idx));
}

// Tensor-rank checkerboard: parity from (coord0 + coord1 + coord2 + coord3),
// matching FillBuffer's per-element iteration over a 4-D shape. For ranks < 4
// the caller passes upper strides of 1 so the unused coords collapse to 0.
template<typename T>
__global__ void checkerboardTensorTypedKernel(T *data, size_t n_elements, size_t s0, size_t s1, size_t s2, T hi, T lo)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n_elements)
    {
        return;
    }
    size_t c0 = idx / s0;
    size_t r0 = idx - c0 * s0;
    size_t c1 = r0 / s1;
    size_t r1 = r0 - c1 * s1;
    size_t c2 = r1 / s2;
    size_t c3 = r1 - c2 * s2;
    data[idx] = ((c0 + c1 + c2 + c3) & 1ull) ? hi : lo;
}

// 2-D image checkerboard: parity from (h + w) only — every BT-component within
// a pixel gets the same scalar value, matching FillImageBatch's host pattern
// (one VT-sized write per pixel with all components = val).
template<typename T>
__global__ void checkerboardImageTypedKernel(T *data, size_t n_elements, size_t row_pitch_T, size_t pixel_size_T, T hi,
                                             T lo)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n_elements)
    {
        return;
    }
    size_t h      = idx / row_pitch_T;
    size_t in_row = idx - h * row_pitch_T;
    size_t w      = in_row / pixel_size_T;
    data[idx]     = ((h + w) & 1ull) ? hi : lo;
}

constexpr int kThreads = 256;

inline size_t numBlocks(size_t n_elements)
{
    return (n_elements + kThreads - 1) / kThreads;
}

} // namespace

template<typename T>
void launchRandomFillTyped(T *dptr, size_t n_elements, std::uint64_t seed, cudaStream_t stream)
{
    if (n_elements == 0 || dptr == nullptr)
    {
        return;
    }
    randomFillTypedKernel<T><<<numBlocks(n_elements), kThreads, 0, stream>>>(dptr, n_elements, seed);
}

template<typename T>
void launchCheckerboardTensorTyped(T *dptr, size_t n_elements, size_t s0, size_t s1, size_t s2, T hi, T lo,
                                   cudaStream_t stream)
{
    if (n_elements == 0 || dptr == nullptr)
    {
        return;
    }
    if (s0 == 0)
    {
        s0 = n_elements;
    }
    if (s1 == 0)
    {
        s1 = 1;
    }
    if (s2 == 0)
    {
        s2 = 1;
    }
    checkerboardTensorTypedKernel<T>
        <<<numBlocks(n_elements), kThreads, 0, stream>>>(dptr, n_elements, s0, s1, s2, hi, lo);
}

template<typename T>
void launchCheckerboardImageTyped(T *dptr, size_t n_elements, size_t row_pitch_T, size_t pixel_size_T, T hi, T lo,
                                  cudaStream_t stream)
{
    if (n_elements == 0 || dptr == nullptr || row_pitch_T == 0 || pixel_size_T == 0)
    {
        return;
    }
    checkerboardImageTypedKernel<T>
        <<<numBlocks(n_elements), kThreads, 0, stream>>>(dptr, n_elements, row_pitch_T, pixel_size_T, hi, lo);
}

// Explicit instantiations for the element types the bench uses.
#define CVCUDA_BENCH_INSTANTIATE_FILL(T)                                                                     \
    template void launchRandomFillTyped<T>(T *, size_t, std::uint64_t, cudaStream_t);                        \
    template void launchCheckerboardTensorTyped<T>(T *, size_t, size_t, size_t, size_t, T, T, cudaStream_t); \
    template void launchCheckerboardImageTyped<T>(T *, size_t, size_t, size_t, T, T, cudaStream_t);

CVCUDA_BENCH_INSTANTIATE_FILL(std::uint8_t)
CVCUDA_BENCH_INSTANTIATE_FILL(std::uint16_t)
CVCUDA_BENCH_INSTANTIATE_FILL(std::uint32_t)
CVCUDA_BENCH_INSTANTIATE_FILL(std::int8_t)
CVCUDA_BENCH_INSTANTIATE_FILL(std::int16_t)
CVCUDA_BENCH_INSTANTIATE_FILL(std::int32_t)
CVCUDA_BENCH_INSTANTIATE_FILL(float)
CVCUDA_BENCH_INSTANTIATE_FILL(double)

#undef CVCUDA_BENCH_INSTANTIATE_FILL

} // namespace benchutils
