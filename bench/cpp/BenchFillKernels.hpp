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

#ifndef CVCUDA_BENCH_FILL_KERNELS_HPP
#define CVCUDA_BENCH_FILL_KERNELS_HPP

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace benchutils {

/**
 * Fill a typed device buffer with an LCG-based pseudo-random stream.
 *
 * Replaces the host-side std::uniform_int_distribution + cudaMemcpy path,
 * which on this codebase can spend 1-6 seconds on a single 0.5-1.6 GB fill.
 * The LCG output is uniform-enough for benchmark workloads where we only
 * want varied data so the kernel doesn't take a constant-input fast path —
 * we are NOT relying on statistical randomness.
 *
 * Distribution by element type:
 *   - integer T:  uniform across the full type range (matches the default
 *                 host LcgValues<T>() with no args).
 *   - float T:    uniform in [-1, +1] (matches the default for floating
 *                 point in host LcgValues<T>() with no args).
 *
 * Explicit instantiations are provided in BenchFillKernels.cu for the
 * element types the bench uses: uint8_t, uint16_t, uint32_t, int8_t,
 * int16_t, int32_t, float, double.
 */
template<typename T>
void launchRandomFillTyped(T *dptr, size_t n_elements, std::uint64_t seed, cudaStream_t stream);

/**
 * Fill an N-D tensor with a per-element checkerboard, parity from full coords.
 *
 * Mirrors the host FillTensor / FillBuffer behaviour: iteration is per BT
 * element, parity sums all four coords (n + h + w + c) so adjacent components
 * within a pixel alternate. Strides are passed in T-elements (not bytes); the
 * inner-most stride is 1 (implicit).
 *
 *   coord0 = idx / s0
 *   r0     = idx - coord0 * s0
 *   coord1 = r0  / s1
 *   r1     = r0  - coord1 * s1
 *   coord2 = r1  / s2
 *   coord3 = r1  - coord2 * s2
 *   data[idx] = ((coord0 + coord1 + coord2 + coord3) & 1) ? hi : lo
 *
 * For ranks < 4 the upper strides should be 1 so the unused coords collapse
 * to 0 (matching FillBuffer's degenerate-loop behaviour).
 */
template<typename T>
void launchCheckerboardTensorTyped(T *dptr, size_t n_elements, size_t s0, size_t s1, size_t s2, T hi, T lo,
                                   cudaStream_t stream);

/**
 * Fill a 2-D image buffer with a per-PIXEL uniform checkerboard.
 *
 * Mirrors the host FillImageBatch behaviour: iteration is per pixel, each
 * pixel's VT-sized cell gets all components set to the same scalar value
 * `((h + w) & 1) ? hi : lo`. Required to match the host pattern for
 * multi-channel formats (uchar3 / uchar4 / float3 / float4) where the
 * tensor-rank kernel would alternate within a pixel and produce a
 * different data layout.
 *
 * @param dptr           Device buffer (typed pointer).
 * @param n_elements     Total T elements in the image (= H * row_pitch_T).
 * @param row_pitch_T    Row stride in T-elements (= bytes_per_row / sizeof(T)).
 * @param pixel_size_T   Pixel size in T-elements (= NumElements<VT>).
 */
template<typename T>
void launchCheckerboardImageTyped(T *dptr, size_t n_elements, size_t row_pitch_T, size_t pixel_size_T, T hi, T lo,
                                  cudaStream_t stream);

} // namespace benchutils

#endif // CVCUDA_BENCH_FILL_KERNELS_HPP
