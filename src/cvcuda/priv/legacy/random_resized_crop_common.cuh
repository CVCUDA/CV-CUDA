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

#ifndef CVCUDA_LEGACY_RANDOM_RESIZED_CROP_COMMON_CUH
#define CVCUDA_LEGACY_RANDOM_RESIZED_CROP_COMMON_CUH

#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <cstddef>
#include <type_traits>

namespace nvcv::legacy::cuda_op {

// Three-channel pixels need uint3-sized stores; other packed 8-bit pixels use
// uint4 to keep the vectorized write width naturally aligned.
template<typename T>
using RRC_DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;

// Number of logical pixels carried by one vectorized destination store.
template<typename T>
constexpr int RRC_NIX = sizeof(RRC_DPT<T>) / sizeof(T);

// uint3 has 4-byte alignment despite its 12-byte size; uint4 uses its full width.
template<typename T>
constexpr unsigned int RRC_MSK = (sizeof(RRC_DPT<T>) == sizeof(uint3) ? sizeof(unsigned int) : sizeof(RRC_DPT<T>)) - 1;

// The NIX kernels only pack 8-bit base types; wider types stay on scalar paths.
template<typename T>
constexpr bool RRC_USE_NIX = sizeof(cuda::BaseType<T>) == 1;

template<typename T>
__device__ __forceinline__ void RRCWritePack(T &u, const T (&v)[RRC_NIX<T>])
{
    reinterpret_cast<RRC_DPT<T> &>(u) = reinterpret_cast<const RRC_DPT<T> &>(v);
}

template<typename T>
__device__ __forceinline__ bool RRCCheckRowAlign(T *row)
{
    return (static_cast<unsigned int>(reinterpret_cast<size_t>(row)) & RRC_MSK<T>) == 0;
}

} // namespace nvcv::legacy::cuda_op

#endif // CVCUDA_LEGACY_RANDOM_RESIZED_CROP_COMMON_CUH
