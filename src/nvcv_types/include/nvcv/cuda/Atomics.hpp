/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file Atomics.cuh
 *
 * @brief Defines atomic operations that might not be readily available in CUDA.
 */

#ifndef NVCV_CUDA_ATOMICS_HPP
#define NVCV_CUDA_ATOMICS_HPP

#include "MathWrappers.hpp"
#include "TypeTraits.hpp"

namespace nvcv::cuda {

/**
 * Metafunction to do a generic atomic operation in floating-point types.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_ATOMICS Atomic operations
 * @{
 *
 * @tparam T Type of the values used in the atomic operation.
 * @tparam OP Operation class that defines the operator call to be used as atomics.
 *
 * @param[in, out] address First value to be used in the atomic operation.
 * @param[in] val Second value to be used.
 * @param[in] op Operation to be used.
 */
template<typename T, class OP, class = Require<std::is_floating_point_v<T>>>
__device__ void AtomicOp(T *address, T val, OP op)
{
    using UT = typename std::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long int>;

    UT *intAddress = reinterpret_cast<UT *>(address);
    UT  assumed, old = *intAddress;

    do
    {
        assumed = old;

        auto newVal = op(val, reinterpret_cast<const T &>(assumed));
        old         = atomicCAS(intAddress, assumed, reinterpret_cast<UT &>(newVal));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);
}

/**
 * Metafunction to do a atomic minimum operation that accepts floating-point types.
 *
 * @tparam T Type of the values used in the atomic operation.
 *
 * @param[in, out] a First value to be used in the atomic operation.
 * @param[in] b Second value to be used.
 */
template<typename T>
inline __device__ void AtomicMin(T &a, T b)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        AtomicOp(&a, b, [](T a_, T b_) { return cuda::min(a_, b_); });
    }
    else
    {
        atomicMin(&a, b);
    }
}

/**
 * Metafunction to do a atomic maximum operation that accepts floating-point types.
 *
 * @tparam T Type of the values used in the atomic operation.
 *
 * @param[in, out] a First value to be used in the atomic operation.
 * @param[in] b Second value to be used.
 */
template<typename T>
inline __device__ void AtomicMax(T &a, T b)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        AtomicOp(&a, b, [](T a_, T b_) { return cuda::max(a_, b_); });
    }
    else
    {
        atomicMax(&a, b);
    }
}

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_ATOMICS_HPP
