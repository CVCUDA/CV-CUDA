/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file DropCast.hpp
 *
 * @brief Defines drop cast functionality.
 */

#ifndef NVCV_CUDA_DROP_CAST_HPP
#define NVCV_CUDA_DROP_CAST_HPP

#include "TypeTraits.hpp" // for Require, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_DROPCAST Drop Cast
 * @{
 */

/**
 * Metafunction to drop components of a compound value.
 *
 * The template parameter \p N defines the number of components to cast the CUDA compound type \p T passed
 * as function argument \p v.  This is done by dropping the last components after \p N from \p v.  For instance, an
 * uint3 can have its z component dropped by passing it as function argument to DropCast and the number 2 as
 * template argument (see example below).  The type \p T is not needed as it is inferred from the argument \p v.
 * It is a requirement of the DropCast function that the type \p T has at least N components.
 *
 * @code
 * uint2 dstIdx = DropCast<2>(blockIdx * blockDim + threadIdx);
 * @endcode
 *
 * @tparam N Number of components to return.
 *
 * @param[in] v Value to drop components from.
 *
 * @return The compound value with N components dropping the last, extra components.
 */
template<int N, typename T, class = Require<HasEnoughComponents<T, N>>>
__host__ __device__ auto DropCast(T v)
{
    using RT = MakeType<BaseType<T>, N>;
    if constexpr (std::is_same_v<T, RT>)
    {
        return v;
    }
    else
    {
        RT out{};

        GetElement<0>(out) = GetElement<0>(v);
        if constexpr (nvcv::cuda::NumElements<RT> >= 2)
            GetElement<1>(out) = GetElement<1>(v);
        if constexpr (nvcv::cuda::NumElements<RT> >= 3)
            GetElement<2>(out) = GetElement<2>(v);
        if constexpr (nvcv::cuda::NumElements<RT> == 4)
            GetElement<3>(out) = GetElement<3>(v);

        return out;
    }
}

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_DROP_CAST_HPP
