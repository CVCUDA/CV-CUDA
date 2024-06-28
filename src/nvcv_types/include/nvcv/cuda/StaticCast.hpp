/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file StaticCast.hpp
 *
 * @brief Defines static cast functionality.
 */

#ifndef NVCV_CUDA_STATIC_CAST_HPP
#define NVCV_CUDA_STATIC_CAST_HPP

#include "TypeTraits.hpp" // for Require, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_STATICCAST Static Cast
 * @{
 */

/**
 * Metafunction to static cast all values of a compound to a target type.
 *
 * The template parameter \p T defines the base type (regular C type) to cast all components of the CUDA
 * compound type \p U passed as function argument \p u to the type \p T.  The static cast return type has the base
 * type \p T and the number of components as the compound type \p U.  For instance, an uint3 can be casted to int3
 * by passing it as function argument of StaticCast and the type int as template argument (see example below).  The
 * type \p U is not needed as it is inferred from the argument \u.  It is a requirement of the StaticCast function
 * that the type \p T is of regular C type and the type \p U is of CUDA compound type.
 *
 * @code
 * int3 idx = StaticCast<int>(blockIdx * blockDim + threadIdx);
 * @endcode
 *
 * @tparam T Type to do static cast on each component of \p u.
 *
 * @param[in] u Compound value to static cast each of its components to target type \p T.
 *
 * @return The compound value with all components static casted to type \p T.
 */
template<typename T, typename U, class = Require<HasTypeTraits<T, U> && !IsCompound<T>>>
__host__ __device__ auto StaticCast(U u)
{
    using RT = ConvertBaseTypeTo<T, U>;
    if constexpr (std::is_same_v<U, RT>)
    {
        return u;
    }
    else
    {
        RT out{};

        GetElement<0>(out) = static_cast<T>(GetElement<0>(u));
        if constexpr (nvcv::cuda::NumElements<RT> >= 2)
            GetElement<1>(out) = static_cast<T>(GetElement<1>(u));
        if constexpr (nvcv::cuda::NumElements<RT> >= 3)
            GetElement<2>(out) = static_cast<T>(GetElement<2>(u));
        if constexpr (nvcv::cuda::NumElements<RT> == 4)
            GetElement<3>(out) = static_cast<T>(GetElement<3>(u));

        return out;
    }
}

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_STATIC_CAST_HPP
