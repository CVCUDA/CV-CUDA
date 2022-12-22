/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file SaturateCast.hpp
 *
 * @brief Defines saturate cast functionality.
 */

#ifndef NVCV_CUDA_SATURATE_CAST_HPP
#define NVCV_CUDA_SATURATE_CAST_HPP

#include "TypeTraits.hpp"              // for Require, etc.
#include "detail/SaturateCastImpl.hpp" // for SaturateCastImpl, etc.

namespace nvcv::cuda {

/**
 * @brief Metafunction to saturate cast all elements to a target type
 *
 * @details This function saturate casts (clamping with potential rounding) all elements to the range defined by
 * the template argument type \p T.  For instance, a float4 with any values (can be below 0 and above 255) can be
 * casted to an uchar4 rounding-then-saturating each value to be in between 0 and 255 (see example below).  It is a
 * requirement of SaturateCast that both types have type traits and type \p T must be a regular C type.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_SATURATECAST Saturate cast
 * @{
 *
 * @code
 * using DataType = MakeType<uchar, 4>;
 * using FloatDataType = ConvertBaseTypeTo<float, DataType>;
 * FloatDataType res = ...; // res component values are in [0, 1]
 * DataType pix = SaturateCast<BaseType<DataType>>(res); // pix are in [0, 255]
 * @endcode
 *
 * @tparam T Type that defines the target range to cast
 * @tparam U Type of the source value (with 1 to 4 elements) passed as argument
 *
 * @param[in] u Source value to cast all elements to range of type \p T
 *
 * @return The value with all elements clamped and potentially rounded
 */
template<typename T, typename U, class = Require<HasTypeTraits<T, U> && !IsCompound<T>>>
__host__ __device__ auto SaturateCast(U u)
{
    using RT = ConvertBaseTypeTo<T, U>;
    if constexpr (std::is_same_v<U, RT>)
    {
        return u;
    }
    else
    {
        RT out{};

#pragma unroll
        for (int e = 0; e < NumElements<RT>; ++e)
        {
            GetElement(out, e) = detail::SaturateCastImpl<T, BaseType<U>>(GetElement(u, e));
        }

        return out;
    }
}

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_SATURATE_CAST_HPP
