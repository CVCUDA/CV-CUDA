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
 * @defgroup NVCV_CPP_CUDATOOLS_SATURATECAST Saturate cast
 * @{
 */

/**
 * Metafunction to saturate cast all elements to a target type.
 *
 * This function saturate casts (clamping with potential rounding) all elements to the range defined by
 * the template argument type \p T.  For instance, a float4 with any values (can be below 0 and above 255) can be
 * casted to an uchar4 rounding-then-saturating each value to be in between 0 and 255 (see example below).  It is a
 * requirement of SaturateCast that both types have the same number of components or \p T is a regular C type.
 *
 * @code
 * using DataType = MakeType<uchar, 4>;
 * using FloatDataType = ConvertBaseTypeTo<float, DataType>;
 * FloatDataType res = ...; // res component values are in [0, 1]
 * DataType pix = SaturateCast<DataType>(res); // pix are in [0, 255]
 * @endcode
 *
 * @tparam T Type that defines the target range to cast.
 * @tparam U Type of the source value (with 1 to 4 elements) passed as argument.
 *
 * @param[in] u Source value to cast all elements to range of base type of \p T
 *
 * @return The value with all elements clamped and potentially rounded.
 */
template<typename T, typename U,
         class = Require<(NumComponents<T> == NumComponents<U>) || (NumComponents<T> == 0 && HasTypeTraits<U>)>>
__host__ __device__ auto SaturateCast(U u)
{
    using BU = BaseType<U>;
    using BT = BaseType<T>;
    using RT = ConvertBaseTypeTo<BT, U>;
    if constexpr (std::is_same_v<U, RT>)
    {
        return u;
    }
    else
    {
        RT out{};

        GetElement<0>(out) = detail::SaturateCastImpl<BT, BU>(GetElement<0>(u));
        if constexpr (nvcv::cuda::NumElements<RT> >= 2)
            GetElement<1>(out) = detail::SaturateCastImpl<BT, BU>(GetElement<1>(u));
        if constexpr (nvcv::cuda::NumElements<RT> >= 3)
            GetElement<2>(out) = detail::SaturateCastImpl<BT, BU>(GetElement<2>(u));
        if constexpr (nvcv::cuda::NumElements<RT> == 4)
            GetElement<3>(out) = detail::SaturateCastImpl<BT, BU>(GetElement<3>(u));

        return out;
    }
}

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_SATURATE_CAST_HPP
