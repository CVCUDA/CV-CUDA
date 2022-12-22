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

#ifndef NVCV_CUDA_RANGE_CAST_HPP
#define NVCV_CUDA_RANGE_CAST_HPP

/**
 * @file RangeCast.hpp
 *
 * @brief Defines range cast functionality.
 */

#include "TypeTraits.hpp"           // for Require, etc.
#include "detail/RangeCastImpl.hpp" // for RangeCastImpl, etc.

namespace nvcv::cuda {

/**
 * @brief Metafunction to range cast (scale) all elements to a target range
 *
 * @details This function range casts (that is scales) all elements to the range defined by the template argument
 * type \p T.  For instance, a float4 with all elements between 0 and 1 can be casted to an uchar4 with scaling of
 * each element to be in between 0 and 255 (see example below).  It is a requirement of RangeCast that both types
 * have type traits and type \p T must be a regular C type. Several examples of possible target range giving a
 * source range, depending on the limits of regular C types, for the RangeCast function are as follows:
 *
 * | Source type U  | Target type T |  Source range   |        Target range       |
 * |:--------------:|:-------------:|:---------------:|:-------------------------:|
 * |  signed char   |     float     |   [-128, 127]   |          [-1, 1]          |
 * |     float      | unsigned char |      [0, 1]     |           [0, 255]        |
 * |     short      | unsigned int  | [-32768, 32767] |      [0, 4294967295]      |
 * |    double      |      int      |     [-1, 1]     | [-2147483648, 2147483647] |
 * | unsigned short |     double    |      [0, 65535] |           [0, 1]          |
 *
 * @defgroup NVCV_CPP_CUDATOOLS_RANGECAST Range cast
 * @{
 *
 * @code
 * using DataType = MakeType<uchar, 4>;
 * using FloatDataType = ConvertBaseTypeTo<float, DataType>;
 * FloatDataType res = ...; // res component values are in [0, 1]
 * DataType pix = RangeCast<BaseType<DataType>>(res); // pix are in [0, 255]
 * @endcode
 *
 * @tparam T Type that defines the target range to cast
 * @tparam U Type of the source value (with 1 to 4 elements) passed as argument
 *
 * @param[in] u Source value to cast all elements to range of type \p T
 *
 * @return The value with all elements scaled
 */
template<typename T, typename U, class = Require<HasTypeTraits<T, U> && !IsCompound<T>>>
__host__ __device__ auto RangeCast(U u)
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
            GetElement(out, e) = detail::RangeCastImpl<T, BaseType<U>>(GetElement(u, e));
        }

        return out;
    }
}

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_RANGE_CAST_HPP
