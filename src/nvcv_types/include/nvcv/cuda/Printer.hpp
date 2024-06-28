/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file Printer.hpp
 *
 * @brief Defines printer operator to print CUDA compound types.
 */

#ifndef NVCV_CUDA_PRINTER_HPP
#define NVCV_CUDA_PRINTER_HPP

#include "TypeTraits.hpp" // for Require, etc.

#include <ostream> // for std::ostream, etc.

/**
 * Metaoperator to insert a pixel into an output stream.
 *
 * The pixel may be a CUDA compound type with 1 to 4 components.  This operator returns the output stream
 * changed by an additional string with the name of the type followed by each component value in between
 * parentheses.
 *
 * @code
 * DataType pix = ...;
 * std::cout << pix;
 * @endcode
 *
 * @tparam T Type of the pixel to be inserted in the output stream.
 *
 * @param[in, out] out Output stream to be changed and returned.
 * @param[in] v Pixel value to be inserted formatted in the output stream.
 *
 * @return Output stream with the data type and values.
 */
template<class T, class = nvcv::cuda::Require<nvcv::cuda::IsCompound<T>>>
__host__ std::ostream &operator<<(std::ostream &out, const T &v)
{
    using BT      = nvcv::cuda::BaseType<T>;
    using OutType = std::conditional_t<sizeof(BT) == 1, int, BT>;

    out << nvcv::cuda::GetTypeName<T>() << "(";

    out << static_cast<OutType>(nvcv::cuda::GetElement<0>(v));
    if constexpr (nvcv::cuda::NumComponents<T> >= 2)
        out << ", " << static_cast<OutType>(nvcv::cuda::GetElement<1>(v));
    if constexpr (nvcv::cuda::NumComponents<T> >= 3)
        out << ", " << static_cast<OutType>(nvcv::cuda::GetElement<2>(v));
    if constexpr (nvcv::cuda::NumComponents<T> == 4)
        out << ", " << static_cast<OutType>(nvcv::cuda::GetElement<3>(v));

    out << ")";

    return out;
}

#endif // NVCV_CUDA_PRINTER_HPP
