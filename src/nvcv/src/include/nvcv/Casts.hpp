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

#ifndef NVCV_CASTS_HPP
#define NVCV_CASTS_HPP

#include "detail/CastsImpl.hpp"

#include <type_traits>

namespace nvcv {
/**
 * @brief Helper type alias to deduce the handle type of a given class `T`.
 *
 * This type alias uses std::remove_pointer to strip the pointer off of `T` (if any),
 * and then accesses the `HandleType` nested type within `T`.
 *
 * @tparam T The class type from which to deduce the handle type.
 */
template<class T>
using HandleTypeOf = typename std::remove_pointer<T>::type::HandleType;

/**
 * @brief A templated function to perform a static cast on the handle type of a given class `T`.
 *
 * This function uses the `StaticCast` structure in the `detail` namespace to perform the static cast.
 * The `decltype` keyword is used to deduce the return type of the function.
 *
 * @tparam T The class type to use for the static cast.
 * @param h The handle type instance to be cast.
 * @return Returns the result of the static cast operation.
 */
template<class T>
auto StaticCast(HandleTypeOf<T> h) -> decltype(detail::StaticCast<T>::cast(h))
{
    return detail::StaticCast<T>::cast(h);
}

/**
 * @brief A templated function to perform a dynamic cast on the handle type of a given class `T`.
 *
 * This function uses the `DynamicCast` structure in the `detail` namespace to perform the dynamic cast.
 * The `decltype` keyword is used to deduce the return type of the function.
 *
 * @tparam T The class type to use for the dynamic cast.
 * @param h The handle type instance to be cast.
 * @return Returns the result of the dynamic cast operation.
 */
template<class T>
auto DynamicCast(HandleTypeOf<T> h) -> decltype(detail::StaticCast<T>::cast(h))
{
    return detail::DynamicCast<T>::cast(h);
}

} // namespace nvcv

#endif // NVCV_CASTS_HPP
