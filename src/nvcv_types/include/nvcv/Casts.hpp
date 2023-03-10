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

#ifndef NVCV_CASTS_HPP
#define NVCV_CASTS_HPP

#include "detail/CastsImpl.hpp"

#include <type_traits>

namespace nvcv {

template<class T>
using HandleTypeOf = typename std::remove_pointer<T>::type::HandleType;

template<class T>
auto StaticCast(HandleTypeOf<T> h) -> decltype(detail::StaticCast<T>::cast(h))
{
    return detail::StaticCast<T>::cast(h);
}

template<class T>
auto DynamicCast(HandleTypeOf<T> h) -> decltype(detail::StaticCast<T>::cast(h))
{
    return detail::DynamicCast<T>::cast(h);
}

} // namespace nvcv

#endif // NVCV_CASTS_HPP
