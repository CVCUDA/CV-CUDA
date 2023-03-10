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

#ifndef NVCV_TYPE_TRAITS_HPP
#define NVCV_TYPE_TRAITS_HPP

#include <type_traits>

namespace nvcv { namespace detail {

template<bool Cond, typename T = void>
using EnableIf_t = typename std::enable_if<Cond, T>::type;

template<bool Cond, typename If, typename Else>
using Conditional_t = typename std::conditional<Cond, If, Else>::type;

template<typename T>
using AddPointer_t = typename std::add_pointer<T>::type;

template<typename T>
using AddLRef_t = typename std::add_lvalue_reference<T>::type;

template<typename T>
using AddRRef_t = typename std::add_rvalue_reference<T>::type;

template<typename T>
using RemovePointer_t = typename std::remove_pointer<T>::type;

template<typename T>
using RemoveRef_t = typename std::remove_reference<T>::type;

template<typename T>
using RemoveCV_t = typename std::remove_cv<T>::type;

template<typename T>
using RemoveCVRef_t = RemoveCV_t<RemoveRef_t<T>>;

template<typename R, typename... Args, typename F>
std::is_convertible<decltype(std::declval<F>()(std::declval<Args>()...)), R> IsInvocableRF(F *);

template<typename R, typename... Args>
std::false_type IsInvocableRF(...);

template<typename R, typename Callable, typename... Args>
struct IsInvocableR : decltype(IsInvocableRF<R, Args...>(AddPointer_t<Callable>()))
{
};

template<typename... Args, typename F, typename = decltype(std::declval<F>()(std::declval<Args>()...))>
std::true_type IsInvocableF(F *);

template<typename... Args>
std::false_type IsInvocableF(...);

template<typename Callable, typename... Args>
struct IsInvocable : decltype(IsInvocableF<Args...>(AddPointer_t<Callable>()))
{
};

std::false_type IsStdFunctionF(...);

template<typename T>
std::true_type IsStdFunctionF(const std::function<T> *);

template<typename X>
struct IsStdFunction : decltype(IsStdFunctionF(std::declval<X *>()))
{
};

}} // namespace nvcv::detail

#endif // NVCV_TYPE_TRAITS_HPP
