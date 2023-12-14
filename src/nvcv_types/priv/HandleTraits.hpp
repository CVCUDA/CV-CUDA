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

#ifndef NVCV_PRIV_CORE_HANDLE_TRAITS_HPP
#define NVCV_PRIV_CORE_HANDLE_TRAITS_HPP

#include <nvcv/Fwd.h>
#include <nvcv/alloc/Fwd.h>

namespace nvcv::priv {

template<class T>
struct HandleTraits;

template<>
struct HandleTraits<NVCVImageHandle>
{
    constexpr static bool hasManager = true;
};

template<>
struct HandleTraits<NVCVTensorHandle>
{
    constexpr static bool hasManager = true;
};

template<>
struct HandleTraits<NVCVArrayHandle>
{
    constexpr static bool hasManager = true;
};

template<>
struct HandleTraits<NVCVAllocatorHandle>
{
    constexpr static bool hasManager = true;
};

template<>
struct HandleTraits<NVCVImageBatchHandle>
{
    constexpr static bool hasManager = true;
};

template<class T>
constexpr bool HasObjManager = HandleTraits<T>::hasManager;

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_HANDLE_TRAITS_HPP
