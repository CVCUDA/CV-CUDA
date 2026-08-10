/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef THRESHOLD_UTILS_CUH
#define THRESHOLD_UTILS_CUH

#include "CvCudaUtils.cuh"

#include <cstdint>

template<typename P, typename T>
__device__ __forceinline__ P LoadPacked(const T *ptr)
{
    static_assert(sizeof(P) % sizeof(T) == 0);

    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(P) == 0)
        return *reinterpret_cast<const P *>(ptr);

    P  value;
    T *elements = reinterpret_cast<T *>(&value);
#pragma unroll
    for (int i = 0; i < sizeof(P) / sizeof(T); ++i) elements[i] = ptr[i];
    return value;
}

template<typename P, typename T>
__device__ __forceinline__ void StorePacked(T *ptr, P value)
{
    static_assert(sizeof(P) % sizeof(T) == 0);

    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(P) == 0)
    {
        *reinterpret_cast<P *>(ptr) = value;
        return;
    }

    const T *elements = reinterpret_cast<const T *>(&value);
#pragma unroll
    for (int i = 0; i < sizeof(P) / sizeof(T); ++i) ptr[i] = elements[i];
}

__global__ void triangle_cal(int *histogram, nvcv::cuda::Tensor1DWrap<double, int32_t> thresh);

#endif // THRESHOLD_UTILS_CUH
