/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file HalfTypes.hpp
 *
 * @brief Defines CUDA compound types for the __half base type.
 */

#ifndef NVCV_CUDA_HALF_TYPES_HPP
#define NVCV_CUDA_HALF_TYPES_HPP

#include <cuda_fp16.h>

#include <type_traits> // for std::is_arithmetic_v, etc.

// CUDA provides __half and the 2-component __half2 (with global typedefs half and half2), but no
// 1/3/4-component counterparts.  Define them at global scope, mirroring the built-in CUDA vector
// types (cf. Compat.hpp), so cuda_tools metaprogramming handles half like any other base type.

struct alignas(2) half1
{
    __half x;
};

struct alignas(2) half3
{
    __half x;
    __half y;
    __half z;
};

// 8-byte alignment enables a single 64-bit load/store per half4 value.
struct alignas(8) half4
{
    __half x;
    __half y;
    __half z;
    __half w;
};

// cuda_fp16.h only defines __half OP __half; mixing __half with another arithmetic type is
// ambiguous, as both operands accept one implicit conversion (to __half or through float).
// Define the mixed operators with the usual C++ arithmetic promotion through float, so __half
// behaves in mixed expressions (e.g. pixel * weight) like the other narrow types do.

#define NVCV_CUDA_HALF_MIXED_OPERATOR(OPERATOR)                             \
    template<typename A, class = std::enable_if_t<std::is_arithmetic_v<A>>> \
    inline __host__ __device__ auto operator OPERATOR(__half h, A a)        \
    {                                                                       \
        return __half2float(h) OPERATOR a;                                  \
    }                                                                       \
    template<typename A, class = std::enable_if_t<std::is_arithmetic_v<A>>> \
    inline __host__ __device__ auto operator OPERATOR(A a, __half h)        \
    {                                                                       \
        return a OPERATOR __half2float(h);                                  \
    }

NVCV_CUDA_HALF_MIXED_OPERATOR(+)
NVCV_CUDA_HALF_MIXED_OPERATOR(-)
NVCV_CUDA_HALF_MIXED_OPERATOR(*)
NVCV_CUDA_HALF_MIXED_OPERATOR(/)
NVCV_CUDA_HALF_MIXED_OPERATOR(==)
NVCV_CUDA_HALF_MIXED_OPERATOR(!=)
NVCV_CUDA_HALF_MIXED_OPERATOR(<)
NVCV_CUDA_HALF_MIXED_OPERATOR(>)
NVCV_CUDA_HALF_MIXED_OPERATOR(<=)
NVCV_CUDA_HALF_MIXED_OPERATOR(>=)

#undef NVCV_CUDA_HALF_MIXED_OPERATOR

#endif // NVCV_CUDA_HALF_TYPES_HPP
