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

#ifndef NVCV_UTIL_COMPILER_HPP
#define NVCV_UTIL_COMPILER_HPP

#if defined(__GNUC__) && !defined(__clang__)
#    define NVCV_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#if defined(__clang__)
#    define NVCV_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif

#if defined(_MSC_VER)
#    define NVCV_MSC_VERSION _MSC_VER
#endif

#if defined(__CUDACC__)
#    define NVCV_CUDACC_VERSION (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__)
#endif

#if defined(_WIN32)
#    define NVCV_WINDOWS 1
#endif

#if defined(__unix__)
#    define NVCV_UNIX 1
#endif

#if NVCV_GCC_VERSION || NVCV_CLANG_VERSION
#    define NVCV_FORCE_INLINE    __attribute__((always_inline)) inline
#    define NVCV_NO_INLINE       __attribute__((noinline))
#    define NVCV_UNUSED_FUNCTION __attribute__((unused))
#    define NVCV_RESTRICT        __restrict__
#elif NVCV_MSC_VERSION
#    define NVCV_FORCE_INLINE __forceinline
#    define NVCV_NO_INLINE    __declspec(noinline)
#    define NVCV_UNUSED_FUNCTION
#    define NVCV_RESTRICT __restrict
#else
#    error "unrecognized compiler!"
#endif

#if __NVCC__
#    define NVCV_CUDA_HOST_DEVICE __host__ __device__
#else
#    define NVCV_CUDA_HOST_DEVICE
#endif

#if defined(__has_feature)
#    if __has_feature(address_sanitizer)
#        define NVCV_SANITIZED 1
#    endif
#elif defined(__SANITIZE_ADDRESS__)
#    define NVCV_SANITIZED 1
#endif

#endif // NVCV_UTIL_COMPILER_HPP
