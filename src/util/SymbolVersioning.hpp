/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NVCV_UTIL_SYMBOLVERSIONING_HPP
#define NVCV_UTIL_SYMBOLVERSIONING_HPP

#include "Compiler.hpp"

/* Tools to help defining versioned APIs.
 *
At first, all public functions are defined like this:

Header:
 NVCV_PUBLIC NVCVStatus nvcvFooCreate(uint64_t flags, NVCVFoo *handle);

Implementation:
 NVCV_PROJ_DEFINE_API(NVCV, 1, 0, NVCVStatus, nvcvFooCreate, (uint64_t flags, NVCVFoo *handle))
 {
    implementation 1;
 }

If we need to add a new version of nvcvFooCreate, we update cv-cuda code like this:

Header:
 #if NVCV_API_VERSION == 100
 __asm__(".symver nvcvFooCreate,nvcvFooCreate@NVCV_1.0");
 NVCV_PUBLIC NVCVStatus nvcvFooCreate(uint64_t flags, NVCVFoo *handle);
 #else
 NVCV_PUBLIC NVCVStatus nvcvFooCreate(int32_t size, uint64_t flags, NVCVFoo *handle);
 #endif

Implementation:

 NVCV_PROJ_DEFINE_API_OLD(NVCV, 1, 0, NVCVStatus, nvcvFooCreate(uint64_t flags, NVCVFoo *handle))
 {
    implementation 1;
 }

 NVCV_PROJ_DEFINE_API(NVCV, 1, 1, NVCVStatus, nvcvFooCreate, (int32_t size, uint64_t flags, NVCVFoo *handle))
 {
    implementation 2;
 }

If user defines NVCV_API_VERSION == 100, he will use the first definition and linker will
link to the cvcuda-1.0 API. If nothing is defined, he will use the most recent definition.

Users using dlopen to retrieve our functions will have to use dlvsym to specify
what version to get. Regular dlsym will always pick the most recent one.

void *foo10 = dlvsym(lib, "nvcvFoo", "NVCV_1.0");
void *foo11 = dlvsym(lib, "nvcvFoo", "NVCV_1.1");
void *foo11_tmp = dlsym(lib, "nvcvFoo");
assert(foo11 == foo11_tmp);
*/

#define NVCV_PROJ_FUNCTION_API(FUNC, VER_MAJOR, VER_MINOR) FUNC##_v##VER_MAJOR##_##VER_MINOR

// gcc>=10.0 supports the symver function attribute, let's use it.
#if NVCV_GCC_VERSION >= 100000
#    define NVCV_PROJ_SYMVER(PROJ, FUNC, VER_MAJOR, VER_MINOR, VERTYPE) \
        extern "C" __attribute__((visibility("default")))               \
        __attribute__((__symver__(#FUNC VERTYPE #PROJ "_" #VER_MAJOR "." #VER_MINOR)))
#else
#    define NVCV_PROJ_SYMVER(PROJ, FUNC, VER_MAJOR, VER_MINOR, VERTYPE)                                \
        __asm__(".symver " #FUNC "_v" #VER_MAJOR "_" #VER_MINOR "," #FUNC VERTYPE #PROJ "_" #VER_MAJOR \
                "." #VER_MINOR);                                                                       \
        extern "C" __attribute__((visibility("default")))
#endif

#define NVCV_PROJ_DEFINE_API_HELPER(PROJ, VER_MAJOR, VER_MINOR, VERTYPE, RETTYPE, FUNC, FUNC_VER, ARGS) \
    extern "C" __attribute__((visibility("default"))) RETTYPE FUNC_VER ARGS;                            \
    NVCV_PROJ_SYMVER(PROJ, FUNC, VER_MAJOR, VER_MINOR, VERTYPE)                                         \
    RETTYPE FUNC_VER ARGS

#define NVCV_PROJ_DEFINE_API_OLD(PROJ, VER_MAJOR, VER_MINOR, RETTYPE, FUNC, ARGS) \
    NVCV_PROJ_DEFINE_API_HELPER(PROJ, VER_MAJOR, VER_MINOR, "@", RETTYPE, FUNC,   \
                                NVCV_PROJ_FUNCTION_API(FUNC, VER_MAJOR, VER_MINOR), ARGS)

#define NVCV_PROJ_DEFINE_API(PROJ, VER_MAJOR, VER_MINOR, RETTYPE, FUNC, ARGS)    \
    NVCV_PROJ_DEFINE_API_HELPER(PROJ, VER_MAJOR, VER_MINOR, "@@", RETTYPE, FUNC, \
                                NVCV_PROJ_FUNCTION_API(FUNC, VER_MAJOR, VER_MINOR), ARGS)

#endif // NVCV_UTIL_SYMBOLVERSIONING_HPP
