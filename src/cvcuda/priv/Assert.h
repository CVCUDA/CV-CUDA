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

#ifndef CVCUDA_PRIV_ASSERT_H
#define CVCUDA_PRIV_ASSERT_H

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(NVCV_CUDA_ASSERT)
#    if !defined(NDEBUG) && __NVCC__
#        define NVCV_CUDA_ASSERT(x, ...)                     \
            do                                               \
            {                                                \
                if (!(x))                                    \
                {                                            \
                    printf("E Condition (%s) failed: ", #x); \
                    printf(__VA_ARGS__);                     \
                    asm("trap;");                            \
                }                                            \
            }                                                \
            while (1 == 0)
#    else
#        define NVCV_CUDA_ASSERT(x, ...) \
            do                           \
            {                            \
            }                            \
            while (1 == 0)
#    endif
#endif // !defined(NVCV_CUDA_ASSERT)

#ifdef __cplusplus
}
#endif

#endif // CVCUDA_PRIV_ASSERT_H
