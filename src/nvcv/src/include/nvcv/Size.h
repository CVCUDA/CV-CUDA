/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_SIZE_H
#define NVCV_SIZE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Struct representing a two-dimensional size.
 *
 * This structure is designed to represent a width and height in 2D space.
 */
typedef struct
{
    int32_t w, h;
} NVCVSize2D;

#ifdef __cplusplus
}
#endif

#endif // NVCV_SIZE_H
