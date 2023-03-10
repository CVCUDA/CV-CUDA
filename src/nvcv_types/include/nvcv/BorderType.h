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

#ifndef NVCV_BORDER_TYPE_H
#define NVCV_BORDER_TYPE_H

#ifdef __cplusplus
extern "C"
{
#endif

// @brief Flag to choose the border mode to be used
typedef enum
{
    NVCV_BORDER_CONSTANT   = 0,
    NVCV_BORDER_REPLICATE  = 1,
    NVCV_BORDER_REFLECT    = 2,
    NVCV_BORDER_WRAP       = 3,
    NVCV_BORDER_REFLECT101 = 4,
} NVCVBorderType;

#ifdef __cplusplus
}
#endif

#endif // NVCV_BORDER_TYPE_H
