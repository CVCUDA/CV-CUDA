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

#ifndef NVCV_BORDER_TYPE_H
#define NVCV_BORDER_TYPE_H

#ifdef __cplusplus
extern "C"
{
#endif

/* @brief Flag to choose the border mode to be used
 *
 * This enum is used to specify the type of border to be used for functions
 * that require it. Here are the different types:
 * - `NVCV_BORDER_CONSTANT`: Uses a constant value for borders.
 * - `NVCV_BORDER_REPLICATE`: Replicates the last element for borders.
 * - `NVCV_BORDER_REFLECT`: Reflects the border elements.
 * - `NVCV_BORDER_WRAP`: Wraps the border elements.
 * - `NVCV_BORDER_REFLECT101`: Reflects the border elements, excluding the last element.
 */
typedef enum
{
    NVCV_BORDER_CONSTANT   = 0, ///< Uses a constant value for borders.
    NVCV_BORDER_REPLICATE  = 1, ///< Replicates the last element for borders.
    NVCV_BORDER_REFLECT    = 2, ///< Reflects the border elements.
    NVCV_BORDER_WRAP       = 3, ///< Wraps the border elements.
    NVCV_BORDER_REFLECT101 = 4, ///< Reflects the border elements, excluding the last element.
} NVCVBorderType;

#ifdef __cplusplus
}
#endif

#endif // NVCV_BORDER_TYPE_H
