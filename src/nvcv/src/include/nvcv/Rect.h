/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_RECT_H
#define NVCV_RECT_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    int32_t x;      //!< x coordinate of the top-left corner
    int32_t y;      //!< y coordinate of the top-left corner
    int32_t width;  //!< width of the rectangle
    int32_t height; //!< height of the rectangle
} NVCVRectI;

#ifdef __cplusplus
}
#endif

#endif // NVCV_RECT_H
