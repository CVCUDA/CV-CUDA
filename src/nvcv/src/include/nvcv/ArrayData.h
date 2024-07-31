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

#ifndef NVCV_ARRAYDATA_H
#define NVCV_ARRAYDATA_H

#include <nvcv/ImageFormat.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Stores the array plane contents. */
typedef struct NVCVArrayBufferStridedRec
{
    /** Untyped memory concepts stride
     * Stride represents the number of bytes spanned by one data unit.
     */
    int64_t stride;

    /** Pointer to memory buffer with array contents.
     */
    NVCVByte *basePtr;
} NVCVArrayBufferStrided;

typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_ARRAY_BUFFER_NONE = 0,

    NVCV_ARRAY_BUFFER_CUDA        = 0x11, /* 0001_0001 */
    NVCV_ARRAY_BUFFER_HOST        = 0x24, /* 0010_0100 */
    NVCV_ARRAY_BUFFER_HOST_PINNED = 0x28, /* 0010_1000 */
} NVCVArrayBufferType;

typedef union NVCVArrayBufferRec
{
    NVCVArrayBufferStrided strided;
} NVCVArrayBuffer;

/** Stores information about data characteristics and content. */
typedef struct NVCVArrayDataRec
{
    NVCVDataType dtype;

    int64_t length;
    int64_t capacity;

    /** Type of image batch buffer.
     *  It defines which member of the \ref NVCVArrayBuffer tagged union that
     *  must be used to access the image batch buffer contents. */
    NVCVArrayBufferType bufferType;

    /** Stores the image batch contents. */
    NVCVArrayBuffer buffer;
} NVCVArrayData;

#ifdef __cplusplus
}
#endif

#endif // NVCV_ARRAYDATA_H
