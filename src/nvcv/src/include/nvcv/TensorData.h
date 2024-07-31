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

#ifndef NVCV_TENSORDATA_H
#define NVCV_TENSORDATA_H

#include "TensorLayout.h"

#include <nvcv/ImageFormat.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Stores the tensor plane contents. */
typedef struct NVCVTensorBufferStridedRec
{
    int64_t strides[NVCV_TENSOR_MAX_RANK];

    /** Pointer to memory buffer with tensor contents.
     * Element with type T is addressed by:
     * pelem = basePtr + shape[0]*strides[0] + ... + shape[rank-1]*strides[rank-1];
     */
    NVCVByte *basePtr;
} NVCVTensorBufferStrided;

/** Represents how the image buffer data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_TENSOR_BUFFER_NONE = 0,

    /** GPU-accessible with equal-shape planes in pitch-linear layout. */
    NVCV_TENSOR_BUFFER_STRIDED_CUDA,
} NVCVTensorBufferType;

/** Represents the available methods to access image batch contents.
 * The correct method depends on \ref NVCVTensorData::bufferType. */
typedef union NVCVTensorBufferRec
{
    /** Tensor image batch stored in pitch-linear layout.
     * To be used when \ref NVCVTensorData::bufferType is:
     * - \ref NVCV_TENSOR_BUFFER_STRIDED_CUDA
     */
    NVCVTensorBufferStrided strided;
} NVCVTensorBuffer;

/** Stores information about image batch characteristics and content. */
typedef struct NVCVTensorDataRec
{
    NVCVDataType     dtype;
    NVCVTensorLayout layout;

    int32_t rank;
    int64_t shape[NVCV_TENSOR_MAX_RANK];

    /** Type of image batch buffer.
     *  It defines which member of the \ref NVCVTensorBuffer tagged union that
     *  must be used to access the image batch buffer contents. */
    NVCVTensorBufferType bufferType;

    /** Stores the image batch contents. */
    NVCVTensorBuffer buffer;
} NVCVTensorData;

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORDATA_H
