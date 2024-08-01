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

#ifndef NVCV_IMAGEDATA_H
#define NVCV_IMAGEDATA_H

#include "detail/CudaFwd.h"

#include <nvcv/DataType.h>
#include <nvcv/ImageFormat.h>
#include <stdint.h>

typedef struct NVCVImagePlaneStridedRec
{
    /** Width of this plane in pixels.
     *  + It must be >= 1. */
    int32_t width;

    /** Height of this plane in pixels.
     *  + It must be >= 1. */
    int32_t height;

    /** Difference in bytes of beginning of one row and the beginning of the previous.
         This is used to address every row (and ultimately every pixel) in the plane.
         @code
            T *pix_addr = (T *)(basePtr + rowStride*height)+width;
         @endcode
         where T is the C type related to dataType.

         + It must be at least `(width * bits-per-pixel + 7)/8`.
    */
    int32_t rowStride;

    /** Pointer to the beginning of the first row of this plane.
        This points to the actual plane contents. */
    NVCVByte *basePtr;
} NVCVImagePlaneStrided;

/** Maximum number of data planes an image can have. */
#define NVCV_MAX_PLANE_COUNT (6)

/** Stores the image plane contents. */
typedef struct NVCVImageBufferStridedRec
{
    /** Number of planes.
     *  + Must be >= 1. */
    int32_t numPlanes;

    /** Data of all image planes in pitch-linear layout.
     *  + Only the first \ref numPlanes elements must have valid data. */
    NVCVImagePlaneStrided planes[NVCV_MAX_PLANE_COUNT];
} NVCVImageBufferStrided;

typedef struct NVCVImageBufferCudaArrayRec
{
    /** Number of planes.
     *  + Must be >= 1. */
    int32_t numPlanes;

    /** Data of all image planes in pitch-linear layout.
     *  + Only the first \ref numPlanes elements must have valid data. */
    cudaArray_t planes[NVCV_MAX_PLANE_COUNT];
} NVCVImageBufferCudaArray;

/** Represents how the image data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_IMAGE_BUFFER_NONE = 0,

    /** GPU-accessible with planes in pitch-linear layout. */
    NVCV_IMAGE_BUFFER_STRIDED_CUDA,

    /** Host-accessible with planes in pitch-linear layout. */
    NVCV_IMAGE_BUFFER_STRIDED_HOST,

    /** Buffer stored in a cudaArray_t.
     * Please consult <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays">cudaArray_t</a>
     * for more information. */
    NVCV_IMAGE_BUFFER_CUDA_ARRAY,
} NVCVImageBufferType;

/** Represents the available methods to access image contents.
 * The correct method depends on \ref NVCVImageData::bufferType. */
typedef union NVCVImageBufferRec
{
    /** Image stored in pitch-linear layout.
     * To be used when \ref NVCVImageData::bufferType is:
     * - \ref NVCV_IMAGE_BUFFER_STRIDED_CUDA
     * - \ref NVCV_IMAGE_BUFFER_STRIDED_HOST
     */
    NVCVImageBufferStrided strided;

    /** Image stored in a `cudaArray_t`.
     * To be used when \ref NVCVImageData::bufferType is:
     * - \ref NVCV_IMAGE_BUFFER_CUDA_ARRAY
     */
    NVCVImageBufferCudaArray cudaarray;
} NVCVImageBuffer;

/** Stores information about image characteristics and content. */
typedef struct NVCVImageDataRec
{
    /** Image format. */
    NVCVImageFormat format;

    /** Type of image buffer.
     *  It defines which member of the \ref NVCVImageBuffer tagged union that
     *  must be used to access the image contents. */
    NVCVImageBufferType bufferType;

    /** Stores the image contents. */
    NVCVImageBuffer buffer;
} NVCVImageData;

#endif // NVCV_IMAGEDATA_H
