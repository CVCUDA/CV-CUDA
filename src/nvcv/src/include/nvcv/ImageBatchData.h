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

#ifndef NVCV_IMAGEBATCHDATA_H
#define NVCV_IMAGEBATCHDATA_H

#include "ImageData.h"

#include <nvcv/ImageFormat.h>

/** Stores the image plane in a variable shape image batch. */
typedef struct NVCVImageBatchVarShapeBufferStridedRec
{
    /** Format of all images in the batch.
     * If images don't have all the same format, or the batch is empty,
     * the value is \ref NVCV_IMAGE_FORMAT_NONE . */
    NVCVImageFormat uniqueFormat;

    /** Union of all image dimensions.
     * If 0 and number of images is >= 1, this value
     * must not be relied upon. */
    int32_t maxWidth, maxHeight;

    /** Pointer to an array of formats, one for each image in `imageList`. */
    NVCVImageFormat *formatList;

    /** Pointer to a host-side array of formats, one for each image in `imageList`. */
    const NVCVImageFormat *hostFormatList;

    /** Pointer to all image planes in pitch-linear layout in the image batch.
     * It's an array of `numPlanesPerImage*numImages` planes. The number of planes
     * in the image can be fetched from the image batch's format. With that,
     * plane P of image N can be indexed as imageList[N].planes[P].
     */
    NVCVImageBufferStrided *imageList;
} NVCVImageBatchVarShapeBufferStrided;

/** Stores the tensor plane contents. */
typedef struct NVCVImageBatchTensorBufferStridedRec
{
    /** Distance in bytes from beginning of first plane of one image to the
     *  first plane of the next image.
     *  + Must be >= 1. */
    int64_t imgStride;

    /** Distance in bytes from beginning of one row to the next.
     *  + Must be >= 1. */
    int32_t rowStride;

    /** Dimensions of each image.
     * + Must be >= 1x1 */
    int32_t imgWidth, imgHeight;

    /** Buffer of all image planes in pitch-linear layout.
     *  It assumes all planes have same dimension specified by imgWidth/imgHeight,
     *  and that all planes have the same row pitch.
     *  + Only the first N elements must have valid data, where N is the number of planes
     *    defined by @ref NVCVImageBatchData::format. */
    void *planeBuffer[NVCV_MAX_PLANE_COUNT];
} NVCVImageBatchTensorBufferStrided;

/** Represents how the image buffer data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_IMAGE_BATCH_BUFFER_NONE = 0,

    /** GPU-accessible with variable-shape planes in pitch-linear layout. */
    NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA,
} NVCVImageBatchBufferType;

/** Represents the available methods to access image batch contents.
 * The correct method depends on \ref NVCVImageBatchData::bufferType. */
typedef union NVCVImageBatchBufferRec
{
    /** Varshape image batch stored in pitch-linear layout.
     * To be used when \ref NVCVImageBatchData::bufferType is:
     * - \ref NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_CUDA
     */
    NVCVImageBatchVarShapeBufferStrided varShapeStrided;
} NVCVImageBatchBuffer;

/** Stores information about image batch characteristics and content. */
typedef struct NVCVImageBatchDataRec
{
    /** Number of images in the image batch */
    int32_t numImages;

    /** Type of image batch buffer.
     *  It defines which member of the \ref NVCVImageBatchBuffer tagged union that
     *  must be used to access the image batch buffer contents. */
    NVCVImageBatchBufferType bufferType;

    /** Stores the image batch contents. */
    NVCVImageBatchBuffer buffer;
} NVCVImageBatchData;

#endif // NVCV_IMAGEBATCHDATA_H
