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

/**
 * @file Image.h
 *
 * @brief Public C interface to NVCV image representation.
 */

#ifndef NVCV_IMAGE_H
#define NVCV_IMAGE_H

#include "Export.h"
#include "Fwd.h"
#include "ImageData.h"
#include "Status.h"
#include "alloc/Allocator.h"
#include "alloc/Requirements.h"
#include "detail/CudaFwd.h"

#include <nvcv/ImageFormat.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Underlying image type.
 *
 * Images can have different underlying types depending on the function used to
 * create them.
 * */
typedef enum
{
    /** 2D image. */
    NVCV_TYPE_IMAGE,
    /** Image that wraps an user-allocated image buffer. */
    NVCV_TYPE_IMAGE_WRAPDATA
} NVCVTypeImage;

typedef struct NVCVImage *NVCVImageHandle;

/** Image data cleanup function type */
typedef void (*NVCVImageDataCleanupFunc)(void *ctx, const NVCVImageData *data);

/** Stores the requirements of an image. */
typedef struct NVCVImageRequirementsRec
{
    int32_t         width, height; /*< Image dimensions. */
    NVCVImageFormat format;        /*< Image format. */

    /** Row stride of each plane, in bytes */
    int32_t planeRowStride[NVCV_MAX_PLANE_COUNT];

    int32_t          alignBytes; /*< Alignment/block size in bytes */
    NVCVRequirements mem;        /*< Image resource requirements. */
} NVCVImageRequirements;

/** Calculates the resource requirements needed to create an image.
 *
 * @param [in] width,height Image dimensions.
 *                          + Width and height must be > 0.
 *
 * @param [in] format       Image format.
 *                          + Must not be \ref NVCV_IMAGE_FORMAT_NONE.
 *
 * @param [in] baseAddrAlignment Alignment, in bytes, of the requested memory buffer.
 *                               If 0, use a default suitable for optimized memory access.
 *                               The used alignment is at least the given value.
 *                               + If different from 0, it must be a power-of-two.
 *
 * @param [in] rowAddrAlignment Alignment, in bytes, of each image's row address.
 *                              If 0, use a default suitable for optimized memory access.
 *                              The used alignment is at least the given value.
 *                              Pass 1 for fully packed rows, i.e., no padding
 *                              at the end of each row.
 *                              + If different from 0, it must be a power-of-two.
 *
 * @param [out] reqs        Where the image requirements will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageCalcRequirements(int32_t width, int32_t height, NVCVImageFormat format,
                                                 int32_t baseAddrAlignment, int32_t rowAddrAlignment,
                                                 NVCVImageRequirements *reqs);

/** Constructs and an image instance with given requirements in the given storage.
 *
 * @param [in] reqs Image requirements. Must have been filled in by @ref nvcvImageCalcRequirements.
 *                  + Must not be NULL
 *
 * @param [in] alloc Allocator to be used to allocate needed memory buffers.
 *                   - The following resources are used:
 *                     - host memory: for internal structures.
 *                     - cuda memory: for image contents buffer.
 *                       If NULL, it'll use the internal default allocator.
 *                   + Allocator must not be destroyed while an image still refers to it.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageConstruct(const NVCVImageRequirements *reqs, NVCVAllocatorHandle alloc,
                                          NVCVImageHandle *handle);

/** Wraps an existing image buffer into an NVCV image instance constructed in given storage
 *
 * It allows for interoperation of external image representations with NVCV.
 * The created image type is \ref NVCV_TYPE_IMAGE_WRAPDATA .
 *
 * @param [in] data Image contents.
 *                  + Must not be NULL
 *                  + Buffer type must not be \ref NVCV_IMAGE_BUFFER_NONE.
 *                  + Image dimensions must be >= 1x1
 *
 * @param [in] cleanup Cleanup function to be called when the image is destroyed
 *                     via @ref nvcvImageDecRef
 *                     If NULL, no cleanup function is defined.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @param [out] handle      Where the image instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageWrapDataConstruct(const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup,
                                                  void *ctxCleanup, NVCVImageHandle *handle);

/** Decrements the reference count of an existing image instance.
 *
 * The image is destroyed when its reference count reaches zero.
 *
 * If the image has type @ref NVCV_TYPE_IMAGE_WRAPDATA and has a cleanup function defined,
 * cleanup will be called.
 *
 * @note The image must not be in use in current and future operations.
 *
 * @param [in] handle       Image to be destroyed.
 *                          If NULL, no operation is performed, successfully.
 *                          + The handle must have been created with any of the nvcvImageConstruct functions.
 *
 * @param [out] newRefCount The decremented reference count. If the return value is 0, the object was destroyed.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageDecRef(NVCVImageHandle handle, int *newRefCount);

/** Increments the reference count of an image.
 *
 * @param [in] handle       Image to be retained.
 *
 * @param [out] newRefCount The incremented reference count.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageIncRef(NVCVImageHandle handle, int *newRefCount);

/** Returns the current reference count of an image
 *
 * @param [in] handle       The handle whose reference count is to be obtained.
 *
 * @param [out] newRefCount The reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageRefCount(NVCVImageHandle handle, int *newRefCount);

/** Associates a user pointer to the image handle.
 *
 * This pointer can be used to associate any kind of data with the image object.
 *
 * @param [in] handle Image to be associated with the user pointer.
 *
 * @param [in] userPtr User pointer.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageSetUserPointer(NVCVImageHandle handle, void *userPtr);

/** Returns the user pointer associated with the image handle.
 *
 * If no user pointer was associated, it'll return a pointer to NULL.
 *
 * @param [in] handle Image to be queried.
 *
 * @param [in] outUserPtr Pointer to where the user pointer will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetUserPointer(NVCVImageHandle handle, void **outUserPtr);

/** Returns the underlying image type.
 *
 * @param [in] handle Image to be queried.
 *                    + Must not be NULL.
 * @param [out] type  The image type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetType(NVCVImageHandle handle, NVCVTypeImage *type);

/**
 * Get the image dimensions in pixels.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] width, height Where dimensions will be written to.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetSize(NVCVImageHandle handle, int32_t *width, int32_t *height);

/**
 * Get the image format.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] format Where the image format will be written to.
 *                    + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetFormat(NVCVImageHandle handle, NVCVImageFormat *fmt);

/**
 * Get the allocator associated with an image.
 *
 * This function creates a new reference to the allocator handle. The caller is responsible for freeing it
 * by calling nvcvAllocatorDecRef.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetAllocator(NVCVImageHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the image contents.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] data Where the image buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageExportData(NVCVImageHandle handle, NVCVImageData *data);

#ifdef __cplusplus
}
#endif

#endif // NVCV_IMAGE_H
