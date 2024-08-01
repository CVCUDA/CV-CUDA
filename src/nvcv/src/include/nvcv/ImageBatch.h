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
 * @file ImageBatch.h
 *
 * @brief Public C interface to NVCV image batch representation.
 */

#ifndef NVCV_IMAGEBATCH_H
#define NVCV_IMAGEBATCH_H

#include "Export.h"
#include "Fwd.h"
#include "Image.h"
#include "ImageBatchData.h"
#include "Status.h"
#include "alloc/Allocator.h"
#include "alloc/Requirements.h"
#include "detail/CudaFwd.h"

#include <nvcv/ImageFormat.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Underlying image batch type. */
typedef enum
{
    /** Batch of 2D images of different dimensions. */
    NVCV_TYPE_IMAGEBATCH_VARSHAPE,
    /** Batch of 2D images that have the same dimensions. */
    NVCV_TYPE_IMAGEBATCH_TENSOR,
    /** Image batch that wraps an user-allocated tensor buffer. */
    NVCV_TYPE_IMAGEBATCH_TENSOR_WRAPDATA,
} NVCVTypeImageBatch;

typedef struct NVCVImageBatch *NVCVImageBatchHandle;

/** Image batch data cleanup function type */
typedef void (*NVCVImageBatchDataCleanupFunc)(void *ctx, const NVCVImageBatchData *data);

/** Stores the requirements of an varshape image batch. */
typedef struct NVCVImageBatchVarShapeRequirementsRec
{
    int32_t capacity; /*< Maximum number of images stored. */

    int32_t          alignBytes; /*< Alignment/block size in bytes */
    NVCVRequirements mem;        /*< Image batch resource requirements. */
} NVCVImageBatchVarShapeRequirements;

/** Calculates the resource requirements needed to create a varshape image batch.
 *
 * @param [in] capacity Maximum number of images that fits in the image batch.
 *                      + Must be >= 0.
 *
 * @param [out] reqs  Where the image batch requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeCalcRequirements(int32_t                             capacity,
                                                              NVCVImageBatchVarShapeRequirements *reqs);

/** Constructs a varshape image batch instance with given requirements in the given storage.
 *
 * @param [in] reqs Image batch requirements. Must have been filled in by @ref nvcvImageBatchVarShapeCalcRequirements.
 *                  + Must not be NULL
 *
 * @param [in] alloc        Allocator to be used to allocate needed memory buffers.
 *                          The following resources are used:
 *                          - host memory
 *                          - cuda memory
 *                          If NULL, it'll use the internal default allocator.
 *                          + Allocator must not be destroyed while an image batch still refers to it.
 *
 * @param [out] handle      Where the image batch instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image batch instance.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeConstruct(const NVCVImageBatchVarShapeRequirements *reqs,
                                                       NVCVAllocatorHandle alloc, NVCVImageBatchHandle *handle);

/** Decrements the reference count of an existing image batch instance.
 *
 * The image batch is destroyed when its reference count reaches zero.
 *
 * If the image has type @ref NVCV_TYPE_IMAGEBATCH_TENSOR_WRAPDATA and has a cleanup function defined,
 * cleanup will be called.
 *
 * @param [in] handle       Image batch to be destroyed.
 *                          If NULL, no operation is performed, successfully.
 *                          + The handle must have been created with any of the nvcvImageBatchXXXConstruct functions.
 *
 * @param [out] newRefCount The decremented reference count. If the return value is 0, the object was destroyed.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchDecRef(NVCVImageBatchHandle handle, int *newRefCount);

/** Increments the reference count of an imagebatch.
 *
 * @param [in] handle       Image batch to be retained.
 *
 * @param [out] newRefCount The incremented reference count.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchIncRef(NVCVImageBatchHandle handle, int *newRefCount);

/** Returns the current reference count of an image batch
 *
 * @param [in] handle       The handle whose reference count is to be obtained.
 *
 * @param [out] newRefCount The reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchRefCount(NVCVImageBatchHandle handle, int *newRefCount);

/** Associates a user pointer to the image batch batch handle.
 *
 * This pointer can be used to associate any kind of data with the image batch object.
 *
 * @param [in] handle Image batch to be associated with the user pointer.
 *
 * @param [in] userPtr User pointer.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchSetUserPointer(NVCVImageBatchHandle handle, void *userPtr);

/** Returns the user pointer associated with the image batch handle.
 *
 * If no user pointer was associated, it'll return a pointer to NULL.
 *
 * @param [in] handle Image batch to be queried.
 *
 * @param [in] outUserPtr Pointer to where the user pointer will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetUserPointer(NVCVImageBatchHandle handle, void **outUserPtr);

/** Returns the underlying type of the image batch.
 *
 * @param [in] handle Image batch to be queried.
 *                    + Must not be NULL.
 * @param [out] type  The image batch type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetType(NVCVImageBatchHandle handle, NVCVTypeImageBatch *type);

/** Returns the capacity of the image batch.
 *
 * @param [in] handle Image batch to be queried.
 *                    + Must not be NULL.
 * @param [out] capacity  The capacity of the given image batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetCapacity(NVCVImageBatchHandle handle, int32_t *capacity);

/**
 * Get the allocator associated with an image batch.
 *
 * This function creates a new reference to the allocator handle. The caller is responsible for freeing it
 * by calling nvcvAllocatorDecRef.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetAllocator(NVCVImageBatchHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the number of images in the batch.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] numImages Where the number of images will be written to.
 *                       + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetNumImages(NVCVImageBatchHandle handle, int32_t *numImages);

/**
 * Retrieve the image batch contents.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[in] stream CUDA stream where the export operation will execute.
 *
 * @param[out] data Where the image batch buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchExportData(NVCVImageBatchHandle handle, CUstream stream, NVCVImageBatchData *data);

/**
 * Get the maximum size of the images in the batch.
 *
 * The maximum size of the image batch is defined as the maximum width and height
 * of all images in it. If the batch is empty, its maximum size is (0,0).
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] maxWidth,maxHeight Where the maximum width and height will be stored.
 *                                If NULL, corresponding value won't be returned.
 *                                + Both cannot be NULL
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeGetMaxSize(NVCVImageBatchHandle handle, int32_t *maxWidth,
                                                        int32_t *maxHeight);

/**
 * Get the unique format of the image batch.
 *
 * The unique format of an image batch is defined as being the format of all images in it,
 * if all images have the same format, or \ref NVCV_IMAGE_FORMAT_NONE otherwise.
 * If the batch is empty, its format is \ref NVCV_IMAGE_FORMAT_NONE.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] format Where the unique format will be written to.
 *                    If batch isn't empty and all images have the same format,
 *                    it'll return this format. Or else it returns \ref NVCV_IMAGE_FORMAT_NONE
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeGetUniqueFormat(NVCVImageBatchHandle handle, NVCVImageFormat *format);

/**
 * Push images to the end of the image batch.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] images Pointer to a buffer with the image handles to be added.
 *                   + Must not be NULL.
 *                   + Must point to an array of at least @p numImages image handles.
 *                   + The images must not be destroyed while they're being referenced by the image batch.
 *                   + Image format must indicate a pitch-linear memory layout.
 *
 * @param[in] numImages Number of images in the @p images array.
 *                      + Must be >= 1.
 *                      + Final number of images must not exceed the image batch capacity.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Image batch capacity exceeded.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapePushImages(NVCVImageBatchHandle handle, const NVCVImageHandle *images,
                                                        int32_t numImages);

/**
 * Callback function used to push images.
 *
 * Every time it is called, it'll return the next image in a sequence.
 * It'll return NULL after the last image is returned.
 *
 * @param [in] ctx User context passed by the user.
 * @returns The next NVCVImage handle in the sequence, or NULL if there's no more
 *          images to return.
 */
typedef NVCVImageHandle (*NVCVPushImageFunc)(void *ctx);

/**
 * Push to the end of the batch the images returned from a callback function.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] cbGetImage Function that returns each image that is pushed to the batch.
 *                       It'll keep being called until it return NULL, meaning that there are no more
 *                       images to be returned.
 *                       + Must not be NULL.
 *                       + It must return NULL before the capacity of the batch is exceeded.
 *                       + The images returned must not be destroyed while they're being referenced by the image batch.
 *                       + Image format must indicate a pitch-linear memory layout.
 *
 * @param[in] ctxCallback Pointer passed to the callback function unchanged.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Image batch capacity exceeded.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapePushImagesCallback(NVCVImageBatchHandle handle,
                                                                NVCVPushImageFunc cbPushImage, void *ctxCallback);

/**
 * Pop images from the end of the image batch.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] numImages Number of images in the @p images array.
 *                      + Must be >= 1.
 *                      + Must be <= number of images in the batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_UNDERFLOW        Tried to remove more images that there are in the batch.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapePopImages(NVCVImageBatchHandle handle, int32_t numImages);

/**
 * Clear the contents of the varshape image batch.
 *
 * It sets its size to 0.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeClear(NVCVImageBatchHandle handle);

/**
 * Retrieve the image handles from the varshape image batch.
 *
 * This function creates new references to the image handles. The caller must release them by calling
 * nvcvImageDecRef on all handles returned by this function.
 *
 * @param[in] handle Varshape image batch to be queried
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] begOffset Index offset of the first image to be retrieved.
 *                      To retrieve starting from the first image, pass 0.
 *                      + Must be < number of images in the batch.
 *
 * @param[out] outImages Where the image handles will be written to.
 *                       The caller must free the handles by calling nvcvImageDecRef.
 *
 * @param[in] numImages Number of images to be retrieved.
 *                      + Must be >= 0.
 *                      + Must be begOffset+numImages <= number of images in the batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Tried to retrieve more images that there are in the batch.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeGetImages(NVCVImageBatchHandle handle, int32_t begIndex,
                                                       NVCVImageHandle *outImages, int32_t numImages);

#ifdef __cplusplus
}
#endif

#endif // NVCV_IMAGEBATCH_H
