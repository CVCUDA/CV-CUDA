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
 * @file Array.h
 *
 * @brief Public C interface to NVCV array representation.
 */

#ifndef NVCV_ARRAY_H
#define NVCV_ARRAY_H

#include "ArrayData.h"
#include "Export.h"
#include "Fwd.h"
#include "Status.h"
#include "alloc/Allocator.h"
#include "alloc/Requirements.h"
#include "detail/CudaFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NVCVArray *NVCVArrayHandle;

/** Array data cleanup function type */
typedef void (*NVCVArrayDataCleanupFunc)(void *ctx, const NVCVArrayData *data);

/** Stores the requirements of a array. */
typedef struct NVCVArrayRequirementsRec
{
    /*< Type of each element */
    NVCVDataType dtype;

    /*< Capacity of the array */
    int64_t capacity;

    /*< Distance in bytes between each data unit. */
    int64_t stride;

    /*< Alignment/block size in bytes */
    int32_t alignBytes;

    /*< Array resource requirements. */
    NVCVRequirements mem;
} NVCVArrayRequirements;

/** Calculates the resource requirements needed to create an array.
 *
 * @param [in] capacity Capacity of the array.
 *
 * @param [in] dtype Type of array's elements.
 *
 * @param [in] alignment Alignment, in bytes, of the array elements. If 0, use a
 *                       default suitable for optimized memory access. The used
 *                       alignment is at least the given value. If not 0, it
 *                       must be a positive integer power-of-two (i.e. 2^n).
 *
 * @param [out] reqs Pointer to the array requirements object. Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayCalcRequirements(int64_t capacity, NVCVDataType dtype, int32_t alignment,
                                                 NVCVArrayRequirements *reqs);

/** Calculates the resource requirements needed to create an array with target resource.
 *
 * @param [in] capacity Capacity of the array.
 *
 * @param [in] dtype Type of array's elements.
 *
 * @param [in] alignment Alignment, in bytes, of the array elements. If 0, use a
 *                       default suitable for optimized memory access. The used
 *                       alignment is at least the given value. If not 0, it
 *                       must be a positive integer power-of-two (i.e. 2^n).
 *
 * @param [in] target The target compute resource for where memory allocation for
 *                    the data contained within the array will occure.
 *
 * @param [out] reqs Pointer to the array requirements object. Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayCalcRequirementsWithTarget(int64_t capacity, NVCVDataType dtype, int32_t alignment,
                                                           NVCVResourceType target, NVCVArrayRequirements *reqs);

/** Constructs an array instance with given requirements.
 *
 * @param [in] reqs Array requirements. Must have been filled by one of the
 *                  nvcvArrayCalcRequirements functions.
 *                  + Must not be NULL
 *
 * @param [in] alloc Allocator to be used to allocate needed memory buffers.
 *
 * @param [out] handle Where the array instance handle will be written to.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the array instance.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayConstruct(const NVCVArrayRequirements *reqs, NVCVAllocatorHandle alloc,
                                          NVCVArrayHandle *handle);

/** Constructs an array instance with given requirements on the target resource.
 *
 * @param [in] reqs Array requirements. Must have been filled by one of the
 *                  nvcvArrayCalcRequirements functions.
 *                  + Must not be NULL
 *
 * @param [in] alloc Allocator to be used to allocate needed memory buffers.
 *
 * @param [in] target The target compute resource for where memory allocation for
 *                    the data contained within the array will occure.
 *
 * @param [out] handle Where the array instance handle will be written to.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the array instance.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayConstructWithTarget(const NVCVArrayRequirements *reqs, NVCVAllocatorHandle alloc,
                                                    NVCVResourceType target, NVCVArrayHandle *handle);

/** Wraps an existing array buffer into an NVCV array instance constructed in given storage
 *
 * It allows for interoperation of external array representations with NVCV.
 *
 * @param [in] data Array contents.
 *
 * @param [in] cleanup Cleanup function to be called when the array is destroyed
 *                     via @ref nvcvArrayDecRef.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @param [out] handle Where the array instance handle will be written to.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the array.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayWrapDataConstruct(const NVCVArrayData *data, NVCVArrayDataCleanupFunc cleanup,
                                                  void *ctxCleanup, NVCVArrayHandle *handle);

/** Decrements the reference count of an existing array instance.
 *
 * The array is destroyed when its reference count reaches zero.
 *
 * If the array is wrapping external data and a cleanup function has been defined, defined,
 * it will be called.
 *
 * @note The array object must not be in use in current and future operations.
 *
 * @param [in] handle       Array to be destroyed.
 *
 * @param [out] newRefCount The decremented reference count. If the return value is 0, the object was destroyed.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayDecRef(NVCVArrayHandle handle, int *newRefCount);

/** Increments the reference count of an array.
 *
 * @param [in] handle       Array to be retained.
 *
 * @param [out] newRefCount The incremented reference count.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayIncRef(NVCVArrayHandle handle, int *newRefCount);

/** Returns the current reference count of an array
 *
 * @param [in] handle       The handle whose reference count is to be obtained.
 *
 * @param [out] newRefCount The reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayRefCount(NVCVArrayHandle handle, int *newRefCount);

/** Associates a user pointer to the array handle.
 *
 * This pointer can be used to associate any kind of data with the array object.
 *
 * @param [in] handle Array to be associated with the user pointer.
 *
 * @param [in] userPtr User pointer.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArraySetUserPointer(NVCVArrayHandle handle, void *userPtr);

/** Returns the user pointer associated with the array handle.
 *
 * If no user pointer was associated, it'll return a pointer to NULL.
 *
 * @param [in] handle Array to be queried.
 *
 * @param [in] outUserPtr Pointer to where the user pointer will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayGetUserPointer(NVCVArrayHandle handle, void **outUserPtr);

/**
 * Get the type of the array elements (its data type).
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] type Where the type will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayGetDataType(NVCVArrayHandle handle, NVCVDataType *type);

/**
 * Get the allocator associated with the array.
 *
 * This function creates a new reference to the allocator handle. The caller is responsible for freeing it
 * by calling nvcvAllocatorDecRef.
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayGetAllocator(NVCVArrayHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the array contents.
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] data Where the array buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayExportData(NVCVArrayHandle handle, NVCVArrayData *data);

/**
 * Retrieve the array legnth.
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *                   + Must have been created by @ref nvcvArrayConstruct.
 *
 * @param[out] length Pointer to the output length of the array.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayGetLength(NVCVArrayHandle handle, int64_t *length);

/**
 * Retrieve the array capacity.
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *                   + Must have been created by @ref nvcvArrayConstruct.
 *
 * @param[out] capacity Pointer to the output capacity of the array.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayGetCapacity(NVCVArrayHandle handle, int64_t *capacity);

/**
 * Resizes the array legnth to the specified length up to the capacity.
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *                   + Must have been created by @ref nvcvArrayConstruct.
 *
 * @param[in] length The input length of the array.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayResize(NVCVArrayHandle handle, int64_t length);

/**
 * Retrieve the array target.
 *
 * @param[in] handle Array to be queried.
 *                   + Must not be NULL.
 *                   + Must have been created by @ref nvcvArrayConstruct.
 *
 * @param[out] target Pointer to the output target of the array.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvArrayGetTarget(NVCVArrayHandle handle, NVCVResourceType *target);

#ifdef __cplusplus
}
#endif

#endif // NVCV_ARRAY_H
