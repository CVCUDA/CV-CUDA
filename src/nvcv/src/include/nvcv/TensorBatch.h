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

/**
 * @file TensorBatch.h
 *
 * @brief Public C interface to NVCV representation of a batch of tensors.
 */

#ifndef NVCV_TENSORBATCH_H
#define NVCV_TENSORBATCH_H

#include "Export.h"
#include "Fwd.h"
#include "Image.h"
#include "Status.h"
#include "Tensor.h"
#include "TensorBatchData.h"
#include "TensorLayout.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NVCVTensorBatch *NVCVTensorBatchHandle;

/** Stores the requirements of an varshape tensor. */
typedef struct NVCVTensorBatchRequirementsRec
{
    /*< Maximum number of tensors in the batch */
    int32_t capacity;

    /*< Alignment/block size in bytes */
    int32_t alignBytes;

    /*< Tensor resource requirements. */
    NVCVRequirements mem;
} NVCVTensorBatchRequirements;

/** Calculates the resource requirements needed to create a tensor batch.
 *
 * @param [in] capacity Maximum number of images that fits in the image batch.
 *                      + Must be >= 1.
 *
 * @param [out] reqs  Where the image batch requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchCalcRequirements(int32_t capacity, NVCVTensorBatchRequirements *reqs);

NVCVStatus nvcvTensorBatchConstruct(const NVCVTensorBatchRequirements *req, NVCVAllocatorHandle alloc,
                                    NVCVTensorBatchHandle *outHandle);

NVCVStatus nvcvTensorBatchClear(NVCVTensorBatchHandle handle);

NVCVStatus nvcvTensorBatchPushTensors(NVCVTensorBatchHandle handle, const NVCVTensorHandle *tensors,
                                      int32_t numTensors);

/**
 * Pop tensors from the end of the image batch.
 *
 * @param[in] handle Tensor batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvTensorBatchConstruct.
 *
 * @param[in] numTensors Number of tensors to remove.
 *                       + Must be >= 1.
 *                       + Must be <= number of tensors in the batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_UNDERFLOW        Tried to remove more tensors that there are in the batch.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus nvcvTensorBatchPopTensors(NVCVTensorBatchHandle handle, int32_t numTensors);

/** Allocates multiple tensors and adds them to a TensorBatch
 *
 * This function allocates the storage for multiple tensors, creates the tensors and puts them in the batch.
 *
 * @param batch             a handle to the batch object to which the new tensors will be added
 * @param numTensors        the number of tensors to add
 * @param shapes            the shapes of the tensors to be added
 * @param strides           the strides of the tensors to be added; if NULL, the tensors are densely packed
 * @param tensorAlignment   the alignment, in bytes, of the base pointer of each tensor in the batch
 */
NVCVStatus nvcvTensorBatchPopulate(NVCVTensorBatchHandle batch, int32_t numTensors, const int64_t **shapes,
                                   const int64_t **strides /* optional, dense packing if NULL */,
                                   int32_t         tensorAlignment /* optional, use default if set to 0 */);

/** Gets handles to a range of tensors in the batch
 *
 * This function creates new references to the Tensor handles. The caller must release them by calling
 * nvcvTensorDecRef on all handles returned by this function.
 *
 * @param batch         a hadle to the batch object from which the tensors are exracted
 * @param index         the index of the first handle to get
 * @param outTensors    the array in which the handles are stored; it must have at least
 *                      numTensors handles
 * @param numTensors    the number of tensors to get
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Tried to retrieve more tensors that there are in the batch.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus nvcvTensorBatchGetTensors(NVCVTensorBatchHandle batch, int32_t index, NVCVTensorHandle *outTensors,
                                     int32_t numTensors);

/** Sets a range of tensors in the batch
 *
 * TBD: Do we need/want it?
 *      Should it also extend the bach if index + numTensors > size (but within capacity)?
 */
NVCVStatus nvcvTensorBatchSetTensors(NVCVTensorBatchHandle batch, int32_t index, const NVCVTensorHandle *tensors,
                                     int32_t numTensors);

NVCVStatus nvcvTensorBatchGetAllocator(NVCVTensorBatchHandle batch, NVCVAllocatorHandle *alloc);

NVCVStatus nvcvTensorBatchGetType(NVCVTensorBatchHandle batch, NVCVTensorBufferType *outType);

/**
 * Retrieve the tensor batch contents.
 *
 * @param[in] handle Tensor batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[in] stream CUDA stream where the export operation will execute.
 *
 * @param[out] data Where the tensor batch buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchExportData(NVCVTensorBatchHandle handle, CUstream stream,
                                                 NVCVTensorBatchData *data);

NVCVStatus nvcvTensorBatchGetNumTensors(NVCVTensorBatchHandle batch, int32_t *outNumTensors);

/** Decrements the reference count of an existing TensorBatch instance.
 *
 * The Tensor batch is destroyed when its reference count reaches zero.
 *
 * @param [in] handle       Tensor batch to be destroyed.
 *                          If NULL, no operation is performed, successfully.
 *                          + The handle must have been created with any of the nvcvTensorBatchXXXConstruct functions.
 *
 * @param [out] newRefCount The decremented reference count. If the return value is 0, the object was destroyed.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchDecRef(NVCVTensorBatchHandle handle, int32_t *newRefCount);

/** Increments the reference count of an Tensorbatch.
 *
 * @param [in] handle       Tensor batch to be retained.
 *
 * @param [out] newRefCount The incremented reference count.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchIncRef(NVCVTensorBatchHandle handle, int32_t *newRefCount);

/** Returns the current reference count of an Tensor batch
 *
 * @param [in] handle       The handle whose reference count is to be obtained.
 *
 * @param [out] outRefCount The reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchRefCount(NVCVTensorBatchHandle handle, int32_t *outRefCount);

/** Associates a user pointer to the Tensor batch handle.
 *
 * This pointer can be used to associate any kind of data with the Tensor batch object.
 *
 * @param [in] handle Tensor batch to be associated with the user pointer.
 *
 * @param [in] userPtr User pointer.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchSetUserPointer(NVCVTensorBatchHandle handle, void *userPtr);

/** Returns the user pointer associated with the Tensor batch handle.
 *
 * If no user pointer was associated, it'll return a pointer to NULL.
 *
 * @param [in] handle Tensor batch to be queried.
 *
 * @param [in] outUserPtr Pointer to where the user pointer will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchGetUserPointer(NVCVTensorBatchHandle handle, void **outUserPtr);

/** Returns the capacity of the Tensor batch handle.
 *
 * @param [in] handle Tensor batch to be queried.
 *
 * @param [in] outCapacityPtr Pointer to where the capacity will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchGetCapacity(NVCVTensorBatchHandle handle, int32_t *outCapacityPtr);

/** Returns the data type of the Tensor batch handle.
 *
 * Returns NVCV_DATA_TYPE_NONE for empty batches.
 *
 * @param [in] handle Tensor batch to be queried.
 *
 * @param [in] outDTypePtr Pointer to where the data type will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchGetDType(NVCVTensorBatchHandle handle, NVCVDataType *outDTypePtr);

/** Returns the layout of the Tensor batch handle.
 *
 * Returns the empty layout for empty batches.
 *
 * @param [in] handle Tensor batch to be queried.
 *
 * @param [in] outDTypePtr Pointer to where the layout will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchGetLayout(NVCVTensorBatchHandle handle, NVCVTensorLayout *outLayoutPtr);

/** Returns the rank of tensors in the tensor batch or -1 for an empty batch.
 *
 * @param [in] handle Tensor batch to be queried.
 *
 * @param [in] outRankPtr Pointer to where the rank will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorBatchGetRank(NVCVTensorBatchHandle handle, int32_t *outRankPtr);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORBATCH_H
