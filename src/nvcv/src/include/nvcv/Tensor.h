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
 * @file Tensor.h
 *
 * @brief Public C interface to NVCV tensor representation.
 */

#ifndef NVCV_TENSOR_H
#define NVCV_TENSOR_H

#include "Export.h"
#include "Fwd.h"
#include "Image.h"
#include "Status.h"
#include "TensorData.h"
#include "alloc/Allocator.h"
#include "alloc/Requirements.h"
#include "detail/CudaFwd.h"

#include <nvcv/ImageFormat.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NVCVTensor *NVCVTensorHandle;

/** Tensor data cleanup function type */
typedef void (*NVCVTensorDataCleanupFunc)(void *ctx, const NVCVTensorData *data);

/** Stores the requirements of an varshape tensor. */
typedef struct NVCVTensorRequirementsRec
{
    /*< Type of each element */
    NVCVDataType dtype;

    /*< Tensor dimension layout.
     * It's optional. If layout not available, set it to NVCV_TENSOR_NONE. */
    NVCVTensorLayout layout;

    /*< Rank, a.k.a number of dimensions */
    int32_t rank;

    /*< Shape of the tensor */
    int64_t shape[NVCV_TENSOR_MAX_RANK];

    /*< Distance in bytes between each element of a given dimension. */
    int64_t strides[NVCV_TENSOR_MAX_RANK];

    /*< Alignment/block size in bytes */
    int32_t alignBytes;

    /*< Tensor resource requirements. */
    NVCVRequirements mem;
} NVCVTensorRequirements;

/** Calculates the resource requirements needed to create a tensor with given shape.
 *
 * @param [in] rank Rank of the tensor (its number of dimensions).
 *
 * @param [in] shape Pointer to array with tensor shape.
 *                   It must contain at least 'rank' elements.
 *
 * @param [in] dtype Type of tensor's elements.
 *
 * @param [in] layout Tensor layout.
 *                    Pass NVCV_TENSOR_NONE is layout is not available.
 *                    + Layout rank must be @p rank.
 *
 * @param [in] baseAddrAlignment Alignment, in bytes, of the requested memory buffer.
 *                               If 0, use a default suitable for optimized memory access.
 *                               The used alignment is at least the given value.
 *                               + If different from 0, it must be a power-of-two.
 *
 * @param [in] rowAddrAlignment Alignment, in bytes, of the start of each second-to-last dimension address.
 *                              If 0, use a default suitable for optimized memory access.
 *                              The used alignment is at least the given value.
 *                              Pass 1 for creation of fully packed tensors, i.e., no padding between dimensions.
 *                              + If different from 0, it must be a power-of-two.
 *
 * @param [out] reqs  Where the tensor requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorCalcRequirements(int32_t rank, const int64_t *shape, NVCVDataType dtype,
                                                  NVCVTensorLayout layout, int32_t baseAddrAlignment,
                                                  int32_t rowAddrAlignment, NVCVTensorRequirements *reqs);

/** Calculates the resource requirements needed to create a tensor that holds N images.
 *
 * @param [in] numImages Number of images in the tensor.
 *                       + Must be >= 1.
 *
 * @param [in] width,height Dimensions of each image in the tensor.
 *                          + Must be >= 1x1
 *
 * @param [in] format Format of the images in the tensor.
 *                    + Must not be \ref NVCV_IMAGE_FORMAT_NONE.
 *                    + All planes in must have the same number of channels.
 *                    + No subsampled planes are allowed.
 *
 * @param [in] baseAddrAlignment Alignment, in bytes, of the requested memory buffer.
 *                               If 0, use a default suitable for optimized memory access.
 *                               The used alignment is at least the given value.
 *                               + If different from 0, it must be a power-of-two.
 *
 * @param [in] rowAddrAlignment Alignment, in bytes, of the start of each second-to-last dimension address.
 *                              If 0, use a default suitable for optimized memory access.
 *                              The used alignment is at least the given value.
 *                              Pass 1 for creation of fully packed tensors, i.e., no padding between dimensions.
 *                              + If different from 0, it must be a power-of-two.
 *
 * @param [out] reqs  Where the tensor requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorCalcRequirementsForImages(int32_t numImages, int32_t width, int32_t height,
                                                           NVCVImageFormat format, int32_t baseAddrAlignment,
                                                           int32_t rowAddrAlignment, NVCVTensorRequirements *reqs);

/** Constructs a tensor instance with given requirements in the given storage.
 *
 * @param [in] reqs Tensor requirements. Must have been filled by one of the nvcvTensorCalcRequirements functions.
 *                  + Must not be NULL
 *
 * @param [in] alloc Allocator to be used to allocate needed memory buffers.
 *                   The following resources are used:
 *                   - cuda memory
 *                   If NULL, it'll use the internal default allocator.
 *                   + Allocator must not be destroyed while an tensor still refers to it.
 *
 * @param [out] handle Where the tensor instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the tensor instance.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorConstruct(const NVCVTensorRequirements *reqs, NVCVAllocatorHandle alloc,
                                           NVCVTensorHandle *handle);

/** Wraps an existing tensor buffer into an NVCV tensor instance constructed in given storage
 *
 * It allows for interoperation of external tensor representations with NVCV.
 *
 * @param [in] data Tensor contents.
 *                  + Must not be NULL.
 *                  + Allowed buffer types:
 *                    - \ref NVCV_TENSOR_BUFFER_STRIDED_CUDA
 *
 * @param [in] cleanup Cleanup function to be called when the tensor is destroyed
 *                     via @ref nvcvTensorDecRef.
 *                     If NULL, no cleanup function is defined.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @param [out] handle Where the tensor instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the tensor.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorWrapDataConstruct(const NVCVTensorData *data, NVCVTensorDataCleanupFunc cleanup,
                                                   void *ctxCleanup, NVCVTensorHandle *handle);

/** Wraps an existing NVCV image into an NVCV tensor instance constructed in given storage
 *
 * Tensor layout is inferred from image characteristics.
 *
 * - CHW: For multi-planar images, one-channel image or single-planar images
 *        whose channels have different bit depths.
 * - HWC: For packed (single plane) images with channels with same bit depths
 *
 * When image is single-planar with channels with different bit depths, it's considered
 * to have one channel, but with element type (type) with multiple components with the
 * required bit depth each.
 *
 * The tensor created by this function holds a reference to the image. The image handle can be safely released
 * and the image object will be kept alive at least as long as the tensor that wraps it.
 *
 * @param [in] img Image to be wrapped
 *                 + Must not be NULL.
 *                 + Must not have subsampled planes
 *                 + All planes must have the same data type.
 *                 + Distance in memory between consecutive planes must be > 0.
 *                 + Row pitch of all planes must be the same.
 *                 + Image must not be destroyed while it's referenced by a tensor.
 *                 + Image contents must be cuda-accessible
 *                 + Image format must be pitch-linear.
 *                 + All planes must have the same dimensions.
 *
 * @param [out] handle Where the tensor instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorWrapImageConstruct(NVCVImageHandle img, NVCVTensorHandle *handle);

/** Decrements the reference count of an existing tensor instance.
 *
 * The tensor is destroyed when its reference count reaches zero.
 *
 * If the tensor is wrapping external data and a cleanup function has been defined, defined,
 * it will be called.
 *
 * @note The tensor object must not be in use in current and future operations.
 *
 * @param [in] handle       Tensor to be destroyed.
 *                          If NULL, no operation is performed, successfully.
 *                          + The handle must have been created with any of the nvcvTensorConstruct functions.
 *
 * @param [out] newRefCount The decremented reference count. If the return value is 0, the object was destroyed.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorDecRef(NVCVTensorHandle handle, int *newRefCount);

/** Increments the reference count of an tensor.
 *
 * @param [in] handle       Tensor to be retained.
 *
 * @param [out] newRefCount The incremented reference count.
 *                          Can be NULL, if the caller isn't interested in the new reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorIncRef(NVCVTensorHandle handle, int *newRefCount);

/** Returns the current reference count of an tensor
 *
 * @param [in] handle       The handle whose reference count is to be obtained.
 *
 * @param [out] newRefCount The reference count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT The handle is invalid
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorRefCount(NVCVTensorHandle handle, int *newRefCount);

/** Associates a user pointer to the tensor handle.
 *
 * This pointer can be used to associate any kind of data with the tensor object.
 *
 * @param [in] handle Tensor to be associated with the user pointer.
 *
 * @param [in] userPtr User pointer.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorSetUserPointer(NVCVTensorHandle handle, void *userPtr);

/** Returns the user pointer associated with the tensor handle.
 *
 * If no user pointer was associated, it'll return a pointer to NULL.
 *
 * @param [in] handle Tensor to be queried.
 *
 * @param [in] outUserPtr Pointer to where the user pointer will be stored.
 *                        + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetUserPointer(NVCVTensorHandle handle, void **outUserPtr);

/**
 * Get the type of the tensor elements (its data type).
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] type Where the type will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetDataType(NVCVTensorHandle handle, NVCVDataType *type);

/**
 * Get the tensor layout
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] layout Where the tensor layout will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetLayout(NVCVTensorHandle handle, NVCVTensorLayout *layout);

/**
 * Get the allocator associated with the tensor.
 *
 * This function creates a new reference to the allocator handle. The caller is responsible for freeing it
 * by calling nvcvAllocatorDecRef.
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetAllocator(NVCVTensorHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the tensor contents.
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] data Where the tensor buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorExportData(NVCVTensorHandle handle, NVCVTensorData *data);

/**
 * Retrieve the tensor shape.
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *                   + Must have been created by @ref nvcvTensorConstruct.
 *
 * @param[in,out] rank Number of elements in output shape buffer.
 *                     When function returns, it stores the actual tensor rank..
 *                     Set it to NVCV_TENSOR_MAX_RANK to return the full shape in @shape.
 *                     Set it to 0 if only tensor's rank must be returned.
 *
 * @param[out] shape Where the tensor shape will be written to.
 *                   Must point to a buffer with @p rank elements.
 *                   Elements above actual number of dimensions will be set to 1.
 *                   + If NULL, @p rank must be 0.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetShape(NVCVTensorHandle handle, int32_t *rank, int64_t *shape);

/**
 * Creates a view of a tensor with a different shape and layout.
 *
 * @param[in] handle Tensor to create a view from.
 *                   + Must not be NULL.
 *
 * @param[in] rank Number of elements in the shape buffer argument.
 *                   + Must be a number between 1 and NVCV_TENSOR_MAX_RANK
 *
 * @param[in] shape New shape.
 *                   Must point to a buffer with @p rank elements.
 *                   Elements above actual number of dimensions will be ignored.
 *
 * @param[in] layout New layout.
 *                   Must have @p rank elements or be empty.
 *
 * @param [out] handle Where the tensor instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is invalid.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorReshape(NVCVTensorHandle handle, int32_t rank, const int64_t *shape,
                                         NVCVTensorLayout layout, NVCVTensorHandle *out_handle);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSOR_H
