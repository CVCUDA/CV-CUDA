/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpHQResize.h
 *
 * @brief Defines types and functions to handle the HQResize operation.
 * @defgroup NVCV_C_ALGORITHM_HQ_RESIZE HQ Resize
 * @{
 */

#ifndef CVCUDA_HQ_RESIZE_H
#define CVCUDA_HQ_RESIZE_H

#include "Operator.h"
#include "Types.h"
#include "Workspace.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Rect.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define NVCV_HQ_RESIZE_MAX_RESIZED_NDIM (3)

typedef struct
{
    int32_t extent[NVCV_HQ_RESIZE_MAX_RESIZED_NDIM];
    int32_t ndim;
    int32_t numChannels;
} HQResizeTensorShapeI;

typedef struct
{
    HQResizeTensorShapeI *shape;
    int32_t               size;        // the number of valid elements in the `shape` array
    int32_t               ndim;        // the number of spatial extents in each `shapes` element
    int32_t               numChannels; // the number of innermost channels, -1 if they differ between samples
} HQResizeTensorShapesI;

typedef struct
{
    float lo[NVCV_HQ_RESIZE_MAX_RESIZED_NDIM];
    float hi[NVCV_HQ_RESIZE_MAX_RESIZED_NDIM];
} HQResizeRoiF;

typedef struct
{
    int32_t       size; // the number of valid elements in the `roi` array
    int32_t       ndim; // the number of valid extents in each `roi` element
    HQResizeRoiF *roi;
} HQResizeRoisF;

/** Constructs an instance of the HQResize operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeCreate(NVCVOperatorHandle *handle);

/** Calculates the workspace requirements for Tensor input/ouput.
 *
 * @param [in] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] batchSize The number of samples in the tensor (the size of N extent).
 *
 * @param [in] inputShape The HW or DHW extents of the input tensor, the number of resized extents,
 *                        and the number of channels.
 *                        Supported number of resizes extents are 2 and 3.
 *                        For ndim = 2, a tensor of layout (N)HW(C) is expected to be processed,
 *                        for ndim = 3, a tensor of layout (N)DHW(C) is expected to be processed.
 *
 * @param [in] outputShape The HW or DHW extents of the output tensor and the number of channels.
 *                         The number of extents and channels must be the same as in inputShape.
 *
 * @param [in] minInterpolation The type of interpolation to be used when downsampling an extent
 *                              (i.e. when output extent is smaller than the corresponding input extent).
 *
 * @param [in] magInterpolation The type of interpolation to be used when upsampling an extent
 *                              (i.e. when output extent is bigger than the corresponding input extent).
 *
 * @param [in] antialias Whether to use antialiasing when downsampling.
 *
 * @param [in] roi Optional region of interest for the input, in (D)HW layout.
 *
 * @param [out] reqOut The pointer for workspace requirements struct that will be filled by the call.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or one of the arguments is out of range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeTensorGetWorkspaceRequirements(NVCVOperatorHandle handle, int batchSize,
                                                                      const HQResizeTensorShapeI  inputShape,
                                                                      const HQResizeTensorShapeI  outputShape,
                                                                      const NVCVInterpolationType minInterpolation,
                                                                      const NVCVInterpolationType magInterpolation,
                                                                      bool antialias, const HQResizeRoiF *roi,
                                                                      NVCVWorkspaceRequirements *reqOut);

/** Calculates the workspace requirements for TensorBatch/ImageBatchVarShape input/ouput.
 *
 * @param [in] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] batchSize The number of samples in the tensor batch/image batch.
 *
 * @param [in] inputShapes The list of shapes (HW or DHW extents) in the input batch,
 *                         the number of channels, and the number of extents to be resampled (2 or 3).
 *                         The number of channels can be specified once for the whole batch or each sample
 *                         separately.
 *
 * @param [in] outputShapes The list of shapes (HW or DHW extents) in the output batch,
 *                          the number of channels, and the number of extents to be resampled (2 or 3).
 *                          The number of channels must match the number of channels in the input.
 *
 * @param [in] minInterpolation The type of interpolation to be used when downsampling an extent
 *                              (i.e. when output extent is smaller than the corresponding input extent).
 *
 * @param [in] magInterpolation The type of interpolation to be used when upsampling an extent
 *                              (i.e. when output extent is bigger than the corresponding input extent).
 *
 * @param [in] antialias Whether to use antialiasing when downsampling.
 *
 * @param [in] roi Optional region of interest for the input, in (D)HW layout. The roi can be described
 *                 as a list for each sample or contain a single element to be used for all the samples
 *                 in the batch.
 *
 * @param [out] reqOut The pointer for workspace requirements struct that will be filled by the call.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or one of the arguments is out of range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeTensorBatchGetWorkspaceRequirements(NVCVOperatorHandle handle, int batchSize,
                                                                           const HQResizeTensorShapesI inputShapes,
                                                                           const HQResizeTensorShapesI outputShapes,
                                                                           const NVCVInterpolationType minInterpolation,
                                                                           const NVCVInterpolationType magInterpolation,
                                                                           bool antialias, const HQResizeRoisF roi,
                                                                           NVCVWorkspaceRequirements *reqOut);

/** Calculates the upper bound for workspace requirements. The workspace that meets the returned
 * requirements can be used with any call to the operator as long as: the input dimensionality
 * (2 or 3) matches, the number of samples does not exceed the maxBatchSize, and all the input
 * and output shapes do not exceed the maxShape in any extent (including number of channels).
 *
 * @param [in] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] maxBatchSize The maximal number of samples in the tensor/tensor batch/image batch.
 *
 * @param [in] maxShape The maximal shape of any input or output sample. The number of channels must
 *                      be an upper bound for number of channels in any sample.
 *
 * @param [out] reqOut The pointer for workspace requirements struct that will be filled by the call.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or one of the arguments is out of range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeGetMaxWorkspaceRequirements(NVCVOperatorHandle handle, int maxBatchSize,
                                                                   const HQResizeTensorShapeI maxShape,
                                                                   NVCVWorkspaceRequirements *reqOut);

/** Executes the HQResize operation on the given cuda stream. This operation does not wait for completion.
 *
 *  Limitations:
 *
 *  Input, Output:
 *       Data Layout:         NVCV_TENSOR_[N][D]HW[C]
 *
 *       Number of channels:  Positive integer
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | No (output can be the same or float32).
 *       Channels      | Yes
 *       Width         | No
 *       Height        | No
 *       Samples       | Yes
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] workspace The workspace with memory for intermediate results. The requirements for a given input
 *                       can be acquired with a call to `cvcudaHQResizeTensorGetWorkspaceRequirements` or
 *                       `cvcudaHQResizeGetMaxWorkspaceRequirements`.
 *
 * @param [in] in The input tensor with (N)(D)HW(C) layout.
 *
 * @param [in] out The output tensor with the same layout, number of samples and channels as the in tensor.
 *
 * @param [in] minInterpolation The type of interpolation to be used when downsampling an extent
 *                              (i.e. when output extent is smaller than the corresponding input extent).
 *                              Supported interpolation formats are: `NVCV_INTERP_NEAREST`, `NVCV_INTERP_LINEAR`,
 *                              `NVCV_INTERP_CUBIC`, `NVCV_INTERP_LANCZOS`, and `NVCV_INTERP_GAUSSIAN`.
 *
 * @param [in] magInterpolation The type of interpolation to be used when upsampling an extent
 *                              (i.e. when output extent is bigger than the corresponding input extent).
 *                               Supported interpolation formats are: `NVCV_INTERP_NEAREST`, `NVCV_INTERP_LINEAR`,
 *                              `NVCV_INTERP_CUBIC`, `NVCV_INTERP_LANCZOS`, and `NVCV_INTERP_GAUSSIAN`.
 *
 * @param [in] antialias Whether to use antialiasing when downsampling. The value is ignored for
 *                       `minInterpolation = NVCV_INTERP_NEAREST`.
 *
 * @param [in] roi Optional region of interest for the input, in (D)HW layout.
 *                 If, for some axis, the low bound is bigger than the high bound,
 *                 the image is flipped in that dimension.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                              const NVCVWorkspace *workspace, NVCVTensorHandle in, NVCVTensorHandle out,
                                              const NVCVInterpolationType minInterpolation,
                                              const NVCVInterpolationType magInterpolation, bool antialias,
                                              const HQResizeRoiF *roi);

/** Executes the HQResize operation on the given cuda stream. This operation does not wait for completion.
 *
 *  Limitations:
 *
 *  Input, Output:
 *       Data Layout:    NVCV_TENSOR_HWC
 *
 *       Number of channels: [1, 2, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | No (output can be the same or float32).
 *       Channels      | Yes
 *       Width         | No
 *       Height        | No
 *       Samples       | Yes
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] workspace The workspace with memory for intermediate results. The requirements for a given input
 *                       can be acquired with a call to `cvcudaHQResizeTensorBatchGetWorkspaceRequirements` or
 *                       `cvcudaHQResizeGetMaxWorkspaceRequirements`.
 *
 * @param [in] in The ImageBatchVarShape batch of input samples.
 *
 * @param [in] out The ImageBatchVarShape batch of output samples.
 *
 * @param [in] minInterpolation The type of interpolation to be used when downsampling an extent
 *                              (i.e. when output extent is smaller than the corresponding input extent).
 *                              Supported interpolation formats are: `NVCV_INTERP_NEAREST`, `NVCV_INTERP_LINEAR`,
 *                              `NVCV_INTERP_CUBIC`, `NVCV_INTERP_LANCZOS`, and `NVCV_INTERP_GAUSSIAN`.
 *
 * @param [in] magInterpolation The type of interpolation to be used when upsampling an extent
 *                              (i.e. when output extent is bigger than the corresponding input extent).
 *                              Supported interpolation formats are: `NVCV_INTERP_NEAREST`, `NVCV_INTERP_LINEAR`,
 *                              `NVCV_INTERP_CUBIC`, `NVCV_INTERP_LANCZOS`, and `NVCV_INTERP_GAUSSIAN`.
 *
 * @param [in] antialias Whether to use antialiasing when downsampling. The value is ignored for
 *                       `minInterpolation = NVCV_INTERP_NEAREST`.
 *
 * @param [in] roi Optional region of interest for the input, in (D)HW layout. The roi can be described
 *                 as a list of elements for each sample or a list containing a single element to be used
 *                 for all the samples in the batch. If, for some axis, the low bound is bigger than
 *                 the high bound, the image is flipped in that dimension.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeImageBatchSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                        const NVCVWorkspace *workspace, NVCVImageBatchHandle in,
                                                        NVCVImageBatchHandle        out,
                                                        const NVCVInterpolationType minInterpolation,
                                                        const NVCVInterpolationType magInterpolation, bool antialias,
                                                        const HQResizeRoisF roi);

/** Executes the HQResize operation on the given cuda stream. This operation does not wait for completion.
 *
 *  Limitations:
 *
 *  Input, Output:
 *       Data Layout:         NVCV_TENSOR_[D]HW[C]
 *
 *       Number of channels:  Positive integer
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | No (output can be the same or float32).
 *       Channels      | Yes
 *       Width         | No
 *       Height        | No
 *       Samples       | Yes
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] workspace The workspace with memory for intermediate results. The requirements for a given input
 *                       can be acquired with a call to `cvcudaHQResizeTensorBatchGetWorkspaceRequirements` or
 *                       `cvcudaHQResizeGetMaxWorkspaceRequirements`.
 *
 * @param [in] in The TensorBatch of input samples.
 *
 * @param [in] out The TensorBatch batch of output samples.
 *
 * @param [in] minInterpolation The type of interpolation to be used when downsampling an extent
 *                              (i.e. when output extent is smaller than the corresponding input extent).
 *                              Supported interpolation formats are: `NVCV_INTERP_NEAREST`, `NVCV_INTERP_LINEAR`,
 *                              `NVCV_INTERP_CUBIC`, `NVCV_INTERP_LANCZOS`, and `NVCV_INTERP_GAUSSIAN`.
 *
 * @param [in] magInterpolation The type of interpolation to be used when upsampling an extent
 *                              (i.e. when output extent is bigger than the corresponding input extent).
 *                              Supported interpolation formats are: `NVCV_INTERP_NEAREST`, `NVCV_INTERP_LINEAR`,
 *                              `NVCV_INTERP_CUBIC`, `NVCV_INTERP_LANCZOS`, and `NVCV_INTERP_GAUSSIAN`.
 *
 * @param [in] antialias Whether to use antialiasing when downsampling. The value is ignored for
 *                       `minInterpolation = NVCV_INTERP_NEAREST`.
 *
 * @param [in] roi Optional region of interest for the input, in (D)HW layout. The roi can be described
 *                 as a list of elements for each sample or a list containing a single element to be used
 *                 for all the samples in the batch. If, for some axis, the low bound is bigger than
 *                 the high bound, the image is flipped in that dimension.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHQResizeTensorBatchSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                         const NVCVWorkspace *workspace, NVCVTensorBatchHandle in,
                                                         NVCVTensorBatchHandle       out,
                                                         const NVCVInterpolationType minInterpolation,
                                                         const NVCVInterpolationType magInterpolation, bool antialias,
                                                         const HQResizeRoisF roi);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_HQ_RESIZE_H */
