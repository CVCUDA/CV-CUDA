/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpPillowResize.h
 *
 * @brief Defines types and functions to handle the pillow resize operation.
 * @defgroup NVCV_C_ALGORITHM_PILLOW_RESIZE Pillow Resize
 * @{
 */

#ifndef CVCUDA_PILLOW_RESIZE_H
#define CVCUDA_PILLOW_RESIZE_H

#include "Operator.h"
#include "Types.h"
#include "Workspace.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Size.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the pillow resize operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] fmt Image format
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPillowResizeCreate(NVCVOperatorHandle *handle);

/** Calculates the upper bounds of buffer sizes required to run the operator
 *
 * @param [in] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxBatchSize Maximum batchsize used in this operator.
 * @param [in] maxWidth Maximum input and output image width.
 * @param [in] maxHeight Maximum input and output image height.
 * @param [in] fmt Image format
 * @param [out] reqOut Requirements for the operator's workspace
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or one of the arguments is out of range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPillowResizeGetWorkspaceRequirements(NVCVOperatorHandle handle, int maxBatchSize,
                                                                    int32_t maxInWidth, int32_t maxInHeight,
                                                                    int32_t maxOutWidth, int32_t maxOutHeight,
                                                                    NVCVImageFormat            fmt,
                                                                    NVCVWorkspaceRequirements *reqOut);

/** Calculates the buffer sizes required to run the operator
 *
 * @param [in] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] batchSize The number of images
 * @param [in] inputSizes The sizes of the input images
 * @param [in] outputSizes The sizes of the output images
 * @param [in] fmt Image format
 * @param [out] reqOut Requirements for the operator's workspace
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or one of the arguments is out of range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPillowResizeVarShapeGetWorkspaceRequirements(NVCVOperatorHandle handle, int batchSize,
                                                                            const NVCVSize2D          *inputSizesWH,
                                                                            const NVCVSize2D          *outputSizesWH,
                                                                            NVCVImageFormat            fmt,
                                                                            NVCVWorkspaceRequirements *reqOut);

/** Executes the pillow resize operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
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
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
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
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | Yes
 *       Number        | Yes
 *       Channels      | Yes
 *       Width         | No
 *       Height        | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor / image batch.
 *
 * @param [out] out output tensor / image batch.
 *
 * @param [in] interpolation Interpolation method to be used, see \ref NVCVInterpolationType for more details.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaPillowResizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                  const NVCVWorkspace *workspace, NVCVTensorHandle in,
                                                  NVCVTensorHandle out, NVCVInterpolationType interpolation);

CVCUDA_PUBLIC NVCVStatus cvcudaPillowResizeVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                          const NVCVWorkspace *workspace, NVCVImageBatchHandle in,
                                                          NVCVImageBatchHandle  out,
                                                          NVCVInterpolationType interpolation);
/** @} */
#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_PILLOW_RESIZE_H */
