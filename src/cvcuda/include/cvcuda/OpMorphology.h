/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpMorphology.h
 *
 * @brief Defines types and functions to handle the morphology operation.
 * @defgroup NVCV_C_ALGORITHM_MORPHOLOGY Morphology
 * @{
 */

#ifndef CVCUDA_MORPHOLOGY_H
#define CVCUDA_MORPHOLOGY_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the Morphology operator.
 *
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxVarShapeBatchSize maximum batch size for var shape operator, can be 0 if VarShape is not used.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMorphologyCreate(NVCVOperatorHandle *handle, const int32_t maxVarShapeBatchSize);

/**
 * Executes the morphology operation of Dilates/Erodes on images
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Input/Output dependency
 *
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | Yes
 *      Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] morphType Type of operation to performs Erode/Dilate. \ref NVCVMorphologyType.
 *
 * @param [in] maskWidth Width of the mask to use (set heigh/width to -1 for default of 3,3).
 *
 * @param [in] maskHeight Height of the mask to use (set heigh/width to -1 for default of 3,3).
 *
 * @param [in] anchorX X-offset of the kernel to use (set anchorX/anchorY to -1 for center of kernel).
 *
 * @param [in] anchorY Y-offset of the kernel to use (set anchorX/anchorY to -1 for center of kernel).
 *
 * @param [in] iteration  Number of times to execute the operation, typically set to 1. Setting to higher than 1 is equivelent
 *                       of increasing the kernel mask by (mask_width - 1, mask_height -1) for every iteration.

 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \ref NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMorphologySubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                NVCVTensorHandle out, NVCVMorphologyType morphType, int32_t maskWidth,
                                                int32_t maskHeight, int32_t anchorX, int32_t anchorY, int32_t iteration,
                                                const NVCVBorderType borderMode);

/**
 * Executes the morphology operation of Dilates/Erodes on images, using an array of variable shape images.
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Input/Output dependency
 *
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | Yes (per image in/out pair)
 *      Height        | Yes (per image in/out pair)
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input variable shape tensor.
 *
 * @param [out] out Output variable shape tensor.
 *
 * @param [in] morphType Type of operation to perform (Erode/Dilate). \ref NVCVMorphologyType.
 *
 * @param [in] masks  1D Tensor of NVCV_DATA_TYPE_2S32 mask W/H pairs, where the 1st pair is for image 0, second for image 1, etc.
 *                    Setting values to -1,-1 will create a default 3,3 mask.
 *                    (Note after the operation the tensor values may be modified by kernel)
 *
 * @param [in] anchors 1D Tensor of NVCV_DATA_TYPE_2S32 X/Y pairs, where the 1st pair is for image 0, second for image 1, etc
 *                      Setting values to -1,-1 will anchor the kernel at the center.
 *                      (Note after the operation the tensor values may be modified by kernel)
 *
 * @param [in] iteration Number of times to execute the operation, typically set to 1. Setting to higher than 1 is equivelent
 *                       of increasing the kernel mask by (mask_width - 1, mask_height -1) for every iteration.

 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \ref NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMorphologyVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                        NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                        NVCVMorphologyType morphType, NVCVTensorHandle masks,
                                                        NVCVTensorHandle anchors, int32_t iteration,
                                                        const NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_MORPHOLOGY */
