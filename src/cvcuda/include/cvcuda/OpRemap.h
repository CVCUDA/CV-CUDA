/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpRemap.h
 *
 * @brief Defines types and functions to handle the Remap operation.
 * @defgroup NVCV_C_ALGORITHM_REMAP Remap
 * @{
 */

#ifndef CVCUDA_REMAP_H
#define CVCUDA_REMAP_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/BorderType.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the Remap operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaRemapCreate(NVCVOperatorHandle *handle);

/** Executes the Remap operation on the given cuda stream. This operation does not wait for completion.
 *
 * Remap operation writes in the output an element read in the input in a position determined by the map.  First,
 * the Remap operation reads a map value at each output coordinate.  The map value is either an absolute position,
 * denormalized (NVCV_REMAP_ABSOLUTE) or normalized (NVCV_REMAP_ABSOLUTE_NORMALIZED), (x, y) or it is a relative
 * offset (NVCV_REMAP_RELATIVE_NORMALIZED) (dx, dy) from the normalized output coordinate to a normalized position
 * in the input.  The element at that input position is fetched from input and stored at the output position.
 *
 * @image html remap.svg
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input map:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC]
 *       Channels:       [2]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
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
 *       Channels      | Yes
 *       Width         | No
 *       Height        | No
 *       Samples       | Yes
 *
 *  Input/Map dependency
 *
 *       Property      |  Input == Map
 *      -------------- | -------------
 *       Data Layout   | No
 *       Data Type     | No
 *       Channels      | No
 *       Width         | No
 *       Height        | No
 *       Samples       | Yes or 1
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor to get values from.
 *                + Must not be NULL.
 *
 * @param [out] out Output tensor to set values to.
 *                  + Must not be NULL.
 *
 * @param [in] map Input tensor to get {x, y} (float2) absolute positions (either normalized or not) or relative
 *                 differences to map values from input to output.  For each normalized position in the output, the
 *                 map is read to get the map value used to fetch values from the input tensor and store them at
 *                 that normalized position.  The map value interpretation depends on \ref mapValueType.  The
 *                 input, output and map tensors can have different width and height, but the input and output
 *                 tensors must have the same number of samples.  The number of samples of the map can be either
 *                 equal to the input or one.  In case it is one, the same map is applied to all input samples.
 *                 + Must have float2 (2F32) data type.
 *                 + Must not be NULL.
 *
 * @param [in] inInterp Interpolation type to be used when fetching values from input tensor.
 *             + It may be one of the following interpolation types:
 *               + NVCV_INTERP_NEAREST
 *               + NVCV_INTERP_LINEAR
 *               + NVCV_INTERP_CUBIC
 *
 * @param [in] mapInterp Interpolation type to be used when fetching indices from map tensor.
 *             + It may be one of the following interpolation types:
 *               + NVCV_INTERP_NEAREST
 *               + NVCV_INTERP_LINEAR
 *               + NVCV_INTERP_CUBIC
 *
 * @param [in] mapValueType This determines how the values inside the map are interpreted.  If it is \ref
 *                          NVCV_REMAP_ABSOLUTE the map values are absolute, denormalized positions in the input
 *                          tensor to fetch values from.  If it is \ref NVCV_REMAP_ABSOLUTE_NORMALIZED the map
 *                          values are absolute, normalized positions in [-1, 1] range to fetch values from the
 *                          input tensor resolution agnostic.  If it is \ref NVCV_REMAP_RELATIVE_NORMALIZED the map
 *                          values are relative, normalized offsets to be applied to each output position to fetch
 *                          values from the input tensor, also resolution agnostic.
 *
 * @param [in] alignCorners The remap operation from output to input via the map is done in floating-point domain.
 *                          Set this flag to true so that they are aligned by the center points of their corner pixels.
 *                          Set it to false so that they are aligned by the corner points of their corner pixels.
 *
 * @param [in] border Border type to be used when fetching values from input tensor.
 *             + It may be one of the following border types:
 *               + NVCV_BORDER_CONSTANT
 *               + NVCV_BORDER_REPLICATE
 *               + NVCV_BORDER_REFLECT
 *               + NVCV_BORDER_WRAP
 *               + NVCV_BORDER_REFLECT101
 *
 * @param [in] borderValue Border value used when accessing outside values in input tensor for constant border.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaRemapSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                           NVCVTensorHandle out, NVCVTensorHandle map, NVCVInterpolationType inInterp,
                                           NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType,
                                           int8_t alignCorners, NVCVBorderType border, float4 borderValue);

/**
 * Executes the Remap operation on a batch of images.
 *
 * Apart from input and output image batches, all parameters are the same as \ref cvcudaRemapSubmit.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaRemapVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                   NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                   NVCVTensorHandle map, NVCVInterpolationType inInterp,
                                                   NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType,
                                                   int8_t alignCorners, NVCVBorderType border, float4 borderValue);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_REMAP_H */
