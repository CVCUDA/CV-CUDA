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
 * @file OpMinMaxLoc.h
 *
 * @brief Defines types and functions to handle the MinMaxLoc operation.
 * @defgroup NVCV_C_ALGORITHM_MINMAXLOC MinMaxLoc
 * @{
 */

#ifndef CVCUDA_MINMAXLOC_H
#define CVCUDA_MINMAXLOC_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the MinMaxLoc operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMinMaxLocCreate(NVCVOperatorHandle *handle);

/** Executes the MinMaxLoc operation on the given cuda stream. This operation does not wait for completion.
 *
 * @note The MinMaxLoc operation does not guarantee deterministic output.  The order of output minimum or maximum
 *       locations found is in no particular order and might differ in different runs.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [HWC, NHWC, CHW, NCHW]
 *       Channels:       [1]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.  The expected layout is [HWC] or [NHWC] or [CHW] or [NCHW], where N is the number
 *                of samples, i.e. images with height H and width W and channels C, inside the tensor.
 *
 * @param [out] minVal Output tensor to store minimum values found in the input tensor.  The expected layout is [N]
 *                     or [NC], meaning rank-1 or rank-2 tensor with first dimension as number of samples N, and a
 *                     potential last dimension C with number of channels.
 *                     + It must have the same number of samples as input tensor.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must have data type S32/U32/F32/F64: for input data type S8/S16/S32 use S32;
 *                       for U8/U16/U32 use U32; for all other data types use same data type as input tensor.
 *                     + It may be NULL to disregard finding minimum values and locations.
 *
 * @param [out] minLoc Output tensor to store minimum locations found in the input tensor.  The expected layout is
 *                     [NM] or [NMC], meaning rank-2 or rank-3 tensor with first dimension as number of samples N,
 *                     second dimension as maximum number of locations M to be found, and a potential last
 *                     dimension C with number of channels.
 *                     + It must have the same number of samples as input tensor.
 *                     + It must have 2S32 data type to store (x, y) sample locations, i.e. (x) coordinate on the
 *                       width W and (y) coordinate on the height H of a sample image.
 *                     + It must have a number of elements M equal to the maximum allowed number of locations
 *                       to be found per sample image (see below).
 *                     + It must be NULL if minVal is NULL (see above).
 *
 * @param [out] numMin Output tensor to store the number of minimum locations found in the input tensor.  The
 *                     expected layout is [N] or [NC], meaning rank-1 or rank-2 tensor with first dimension as
 *                     number of samples N, and a potential last dimension C with number of channels.
 *                     + It must have the same number of samples as input tensor.
 *                     + It must have S32 data type to store number of minima found.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must be NULL if minVal is NULL (see above).
 *
 * @param [out] maxVal Output tensor to store maximum values found in the input tensor.  The expected layout is [N]
 *                     or [NC], meaning rank-1 or rank-2 tensor with first dimension as number of samples N, and a
 *                     potential last dimension C with number of channels.
 *                     + It must have the same number of samples as input tensor.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must have data type S32/U32/F32/F64: for input data type S8/S16/S32 use S32;
 *                       for U8/U16/U32 use U32; for all other data types use same data type as input tensor.
 *                     + It may be NULL to disregard finding maximum values and locations.
 *
 * @param [out] maxLoc Output tensor to store maximum locations found in the input tensor.  The expected layout is
 *                     [NM] or [NMC], meaning rank-2 or rank-3 tensor with first dimension as number of samples N,
 *                     second dimension as maximum number of locations M to be found, and a potential last
 *                     dimension C with number of channels.
 *                     + It must have the same number of samples as input tensor.
 *                     + It must have 2S32 data type to store (x, y) sample locations, i.e. (x) coordinate on the
 *                       width W and (y) coordinate on the height H of a sample image.
 *                     + It must have a number of elements M equal to the maximum allowed number of locations
 *                       to be found per sample image (see below).
 *                     + It must be NULL if maxVal is NULL (see above).
 *
 * @param [out] numMax Output tensor to store the number of maximum locations found in the input tensor.  The
 *                     expected layout is [N] or [NC], meaning rank-1 or rank-2 tensor with first dimension as
 *                     number of samples N, and a potential last dimension C with number of channels.
 *                     + It must have the same number of samples as input tensor.
 *                     + It must have S32 data type to store number of maxima found.
 *                     + It must have one element per sample, i.e. number of channels must be 1 in a [NC] tensor.
 *                     + It must be NULL if maxVal is NULL (see above).
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMinMaxLocSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle minVal, NVCVTensorHandle minLoc,
                                               NVCVTensorHandle numMin, NVCVTensorHandle maxVal,
                                               NVCVTensorHandle maxLoc, NVCVTensorHandle numMax);

/**
 * Executes the MinMaxLoc operation on a batch of images.
 *
 * Apart from input image batch, all parameters are the same as \ref cvcudaMinMaxLocSubmit.
 *
 * @param[in] in Input image batch.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMinMaxLocVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle in, NVCVTensorHandle minVal,
                                                       NVCVTensorHandle minLoc, NVCVTensorHandle numMin,
                                                       NVCVTensorHandle maxVal, NVCVTensorHandle maxLoc,
                                                       NVCVTensorHandle numMax);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_MINMAXLOC_H */
