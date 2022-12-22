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
 * @file OpLaplacian.h
 *
 * @brief Defines types and functions to handle the Laplacian operation.
 * @defgroup NVCV_C_ALGORITHM_LAPLACIAN Laplacian
 * @{
 */

#ifndef CVCUDA_LAPLACIAN_H
#define CVCUDA_LAPLACIAN_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the Laplacian.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaLaplacianCreate(NVCVOperatorHandle *handle);

/** Executes the Laplacian operation on the given cuda stream.  This operation does not wait for completion.
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
 * @param [in] ksize Aperture size used to compute the second-derivative filters, it can be 1 or 3.
 *
 * @param [in] scale Scale factor for the Laplacian values (use 1 for no scale).
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaLaplacianSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle out, int32_t ksize, float scale,
                                               NVCVBorderType borderMode);

/**
 * Executes the Laplacian operation on a batch of images.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 * @param[in] ksize Aperture size to compute second-derivative filters, either 1 or 3 per image, as a 1D Tensor of int.
 *                  + Must be of pixel type NVCV_DATA_TYPE_S32
 * @param[in] scale Scale factor Laplacian values as a 1D Tensor of float.
 *                  + Must be of pixel type NVCV_DATA_TYPE_F32
 * @param[in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaLaplacianVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                       NVCVTensorHandle ksize, NVCVTensorHandle scale,
                                                       NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_LAPLACIAN_H */
