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
 * @file OpCvtColor.h
 *
 * @brief Defines types and functions to handle the CvtColor (convert color) operation.
 * @defgroup NVCV_C_ALGORITHM_CVTCOLOR CvtColor
 * @{
 */

#ifndef CVCUDA_CVTCOLOR_H
#define CVCUDA_CVTCOLOR_H

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

/** Constructs an instance of the CvtColor (convert color operation).
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCvtColorCreate(NVCVOperatorHandle *handle);

/** Executes the CvtColor (convert color) operation on the given cuda stream.  This operation does not wait for completion.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] code Color conversion code, \ref NVCVColorConversionCode.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCvtColorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                              NVCVTensorHandle out, NVCVColorConversionCode code);

CVCUDA_PUBLIC NVCVStatus cvcudaCvtColorVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                      NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                      NVCVColorConversionCode code);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_CVTCOLOR_H */
