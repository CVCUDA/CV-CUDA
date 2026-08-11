/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpGammaContrast.h
 *
 * @brief Defines types and functions to handle the GammaContrast operation.
 * @defgroup NVCV_C_ALGORITHM_GAMMA_CONTRAST Gamma Contrast
 * @{
 */

#ifndef CVCUDA_GAMMA_CONTRAST_H
#define CVCUDA_GAMMA_CONTRAST_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/RoundMode.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the GammaContrast.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxVarShapeBatchSize is the positive maximum batch size for the operator.
 *
 * @param [in] maxVarShapeChannelCount is the positive maximum channel count for the operator.
 *
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or an input limit is not positive.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaGammaContrastCreate(NVCVOperatorHandle *handle, const int32_t maxVarShapeBatchSize,
                                                   const int32_t maxVarShapeChannelCount);

/** Executes the GammaContrast operation on the given cuda stream.  This operation does not wait for completion.
 *
 * Limitations:
 *
 * Input image batch:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels:       [1, 2, 3, 4]  (planar layouts kNCHW/kCHW exclude 2 channels)
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      16bit Float    | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output image batch:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels:       [1, 2, 3, 4]  (planar layouts kNCHW/kCHW exclude 2 channels)
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      16bit Float    | No
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
 * @param [in] in Input image batch.
 *
 * @param [out] out Output image batch.
 *
 * @param [in] gamma 1D tensor with the gamma value for each image / image channel.
 *
 *
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaGammaContrastVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                           NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                           NVCVTensorHandle gamma);

/** Executes the GammaContrast operation on a tensor input/output (interleaved (N)HWC or planar
 * (N)CHW layout). This operation does not wait for completion.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in  Input tensor.
 * @param [out] out Output tensor.
 * @param [in] gamma 1D tensor with the gamma value for each sample / sample channel.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaGammaContrastSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                   NVCVTensorHandle out, NVCVTensorHandle gamma);

/** Executes the GammaContrast operation on a tensor input/output using host-scalar gamma and gain
 * (interleaved (N)HWC or planar (N)CHW layout). This operation does not wait for completion.
 *
 * Unlike #cvcudaGammaContrastSubmit, gamma and gain are plain host floats passed by value into the
 * kernel launch, so no gamma tensor is allocated and no host->device copy is performed. The same
 * gamma/gain is applied to every sample and channel, computing out = gain * in**gamma (the
 * torchvision adjust_gamma formula). With gain == 1.0f and roundMode == #NVCV_ROUND_NEAREST, the
 * result is bit-exact with #cvcudaGammaContrastSubmit fed a gamma tensor filled with the same value.
 * Data layout, channel, and data-type support match #cvcudaGammaContrastSubmit.
 *
 * Because no gamma scratch is used, the max-batch/max-channel capacities given to
 * #cvcudaGammaContrastCreate do not constrain this function (they only bound the gamma-tensor
 * staging used by the tensor and var-shape submits). gamma and gain are not range-validated;
 * a negative gamma follows powf semantics (NaN for fractional exponents of negative inputs).
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in  Input tensor.
 * @param [out] out Output tensor.
 * @param [in] gamma Host-scalar gamma exponent applied to every sample/channel.
 * @param [in] gain  Host-scalar output gain applied to every sample/channel.
 * @param [in] roundMode Rounding mode used for integer outputs, cf. \ref NVCVRoundMode.
 *                       Use #NVCV_ROUND_NEAREST to preserve the tensor-gamma behavior or
 *                       #NVCV_ROUND_TRUNCATE to truncate toward zero. Floating-point outputs are unaffected.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaGammaContrastScalarSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                         NVCVTensorHandle in, NVCVTensorHandle out, float gamma,
                                                         float gain, NVCVRoundMode roundMode);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* CVCUDA_GAMMA_CONTRAST_H */
