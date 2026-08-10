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
 * @file OpNormalize.h
 *
 * @brief Defines types and functions to handle the normalize operation.
 * @defgroup NVCV_C_ALGORITHM_NORMALIZE Normalize
 * @{
 */

#ifndef CVCUDA_NORMALIZE_H
#define CVCUDA_NORMALIZE_H

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

// @brief Flag to be used by normalize operation to indicate scale is standard deviation instead.
#define CVCUDA_NORMALIZE_SCALE_IS_STDDEV (1 << 0)

/** Constructs and an instance of the normalize operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaNormalizeCreate(NVCVOperatorHandle *handle);

/**
 * Executes the normalize operation on the given cuda stream. This operation does not
 * wait for completion.
 *
 * Data normalization is done using externally provided base (typically: mean or min) and scale (typically
 * reciprocal of standard deviation or 1/(max-min)). The normalization follows the formula:
 * ```
 * out[data_idx] = (in[data_idx] - base[param_idx]) * scale[param_idx] * global_scale + shift
 * ```
 * Where `data_idx` is a position in the data tensor (in, out) and `param_idx` is a position
 * in the base and scale tensors (see below for details). The two additional constants,
 * `global_scale` and `shift` can be used to adjust the result to the dynamic range and resolution
 * of the output type.
 *
 * The `scale` parameter may also be interpreted as standard deviation - in that case, its
 * reciprocal is used and optionally, a regularizing term is added to the variance.
 * ```
 * m = 1 / sqrt(square(stddev[param_idx]) + epsilon)
 * out[data_idx] = (in[data_idx] - mean[param_idx]) * m * global_scale + shift
 * ```
 *
 * `param_idx` is calculated as follows (where axis = N,H,W,C):
 * ```
 * param_idx[axis] = param_shape[axis] == 1 ? 0 : data_idx[axis]
 * ```
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      16bit Float    | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
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
 * Scale/Base Tensor:
 *
 *      Scale and Base may be a tensor the same shape as the input/output tensors, or it can be a scalar each dimension.
 *      Dimensions with extent 1 are broadcast over the corresponding input axis.
 *
 *      For varshape variant with interleaved input, scale and base may represent either a scalar with shape [1,1,1,1],
 *      or a tensor with shape [1,1,1,C], where C is the number of channels in the input format. Varshape scale and
 *      base are always broadcast over the image batch.
 *
 *      For planar (kNCHW / kCHW) tensor input, scale and base must use a planar layout that matches the input
 *      for non-scalar parameters (e.g. [1,C,1,1] for per-channel parameters). Tensor input supports the same
 *      per-axis broadcasting rule above. For planar varshape input, scale and base must be batch and spatial
 *      broadcasted, so use [1,1,1,1] for scalar parameters or [1,C,1,1] for per-channel parameters. A truly
 *      scalar parameter tensor (numChannels == 1) is also accepted in either layout because both encode the same
 *      bytes.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [in] base Base tensor.
 *
 * @param [in] scale Scale tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] global_scale Additional scale value to be used in addition to scale.
 *
 * @param [in] shift Additional bias value to be used in addition to base.
 *
 * @param [in] epsilon Epsilon to use when \p CVCUDA_NORMALIZE_SCALE_IS_STDDEV flag is set as a regularizing term to be
 *                     added to variance.
 *
 * @param [in] flags Algorithm flags, use \p CVCUDA_NORMALIZE_SCALE_IS_STDDEV if scale passed as argument
 *                   is standard deviation instead or 0 if it is scaling.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaNormalizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle base, NVCVTensorHandle scale, NVCVTensorHandle out,
                                               float global_scale, float shift, float epsilon, uint32_t flags);

CVCUDA_PUBLIC NVCVStatus cvcudaNormalizeVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle in, NVCVTensorHandle base,
                                                       NVCVTensorHandle scale, NVCVImageBatchHandle out,
                                                       float global_scale, float shift, float epsilon, uint32_t flags);
/** @} */

/**
 * Executes the normalize operation on the given cuda stream, taking base and scale by value instead of
 * as tensors. This operation does not wait for completion.
 *
 * This is a "tensor-free" variant of #cvcudaNormalizeSubmit: the base (typically: mean or min) and
 * scale (typically reciprocal of standard deviation or 1/(max-min)) values are passed by value in
 * \p base and \p scale (each a `float4` holding up to four channels) instead of being read from device
 * tensors, so the caller does not allocate or upload any parameter tensor (avoiding a host-to-device
 * copy). The normalization follows the same formula:
 * ```
 * out[data_idx] = (in[data_idx] - base[channel]) * scale[channel] * global_scale + shift
 * ```
 * The two additional constants, `global_scale` and `shift`, can be used to adjust the result to the
 * dynamic range and resolution of the output type.
 *
 * The `scale` parameter may also be interpreted as standard deviation - in that case, its reciprocal is
 * used and optionally, a regularizing term is added to the variance.
 * ```
 * m = 1 / sqrt(square(stddev[channel]) + epsilon)
 * out[data_idx] = (in[data_idx] - mean[channel]) * m * global_scale + shift
 * ```
 *
 * `baseChannels` and `scaleChannels` give how many leading lanes of the corresponding `float4` are
 * meaningful:
 * ```
 * channel_idx = count == 1 ? 0 : channel
 * ```
 * A count of 1 broadcasts a single value to every channel; a count equal to the input channel count
 * supplies one value per channel. Any other count is rejected. Results are bit-identical to
 * #cvcudaNormalizeSubmit for the same values.
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      16bit Float    | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
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
 * Base/Scale:
 *
 *      Base and scale are supplied by value (up to four channels packed in a float4). Each may represent
 *      a scalar broadcast over every channel (baseChannels / scaleChannels == 1) or a per-channel value
 *      (count equal to the number of channels in the input). Interleaved (kNHWC / kHWC) and planar
 *      (kNCHW / kCHW) tensor input are both supported. Unlike #cvcudaNormalizeSubmit, per-axis spatial
 *      (width/height) parameters are not available; use #cvcudaNormalizeSubmit with parameter tensors
 *      for spatial parameters. CUDA launch limits require N <= 65535 for interleaved input and
 *      N*C <= 65535 for planar input.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [in] base Base values, passed by value (up to four channels in a float4).
 *
 * @param [in] scale Scale values, passed by value (up to four channels in a float4).
 *
 * @param [in] baseChannels Number of meaningful \p base lanes: 1 (broadcast) or the input channel count.
 *
 * @param [in] scaleChannels Number of meaningful \p scale lanes: 1 (broadcast) or the input channel count.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] global_scale Additional scale value to be used in addition to scale.
 *
 * @param [in] shift Additional bias value to be used in addition to base.
 *
 * @param [in] epsilon Epsilon to use when \p CVCUDA_NORMALIZE_SCALE_IS_STDDEV flag is set as a regularizing term to be
 *                     added to variance.
 *
 * @param [in] flags Algorithm flags, use \p CVCUDA_NORMALIZE_SCALE_IS_STDDEV if scale passed as argument
 *                   is standard deviation instead or 0 if it is scaling.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaNormalizeScalarSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                     NVCVTensorHandle in, float4 base, float4 scale,
                                                     int32_t baseChannels, int32_t scaleChannels, NVCVTensorHandle out,
                                                     float global_scale, float shift, float epsilon, uint32_t flags);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_NORMALIZE_H */
