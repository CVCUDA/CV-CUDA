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
 * @file OpCropFlipNormalizeReformat.h
 *
 * @brief Defines types and functions to handle the CropFlipNormalizeReformat operation.
 * @defgroup NVCV_C_ALGORITHM_CROP_FLIP_NORMALIZE_REFORMAT CropFlipNormalizeReformat
 * @{
 */

#ifndef CVCUDA_CROP_FLIP_NORMALIZE_REFORMAT_H
#define CVCUDA_CROP_FLIP_NORMALIZE_REFORMAT_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/BorderType.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Rect.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the SliceFlipNormalize operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCropFlipNormalizeReformatCreate(NVCVOperatorHandle *handle);

/** Executes the CropFlipNormalizeReformat operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 * This operation performs the following steps:
 *    1. Pad and Crop the input image to the specified rectangle.
 *    2. Flip the cropped image horizontally and/or vertically.
 *    3. Normalize the flipped image using the provided base and scale.
 *    4. Convert the normalized image to the specified output data type.
 *    5. Reformat the normalized image to the specified output layout.
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
 * For the Crop operation, the input image is cropped to the specified rectangle. The rectangle is [crop_x, crop_y, crop_x + crop_width, crop_y + crop_height].
 * Where the crop_x and crop_y can be negative, in which case the image is padded. The padding method is specified by the borderMode parameter.
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, KCHW]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | Yes
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC, kNCHW, KCHW]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | Yes
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Input/Output dependency
 *
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes/No
 *      Data Type     | Yes/No
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | Yes/No
 *      Height        | Yes/No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input image batch.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] cropRect crop rectangle tensor which has shape of [batch_size, 1, 1, 4] in reference to the input tensor.
 *               The crop value of [crop_x, crop_y, crop_width, crop_height] stored in the final dimension of the crop tensor
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @param [in] borderValue Border value to be used for constant border mode \p NVCV_BORDER_CONSTANT.
 *
 * @param [in] flipCode a tensor flag to specify how to flip the array; 0 means flipping
 *      around the x-axis and positive value (for example, 1) means flipping
 *      around y-axis. Negative value (for example, -1) means flipping around
 *      both axes.
 *
 * @param [in] base Tensor providing base values for normalization.
 *
 * @param [in] scale Tensor providing scale values for normalization.
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

CVCUDA_PUBLIC NVCVStatus cvcudaCropFlipNormalizeReformatSubmit(
    NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
    NVCVTensorHandle cropRect, NVCVBorderType borderMode, float borderValue, NVCVTensorHandle flipCode,
    NVCVTensorHandle base, NVCVTensorHandle scale, float global_scale, float shift, float epsilon, uint32_t flags);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_CROP_FLIP_NORMALIZE_REFORMAT_H */
