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
 * @file OpErase.h
 *
 * @brief Defines types and functions to handle the erase operation.
 * @defgroup NVCV_C_ALGORITHM_ERASE Erase
 * @{
 */

#ifndef CVCUDA_ERASE_H
#define CVCUDA_ERASE_H

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

/** Constructs and an instance of the erase operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] max_num_erasing_area the maximum number of areas that will be erased.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
*/

CVCUDA_PUBLIC NVCVStatus cvcudaEraseCreate(NVCVOperatorHandle *handle, int32_t max_num_erasing_area);

/** Executes the erase operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (planar kNCHW/kCHW: [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       16bit Float    | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (planar kNCHW/kCHW: [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       16bit Float    | No
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
 *       Width         | Yes
 *       Height        | Yes
 *
 *  In-place execution is selected by passing the same handle for `in` and `out`.
 *  There is no separate in-place parameter. When `in` and `out` are different handles,
 *  the operation copies `in` to `out` before applying the erase areas. Passing different
 *  handles that partially alias the same storage is not a supported aliasing mode.
 *
 *  anchor Tensor
 *
 *      Must be 'N' (dim = 1) with N = number of erasing area.
 *      Data Type must be 32bit Signed.
 *      DataType must be TYPE_2S32.
 *
 *  erasing Tensor
 *
 *      Must be 'N' (dim = 1) with N = number of erasing area.
 *      Data Type must be 32bit Signed.
 *      DataType must be TYPE_3S32.
 *
 *  imgIdx Tensor
 *
 *      Must be 'N' (dim = 1) with N = number of erasing area.
 *      Data Type must be 32bit Signed.
 *      DataType must be TYPE_S32.
 *
 *  values Tensor
 *
 *      Must be 'N' (dim = 1) with W = number of erasing area * 4.
 *      Data Type must be 32bit Float.
 *      DataType must be TYPE_F32.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor / image batch.
 *
 * @param [out] out output tensor / image batch.
 *
 * @param [in] anchor an array of size num_erasing_area that gives the x coordinate and y coordinate of the top left point in the erasing areas.
 *
 * @param [in] erasing an array of size num_erasing_area that gives the widths of the erasing areas, the heights of the erasing areas and
 *              integers in range 0-15, each of whose bits indicates whether or not the corresponding channel need to be erased.
 *
 * @param [in] values an array of size num_erasing_area*4 that gives the filling value for each erase area.
 *
 * @param [in] imgIdx an array of size num_erasing_area that maps a erase area idx to img idx in the batch.
 *
 * @param [in] random an boolean for random op.
 *
 * @param [in] seed random seed for random filling erase area.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
*/
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaEraseSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                           NVCVTensorHandle out, NVCVTensorHandle anchor, NVCVTensorHandle erasing,
                                           NVCVTensorHandle values, NVCVTensorHandle imgIdx, int8_t random,
                                           uint32_t seed);

CVCUDA_PUBLIC NVCVStatus cvcudaEraseVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                   NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                   NVCVTensorHandle anchor, NVCVTensorHandle erasing,
                                                   NVCVTensorHandle values, NVCVTensorHandle imgIdx, int8_t random,
                                                   uint32_t seed);

/** Executes a torchvision-compatible single-region erase on the given CUDA stream.
 *  This operation does not wait for completion.
 *
 *  The operation is equivalent to the following assignment on the logical planar view of the input:
 *
 *      out = in.clone()
 *      out[..., i:i+h, j:j+w] = values
 *
 *  Passing the same tensor handle for @p in and @p out performs the assignment in place. Negative,
 *  empty, and out-of-bounds regions follow Python's step-one slice rules. For interleaved layouts,
 *  @p values is still broadcast against the logical planar shape `(N, C, h, w)` or `(C, h, w)`.
 *
 *  Reference: torchvision.transforms.v2.functional.erase.
 *
 *  This overload is independent of the `max_num_erasing_area` capacity supplied to
 *  #cvcudaEraseCreate.
 *
 *  Limitations:
 *
 *  Input / Output:
 *       Container:      Tensor only
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       64bit Unsigned | Yes
 *       64bit Signed   | Yes
 *       16bit Float    | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *  Values:
 *       Rank must not exceed four. Dimensions are right-aligned against the logical `(N, C, H, W)`
 *       shape, with `N = 1` for unbatched inputs, and each must equal the corresponding
 *       selected-region dimension or be one for broadcasting. The dtype must match the input dtype
 *       or be 32-bit float. Float-to-integer conversion matches CUDA PyTorch assignment semantics,
 *       including for non-finite and out-of-range values.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image tensor.
 * @param [out] out Output image tensor. Shape, layout, and dtype must match @p in.
 * @param [in] i Vertical slice start.
 * @param [in] j Horizontal slice start.
 * @param [in] h Vertical slice extent used to form the exclusive stop `i + h`.
 * @param [in] w Horizontal slice extent used to form the exclusive stop `j + w`.
 * @param [in] values Value tensor to broadcast into the selected region.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT A tensor or parameter violates the contract above.
 * @retval #NVCV_SUCCESS Operation submitted successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaEraseRegionSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, int64_t i, int64_t j, int64_t h, int64_t w,
                                                 NVCVTensorHandle values);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_ERASE_H */
