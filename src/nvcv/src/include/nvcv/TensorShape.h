/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TENSORSHAPE_H
#define NVCV_TENSORSHAPE_H

#include "TensorLayout.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Permute the input shape with given layout to a different layout.
 *
 * @param[in] srcLayout The layout of the source shape.
 *
 * @param[in] srcShape The shape to be permuted.
 *                     + Must not be NULL.
 *                     + Number of dimensions must be equal to dimensions in @p srcLayout
 *
 * @param[in] dstLayout The layout of the destination shape.
 *
 * @param[out] dstShape Where the permutation will be written to.
 *                      + Must not be NULL.
 *                      + Number of dimensions must be equal to dimensions in @p dstLayout
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorShapePermute(NVCVTensorLayout srcLayout, const int64_t *srcShape,
                                              NVCVTensorLayout dstLayout, int64_t *dstShape);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORSHAPE_H
