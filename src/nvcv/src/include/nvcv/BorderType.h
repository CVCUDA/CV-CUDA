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

#ifndef NVCV_BORDER_TYPE_H
#define NVCV_BORDER_TYPE_H

#ifdef __cplusplus
extern "C"
{
#endif

/* @brief Flag to choose the border mode to be used
 *
 * This enum is used to specify the type of border to be used for functions
 * that require it. Here are the different types:
 * - `NVCV_BORDER_CONSTANT`: Uses a constant value for borders.
 * - `NVCV_BORDER_REPLICATE`: Replicates the last element for borders.
 * - `NVCV_BORDER_REFLECT`: Reflects the border elements.
 * - `NVCV_BORDER_WRAP`: Wraps the border elements.
 * - `NVCV_BORDER_REFLECT101`: Reflects the border elements, excluding the last element.
 * - `NVCV_BORDER_CONSTANT_COVERAGE`: Blends the constant value in by coverage.
 */
typedef enum
{
    NVCV_BORDER_CONSTANT   = 0, ///< Uses a constant value for borders.
    NVCV_BORDER_REPLICATE  = 1, ///< Replicates the last element for borders.
    NVCV_BORDER_REFLECT    = 2, ///< Reflects the border elements.
    NVCV_BORDER_WRAP       = 3, ///< Wraps the border elements.
    NVCV_BORDER_REFLECT101 = 4, ///< Reflects the border elements, excluding the last element.

    /** Blends the constant value in by how much of the sampling footprint fell inside the tensor.
     *
     * With `coverage` the sum of the interpolation weights whose taps landed inside, the sampled
     * value is `inside * coverage + borderValue * (1 - coverage)`, where `inside` is the
     * interpolation evaluated with a zero border. `NVCV_BORDER_CONSTANT` instead feeds
     * `borderValue` straight into the interpolation, giving `inside + borderValue * (1 - coverage)`;
     * the two agree wherever the footprint is fully inside and differ only at the edges.
     *
     * Supported interpolation modes are NEAREST, LINEAR and CUBIC; AREA is not supported.
     * Nearest-neighbour interpolation takes `borderValue` when `coverage < 0.5`. Cubic coverage
     * uses signed interpolation weights without clamping their sum to [0, 1].
     *
     * `borderValue` is blended at full precision and the result is rounded once, so a fractional
     * value keeps its fraction on an integer image.
     *
     * This mode describes how to sample, so only operators that both interpolate and take a
     * border mode accept it: WarpAffine, WarpPerspective and Remap, and only on their tensor
     * paths. Their variable-shape paths and operators that merely read neighbours, such as the
     * filters and CopyMakeBorder, reject it rather than silently treating it as
     * `NVCV_BORDER_CONSTANT`.
     */
    NVCV_BORDER_CONSTANT_COVERAGE = 5,
} NVCVBorderType;

#ifdef __cplusplus
}
#endif

#endif // NVCV_BORDER_TYPE_H
