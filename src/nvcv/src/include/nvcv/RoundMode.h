/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_ROUND_MODE_H
#define NVCV_ROUND_MODE_H

#ifdef __cplusplus
extern "C"
{
#endif

/* @brief Selects how floating-point values are rounded when converting to an integer type.
 *
 * Only affects integer outputs; conversions whose output type is floating-point are unaffected.
 *
 * The two _WRAP variants are defined at the enum level for use by operators that implement
 * modular reduction (e.g. ConvertTo).  In those operators the rounded value is reduced modulo
 * the destination's representable range instead of saturating to it; for unsigned destinations
 * negative intermediate results map to zero rather than wrapping (unsigned asymmetry).
 * Operators that do not support wrap modes (e.g. GammaContrast) reject these values with
 * NVCV_ERROR_INVALID_ARGUMENT.
 */
typedef enum
#ifdef __cplusplus
    : unsigned int
#endif
{
    NVCV_ROUND_NEAREST       = 0, ///< Round to nearest, ties to even (default).
    NVCV_ROUND_TRUNCATE      = 1, ///< Truncate toward zero (drop the fractional part).
    NVCV_ROUND_NEAREST_WRAP  = 2, ///< Round to nearest (ties to even), then wrap modulo destination range.
    NVCV_ROUND_TRUNCATE_WRAP = 3, ///< Truncate toward zero, then wrap modulo destination range.
} NVCVRoundMode;

#ifdef __cplusplus
}
#endif

#endif // NVCV_ROUND_MODE_H
