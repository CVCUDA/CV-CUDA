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

#ifndef MORPHOLOGY_UTILS_CUH
#define MORPHOLOGY_UTILS_CUH

#include <cuda_fp16.h>
#include <cvcuda/Types.h>                   // for NVCVMorphologyType, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp> // for HalfLowest, HalfMax, etc.

#include <limits>
#include <type_traits>

// Identity element of the running min/max (also the constant-border value): dilate starts from
// the lowest representable value, erode from the highest. std::numeric_limits has no __half
// specialization (its primary template would silently yield zeros), so half uses the
// function-form limits from TypeTraits.hpp. Shared by the tensor and var-shape morphology files.
template<typename BT>
inline BT MorphIdentityValue(NVCVMorphologyType morph_type)
{
    if constexpr (std::is_same_v<BT, __half>)
    {
        return (morph_type == NVCVMorphologyType::NVCV_DILATE) ? nvcv::cuda::HalfLowest() : nvcv::cuda::HalfMax();
    }
    else
    {
        return (morph_type == NVCVMorphologyType::NVCV_DILATE) ? std::numeric_limits<BT>::min()
                                                               : std::numeric_limits<BT>::max();
    }
}

#endif // MORPHOLOGY_UTILS_CUH
