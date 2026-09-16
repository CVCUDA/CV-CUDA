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

#ifndef CVCUDA_PRIV_PHOTOMETRIC_BOUND_CUH
#define CVCUDA_PRIV_PHOTOMETRIC_BOUND_CUH

#include <cvcuda/cuda_tools/TypeTraits.hpp>

namespace cvcuda::priv {

// Photometric upper bound (white level) of an image with base type BT: dtype max for integers,
// 1 for the float family (incl. __half). Matches the torchvision.transforms.v2 / OpenCV
// convention used by the photometric ops (Invert, Solarize, AutoContrast, BrightnessContrast).
// R is the type the bound is consumed in, e.g. the float/double intermediate of a mixed-dtype
// pipeline; the decision is always keyed on BT.
template<typename BT, typename R = BT>
inline __host__ __device__ R PhotometricUpperBound()
{
    if constexpr (nvcv::cuda::detail::IsFloatingPointV<BT>)
    {
        return R(1);
    }
    else
    {
        return static_cast<R>(nvcv::cuda::TypeTraits<BT>::max);
    }
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PHOTOMETRIC_BOUND_CUH
