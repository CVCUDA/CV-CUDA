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

/**
 * @file AdjustColorCommon.cuh
 *
 * @brief Device helpers for torchvision-compatible image-range clamping and AdjustContrast.
 */

#ifndef CVCUDA_PRIV_ADJUST_COLOR_COMMON_CUH
#define CVCUDA_PRIV_ADJUST_COLOR_COMMON_CUH

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <type_traits>

namespace cvcuda::priv::adjust {

namespace cuda = nvcv::cuda;

// torchvision's per-dtype clamp bound (``_max_value``): 1.0 for floating-point images and the
// dtype maximum for integer images. AdjustContrast uses this bound to match torchvision's
// image-domain ``_blend`` behavior.
template<typename BT>
inline __host__ __device__ float Bound()
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        return 1.0f;
    }
    else
    {
        return static_cast<float>(cuda::TypeTraits<BT>::max);
    }
}

// torchvision BT.601 luma weights used by adjust_contrast's grayscale conversion. These differ
// from cvtcolor's RGB2GRAY red weight (0.299) so the mean matches torchvision exactly.
inline constexpr float kLumaR = 0.2989f;
inline constexpr float kLumaG = 0.587f;
inline constexpr float kLumaB = 0.114f;

// torchvision ``_blend`` for one channel component, evaluated in the float domain then clamped to
// [0, bound] and saturate-cast back to the base type:
//   out = SaturateCast(clamp(ratio * in + (1 - ratio) * other, 0, bound))
// adjust_contrast passes the per-image grayscale mean as ``other``. Integer results round-to-nearest
// via SaturateCast (torchvision truncates, so integer outputs may differ by <=1 LSB); float results
// are the bit-exact clamped affine.
template<typename BT>
inline __device__ BT Blend(BT v, float ratio, float other, float bound)
{
    float pixel = ratio * static_cast<float>(v) + (1.0f - ratio) * other;
    pixel       = pixel < 0.0f ? 0.0f : (pixel > bound ? bound : pixel);
    return cuda::SaturateCast<BT>(pixel);
}

// torchvision adjust_contrast grayscale value from explicit R, G, B base components (used by the
// planar reduction path where the channels live in separate planes):
//   gray = 0.2989 R + 0.587 G + 0.114 B; floored for integer inputs, matching torchvision's
//   ``_rgb_to_grayscale_image(..., preserve_dtype=False)`` followed by ``floor_()``.
template<typename BT>
inline __device__ float GrayFromRGB(BT r, BT g, BT b)
{
    float gray = kLumaR * static_cast<float>(r) + kLumaG * static_cast<float>(g) + kLumaB * static_cast<float>(b);
    if constexpr (std::is_integral_v<BT>)
    {
        gray = floorf(gray);
    }
    return gray;
}

// torchvision adjust_contrast grayscale value for one interleaved pixel, in the float domain:
//   1 channel  -> the channel value cast to float.
//   3 channels -> GrayFromRGB of the three interleaved components.
template<typename T>
inline __device__ float GrayValue(T pixel)
{
    using BT                         = cuda::BaseType<T>;
    static constexpr int numChannels = cuda::NumElements<T>;
    if constexpr (numChannels == 1)
    {
        return static_cast<float>(cuda::GetElement(pixel, 0));
    }
    else
    {
        return GrayFromRGB<BT>(cuda::GetElement(pixel, 0), cuda::GetElement(pixel, 1), cuda::GetElement(pixel, 2));
    }
}

} // namespace cvcuda::priv::adjust

#endif // CVCUDA_PRIV_ADJUST_COLOR_COMMON_CUH
