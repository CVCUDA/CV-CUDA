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

#ifndef CVCUDA_PRIV_COPY_MAKE_BORDER_POLICY_HPP
#define CVCUDA_PRIV_COPY_MAKE_BORDER_POLICY_HPP

#include <cstdint>

namespace cvcuda::priv {

constexpr bool UsePackedRGB8Reflect101ForSM(int sm)
{
    return sm != 120;
}

static_assert(!UsePackedRGB8Reflect101ForSM(120));
static_assert(UsePackedRGB8Reflect101ForSM(119));

constexpr bool CanUseReflect101RowCarrier(int pixelBytes, int cols, int outCols, int left)
{
    const int64_t right = static_cast<int64_t>(outCols) - left - cols;
    return (pixelBytes == 3 || pixelBytes == 4) && cols > 1 && left >= 0 && right >= 0 && (left > 0 || right > 0)
        && left < cols && right < cols && (static_cast<int64_t>(cols) * pixelBytes) % 16 == 0
        && (static_cast<int64_t>(outCols) * pixelBytes) % 16 == 0
        && (static_cast<int64_t>(left) * pixelBytes) % 16 == 0;
}

constexpr bool UseLegacyPlanarReflect101VectorForSM(int sm)
{
    return sm == 120;
}

constexpr bool UsePlanarRGBF32Reflect101Vector(int64_t planeElements, int samples, int cacheBytes)
{
    return samples > 0 && cacheBytes > 0
        && planeElements > cacheBytes / (static_cast<int64_t>(sizeof(float)) * samples);
}

constexpr bool UseRGB8Reflect101TensorRowCarrier(int64_t pixelsPerSample, int samples, int cacheBytes)
{
    return samples > 0 && cacheBytes > 0 && pixelsPerSample > cacheBytes / (int64_t{3} * samples);
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_COPY_MAKE_BORDER_POLICY_HPP
