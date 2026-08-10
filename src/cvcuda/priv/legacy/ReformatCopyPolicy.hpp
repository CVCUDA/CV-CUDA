/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CVCUDA_PRIV_LEGACY_REFORMAT_COPY_POLICY_HPP
#define CVCUDA_PRIV_LEGACY_REFORMAT_COPY_POLICY_HPP

#include <cstddef>
#include <cstdint>

namespace nvcv::legacy::cuda_op::detail {

inline constexpr std::size_t kSingleSampleMinRowBytes = 256;
inline constexpr std::size_t kMultiSampleMinRowBytes  = 288;
inline constexpr std::size_t kLimitedBatchMinPixels   = 1600 * 900;
inline constexpr std::size_t kAnyBatchMinPixels       = 1792 * 1056;

constexpr bool ShouldUsePitchedCopy(std::uint32_t numSamples, std::size_t pixelsPerSample,
                                    std::size_t rowBytes) noexcept
{
    if (numSamples == 0)
    {
        return true;
    }

    if (numSamples == 1)
    {
        return rowBytes >= kSingleSampleMinRowBytes;
    }

    return rowBytes >= kMultiSampleMinRowBytes
        && (pixelsPerSample >= kAnyBatchMinPixels || (numSamples <= 8 && pixelsPerSample >= kLimitedBatchMinPixels));
}

} // namespace nvcv::legacy::cuda_op::detail

#endif // CVCUDA_PRIV_LEGACY_REFORMAT_COPY_POLICY_HPP
