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

#ifndef CVCUDA_PRIV_PLANAR_SAMPLE_DECODE_HPP
#define CVCUDA_PRIV_PLANAR_SAMPLE_DECODE_HPP

#include <cstdint>

namespace cvcuda::priv {

// A planar var-shape launch folds (sample, plane) into gridDim.z, so every thread divides
// blockIdx.z by the channel count to recover its sample. Dividing by a value the compiler only sees
// at run time costs a reciprocal sequence of roughly twenty instructions on the vector datapath; a
// multiplier the host computes once turns it into a single multiply-high.
//
// `z` is pre-shifted by kPlanarSampleShift before the multiply so the multiplier still fits in 32
// bits for a channel count of 1, which is the tightest case. Fifteen is the smallest shift for
// which that holds.
constexpr int kPlanarSampleShift = 15;

// The largest blockIdx.z a planar launch can present. gridDim.z is capped at kMaxGridZ, so the
// decode only has to be exact below this.
constexpr uint32_t kPlanarSampleMaxZ = 65536;

// ceil(2^(32 - kPlanarSampleShift) / channels). Paired with PlanarSampleFromZ this reproduces
// `z / channels` exactly for every z < kPlanarSampleMaxZ and every channel count
// filter::ValidateChannelCount admits (1, 3 or 4); tests/cvcuda/unit/TestPlanarSampleDecode.cpp
// checks that exhaustively.
constexpr uint32_t PlanarSampleReciprocal(int channels)
{
    const uint64_t numerator = uint64_t{1} << (32 - kPlanarSampleShift);
    return static_cast<uint32_t>((numerator + static_cast<uint64_t>(channels) - 1) / static_cast<uint64_t>(channels));
}

// Host-side twin of the device decode, so a test can pin the identity without launching a kernel.
constexpr uint32_t PlanarSampleFromZ(uint32_t z, uint32_t reciprocal)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(z << kPlanarSampleShift) * reciprocal) >> 32);
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PLANAR_SAMPLE_DECODE_HPP
