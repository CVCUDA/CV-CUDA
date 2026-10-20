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

#include "Definitions.hpp"

#include <cvcuda/priv/PlanarSampleDecode.hpp>

namespace priv = cvcuda::priv;

// The planar filter kernels replace `blockIdx.z / channels` with this multiply-high. It is only a
// valid substitution while it agrees with the division everywhere the launch can reach, so check
// every z the grid-z cap allows rather than sampling.
TEST(PlanarSampleDecode, MatchesDivisionOverTheWholeGridZRange)
{
    for (int channels : {1, 3, 4})
    {
        const uint32_t reciprocal = priv::PlanarSampleReciprocal(channels);

        for (uint32_t z = 0; z < priv::kPlanarSampleMaxZ; ++z)
        {
            ASSERT_EQ(priv::PlanarSampleFromZ(z, reciprocal), z / static_cast<uint32_t>(channels))
                << "channels=" << channels << " z=" << z;
        }
    }
}

// The pre-shift is what keeps the multiplier inside 32 bits; a single channel is the tightest case
// because it needs the largest multiplier. If a future edit raises the shift, this catches it.
TEST(PlanarSampleDecode, ReciprocalFitsIn32BitsForEveryChannelCount)
{
    for (int channels : {1, 3, 4})
    {
        const uint64_t exact = (uint64_t{1} << (32 - priv::kPlanarSampleShift)) / channels;
        EXPECT_LE(exact, uint64_t{0xFFFFFFFF}) << "channels=" << channels;
        EXPECT_GT(priv::PlanarSampleReciprocal(channels), 0u) << "channels=" << channels;
    }
}

// The shifted product must not overflow the 64-bit intermediate the decode multiplies through.
TEST(PlanarSampleDecode, ShiftedInputStaysInRange)
{
    const uint64_t largest = static_cast<uint64_t>(priv::kPlanarSampleMaxZ - 1) << priv::kPlanarSampleShift;
    EXPECT_LE(largest, uint64_t{0xFFFFFFFF});
}
