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

#ifndef CVCUDA_PRIV_FLIP_POLICY_HPP
#define CVCUDA_PRIV_FLIP_POLICY_HPP

namespace cvcuda::priv {

// Groups per thread in the var-shape U8 group copy. The copy is latency-bound, so the count sets
// how many independent group moves a thread keeps in flight. Blackwell absorbs sixteen on the
// interleaved path (GB300, uint8 var-shape 256x1080p: 557 -> 500 us), but the same sixteen more
// than doubles the planar path there (uchar3 NCHW var-shape 128x1080p: +103%): a planar submit
// already multiplies the grid by the plane count, so sixteen groups per thread starve it of blocks
// instead of hiding latency. Below Blackwell the higher count loses outright -- it cost A100
// 10.6-12.8% under both CUDA majors, with H100 flat either way.
constexpr int FlipVarShapeU8GroupsPerThread(int sm, bool planar)
{
    return (sm >= 100 && !planar) ? 16 : 4;
}

// A device whose compute capability could not be queried reports 0 and must land on the
// conservative count, never on the Blackwell-only one.
static_assert(FlipVarShapeU8GroupsPerThread(0, false) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(80, false) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(89, false) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(90, false) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(100, false) == 16);
static_assert(FlipVarShapeU8GroupsPerThread(103, false) == 16);

// The planar path keeps the conservative count on every architecture, Blackwell included.
static_assert(FlipVarShapeU8GroupsPerThread(80, true) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(90, true) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(100, true) == 4);
static_assert(FlipVarShapeU8GroupsPerThread(103, true) == 4);

// Six- and twelve-byte pixel reversals retain the per-pixel kernel: their wider gather
// windows cost more than the carrier saves. Vertical flips only copy rows.
constexpr bool FlipReversalUsesCarrierStream(int pixelBytes)
{
    return pixelBytes != 6 && pixelBytes != 12;
}

static_assert(FlipReversalUsesCarrierStream(1));
static_assert(FlipReversalUsesCarrierStream(3));
static_assert(!FlipReversalUsesCarrierStream(6));
static_assert(FlipReversalUsesCarrierStream(16));
static_assert(!FlipReversalUsesCarrierStream(12));

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_FLIP_POLICY_HPP
