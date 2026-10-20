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

#ifndef CVCUDA_PRIV_FILTER_SCHEDULE_POLICY_HPP
#define CVCUDA_PRIV_FILTER_SCHEDULE_POLICY_HPP

namespace cvcuda::priv {

// Which schedule a box/convolution filter submit runs. Resolved once per submit from the current
// device's SM and the pixel it is about to move.
//
// `kDirect` is one thread per output pixel reading its whole window through the raw wrap wherever
// that window is interior -- the schedule the legacy layer used. `kTiled` stages an
// (outputs x block + halo) tile in shared memory first, then reads the taps back out of it.
//
// Tiling trades global traffic for shared traffic plus a block barrier: on the 5x5 var-shape
// benchmark it cuts global loads from 73 to 24 but adds 67 shared loads, 8 shared stores and one
// barrier that `kDirect` never pays. Whether that trade pays depends on how well the overlapping
// windows already hit L1/L2 -- which is an architecture property -- and on how many bytes each of
// those 67 shared accesses moves, which is an element-width property. Both inputs matter, so
// neither alone decides the schedule.
enum class FilterSchedule
{
    kDirect,
    kTiled
};

// `elemBytes` is the width of one channel value, `numComponents` the channel count, so
// elemBytes * numComponents is the pixel. A planar submit works one plane at a time, so its tile
// holds scalars and only `elemBytes` matters; an interleaved submit stages whole pixels.
//
// Measured against the pre-refactor baseline on the 5x5 REPLICATE var-shape benchmark. Both
// schedules were run on every SKU, so these are the two columns the rule chooses between:
//
//   planar, 4-byte planes        tiled           direct
//     A100  float3               +6.2%           ~0%  (post-fix; was +15.9% before fix)
//     A100  float4               +6.2%           ~0%  (post-fix; was +14.0% before fix)
//     H100  float3               +15.6%          +3.4% to +5.6%
//     H100  float4               +17.7%          +1.6% to +3.7%
//     GB300 float3 / float4      -6.5% / -5.3%   -4.0% / -4.5%
//
//   planar, 1-byte planes  tiled wins: A100 uchar3 -19% to -26%, H100 uchar3 -5% to -22%
//   interleaved, 4-component >= 8 B   A100 half4 +11.8% tiled; GB300 float4 ~+14% tiled
//   interleaved, everything else      A100 uint8 -33%, uchar3 -45%, half3 -36%, float3 -21% tiled
//
// The original +14-16% direct measurements on A100 were caused by two extra src.width/height
// global loads and the BSSY/BSYNC convergence barrier the compiler inserted around them --
// roughly 30 instructions of overhead per thread. After eliminating those loads (AverageBlur
// output always matches input extent, so dst dimensions suffice for the interior test), the
// direct path compiles to the same structure as the legacy kernel and the overhead disappears.
// A100 now prefers direct by ~6 points over tiled (32 REG vs 40 REG, 100% vs 75% occupancy).
// Blackwell remains faster than legacy on both schedules; tiled stays there for consistency.
constexpr FilterSchedule FilterScheduleForSM(int sm, int elemBytes, int numComponents, bool planar)
{
    if (planar)
    {
        return (elemBytes >= 4 && (sm == 90 || sm == 80)) ? FilterSchedule::kDirect : FilterSchedule::kTiled;
    }
    return (numComponents == 4 && elemBytes * numComponents >= 8) ? FilterSchedule::kDirect : FilterSchedule::kTiled;
}

// Planar: 4-byte planes go direct on Ampere (A100) and Hopper; Blackwell keeps the tile.
static_assert(FilterScheduleForSM(90, 4, 3, true) == FilterSchedule::kDirect);
static_assert(FilterScheduleForSM(90, 4, 4, true) == FilterSchedule::kDirect);
static_assert(FilterScheduleForSM(80, 4, 3, true) == FilterSchedule::kDirect);
static_assert(FilterScheduleForSM(80, 4, 4, true) == FilterSchedule::kDirect);
static_assert(FilterScheduleForSM(100, 4, 3, true) == FilterSchedule::kTiled);
static_assert(FilterScheduleForSM(120, 4, 4, true) == FilterSchedule::kTiled);

// Planar: narrow planes stay tiled everywhere.
static_assert(FilterScheduleForSM(80, 1, 3, true) == FilterSchedule::kTiled);
static_assert(FilterScheduleForSM(90, 1, 3, true) == FilterSchedule::kTiled);
static_assert(FilterScheduleForSM(90, 2, 3, true) == FilterSchedule::kTiled);

// Interleaved: wide 4-component pixels never tile; uchar4 is narrow enough that it still pays.
static_assert(FilterScheduleForSM(80, 2, 4, false) == FilterSchedule::kDirect); // half4, 8 B
static_assert(FilterScheduleForSM(90, 4, 4, false) == FilterSchedule::kDirect); // float4, 16 B
static_assert(FilterScheduleForSM(100, 4, 4, false) == FilterSchedule::kDirect);
static_assert(FilterScheduleForSM(80, 1, 4, false) == FilterSchedule::kTiled); // uchar4, 4 B
static_assert(FilterScheduleForSM(80, 1, 1, false) == FilterSchedule::kTiled); // uint8
static_assert(FilterScheduleForSM(80, 4, 3, false) == FilterSchedule::kTiled); // float3, 12 B
static_assert(FilterScheduleForSM(80, 2, 3, false) == FilterSchedule::kTiled); // half3

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_FILTER_SCHEDULE_POLICY_HPP
