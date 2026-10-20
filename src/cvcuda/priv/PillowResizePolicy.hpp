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

#ifndef CVCUDA_PRIV_PILLOW_RESIZE_POLICY_HPP
#define CVCUDA_PRIV_PILLOW_RESIZE_POLICY_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cvcuda::priv {

// The fused downscale kernel trades the intermediate's DRAM round-trip for recomputing each
// horizontal sum once per vertical tap. That tradeoff is not monotonic across GPU generations, so
// downscale fusion stays on the architectures it was measured to win on; every other architecture
// takes the separable path. `sm` is the compute-capability encoding (major * 10 + minor).
constexpr bool PillowResizeFusesDownscaleForSM(int sm)
{
    const int major = sm / 10;
    return major == 8 || major == 9;
}

static_assert(PillowResizeFusesDownscaleForSM(80));
static_assert(PillowResizeFusesDownscaleForSM(89));
static_assert(PillowResizeFusesDownscaleForSM(90));
static_assert(!PillowResizeFusesDownscaleForSM(75));
static_assert(!PillowResizeFusesDownscaleForSM(100));
static_assert(!PillowResizeFusesDownscaleForSM(103));

// Wider spans amortize bounds and coefficient setup, at the cost of more live accumulators.
constexpr int kPillowVerticalVecBytes           = 16;
constexpr int kPillowNarrowVecElems             = 4;
constexpr int kPillowHorizontalOutputsPerThread = 4;

constexpr bool PillowWideSpanKeepsBlockBusy(int elements, int span, int block_x)
{
    const int threads = elements / span + (elements % span != 0);
    const int slots   = ((threads + block_x - 1) / block_x) * block_x;
    return slots - threads <= slots / 8;
}

static_assert(PillowWideSpanKeepsBlockBusy(1920 * 3, 16, 32));
static_assert(!PillowWideSpanKeepsBlockBusy(640, 16, 32));

constexpr int kPillowU8TileWidth   = 64;
constexpr int kPillowU8TileHeight  = 32;
constexpr int kPillowU8TileMaxRows = 96;
constexpr int kPillowU8TileThreads = 256;

constexpr bool PillowU8FusedTileFits(int inputHeight, int outputHeight, int kernelSize)
{
    if (inputHeight <= 0 || outputHeight <= 0 || kernelSize <= 0)
        return false;
    const long long centerSpan
        = (static_cast<long long>(kPillowU8TileHeight) * inputHeight + outputHeight - 1) / outputHeight;
    const long long floatRoundoff = (static_cast<long long>(inputHeight) + (1LL << 21) - 1) / (1LL << 21);
    return centerSpan + kernelSize + floatRoundoff + 4 <= kPillowU8TileMaxRows;
}

struct PillowStorageRange
{
    uintptr_t begin = 0;
    uintptr_t end   = 0;
};

inline bool PillowPlaneStorageRange(uintptr_t base, int64_t height, int64_t rowStride, int64_t rowBytes,
                                    PillowStorageRange &range)
{
    range = {};
    if (base == 0 || height <= 0 || rowStride <= 0 || rowBytes <= 0 || rowBytes > rowStride)
        return false;
    constexpr uintptr_t limit = std::numeric_limits<uintptr_t>::max();
    if (static_cast<uint64_t>(rowBytes) > limit || static_cast<uint64_t>(rowStride) > limit
        || static_cast<uint64_t>(height - 1) > (limit - static_cast<uintptr_t>(rowBytes)) / rowStride)
        return false;
    const uintptr_t bytes
        = static_cast<uintptr_t>(height - 1) * static_cast<uintptr_t>(rowStride) + static_cast<uintptr_t>(rowBytes);
    if (bytes > limit - base)
        return false;
    range = {base, base + bytes};
    return true;
}

inline bool PillowStorageRangesDisjoint(const PillowStorageRange &first, const PillowStorageRange &second)
{
    return first.begin != 0 && first.end > first.begin && second.begin != 0 && second.end > second.begin
        && (first.end <= second.begin || second.end <= first.begin);
}

inline bool PillowStorageRangeDisjointFromAll(const PillowStorageRange &output, const PillowStorageRange *sourceStorage,
                                              size_t sourceCount)
{
    if (sourceStorage == nullptr || sourceCount == 0
        || sourceCount > std::numeric_limits<size_t>::max() / sizeof(PillowStorageRange))
        return false;
    const auto *bytes = reinterpret_cast<const unsigned char *>(sourceStorage);
    for (size_t source = 0; source < sourceCount; ++source)
    {
        PillowStorageRange input;
        std::memcpy(&input, bytes + source * sizeof(input), sizeof(input));
        if (!PillowStorageRangesDisjoint(input, output))
            return false;
    }
    return true;
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PILLOW_RESIZE_POLICY_HPP
