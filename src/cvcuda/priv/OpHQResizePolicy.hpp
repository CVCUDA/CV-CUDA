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

#ifndef CVCUDA_PRIV_OP_HQ_RESIZE_POLICY_HPP
#define CVCUDA_PRIV_OP_HQ_RESIZE_POLICY_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>

namespace cvcuda::priv::hq_resize {

inline constexpr int kResizeBlockDimX         = 32;
inline constexpr int kResizeBlockDimY         = 8;
inline constexpr int kResizeBlockDimZ         = 1;
inline constexpr int kCubicFilterCoefficients = 129;
// Each thread block covers lanes * volume(blockDim) output elements. More
// lanes reduce the grid size, which may reduce parallelism for small images,
// while improving reuse of filter coefficients computed once for all lanes.
inline constexpr int kDefaultResizeBlockLanes             = 8;
inline constexpr int kU8CubicMagnificationBlockLanes      = 64;
inline constexpr int kF16CubicMagnificationBlockLanes     = 64;
inline constexpr int kF16CubicMinimumBlocksPerSM          = 16;
inline constexpr int kF32CubicMagnificationBlockLanes     = 16;
inline constexpr int kF32CubicMagnificationSM89BlockLanes = 8;

constexpr bool CanVectorizeF32DirectLinear2D(int outWidth, int outHeight, int numSamples)
{
    constexpr int kMinOutputWidth = 4 * kResizeBlockDimX;
    constexpr int kMinBlocks      = 512;
    if (outWidth < kMinOutputWidth || outHeight <= 0 || numSamples <= 0)
    {
        return false;
    }
    const int64_t blocksX = 1 + (outWidth - 1) / kMinOutputWidth;
    const int64_t blocksY = 1 + (outHeight - 1) / kResizeBlockDimY;
    return blocksX * blocksY >= 1 + (kMinBlocks - 1) / numSamples;
}

constexpr int GetF32DirectLinear2DLanes(int inWidth, int inHeight, int outWidth, int outHeight, int numSamples)
{
    if (!CanVectorizeF32DirectLinear2D(outWidth, outHeight, numSamples))
    {
        return 1;
    }
    return outWidth <= (1 << 22) && outHeight <= (1 << 22)
                && static_cast<int64_t>(inWidth) == 2 * static_cast<int64_t>(outWidth)
                && static_cast<int64_t>(inHeight) == 2 * static_cast<int64_t>(outHeight)
             ? 2
             : 4;
}

constexpr bool HasF16CubicBlockPopulation(int outWidth, int outHeight, int blockWidth, int blockHeight, int numSamples,
                                          int64_t minimumBlocks)
{
    // RunPass computes its per-axis grid with int32 ceil division. A wider
    // block must not overflow the (extent + block) intermediate before subtracting one.
    if (outWidth <= 0 || outHeight <= 0 || blockWidth <= 0 || blockHeight <= 0 || numSamples <= 0 || minimumBlocks <= 0
        || outWidth > std::numeric_limits<int>::max() - blockWidth
        || outHeight > std::numeric_limits<int>::max() - blockHeight)
    {
        return false;
    }
    const int64_t blocksX = (static_cast<int64_t>(outWidth) + blockWidth - 1) / blockWidth;
    const int64_t blocksY = (static_cast<int64_t>(outHeight) + blockHeight - 1) / blockHeight;
    return blocksX * blocksY >= (minimumBlocks + numSamples - 1) / numSamples;
}

static_assert(!HasF16CubicBlockPopulation(3840, 1080, 32, 512, 5, 16 * 114));
static_assert(HasF16CubicBlockPopulation(3840, 1080, 32, 512, 6, 16 * 114));
static_assert(!HasF16CubicBlockPopulation(2147481600, 2, 2048, 2, 1, 16 * 114));
static_assert(HasF16CubicBlockPopulation(2147481599, 2, 2048, 2, 1, 16 * 114));
static_assert(!HasF16CubicBlockPopulation(32, 2147483136, 32, 512, 1, 16 * 114));
static_assert(HasF16CubicBlockPopulation(32, 2147483135, 32, 512, 1, 16 * 114));

constexpr int GetF32CubicMagnificationBlockLanes(int computeCapability)
{
    // The wider tile wins on the measured SM80/SM86/SM90 devices, but regresses SM89.
    // Keep the current behavior everywhere else to avoid changing unmeasured architectures.
    return computeCapability == 89 ? kF32CubicMagnificationSM89BlockLanes : kF32CubicMagnificationBlockLanes;
}

static_assert(GetF32CubicMagnificationBlockLanes(80) == 16);
static_assert(GetF32CubicMagnificationBlockLanes(86) == 16);
static_assert(GetF32CubicMagnificationBlockLanes(89) == 8);
static_assert(GetF32CubicMagnificationBlockLanes(90) == 16);
static_assert(GetF32CubicMagnificationBlockLanes(100) == 16);

// Lane widening was characterized on one 188-SM compute-capability 12.0
// configuration using the hqresize_vlm_center_crop_varshape_rgb8_advanced,
// hqresize_vlm_center_crop_batch_boundary_rgb8_advanced, and
// hqresize_vlm_center_crop_no_antialias_fallback_rgb8_advanced sweeps. The SM
// count is part of that validated device
// fingerprint: the population guard below cannot establish profitability on
// different hardware, so broadening this gate requires equivalent sweeps. Add
// another validated fingerprint here after such a sweep; keep this exact row
// until then rather than generalizing it from architecture or SM count alone.
inline constexpr int kValidatedComputeCapability   = 120;
inline constexpr int kValidatedMultiprocessorCount = 188;
// Support 28 is the largest cubic filter footprint in the measured
// 1080-to-160 contraction. Larger filters retain the default layout until
// separately characterized.
inline constexpr int kMaximumValidatedAdaptiveHorizontalFilterSupport = 28;

constexpr bool IsValidatedAdaptiveHorizontalDevice(int computeCapability, int multiprocessorCount)
{
    return computeCapability == kValidatedComputeCapability && multiprocessorCount == kValidatedMultiprocessorCount;
}

constexpr bool CanUseAdaptiveHorizontalLayout(bool antialias, bool lanesPinned, bool packedRgb8, int computeCapability,
                                              int multiprocessorCount)
{
    return antialias && !lanesPinned && packedRgb8
        && IsValidatedAdaptiveHorizontalDevice(computeCapability, multiprocessorCount);
}

struct AdaptiveHorizontalWork
{
    int   outWidth;
    int   outHeight;
    int   filterSupport;
    float horizontalScale;
};

template<typename SampleDesc, typename IsSharedMemoryFilter>
constexpr std::optional<AdaptiveHorizontalWork> DescribeAdaptiveHorizontalSample(
    const SampleDesc &sample, IsSharedMemoryFilter &&isSharedMemoryFilter)
{
    constexpr int kHorizontalPass = 1;
    const auto   &flt             = sample.filter[kHorizontalPass];
    const float   scale           = sample.scale.y < 0 ? -sample.scale.y : sample.scale.y;
    const auto    shape           = sample.shapes[kHorizontalPass + 1];
    // The upper bound rejects infinity; the contraction comparison also rejects NaN.
    if (sample.processingOrder.x != 1 || sample.processingOrder.y != 0
        || !isSharedMemoryFilter(sample.filterKind[kHorizontalPass]) || flt.numCoeffs != kCubicFilterCoefficients
        || !(scale > 1.0f && scale <= std::numeric_limits<float>::max()))
    {
        return std::nullopt;
    }
    return AdaptiveHorizontalWork{shape.x, shape.y, flt.support(), scale};
}

struct AdaptiveHorizontalBatchWork
{
    std::optional<AdaptiveHorizontalWork> first;
    int                                   numSamples{};
    bool                                  homogeneous{true};
};

constexpr bool IsSameAdaptiveHorizontalWork(const std::optional<AdaptiveHorizontalWork> &lhs,
                                            const std::optional<AdaptiveHorizontalWork> &rhs)
{
    if (lhs.has_value() != rhs.has_value())
    {
        return false;
    }
    if (!lhs)
    {
        return true;
    }
    if (lhs->outWidth != rhs->outWidth || lhs->outHeight != rhs->outHeight || lhs->filterSupport != rhs->filterSupport)
    {
        return false;
    }

    constexpr float kScaleTolerance = 8 * std::numeric_limits<float>::epsilon();
    const auto      absValue        = [](float value)
    {
        return value < 0 ? -value : value;
    };
    const float scaleMagnitude = std::max(absValue(lhs->horizontalScale), absValue(rhs->horizontalScale));
    return absValue(lhs->horizontalScale - rhs->horizontalScale) <= kScaleTolerance * scaleMagnitude;
}

constexpr void AddAdaptiveHorizontalWork(AdaptiveHorizontalBatchWork                 &batch,
                                         const std::optional<AdaptiveHorizontalWork> &work)
{
    if (batch.numSamples == 0)
    {
        batch.first = work;
    }
    else
    {
        batch.homogeneous = batch.homogeneous && IsSameAdaptiveHorizontalWork(batch.first, work);
    }
    ++batch.numSamples;
}

constexpr int GetAdaptiveHorizontalBlockLanes(const AdaptiveHorizontalWork &work, int numSamples,
                                              int multiprocessorCount)
{
    constexpr int   kMaximumHorizontalTiles = 11;
    constexpr int   kMaximumAdaptiveRows    = kResizeBlockDimY * 32;
    // With the validated device, packed RGB8 cubic-antialias filter, and 1080p
    // input held fixed, the three named VLM sweeps above cross from parity at
    // 3.75 to a clear win immediately above it as crop size and batch vary.
    constexpr float kMinimumReuseScore  = 3.75f;
    constexpr int   kMinimumBlocksPerSM = 2;

    if (work.outWidth <= 0 || work.outHeight <= 0 || work.filterSupport <= 0
        || work.filterSupport > kMaximumValidatedAdaptiveHorizontalFilterSupport || numSamples <= 0
        || multiprocessorCount <= 0)
    {
        return kDefaultResizeBlockLanes;
    }

    const int64_t blocksX = (static_cast<int64_t>(work.outWidth) + kResizeBlockDimX - 1) / kResizeBlockDimX;
    // The measured sweep covered widths through 352 px (11 32-wide tiles).
    // Wider outputs retain the default layout until separately characterized.
    // Likewise, once the widest layout needs a second 256-row group, lane 32
    // is only noise-level faster and lane 16 regresses.
    if (blocksX > kMaximumHorizontalTiles || work.outHeight > kMaximumAdaptiveRows)
    {
        return kDefaultResizeBlockLanes;
    }

    const int64_t minBlocks        = static_cast<int64_t>(kMinimumBlocksPerSM) * multiprocessorCount;
    const int64_t defaultBlockRows = std::min<int64_t>(kResizeBlockDimY * kDefaultResizeBlockLanes, work.outHeight);
    const int64_t defaultBlocksY   = (work.outHeight + defaultBlockRows - 1) / defaultBlockRows;
    const int64_t defaultSlots     = defaultBlocksY * ((defaultBlockRows + kResizeBlockDimY - 1) / kResizeBlockDimY);
    constexpr std::array candidates{32, 16};
    int64_t              previousBlockRows = -1;
    for (int lanes : candidates)
    {
        const int64_t blockRows = std::min<int64_t>(kResizeBlockDimY * lanes, work.outHeight);
        if (blockRows == defaultBlockRows || blockRows == previousBlockRows)
        {
            continue;
        }
        previousBlockRows            = blockRows;
        const int64_t blocksY        = (work.outHeight + blockRows - 1) / blockRows;
        const int64_t candidateSlots = blocksY * ((blockRows + kResizeBlockDimY - 1) / kResizeBlockDimY);

        // Widen when horizontal contraction exceeds the measured 3.75x
        // crossover, corrected for row-tail waste in the candidate layout.
        // The population guard separately rejects under-filled grids.
        const bool profitable = static_cast<double>(work.horizontalScale) * static_cast<double>(defaultSlots)
                              > static_cast<double>(kMinimumReuseScore) * static_cast<double>(candidateSlots);
        if (profitable && blocksX * blocksY * numSamples >= minBlocks)
        {
            return lanes;
        }
    }
    // Narrower layouts regress under-filled workloads, so the existing
    // eight-lane layout remains the fallback.
    return kDefaultResizeBlockLanes;
}

constexpr int GetAdaptiveHorizontalBlockLanes(const AdaptiveHorizontalBatchWork &batch, int multiprocessorCount)
{
    if (!batch.homogeneous || !batch.first)
    {
        return kDefaultResizeBlockLanes;
    }
    return GetAdaptiveHorizontalBlockLanes(*batch.first, batch.numSamples, multiprocessorCount);
}

template<typename SampleDesc, typename SetupBlockLayoutPass>
void SetupAdaptiveHorizontalBlockLayout(SampleDesc *sampleDescs, // NOSONAR: C++17 has no std::span.
                                        const AdaptiveHorizontalBatchWork &batch, bool wideStride,
                                        int multiprocessorCount, SetupBlockLayoutPass &&setupBlockLayoutPass)
{
    if (wideStride)
    {
        return;
    }

    const int lanes = GetAdaptiveHorizontalBlockLanes(batch, multiprocessorCount);
    if (lanes == kDefaultResizeBlockLanes)
    {
        return;
    }

    constexpr int kHorizontalPass = 1;
    for (int sampleIdx = 0; sampleIdx < batch.numSamples; ++sampleIdx)
    {
        setupBlockLayoutPass(sampleDescs[sampleIdx], kHorizontalPass, lanes);
    }
}

enum class DirectTensorPath
{
    Other,
    LinearExpand2x,
    CubicContract2x,
    CubicExpand2x,
};

struct DirectTensorPathDesc
{
    bool  linear;
    bool  cubic;
    bool  xFirst;
    float originX;
    float originY;
    float scaleX;
    float scaleY;
    int   inWidth;
    int   inHeight;
    int   outWidth;
    int   outHeight;
};

constexpr DirectTensorPath ClassifyDirectTensorPath(const DirectTensorPathDesc &desc)
{
    const bool zeroOrigin = desc.originX == 0.f && desc.originY == 0.f;
    const bool expand2x   = desc.scaleX == 0.5f && desc.scaleY == 0.5f && desc.outWidth == 2 * desc.inWidth
                       && desc.outHeight == 2 * desc.inHeight;
    const bool contract2x = desc.scaleX == 2.f && desc.scaleY == 2.f && desc.inWidth == 2 * desc.outWidth
                         && desc.inHeight == 2 * desc.outHeight;

    if (desc.linear && zeroOrigin && expand2x)
    {
        return DirectTensorPath::LinearExpand2x;
    }
    if (desc.cubic && !desc.xFirst && zeroOrigin && contract2x)
    {
        return DirectTensorPath::CubicContract2x;
    }
    if (desc.cubic && desc.xFirst && zeroOrigin && expand2x)
    {
        return DirectTensorPath::CubicExpand2x;
    }
    return DirectTensorPath::Other;
}

constexpr bool UseDirectTensorPathForSM(int sm, DirectTensorPath path, bool isFloat, int numChannels)
{
    if (sm == 75)
    {
        return path != DirectTensorPath::CubicContract2x || !isFloat || (numChannels != 1 && numChannels != 3);
    }

    if (sm != 100 && sm != 103)
    {
        return true;
    }

    switch (path)
    {
    case DirectTensorPath::Other:
        return true;
    case DirectTensorPath::LinearExpand2x:
        return !(isFloat && numChannels == 3);
    case DirectTensorPath::CubicContract2x:
        return numChannels != 3;
    case DirectTensorPath::CubicExpand2x:
        return sm != 100 || !(isFloat && numChannels == 1);
    }
    return true;
}

constexpr bool ShouldUseDirectTensorPathForSM(int sm, const DirectTensorPathDesc &desc, bool isFloat, int numChannels)
{
    return UseDirectTensorPathForSM(sm, ClassifyDirectTensorPath(desc), isFloat, numChannels);
}

static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::LinearExpand2x, true, 3));
static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::CubicContract2x, true, 3));
static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::CubicContract2x, false, 3));
static_assert(!UseDirectTensorPathForSM(100, DirectTensorPath::CubicExpand2x, true, 1));
static_assert(!UseDirectTensorPathForSM(103, DirectTensorPath::LinearExpand2x, true, 3));
static_assert(!UseDirectTensorPathForSM(103, DirectTensorPath::CubicContract2x, true, 3));
static_assert(!UseDirectTensorPathForSM(103, DirectTensorPath::CubicContract2x, false, 3));
static_assert(UseDirectTensorPathForSM(103, DirectTensorPath::CubicExpand2x, true, 1));
static_assert(!UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, true, 1));
static_assert(!UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, true, 3));
static_assert(UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, false, 1));
static_assert(UseDirectTensorPathForSM(75, DirectTensorPath::CubicContract2x, true, 4));
static_assert(UseDirectTensorPathForSM(75, DirectTensorPath::LinearExpand2x, true, 3));
static_assert(UseDirectTensorPathForSM(80, DirectTensorPath::CubicContract2x, true, 1));

} // namespace cvcuda::priv::hq_resize

#endif // CVCUDA_PRIV_OP_HQ_RESIZE_POLICY_HPP
