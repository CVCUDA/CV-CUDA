/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/TensorDataUtils.hpp>
#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpLabel.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

// ----------------------- Basic utility definitions ---------------------------

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;
namespace test = nvcv::test;
namespace type = nvcv::test::type;

using U8 = uint8_t;

using RawBufferType = std::vector<U8>;

// --------------------- Reference (gold) computations -------------------------

namespace ref {

inline long NumElements(const long4_16a &shape)
{
    return shape.x * shape.y * shape.z * shape.w;
}

inline long4_16a CoordFromIndex(long idx, const long4_16a &shape)
{
    long w = idx % shape.w;
    idx /= shape.w;
    long z = idx % shape.z;
    idx /= shape.z;
    long y = idx % shape.y;
    idx /= shape.y;
    return long4_16a{idx, y, z, w};
}

template<typename Func>
void ForEachCoord(const long4_16a &shape, const Func &func)
{
    for (long idx = 0; idx < NumElements(shape); ++idx)
    {
        func(CoordFromIndex(idx, shape));
    }
}

template<typename Func>
void ForEachSampleCoord(const long4_16a &shape, long sample, const Func &func)
{
    long sampleElements = shape.y * shape.z * shape.w;
    for (long idx = 0; idx < sampleElements; ++idx)
    {
        long linear = idx;
        long w      = linear % shape.w;
        linear /= shape.w;
        long z = linear % shape.z;
        linear /= shape.z;
        func(long4_16a{sample, linear, z, w});
    }
}

template<typename DT>
DT PositionLabel(const long4_16a &coord, const long4_16a &strides)
{
    return static_cast<DT>(coord.y * strides.y / sizeof(DT) + coord.z * strides.z / sizeof(DT) + coord.w);
}

// Pre-filter step is to binarize srcVec using threshold range [min, max] -> 1, zero otherwise
template<typename ST>
inline void Binarize(RawBufferType &srcVec, const RawBufferType &minVec, const RawBufferType &maxVec,
                     const long4_16a &srcStrides, const long1 &minStrides, const long1 &maxStrides,
                     const long4_16a &shape)
{
    bool hasMinThresh = minStrides.x > 0;
    bool hasMaxThresh = maxStrides.x > 0;

    ForEachCoord(shape,
                 [&srcVec, &minVec, &maxVec, &srcStrides, &minStrides, &maxStrides, hasMinThresh,
                  hasMaxThresh](const long4_16a &curCoord)
                 {
                     ST minThresh = hasMinThresh ? util::ValueAt<ST>(minVec, minStrides, long1{curCoord.x}) : 0;
                     ST maxThresh = hasMaxThresh ? util::ValueAt<ST>(maxVec, maxStrides, long1{curCoord.x}) : 0;
                     ST value     = util::ValueAt<ST>(srcVec, srcStrides, curCoord);

                     if (hasMinThresh && hasMaxThresh)
                     {
                         value = (value < minThresh || value > maxThresh) ? 0 : 1;
                     }
                     else if (hasMinThresh)
                     {
                         value = (value < minThresh) ? 0 : 1;
                     }
                     else if (hasMaxThresh)
                     {
                         value = (value > maxThresh) ? 0 : 1;
                     }

                     util::ValueAt<ST>(srcVec, srcStrides, curCoord) = value;
                 });
}

template<typename ST, typename DT>
inline void LabelComponent(RawBufferType &tmpVec, RawBufferType &dstVec, const RawBufferType &srcVec,
                           const long4_16a &tmpStrides, const long4_16a &dstStrides, const long4_16a &srcStrides,
                           const long4_16a &shape, const long4_16a &curCoord, ST value, DT label);

template<typename ST, typename DT>
void LabelPixel(RawBufferType &tmpVec, RawBufferType &dstVec, const RawBufferType &srcVec, const long4_16a &tmpStrides,
                const long4_16a &dstStrides, const long4_16a &srcStrides, const long4_16a &shape,
                const long4_16a &curCoord)
{
    if (util::ValueAt<U8>(tmpVec, tmpStrides, curCoord) == 1)
    {
        return;
    }

    ST value = util::ValueAt<ST>(srcVec, srcStrides, curCoord);
    DT label = PositionLabel<DT>(curCoord, dstStrides);

    LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape, curCoord, value, label);
}

// Label each component with label in dstVec matching value in srcVec, marking labeled elements as 1 in tmpVec
// (since this function is called recursively, using big input sizes may lead to stack overflow)
template<typename ST, typename DT>
inline void LabelComponent(RawBufferType &tmpVec, RawBufferType &dstVec, const RawBufferType &srcVec,
                           const long4_16a &tmpStrides, const long4_16a &dstStrides, const long4_16a &srcStrides,
                           const long4_16a &shape, const long4_16a &curCoord, ST value, DT label)
{
    if (util::ValueAt<U8>(tmpVec, tmpStrides, curCoord) == 1)
    {
        return; // The element was already labeled, skip it
    }
    if (value != util::ValueAt<ST>(srcVec, srcStrides, curCoord))
    {
        return; // The element is not in the same labeled region, skip it
    }

    // Set element label in dstVec and mark it as labeled in tmpVec
    util::ValueAt<DT>(dstVec, dstStrides, curCoord) = label;
    util::ValueAt<U8>(tmpVec, tmpStrides, curCoord) = 1;

    // For each neighbor, recursively call label component to label each neighbor
    if (curCoord.y > 0)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4_16a{curCoord.x, curCoord.y - 1, curCoord.z, curCoord.w}, value, label);
    }
    if (curCoord.y < shape.y - 1)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4_16a{curCoord.x, curCoord.y + 1, curCoord.z, curCoord.w}, value, label);
    }
    if (curCoord.z > 0)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4_16a{curCoord.x, curCoord.y, curCoord.z - 1, curCoord.w}, value, label);
    }
    if (curCoord.z < shape.z - 1)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4_16a{curCoord.x, curCoord.y, curCoord.z + 1, curCoord.w}, value, label);
    }
    if (curCoord.w > 0)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4_16a{curCoord.x, curCoord.y, curCoord.z, curCoord.w - 1}, value, label);
    }
    if (curCoord.w < shape.w - 1)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4_16a{curCoord.x, curCoord.y, curCoord.z, curCoord.w + 1}, value, label);
    }
}

// Label N volumes in NDHW tensor stored in srcVec yielding dstVec, with corresponding srcStrides/dstStrides
// - ST is the source type, the data type of the input tensor in srcVec
// - DT is the destination type, the data type of the output tensor in dstVec
template<typename ST, typename DT>
void Label(RawBufferType &dstVec, const RawBufferType &srcVec, const long4_16a &dstStrides, const long4_16a &srcStrides,
           const long4_16a &shape)
{
    // Use a temporary NDHW tensor stored in tmpVec to set elements already labeled, initially zeroes (all unlabeled)
    RawBufferType tmpVec(shape.x * shape.y * shape.z * shape.w, 0);

    // The temporary tensor is packed and each element is a single byte, thus:
    long4_16a tmpStrides{shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1};

    // For all elements in input tensor
    ForEachCoord(shape,
                 [&tmpVec, &dstVec, &srcVec, &tmpStrides, &dstStrides, &srcStrides, &shape](const long4_16a &curCoord)
                 { LabelPixel<ST, DT>(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape, curCoord); });
}

// Replace labels assigned to regions marked as background in source, and fix a potential region labeled with
// background label in destination by another label (since background label is a reserved label)
template<typename ST, typename DT>
void ReplaceBgLabels(RawBufferType &dstVec, const RawBufferType &srcVec, const RawBufferType &bglVec,
                     const long4_16a &dstStrides, const long4_16a &srcStrides, const long1 &bglStrides,
                     const long4_16a &shape)
{
    ForEachCoord(shape,
                 [&dstVec, &srcVec, &bglVec, &dstStrides, &srcStrides, &bglStrides](const long4_16a &curCoord)
                 {
                     ST backgroundLabel = util::ValueAt<ST>(bglVec, bglStrides, long1{curCoord.x});
                     ST value           = util::ValueAt<ST>(srcVec, srcStrides, curCoord);
                     DT label           = util::ValueAt<DT>(dstVec, dstStrides, curCoord);

                     if (value == backgroundLabel)
                     {
                         // The current value is a background label, write it to output
                         util::ValueAt<DT>(dstVec, dstStrides, curCoord) = static_cast<DT>(backgroundLabel);
                     }
                     else if (label == (DT)backgroundLabel)
                     {
                         // If the label assigned happens to be the same as the background label, replace it by
                         // another label that is never assigned outside the possible offsets
                         util::ValueAt<DT>(dstVec, dstStrides, curCoord) = static_cast<DT>(dstStrides.x / sizeof(DT));
                     }
                 });
}

// Get the unique set of labels from output in dstVec, disregarding background labels
template<typename ST, typename DT>
void GetLabels(std::vector<std::set<DT>> &labels, const RawBufferType &dstVec, const RawBufferType &bglVec,
               const long4_16a &dstStrides, const long1 &bglStrides, const long4_16a &dstShape)
{
    bool hasBgLabel = bglStrides.x > 0;

    ForEachCoord(dstShape,
                 [&labels, &dstVec, &bglVec, &dstStrides, &bglStrides, hasBgLabel](const long4_16a &curCoord)
                 {
                     ST backgroundLabel = hasBgLabel ? util::ValueAt<ST>(bglVec, bglStrides, long1{curCoord.x}) : 0;
                     DT label           = util::ValueAt<DT>(dstVec, dstStrides, curCoord);

                     if (hasBgLabel && label == static_cast<DT>(backgroundLabel))
                     {
                         return; // ignore (do not get) background labels
                     }

                     labels[curCoord.x].insert(label);
                 });
}

// Get the unique set of labels from statistics in staVec
template<typename DT>
void GetLabels(std::vector<std::set<DT>> &labels, const RawBufferType &cntVec, const RawBufferType &staVec,
               const long1 &cntStrides, const long3 &staStrides, long numSamples)
{
    for (long x = 0; x < numSamples; ++x)
    {
        long numLabels = util::ValueAt<DT>(cntVec, cntStrides, long1{x});

        for (long y = 0; y < numLabels; ++y)
        {
            DT label = util::ValueAt<DT>(staVec, staStrides, long3{x, y, 0});

            labels[x].insert(label);
        }
    }
}

// Count how many different labels were found
template<typename DT>
void CountLabels(RawBufferType &cntVec, const long1 &cntStrides, const std::vector<std::set<DT>> &labels,
                 long numSamples)
{
    for (long x = 0; x < numSamples; ++x)
    {
        util::ValueAt<DT>(cntVec, cntStrides, long1{x}) = static_cast<DT>(labels[x].size());
    }
}

// Sort statistics according to region index as test stats have no imposed ordering, it allows comparing against gold
template<typename DT>
void SortStats(std::vector<std::vector<std::vector<DT>>> &stats, std::vector<std::set<DT>> &labels,
               const RawBufferType &staVec, const long3 &staStrides, const long3 &staShape)
{
    for (long x = 0; x < staShape.x; ++x)
    {
        long numLabels = labels[x].size();

        stats[x].resize(numLabels);

        for (long y = 0; y < numLabels; ++y)
        {
            DT   label = util::ValueAt<DT>(staVec, staStrides, long3{x, y, 0});
            auto fit   = labels[x].find(label);

            long regionIdx = std::distance(labels[x].cbegin(), fit);
            ASSERT_LE(regionIdx, numLabels) << "E idx " << regionIdx << " >= " << numLabels;

            stats[x][regionIdx].resize(staShape.z);

            for (long z = 0; z < staShape.z; ++z)
            {
                stats[x][regionIdx][z] = util::ValueAt<DT>(staVec, staStrides, long3{x, y, z});
            }
        }
    }
}

template<typename ST, typename DT>
bool FindLabelRegion(long &regionIdx, DT &label, bool &usesPositionStats, const RawBufferType &dstVec,
                     const RawBufferType &bglVec, const long4_16a &dstStrides, const long1 &bglStrides,
                     const std::vector<std::set<DT>> &labels, const long4_16a &curCoord, DT endLabel, bool hasBgLabel)
{
    ST backgroundLabel = hasBgLabel ? util::ValueAt<ST>(bglVec, bglStrides, long1{curCoord.x}) : 0;

    label    = util::ValueAt<DT>(dstVec, dstStrides, curCoord);
    auto fit = labels[curCoord.x].find(label);
    if (fit == labels[curCoord.x].end())
    {
        return false;
    }

    DT posLabel = PositionLabel<DT>(curCoord, dstStrides);
    usesPositionStats
        = (hasBgLabel && label == endLabel && posLabel == static_cast<DT>(backgroundLabel)) || label == posLabel;
    regionIdx = std::distance(labels[curCoord.x].cbegin(), fit);

    return true;
}

template<typename MT>
bool IsInsideMask(const RawBufferType &mskVec, const long4_16a &mskStrides, long maskN, const long4_16a &curCoord)
{
    return util::ValueAt<MT>(mskVec, mskStrides,
                             long4_16a{maskN == 1 ? 0 : curCoord.x, curCoord.y, curCoord.z, curCoord.w})
        != 0;
}

template<typename DT>
void InitializeRegionStats(std::vector<DT> &regionStats, DT label, const long4_16a &curCoord, DT regionMark,
                           int numStats)
{
    regionStats.resize(numStats);
    regionStats[0] = label;
    regionStats[1] = static_cast<DT>(curCoord.w);
    regionStats[2] = static_cast<DT>(curCoord.z);

    if (numStats == 7)
    {
        regionStats[3] = 1;
        regionStats[4] = 1;
        regionStats[5] = 1;
        regionStats[6] = regionMark;
        return;
    }

    regionStats[3] = static_cast<DT>(curCoord.y);
    regionStats[4] = 1;
    regionStats[5] = 1;
    regionStats[6] = 1;
    regionStats[7] = 1;
    regionStats[8] = regionMark;
}

template<typename DT>
void UpdateRegionStats(std::vector<DT> &regionStats, const long4_16a &curCoord, int numStats)
{
    auto bboxAreaW = static_cast<DT>(std::abs(static_cast<long>(regionStats[1]) - curCoord.w) + 1);
    auto bboxAreaH = static_cast<DT>(std::abs(static_cast<long>(regionStats[2]) - curCoord.z) + 1);

    if (numStats == 7)
    {
        regionStats[3] = std::max(regionStats[3], bboxAreaW);
        regionStats[4] = std::max(regionStats[4], bboxAreaH);
        regionStats[5] += 1;
        return;
    }

    auto bboxAreaD = static_cast<DT>(std::abs(static_cast<long>(regionStats[3]) - curCoord.y) + 1);

    regionStats[4] = std::max(regionStats[4], bboxAreaW);
    regionStats[5] = std::max(regionStats[5], bboxAreaH);
    regionStats[6] = std::max(regionStats[6], bboxAreaD);
    regionStats[7] += 1;
}

// Compute statistics of labeled regions
template<typename ST, typename DT, typename MT>
void ComputeStats(std::vector<std::vector<std::vector<DT>>> &stats, const RawBufferType &dstVec,
                  const RawBufferType &mskVec, const RawBufferType &bglVec, const long4_16a &dstStrides,
                  const long4_16a &mskStrides, const long1 &bglStrides, const std::vector<std::set<DT>> &labels,
                  const long4_16a &shape, long maskN, int numStats)
{
    // One-element-after-the-end label is a special label assigned to a region which got the background label
    auto endLabel = static_cast<DT>(dstStrides.x / sizeof(DT));

    bool hasMask    = mskStrides.x > 0;
    bool hasBgLabel = bglStrides.x > 0;

    for (long x = 0; x < shape.x; ++x)
    {
        stats[x].resize(labels[x].size());
    }

    ForEachCoord(shape,
                 [&stats, &dstVec, &mskVec, &bglVec, &dstStrides, &mskStrides, &bglStrides, &labels, endLabel, hasMask,
                  hasBgLabel, maskN, numStats](const long4_16a &curCoord)
                 {
                     long regionIdx;
                     DT   label;
                     bool usesPositionStats;
                     if (!FindLabelRegion<ST>(regionIdx, label, usesPositionStats, dstVec, bglVec, dstStrides,
                                              bglStrides, labels, curCoord, endLabel, hasBgLabel))
                     {
                         return; // this label is to be ignored
                     }

                     if (!usesPositionStats)
                     {
                         return;
                     }

                     DT regionMark = 0; // region has no marks

                     // If has mask and the element is inside the mask
                     if (hasMask && IsInsideMask<MT>(mskVec, mskStrides, maskN, curCoord))
                     {
                         regionMark = 2; // mark the region as inside the mask (= 2)
                     }

                     InitializeRegionStats(stats[curCoord.x][regionIdx], label, curCoord, regionMark, numStats);
                 });

    ForEachCoord(
        shape,
        [&stats, &dstVec, &mskVec, &bglVec, &dstStrides, &mskStrides, &bglStrides, &labels, endLabel, hasMask,
         hasBgLabel, maskN, numStats](const long4_16a &curCoord)
        {
            long regionIdx;
            DT   label;
            bool usesPositionStats;
            if (!FindLabelRegion<ST>(regionIdx, label, usesPositionStats, dstVec, bglVec, dstStrides, bglStrides,
                                     labels, curCoord, endLabel, hasBgLabel))
            {
                return;
            }

            if (usesPositionStats)
            {
                return; // statistics for this element was already computed
            }

            std::vector<DT> &regionStats = stats[curCoord.x][regionIdx];

            // If has mask and the region has no marks (it is no marked as inside mask)
            if (hasMask && regionStats[numStats - 1] == 0 && IsInsideMask<MT>(mskVec, mskStrides, maskN, curCoord))
            {
                regionStats[numStats - 1] = 2; // mark the region as inside mask (= 2)
            }

            UpdateRegionStats(regionStats, curCoord, numStats);
        });
}

// Remove islands (regions with less than minimum size in mszVec) from dstVec based on statistics
template<typename ST, typename DT>
void RemoveIslands(std::vector<std::set<DT>> &labels, RawBufferType &dstVec, const RawBufferType &bglVec,
                   const RawBufferType &mszVec, const long4_16a &dstStrides, const long1 &bglStrides,
                   const long1 &mszStrides, std::vector<std::vector<std::vector<DT>>> &stats, const long4_16a &shape,
                   int numStats)
{
    ForEachCoord(shape,
                 [&labels, &dstVec, &bglVec, &mszVec, &dstStrides, &bglStrides, &mszStrides, &stats,
                  numStats](const long4_16a &curCoord)
                 {
                     ST   backgroundLabel = util::ValueAt<ST>(bglVec, bglStrides, long1{curCoord.x});
                     DT   minSize         = util::ValueAt<DT>(mszVec, mszStrides, long1{curCoord.x});
                     DT   label           = util::ValueAt<DT>(dstVec, dstStrides, curCoord);
                     auto fit             = labels[curCoord.x].find(label); // result of find iterator
                     if (fit == labels[curCoord.x].end())
                     {
                         return; // this label is to be ignored
                     }

                     long regionIdx  = std::distance(labels[curCoord.x].cbegin(), fit);
                     DT   regionSize = stats[curCoord.x][regionIdx][numStats - 2];

                     // If region size is smaller than minimum size (it is an island) and the region is not marked
                     // as inside the mask (= 2), then remove the island and mark it as removed
                     if (regionSize < minSize && stats[curCoord.x][regionIdx][numStats - 1] != 2)
                     {
                         util::ValueAt<DT>(dstVec, dstStrides, curCoord) = backgroundLabel;

                         stats[curCoord.x][regionIdx][numStats - 1] = 1;
                     }
                 });
}

// Relabel replaces index-based labels by consecutive region indices
template<typename ST, typename DT>
void Relabel(RawBufferType &dstVec, const RawBufferType &bglVec, const RawBufferType &staVec,
             const RawBufferType &cntVec, const long4_16a &dstStrides, const long1 &bglStrides, const long3 &staStrides,
             const long1 &cntStrides, const long4_16a &shape)
{
    for (long x = 0; x < shape.x; ++x)
    {
        ST backgroundLabel = util::ValueAt<ST>(bglVec, bglStrides, long1{x});

        std::map<DT, DT> origLabelToRegionIdx;

        DT numLabels = util::ValueAt<DT>(cntVec, cntStrides, long1{x});

        for (DT y = 0; y < numLabels; ++y)
        {
            DT origLabel = util::ValueAt<DT>(staVec, staStrides, long3{x, y, 0});
            origLabelToRegionIdx.insert({origLabel, y});
        }
        ForEachSampleCoord(shape, x,
                           [&dstVec, &dstStrides, &origLabelToRegionIdx, backgroundLabel](const long4_16a &curCoord)
                           {
                               DT label = util::ValueAt<DT>(dstVec, dstStrides, curCoord);

                               if (label == (DT)backgroundLabel)
                               {
                                   return;
                               }

                               DT regionIdx = origLabelToRegionIdx[label];

                               if (regionIdx >= (DT)backgroundLabel)
                               {
                                   regionIdx += 1; // increment region indices to skip background labels
                               }

                               util::ValueAt<DT>(dstVec, dstStrides, curCoord) = regionIdx;
                           });
    }
}

} // namespace ref

// ----------------------------- Start tests -----------------------------------

// clang-format off

#define NVCV_SHAPE(w, h, d, n) (int4{w, h, d, n})

#define NVCV_TEST_ROW(InShape, DataType, Type, HasBgLabel, HasMinThresh, HasMaxThresh, DoPostFilters, DoRelabel)       \
    type::Types<type::Value<InShape>, type::Value<DataType>, Type, type::Value<HasBgLabel>, type::Value<HasMinThresh>, \
                type::Value<HasMaxThresh>, type::Value<DoPostFilters>, type::Value<DoRelabel>>

// DoPostFilters: (0) none; (1) count regions; (2) + compute statistics; (3) + island removal; (4) + masked.

NVCV_TYPED_TEST_SUITE(OpLabel, type::Types<
    NVCV_TEST_ROW(NVCV_SHAPE(33, 16, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, false, false, false, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(23, 81, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, false, true, false, 1, false),
    NVCV_TEST_ROW(NVCV_SHAPE(13, 14, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, false, true, true, 2, false),
    NVCV_TEST_ROW(NVCV_SHAPE(32, 43, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, true, false, false, 3, false),
    NVCV_TEST_ROW(NVCV_SHAPE(13, 52, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, true, false, false, 4, false),
    NVCV_TEST_ROW(NVCV_SHAPE(22, 12, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, false, false, true, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(15, 16, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, true, false, true, 1, false),
    NVCV_TEST_ROW(NVCV_SHAPE(14, 26, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, true, true, false, 2, true),
    NVCV_TEST_ROW(NVCV_SHAPE(40, 17, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, true, true, true, 4, true),
    NVCV_TEST_ROW(NVCV_SHAPE(28, 73, 1, 3), NVCV_DATA_TYPE_U16, uint16_t, true, true, true, 3, true),
    NVCV_TEST_ROW(NVCV_SHAPE(19, 61, 1, 3), NVCV_DATA_TYPE_U16, uint16_t, true, true, true, 4, true),
    NVCV_TEST_ROW(NVCV_SHAPE(35, 19, 1, 2), NVCV_DATA_TYPE_U32, uint32_t, false, false, false, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(23, 21, 12, 1), NVCV_DATA_TYPE_U32, uint32_t, false, false, false, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(33, 41, 22, 1), NVCV_DATA_TYPE_U32, uint32_t, false, false, false, 1, false),
    NVCV_TEST_ROW(NVCV_SHAPE(25, 38, 13, 2), NVCV_DATA_TYPE_S8, int8_t, true, false, false, 2, false),
    NVCV_TEST_ROW(NVCV_SHAPE(25, 18, 13, 1), NVCV_DATA_TYPE_S8, int8_t, true, false, false, 3, false),
    NVCV_TEST_ROW(NVCV_SHAPE(45, 17, 11, 1), NVCV_DATA_TYPE_S8, int8_t, true, false, false, 4, false),
    NVCV_TEST_ROW(NVCV_SHAPE(22, 37, 19, 2), NVCV_DATA_TYPE_S16, int16_t, true, true, false, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(18, 27, 3, 1), NVCV_DATA_TYPE_S32, int32_t, true, false, true, 1, false),
    NVCV_TEST_ROW(NVCV_SHAPE(17, 29, 5, 2), NVCV_DATA_TYPE_U8, uint8_t, true, true, true, 2, false),
    NVCV_TEST_ROW(NVCV_SHAPE(16, 28, 4, 3), NVCV_DATA_TYPE_U8, uint8_t, true, true, true, 3, true),
    NVCV_TEST_ROW(NVCV_SHAPE(17, 27, 5, 2), NVCV_DATA_TYPE_U8, uint8_t, true, true, true, 4, true),
    NVCV_TEST_ROW(NVCV_SHAPE(40, 17, 5, 2), NVCV_DATA_TYPE_U8, uint8_t, true, true, true, 4, true),
    // Widths above one CUDA block exercise the X-reduction threshold paths for both 2D and 3D inputs.
    NVCV_TEST_ROW(NVCV_SHAPE(48, 17, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, false, true, false, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(49, 17, 1, 1), NVCV_DATA_TYPE_U8, uint8_t, false, false, true, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(48, 17, 5, 1), NVCV_DATA_TYPE_U8, uint8_t, false, true, false, 0, false),
    NVCV_TEST_ROW(NVCV_SHAPE(49, 17, 5, 1), NVCV_DATA_TYPE_U8, uint8_t, false, false, true, 0, false)
>);

// clang-format on

TYPED_TEST(OpLabel, correct_output)
{
    // First setup: get test parameters, create input and output tensors and get their data accesses

    int4           shape{type::GetValue<TypeParam, 0>};
    nvcv::DataType srcDT{type::GetValue<TypeParam, 1>};
    nvcv::DataType dstDT{srcDT.dataKind() == nvcv::DataKind::SIGNED ? nvcv::TYPE_S32 : nvcv::TYPE_U32};
    nvcv::DataType mskDT{srcDT.dataKind() == nvcv::DataKind::SIGNED ? nvcv::TYPE_S8 : nvcv::TYPE_U8};

    // Testing dstDT/mskDT with S32/S8 when srcDT is signed
    // DstT must be U32 even though dstDT may be S32 (ref. code expects it as U32 since it treated it as a mask)
    // MskT must be U8 even though mskDT may be S8 (ref. code only check if it is zero as outside the mask)

    using SrcT = type::GetType<TypeParam, 2>;
    using DstT = uint32_t;
    using MskT = uint8_t;

    bool hasBgLabel    = type::GetValue<TypeParam, 3>;
    bool hasMinThresh  = type::GetValue<TypeParam, 4>;
    bool hasMaxThresh  = type::GetValue<TypeParam, 5>;
    int  doPostFilters = type::GetValue<TypeParam, 6>;
    bool doRelabel     = type::GetValue<TypeParam, 7>;

    // @note The tensors below are defined as: input or source (src), output or destination (dst), background
    // labels (bgl), minimum threshold (min), maximum threshold (max), minimum size for islands removal (msz),
    // count of labeled regions (count) and statistics computed per labeled region (sta)

    nvcv::Tensor srcTensor;
    nvcv::Tensor dstTensor;
    nvcv::Tensor bglTensor;
    nvcv::Tensor minTensor;
    nvcv::Tensor maxTensor;
    nvcv::Tensor mszTensor;
    nvcv::Tensor cntTensor;
    nvcv::Tensor staTensor;
    nvcv::Tensor mskTensor;

    nvcv::Optional<nvcv::TensorDataStridedCuda> srcData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> dstData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> bglData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> minData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> maxData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> mszData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> cntData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> staData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> mskData;

    NVCVConnectivityType connectivity = (shape.z == 1) ? NVCV_CONNECTIVITY_4_2D : NVCV_CONNECTIVITY_6_3D;
    NVCVLabelType        assignLabels = doRelabel ? NVCV_LABEL_SEQUENTIAL : NVCV_LABEL_FAST;
    NVCVLabelMaskType    maskType     = NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY; // this is the only mask type allowed

    long maskN{shape.w % 2 == 1 ? 1 : shape.w}; // test a single mask for all N when src/dst N is odd

    long4_16a mskShape{maskN, shape.z, shape.y, shape.x}; // mskShape is NDHW whereas shape is WHDN

    long3 staShape{shape.w, 10000, (shape.z == 1) ? 7 : 9};

    // clang-format off

    if (shape.w == 1) // tensors without N in layout (single-sample problem)
    {
        if (shape.z == 1) // tensors without D in layout (2D problem)
        {
            srcTensor = nvcv::Tensor({{shape.y, shape.x}, "HW"}, srcDT);
        }
        else // tensors with D in layout (3D problem)
        {
            srcTensor = nvcv::Tensor({{shape.z, shape.y, shape.x}, "DHW"}, srcDT);
        }
    }
    else // tensors with N in layout (batched problem)
    {
        if (shape.z == 1) // tensors without D in layout (2D problem)
        {
            srcTensor = nvcv::Tensor({{shape.w, shape.y, shape.x}, "NHW"}, srcDT);
        }
        else // tensors with D in layout (3D problem)
        {
            srcTensor = nvcv::Tensor({{shape.w, shape.z, shape.y, shape.x}, "NDHW"}, srcDT);
        }
    }

    if (hasBgLabel)
    {
        bglTensor = nvcv::Tensor({{shape.w}, "N"}, srcDT);

        bglData = bglTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(bglData);
    }
    if (hasMinThresh)
    {
        minTensor = nvcv::Tensor({{shape.w}, "N"}, srcDT);

        minData = minTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(minData);
    }
    if (hasMaxThresh)
    {
        maxTensor = nvcv::Tensor({{shape.w}, "N"}, srcDT);

        maxData = maxTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(maxData);
    }
    if (doPostFilters >= 1)
    {
        cntTensor = nvcv::Tensor({{shape.w}, "N"}, dstDT);

        cntData = cntTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(cntData);
    }
    if (doPostFilters >= 2)
    {
        staTensor = nvcv::Tensor({{staShape.x, staShape.y, staShape.z}, "NMA"}, dstDT);

        staData = staTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(staData);
    }
    if (doPostFilters >= 3)
    {
        mszTensor = nvcv::Tensor({{shape.w}, "N"}, dstDT);

        mszData = mszTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(mszData);
    }
    if (doPostFilters >= 4)
    {
        mskTensor = nvcv::Tensor({{mskShape.x, mskShape.y, mskShape.z, mskShape.w}, "NDHW"}, mskDT);

        mskData = mskTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(mskData);
    }

    // clang-format on

    dstTensor = nvcv::Tensor(srcTensor.shape(), dstDT);

    srcData = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);

    dstData = dstTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(dstData);

    // Second setup: get tensors shape, strides and buffer sizes

    int4 ids{srcTensor.layout().find('N'), srcTensor.layout().find('D'), srcTensor.layout().find('H'),
             srcTensor.layout().find('W')};

    long4_16a srcShape{shape.w, shape.z, shape.y, shape.x}; // srcShape is NDHW whereas shape is WHDN

    long4_16a srcStrides{0, 0, srcData->stride(ids.z), srcData->stride(ids.w)};
    long4_16a dstStrides{0, 0, dstData->stride(ids.z), dstData->stride(ids.w)};
    long1     bglStrides{bglTensor ? bglData->stride(0) : 0};
    long1     minStrides{minTensor ? minData->stride(0) : 0};
    long1     maxStrides{maxTensor ? maxData->stride(0) : 0};
    long1     mszStrides{mszTensor ? mszData->stride(0) : 0};
    long1     cntStrides{cntTensor ? cntData->stride(0) : 0};
    long3 staStrides = staTensor ? long3{staData->stride(0), staData->stride(1), staData->stride(2)} : long3{0, 0, 0};
    long4_16a mskStrides{0, 0, 0, 0};

    if (mskTensor)
    {
        int4 maskIds{mskTensor.layout().find('N'), mskTensor.layout().find('D'), mskTensor.layout().find('H'),
                     mskTensor.layout().find('W')};

        mskStrides = long4_16a{mskData->stride(maskIds.x), mskData->stride(maskIds.y), mskData->stride(maskIds.z),
                               mskData->stride(maskIds.w)};
    }

    srcStrides.y = (ids.y == -1) ? srcStrides.z * srcShape.z : srcData->stride(ids.y);
    srcStrides.x = (ids.x == -1) ? srcStrides.y * srcShape.y : srcData->stride(ids.x);
    dstStrides.y = (ids.y == -1) ? dstStrides.z * srcShape.z : dstData->stride(ids.y);
    dstStrides.x = (ids.x == -1) ? dstStrides.y * srcShape.y : dstData->stride(ids.x);

    long srcBufSize = srcStrides.x * srcShape.x;
    long dstBufSize = dstStrides.x * srcShape.x;
    long mskBufSize = mskStrides.x * mskShape.x;
    long bglBufSize = bglStrides.x * srcShape.x;
    long minBufSize = minStrides.x * srcShape.x;
    long maxBufSize = maxStrides.x * srcShape.x;
    long mszBufSize = mszStrides.x * srcShape.x;
    long cntBufSize = cntStrides.x * srcShape.x;
    long staBufSize = staStrides.x * srcShape.x;

    // Third setup: generate raw buffer data and copy them into tensors

    RawBufferType srcVec(srcBufSize);
    RawBufferType mskVec(mskBufSize);
    RawBufferType bglVec(bglBufSize);
    RawBufferType minVec(minBufSize);
    RawBufferType maxVec(maxBufSize);
    RawBufferType mszVec(mszBufSize);

    std::default_random_engine rng(0);

    std::uniform_int_distribution<SrcT> srcRandom(0, 6);
    std::uniform_int_distribution<MskT> mskRandom(0, 1);
    std::uniform_int_distribution<SrcT> bglRandom(0, (minTensor || maxTensor) ? 1 : 6);
    std::uniform_int_distribution<SrcT> minRandom(1, 3);
    std::uniform_int_distribution<SrcT> maxRandom(3, 5);

    // clang-format off

    ref::ForEachCoord(srcShape,
                      [&srcVec, &srcStrides, &srcRandom, &rng](const long4_16a &curCoord)
                      {
                          util::ValueAt<SrcT>(srcVec, srcStrides, curCoord) = srcRandom(rng);
                      });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));

    if (bglTensor)
    {
        for (long x = 0; x < srcShape.x; ++x)
            util::ValueAt<SrcT>(bglVec, bglStrides, long1{x}) = bglRandom(rng);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(bglData->basePtr(), bglVec.data(), bglBufSize, cudaMemcpyHostToDevice));
    }
    if (minTensor)
    {
        for (long x = 0; x < srcShape.x; ++x)
            util::ValueAt<SrcT>(minVec, minStrides, long1{x}) = minRandom(rng);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(minData->basePtr(), minVec.data(), minBufSize, cudaMemcpyHostToDevice));
    }
    if (maxTensor)
    {
        for (long x = 0; x < srcShape.x; ++x)
            util::ValueAt<SrcT>(maxVec, maxStrides, long1{x}) = maxRandom(rng);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(maxData->basePtr(), maxVec.data(), maxBufSize, cudaMemcpyHostToDevice));
    }
    if (mszTensor)
    {
        for (long x = 0; x < srcShape.x; ++x)
            util::ValueAt<DstT>(mszVec, mszStrides, long1{x}) = 2;

        ASSERT_EQ(cudaSuccess, cudaMemcpy(mszData->basePtr(), mszVec.data(), mszBufSize, cudaMemcpyHostToDevice));
    }
    if (mskTensor)
    {
        ref::ForEachCoord(mskShape,
                          [&mskVec, &mskStrides, &mskRandom, &rng](const long4_16a &curCoord)
                          {
                              util::ValueAt<MskT>(mskVec, mskStrides, curCoord) = mskRandom(rng);
                          });

        ASSERT_EQ(cudaSuccess, cudaMemcpy(mskData->basePtr(), mskVec.data(), mskBufSize, cudaMemcpyHostToDevice));
    }

    // clang-format on

    // After all above setups are done, run the operator, synchronize the stream and copy its results back to host

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Label op;
    EXPECT_NO_THROW(op(stream, srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                       mskTensor, connectivity, assignLabels, maskType));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // The operator's results are named as test that must be equal to gold, the three outputs are: labels (lab),
    // count (cnt) and statistics (sta); gold statistics are not written as raw buffer, only in 3-vector form

    RawBufferType labTestVec(dstBufSize, 0);
    RawBufferType labGoldVec(dstBufSize, 0);
    RawBufferType cntTestVec(cntBufSize, 0);
    RawBufferType cntGoldVec(cntBufSize, 0);
    RawBufferType staTestVec(staBufSize, 0);

    std::vector<std::set<DstT>> testLabels(srcShape.x);
    std::vector<std::set<DstT>> goldLabels(srcShape.x);

    std::vector<std::vector<std::vector<DstT>>> testStats(srcShape.x);
    std::vector<std::vector<std::vector<DstT>>> goldStats(srcShape.x);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(labTestVec.data(), dstData->basePtr(), dstBufSize, cudaMemcpyDeviceToHost));

    // To generate the gold data, the reference code (in ref namespace) is used in a specific sequence of steps:
    // (1) pre-filter binarization uses min/max thresholds (if present) to replace input mask to binary; (2) the
    // label operation itself; (3) background labels are replaced (if present); (4) get all original gold labels;
    // (5) count the labels got; (6) compute statistics of the labeled regions; (7) get all original test labels;
    // (8) remove islands as post-filter step (if minSize tensor is present); (9) relabel to replace non-sequential
    // labels to consecutive region indices; (10) sort test statistics to be able to compare against gold.

    // In-between the generation of gold data, EXPECT_EQ is used to compare test data against gold.

    if (minTensor || maxTensor)
    {
        ref::Binarize<SrcT>(srcVec, minVec, maxVec, srcStrides, minStrides, maxStrides, srcShape);
    }

    ref::Label<SrcT, DstT>(labGoldVec, srcVec, dstStrides, srcStrides, srcShape);

    if (bglTensor)
    {
        ref::ReplaceBgLabels<SrcT, DstT>(labGoldVec, srcVec, bglVec, dstStrides, srcStrides, bglStrides, srcShape);
    }

    ref::GetLabels<SrcT, DstT>(goldLabels, labGoldVec, bglVec, dstStrides, bglStrides, srcShape);

    if (cntTensor)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(cntTestVec.data(), cntData->basePtr(), cntBufSize, cudaMemcpyDeviceToHost));

        ref::CountLabels<DstT>(cntGoldVec, cntStrides, goldLabels, srcShape.x);
    }

    EXPECT_EQ(cntTestVec, cntGoldVec);

    if (staTensor)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(staTestVec.data(), staData->basePtr(), staBufSize, cudaMemcpyDeviceToHost));

        ref::ComputeStats<SrcT, DstT, MskT>(goldStats, labGoldVec, mskVec, bglVec, dstStrides, mskStrides, bglStrides,
                                            goldLabels, srcShape, maskN, static_cast<int>(staShape.z));

        ref::GetLabels<DstT>(testLabels, cntTestVec, staTestVec, cntStrides, staStrides, srcShape.x);
    }
    else
    {
        ref::GetLabels<SrcT, DstT>(testLabels, labTestVec, bglVec, dstStrides, bglStrides, srcShape);
    }

    EXPECT_EQ(testLabels, goldLabels);

    if (mszTensor)
    {
        ref::RemoveIslands<SrcT, DstT>(goldLabels, labGoldVec, bglVec, mszVec, dstStrides, bglStrides, mszStrides,
                                       goldStats, srcShape, static_cast<int>(staShape.z));
    }

    if (doRelabel)
    {
        ref::Relabel<SrcT, DstT>(labGoldVec, bglVec, staTestVec, cntTestVec, dstStrides, bglStrides, staStrides,
                                 cntStrides, srcShape);
    }

    if (staTensor)
    {
        ref::SortStats<DstT>(testStats, testLabels, staTestVec, staStrides, staShape);
    }

    EXPECT_EQ(testStats, goldStats);

    EXPECT_EQ(labTestVec, labGoldVec);
}

template<typename T>
void UploadVectorTensor(const nvcv::Tensor &tensor, const std::vector<T> &values)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(data);
    ASSERT_EQ(static_cast<int64_t>(values.size()), data->shape(0));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(data->basePtr(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void DownloadVectorTensor(const nvcv::Tensor &tensor, std::vector<T> &values)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(data);
    values.resize(data->shape(0));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(values.data(), data->basePtr(), values.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
::testing::AssertionResult DownloadImageTensor(const nvcv::Tensor &tensor, int numSamples, std::vector<T> &values)
{
    values.clear();
    int sample = 0;
    try
    {
        for (; sample < numSamples; ++sample)
        {
            std::vector<T> sampleValues;
            util::GetImageVectorFromTensor<T>(tensor.exportData(), sample, sampleValues);
            values.insert(values.end(), sampleValues.begin(), sampleValues.end());
        }
    }
    catch (const util::TensorDataUtilsError &e)
    {
        return ::testing::AssertionFailure() << "sample " << sample << ": " << e.what();
    }
    return ::testing::AssertionSuccess();
}

static std::vector<uint8_t> MakeLabelParityInput(int width, int height)
{
    std::vector<uint8_t> values(width * height);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = static_cast<uint8_t>((i * 7 + 13) % 7);
    }
    return values;
}

static ::testing::AssertionResult UploadLabelParityInputs(nvcv::Tensor &interleaved, nvcv::Tensor &planar,
                                                          std::vector<uint8_t> &values, int numSamples)
{
    int         sample     = 0;
    const char *tensorName = "interleaved";
    try
    {
        for (; sample < numSamples; ++sample)
        {
            tensorName = "interleaved";
            util::SetImageTensorFromVector<uint8_t>(interleaved.exportData(), values, sample);

            tensorName = "planar";
            util::SetImageTensorFromVector<uint8_t>(planar.exportData(), values, sample);
        }
    }
    catch (const util::TensorDataUtilsError &e)
    {
        return ::testing::AssertionFailure() << tensorName << " sample " << sample << ": " << e.what();
    }
    return ::testing::AssertionSuccess();
}

static nvcv::TensorShape MakeLabelTensorShape(int numSamples, int width, int height, const std::string &layout)
{
    if (layout == "HWC" || layout == "CHW")
    {
        return layout == "HWC" ? nvcv::TensorShape{{height, width, 1}, layout.c_str()}
                               : nvcv::TensorShape{{1, height, width}, layout.c_str()};
    }

    return layout == "NHWC" ? nvcv::TensorShape{{numSamples, height, width, 1}, layout.c_str()}
                            : nvcv::TensorShape{{numSamples, 1, height, width}, layout.c_str()};
}

static void RunLabelPlanarParityCase(int numSamples, int width, int height, const std::string &interleavedLayout,
                                     const std::string &planarLayout)
{
    nvcv::TensorShape interleavedShape = MakeLabelTensorShape(numSamples, width, height, interleavedLayout);
    nvcv::TensorShape planarShape      = MakeLabelTensorShape(numSamples, width, height, planarLayout);

    nvcv::Tensor srcInterleaved(interleavedShape, nvcv::TYPE_U8);
    nvcv::Tensor dstInterleaved(interleavedShape, nvcv::TYPE_U32);
    nvcv::Tensor srcPlanar(planarShape, nvcv::TYPE_U8);
    nvcv::Tensor dstPlanar(planarShape, nvcv::TYPE_U32);

    nvcv::Tensor bgInterleaved({{numSamples}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor minInterleaved({{numSamples}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor maxInterleaved({{numSamples}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor countInterleaved({{numSamples}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor bgPlanar({{numSamples}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor minPlanar({{numSamples}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor maxPlanar({{numSamples}, "N"}, nvcv::TYPE_U8);
    nvcv::Tensor countPlanar({{numSamples}, "N"}, nvcv::TYPE_U32);

    auto srcValues = MakeLabelParityInput(width, height);
    ASSERT_TRUE(UploadLabelParityInputs(srcInterleaved, srcPlanar, srcValues, numSamples));

    std::vector<uint8_t> bgValues(numSamples, 0);
    std::vector<uint8_t> minValues(numSamples, 1);
    std::vector<uint8_t> maxValues(numSamples, 5);
    UploadVectorTensor(bgInterleaved, bgValues);
    UploadVectorTensor(bgPlanar, bgValues);
    UploadVectorTensor(minInterleaved, minValues);
    UploadVectorTensor(minPlanar, minValues);
    UploadVectorTensor(maxInterleaved, maxValues);
    UploadVectorTensor(maxPlanar, maxValues);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Label op;
    ASSERT_NO_THROW(op(stream, srcInterleaved, dstInterleaved, bgInterleaved, minInterleaved, maxInterleaved,
                       nvcv::Tensor{nullptr}, countInterleaved, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                       NVCV_CONNECTIVITY_4_2D, NVCV_LABEL_FAST, NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY));
    ASSERT_NO_THROW(op(stream, srcPlanar, dstPlanar, bgPlanar, minPlanar, maxPlanar, nvcv::Tensor{nullptr}, countPlanar,
                       nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr}, NVCV_CONNECTIVITY_4_2D, NVCV_LABEL_FAST,
                       NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint32_t> dstInterleavedValues;
    std::vector<uint32_t> dstPlanarValues;
    ASSERT_TRUE(DownloadImageTensor(dstInterleaved, numSamples, dstInterleavedValues));
    ASSERT_TRUE(DownloadImageTensor(dstPlanar, numSamples, dstPlanarValues));
    EXPECT_EQ(dstInterleavedValues, dstPlanarValues);

    std::vector<uint32_t> countInterleavedValues;
    std::vector<uint32_t> countPlanarValues;
    DownloadVectorTensor(countInterleaved, countInterleavedValues);
    DownloadVectorTensor(countPlanar, countPlanarValues);
    EXPECT_EQ(countInterleavedValues, countPlanarValues);
}

// clang-format off
NVCV_TEST_SUITE_P(OpLabelPlanar, test::ValueList<int, int, int, std::string, std::string>{
    // samples, width, height, interleavedLayout, planarLayout
    {1, 19, 17,  "HWC",  "CHW"},
    {2, 23, 13, "NHWC", "NCHW"},
});

// clang-format on

TEST_P(OpLabelPlanar, tensor_matches_interleaved)
{
    RunLabelPlanarParityCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                             GetParamValue<4>());
}

struct OpLabel_Negative : public ::testing::Test // NOSONAR: negative tests keep shared invalid fixtures together.
{
    void SetUp() override
    {
        shape = {33, 16, 1, 1};
        srcDT = nvcv::TYPE_U8;
        dstDT = nvcv::TYPE_U32;
        mskDT = nvcv::TYPE_U8;

        connectivity = NVCV_CONNECTIVITY_4_2D;
        assignLabels = NVCV_LABEL_SEQUENTIAL;
        maskType     = NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY; // this is the only mask type allowed

        mskShape = {1, shape.z, shape.y, shape.x}; // mskShape is NDHW whereas shape is WHDN
        staShape = {1, 10000, (shape.z == 1) ? 7 : 9};

        // clang-format off
        srcTensor = nvcv::Tensor({{shape.y, shape.x}, "HW"}, srcDT);

        bglTensor = nvcv::Tensor({{shape.w}, "N"}, srcDT);
        minTensor = nvcv::Tensor({{shape.w}, "N"}, srcDT);
        maxTensor = nvcv::Tensor({{shape.w}, "N"}, srcDT);
        cntTensor = nvcv::Tensor({{shape.w}, "N"}, dstDT);
        staTensor = nvcv::Tensor({{staShape.x, staShape.y, 7}, "NMA"}, dstDT);
        staTensor3D = nvcv::Tensor({{staShape.x, staShape.y, 9}, "NMA"}, dstDT);
        mszTensor = nvcv::Tensor({{shape.w}, "N"}, dstDT);
        mskTensor = nvcv::Tensor({{mskShape.x, mskShape.y, mskShape.z, mskShape.w}, "NDHW"}, mskDT);

        dstTensor = nvcv::Tensor(srcTensor.shape(), dstDT);
        // clang-format on

        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    void runOpLabelNegativeTest(const nvcv::Tensor &testSrcTensor, const nvcv::Tensor &testDstTensor,
                                const nvcv::Tensor &testBglTensor, const nvcv::Tensor &testMinTensor,
                                const nvcv::Tensor &testMaxTensor, const nvcv::Tensor &testMszTensor,
                                const nvcv::Tensor &testCntTensor, const nvcv::Tensor &testStaTensor,
                                const nvcv::Tensor &testMskTensor, NVCVConnectivityType testConnectivity,
                                NVCVLabelType testAssignLabels, NVCVLabelMaskType testMaskType)
    {
        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall(
                [this, &testSrcTensor, &testDstTensor, &testBglTensor, &testMinTensor, &testMaxTensor, &testMszTensor,
                 &testCntTensor, &testStaTensor, &testMskTensor, &testConnectivity, &testAssignLabels, &testMaskType]
                {
                    op(stream, testSrcTensor, testDstTensor, testBglTensor, testMinTensor, testMaxTensor, testMszTensor,
                       testCntTensor, testStaTensor, testMskTensor, testConnectivity, testAssignLabels, testMaskType);
                }));
        std::array<char, 1024> msg;
        nvcvGetLastErrorMessage(msg.data(), msg.size());
        std::cout << "\033[33m" << msg.data() << "\033[0m" << std::endl;
    }

    int4           shape;
    nvcv::DataType srcDT;
    nvcv::DataType dstDT;
    nvcv::DataType mskDT;

    NVCVConnectivityType connectivity;
    NVCVLabelType        assignLabels;
    NVCVLabelMaskType    maskType;

    long4_16a mskShape;
    long3     staShape;

    nvcv::Tensor srcTensor;

    nvcv::Tensor bglTensor;
    nvcv::Tensor minTensor;
    nvcv::Tensor maxTensor;
    nvcv::Tensor cntTensor;
    nvcv::Tensor staTensor;
    nvcv::Tensor staTensor3D;
    nvcv::Tensor mszTensor;
    nvcv::Tensor mskTensor;

    nvcv::Tensor dstTensor;

    cudaStream_t  stream;
    cvcuda::Label op;
};

// clang-format off
TEST_F(OpLabel_Negative, InvalidSourceLayout)
{
    nvcv::Tensor srcTensorInvalidLayout({{1, shape.y, shape.x},"WHC"},srcDT);
    runOpLabelNegativeTest(srcTensorInvalidLayout, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidSourceChannelNum)
{
    nvcv::Tensor srcTensorInvalidChannelNum({{shape.y, shape.x, 3},"HWC"},srcDT);
    nvcv::Tensor dstTensorHWCLayout({{shape.y, shape.x, 3},"HWC"},dstDT);
    runOpLabelNegativeTest(srcTensorInvalidChannelNum, dstTensorHWCLayout, bglTensor, minTensor, maxTensor, mszTensor,
                           cntTensor, staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, Tensor3DWith2DConnectivity)
{
    nvcv::Tensor srcTensor3D({{shape.w, 2, shape.y, shape.x},"NDHW"},srcDT);
    nvcv::Tensor dstTensor3D(srcTensor3D.shape(), dstDT);
    runOpLabelNegativeTest(srcTensor3D, dstTensor3D, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, Tensor2DWith3DConnectivity)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensor, NVCV_CONNECTIVITY_6_3D, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidDestinationLayout)
{
    nvcv::Tensor dstTensorHWCLayout({{shape.y, shape.x, 3},"HWC"},dstDT);
    runOpLabelNegativeTest(srcTensor, dstTensorHWCLayout, bglTensor, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidDestinationDataType)
{
    nvcv::Tensor dstTensorInvalidDtype(srcTensor.shape(), nvcv::TYPE_F32);
    runOpLabelNegativeTest(srcTensor, dstTensorInvalidDtype, bglTensor, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidBgLabelShape)
{
    nvcv::Tensor bglTensorInvalidShape({{shape.w, 2}, "NC"},srcDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensorInvalidShape, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidBgLabelDataType)
{
    nvcv::Tensor bglTensorInvalidDtype({{shape.w}, "N"}, nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensorInvalidDtype, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMinThreshShape)
{
    nvcv::Tensor minTensorInvalidShape({{shape.w, 2},"NC"},srcDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensorInvalidShape, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMinThreshDataType)
{
    nvcv::Tensor minTensorInvalidDtype({{shape.w}, "N"}, nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensorInvalidDtype, maxTensor, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMaxThreshShape)
{
    nvcv::Tensor maxTensorInvalidShape({{shape.w, 2},"NC"},srcDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensorInvalidShape, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMaxThreshDataType)
{
    nvcv::Tensor maxTensorInvalidDtype({{shape.w}, "N"}, nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensorInvalidDtype, mszTensor, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidCountShape)
{
    nvcv::Tensor cntTensorInvalidShape({{shape.w, 2},"NC"},dstDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensorInvalidShape,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidCountDataType)
{
    nvcv::Tensor cntTensorInvalidDtype({{shape.w}, "N"}, nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensorInvalidDtype,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, StatusWithoutCount)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, nvcv::Tensor{nullptr}, staTensor,
                           mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, reLabelWithoutStatus)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, nvcv::Tensor{nullptr},
                           mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidStatsShape)
{
    nvcv::Tensor staTensorInvalidShape({{shape.w, 2},"NC"},dstDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensorInvalidShape, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidStatsDataType)
{
    nvcv::Tensor staTensorInvalidDtype({{staShape.x, staShape.y, staShape.z},"NMA"},nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor,
                           staTensorInvalidDtype, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, minSizeWithoutBgLabel)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, nvcv::Tensor{nullptr}, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMinSizeShape)
{
    nvcv::Tensor minSizeInvalidShape({{shape.w, 2},"NC"},dstDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, minSizeInvalidShape, cntTensor,
                           staTensor, mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMinSizeDataType)
{
    nvcv::Tensor minSizeInvalidDtype({{shape.w}, "N"}, nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, minSizeInvalidDtype, cntTensor,
                           staTensor, minSizeInvalidDtype, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, MaskWithoutMinSize)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, nvcv::Tensor{nullptr}, cntTensor, staTensor,
                           mskTensor, connectivity, assignLabels, maskType);
}

#ifndef ENABLE_SANITIZER
TEST_F(OpLabel_Negative, InvalidMaskType)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensor, connectivity, assignLabels, static_cast<NVCVLabelMaskType>(255));
}
#endif

TEST_F(OpLabel_Negative, InvalidMaskShape)
{
    nvcv::Tensor mskTensorInvalidN({{mskShape.x + 13, mskShape.y, mskShape.z, mskShape.w},"NDHW"},mskDT);
    nvcv::Tensor mskTensorInvalidH({{mskShape.x, mskShape.y, mskShape.z + 12, mskShape.w},"NDHW"},mskDT);
    nvcv::Tensor mskTensorInvalidW({{mskShape.x, mskShape.y, mskShape.z, mskShape.w + 11},"NDHW"},mskDT);
    nvcv::Tensor mskTensorInvalidD({{mskShape.x, mskShape.y + 10, mskShape.z, mskShape.w},"NDHW"},mskDT);
    nvcv::Tensor mskTensorInvalidC({{mskShape.x, mskShape.y, mskShape.z, mskShape.w, 9},"NDHWC"},mskDT);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensorInvalidN, connectivity, assignLabels, maskType);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensorInvalidH, connectivity, assignLabels, maskType);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensorInvalidW, connectivity, assignLabels, maskType);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensorInvalidD, connectivity, assignLabels, maskType);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensorInvalidC, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, InvalidMaskDataType)
{
    nvcv::Tensor mskTensorInvalidDtype({{mskShape.x, mskShape.y, mskShape.z, mskShape.w},"NDHW"},nvcv::TYPE_U16);
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensorInvalidDtype, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, Tensor3DWith2DMask)
{
    nvcv::Tensor srcTensor3D({{shape.w, 2, shape.y, shape.x},"NDHW"},srcDT);
    nvcv::Tensor dstTensor3D(srcTensor3D.shape(), dstDT);
    nvcv::Tensor mskTensor2D({{1, mskShape.z, mskShape.w, 1},"NHWC"}, mskDT);
    runOpLabelNegativeTest(srcTensor3D, dstTensor3D, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor3D,
                           mskTensor2D, NVCV_CONNECTIVITY_6_3D, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, fullConnectivity)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
                           mskTensor, NVCV_CONNECTIVITY_8_2D, assignLabels, maskType);
}

// clang-format on

TEST_F(OpLabel_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaLabelCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
