/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpLabel.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <set>
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

// Pre-filter step is to binarize srcVec using threshold range [min, max] -> 1, zero otherwise
template<typename ST>
inline void Binarize(RawBufferType &srcVec, const RawBufferType &minVec, const RawBufferType &maxVec,
                     const long4 &srcStrides, const long1 &minStrides, const long1 &maxStrides, const long4 &shape)
{
    bool hasMinThresh = minStrides.x > 0;
    bool hasMaxThresh = maxStrides.x > 0;

    for (long x = 0; x < shape.x; ++x)
    {
        ST minThresh = hasMinThresh ? util::ValueAt<ST>(minVec, minStrides, long1{x}) : 0;
        ST maxThresh = hasMaxThresh ? util::ValueAt<ST>(maxVec, maxStrides, long1{x}) : 0;

        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    long4 curCoord{x, y, z, w};

                    ST value = util::ValueAt<ST>(srcVec, srcStrides, curCoord);

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
                }
            }
        }
    }
}

// Label each component with label in dstVec matching value in srcVec, marking labeled elements as 1 in tmpVec
// (since this function is called recursively, using big input sizes may lead to stack overflow)
template<typename ST, typename DT>
inline void LabelComponent(RawBufferType &tmpVec, RawBufferType &dstVec, const RawBufferType &srcVec,
                           const long4 &tmpStrides, const long4 &dstStrides, const long4 &srcStrides,
                           const long4 &shape, const long4 &curCoord, ST value, DT label)
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
                       long4{curCoord.x, curCoord.y - 1, curCoord.z, curCoord.w}, value, label);
    }
    if (curCoord.y < shape.y - 1)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4{curCoord.x, curCoord.y + 1, curCoord.z, curCoord.w}, value, label);
    }
    if (curCoord.z > 0)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4{curCoord.x, curCoord.y, curCoord.z - 1, curCoord.w}, value, label);
    }
    if (curCoord.z < shape.z - 1)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4{curCoord.x, curCoord.y, curCoord.z + 1, curCoord.w}, value, label);
    }
    if (curCoord.w > 0)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4{curCoord.x, curCoord.y, curCoord.z, curCoord.w - 1}, value, label);
    }
    if (curCoord.w < shape.w - 1)
    {
        LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape,
                       long4{curCoord.x, curCoord.y, curCoord.z, curCoord.w + 1}, value, label);
    }
}

// Label N volumes in NDHW tensor stored in srcVec yielding dstVec, with corresponding srcStrides/dstStrides
// - ST is the source type, the data type of the input tensor in srcVec
// - DT is the destination type, the data type of the output tensor in dstVec
template<typename ST, typename DT>
void Label(RawBufferType &dstVec, const RawBufferType &srcVec, const long4 &dstStrides, const long4 &srcStrides,
           const long4 &shape)
{
    // Use a temporary NDHW tensor stored in tmpVec to set elements already labeled, initially zeroes (all unlabeled)
    RawBufferType tmpVec(shape.x * shape.y * shape.z * shape.w, 0);

    // The temporary tensor is packed and each element is a single byte, thus:
    long4 tmpStrides{shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1};

    // For all elements in input tensor
    for (long x = 0; x < shape.x; ++x)
    {
        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    long4 curCoord{x, y, z, w};

                    if (util::ValueAt<U8>(tmpVec, tmpStrides, curCoord) == 1)
                    {
                        continue; // The element was already labeled, skip it
                    }

                    // Get current value from input tensor and set label as a 1D flattened (global) position
                    ST value = util::ValueAt<ST>(srcVec, srcStrides, curCoord);
                    DT label = y * dstStrides.y / sizeof(DT) + z * dstStrides.z / sizeof(DT) + w;

                    // Recursively call to label component
                    LabelComponent(tmpVec, dstVec, srcVec, tmpStrides, dstStrides, srcStrides, shape, curCoord, value,
                                   label);
                }
            }
        }
    }
}

// Replace labels assigned to regions marked as background in source, and fix a potential region labeled with
// background label in destination by another label (since background label is a reserved label)
template<typename ST, typename DT>
void ReplaceBgLabels(RawBufferType &dstVec, const RawBufferType &srcVec, const RawBufferType &bglVec,
                     const long4 &dstStrides, const long4 &srcStrides, const long1 &bglStrides, const long4 &shape)
{
    for (long x = 0; x < shape.x; ++x)
    {
        ST backgroundLabel = util::ValueAt<ST>(bglVec, bglStrides, long1{x});

        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    long4 curCoord{x, y, z, w};

                    ST value = util::ValueAt<ST>(srcVec, srcStrides, curCoord);
                    DT label = util::ValueAt<DT>(dstVec, dstStrides, curCoord);

                    if (value == backgroundLabel)
                    {
                        // The current value is a background label, write it to output
                        util::ValueAt<DT>(dstVec, dstStrides, curCoord) = (DT)backgroundLabel;
                    }
                    else if (label == (DT)backgroundLabel)
                    {
                        // If the label assigned happens to be the same as the background label, replace it by
                        // another label that is never assigned outside the possible offsets
                        util::ValueAt<DT>(dstVec, dstStrides, curCoord) = dstStrides.x / sizeof(DT);
                    }
                }
            }
        }
    }
}

// Get the unique set of labels from output in dstVec, disregarding background labels
template<typename ST, typename DT>
void GetLabels(std::vector<std::set<DT>> &labels, const RawBufferType &dstVec, const RawBufferType &bglVec,
               const long4 &dstStrides, const long1 &bglStrides, const long4 &dstShape)
{
    bool hasBgLabel = bglStrides.x > 0;

    for (long x = 0; x < dstShape.x; ++x)
    {
        ST backgroundLabel = hasBgLabel ? util::ValueAt<ST>(bglVec, bglStrides, long1{x}) : 0;

        for (long y = 0; y < dstShape.y; ++y)
        {
            for (long z = 0; z < dstShape.z; ++z)
            {
                for (long w = 0; w < dstShape.w; ++w)
                {
                    DT label = util::ValueAt<DT>(dstVec, dstStrides, long4{x, y, z, w});

                    if (hasBgLabel && label == (DT)backgroundLabel)
                    {
                        continue; // ignore (do not get) background labels
                    }

                    labels[x].insert(label);
                }
            }
        }
    }
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
        util::ValueAt<DT>(cntVec, cntStrides, long1{x}) = (DT)labels[x].size();
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

// Compute statistics of labeled regions
template<typename ST, typename DT, typename MT>
void ComputeStats(std::vector<std::vector<std::vector<DT>>> &stats, const RawBufferType &dstVec,
                  const RawBufferType &mskVec, const RawBufferType &bglVec, const long4 &dstStrides,
                  const long4 &mskStrides, const long1 &bglStrides, const std::vector<std::set<DT>> &labels,
                  const long4 &shape, long maskN, int numStats)
{
    // One-element-after-the-end label is a special label assigned to a region which got the background label
    DT endLabel = dstStrides.x / sizeof(DT);

    bool hasMask    = mskStrides.x > 0;
    bool hasBgLabel = bglStrides.x > 0;

    for (long x = 0; x < shape.x; ++x)
    {
        ST backgroundLabel = hasBgLabel ? util::ValueAt<ST>(bglVec, bglStrides, long1{x}) : 0;

        stats[x].resize(labels[x].size());

        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    DT   label = util::ValueAt<DT>(dstVec, dstStrides, long4{x, y, z, w});
                    auto fit   = labels[x].find(label); // result of find iterator
                    if (fit == labels[x].end())
                    {
                        continue; // this label is to be ignored
                    }

                    DT posLabel = y * dstStrides.y / sizeof(DT) + z * dstStrides.z / sizeof(DT) + w;

                    if ((hasBgLabel && label == endLabel && posLabel == (DT)backgroundLabel) || label == posLabel)
                    {
                        long regionIdx  = std::distance(labels[x].cbegin(), fit);
                        DT   regionMark = 0; // region has no marks

                        // If has mask and the element is inside the mask
                        if (hasMask && util::ValueAt<MT>(mskVec, mskStrides, long4{maskN == 1 ? 0 : x, y, z, w}) != 0)
                        {
                            regionMark = 2; // mark the region as inside the mask (= 2)
                        }

                        stats[x][regionIdx].resize(numStats);
                        stats[x][regionIdx][0] = label;
                        stats[x][regionIdx][1] = w;
                        stats[x][regionIdx][2] = z;

                        if (numStats == 7)
                        {
                            stats[x][regionIdx][3] = 1;
                            stats[x][regionIdx][4] = 1;
                            stats[x][regionIdx][5] = 1;
                            stats[x][regionIdx][6] = regionMark;
                        }
                        else
                        {
                            stats[x][regionIdx][3] = y;
                            stats[x][regionIdx][4] = 1;
                            stats[x][regionIdx][5] = 1;
                            stats[x][regionIdx][6] = 1;
                            stats[x][regionIdx][7] = 1;
                            stats[x][regionIdx][8] = regionMark;
                        }
                    }
                }
            }
        }
        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    DT   label = util::ValueAt<DT>(dstVec, dstStrides, long4{x, y, z, w});
                    auto fit   = labels[x].find(label);
                    if (fit == labels[x].end())
                    {
                        continue;
                    }

                    DT posLabel = y * dstStrides.y / sizeof(DT) + z * dstStrides.z / sizeof(DT) + w;

                    if ((hasBgLabel && label == endLabel && posLabel == (DT)backgroundLabel) || label == posLabel)
                    {
                        continue; // statistics for this element was already computed
                    }

                    long regionIdx = std::distance(labels[x].cbegin(), fit);
                    DT   bboxAreaW = std::abs(stats[x][regionIdx][1] - w) + 1;
                    DT   bboxAreaH = std::abs(stats[x][regionIdx][2] - z) + 1;

                    // If has mask and the region has no marks (it is no marked as inside mask)
                    if (hasMask && stats[x][regionIdx][numStats - 1] == 0)
                    {
                        // If element is inside mask
                        if (util::ValueAt<MT>(mskVec, mskStrides, long4{maskN == 1 ? 0 : x, y, z, w}) != 0)
                        {
                            stats[x][regionIdx][numStats - 1] = 2; // mark the region as inside mask (= 2)
                        }
                    }

                    if (numStats == 7)
                    {
                        stats[x][regionIdx][3] = std::max(stats[x][regionIdx][3], bboxAreaW);
                        stats[x][regionIdx][4] = std::max(stats[x][regionIdx][4], bboxAreaH);
                        stats[x][regionIdx][5] += 1;
                    }
                    else
                    {
                        DT bboxAreaD = std::abs(stats[x][regionIdx][3] - y) + 1;

                        stats[x][regionIdx][4] = std::max(stats[x][regionIdx][4], bboxAreaW);
                        stats[x][regionIdx][5] = std::max(stats[x][regionIdx][5], bboxAreaH);
                        stats[x][regionIdx][6] = std::max(stats[x][regionIdx][6], bboxAreaD);
                        stats[x][regionIdx][7] += 1;
                    }
                }
            }
        }
    }
}

// Remove islands (regions with less than minimum size in mszVec) from dstVec based on statistics
template<typename ST, typename DT>
void RemoveIslands(std::vector<std::set<DT>> &labels, RawBufferType &dstVec, const RawBufferType &bglVec,
                   const RawBufferType &mszVec, const long4 &dstStrides, const long1 &bglStrides,
                   const long1 &mszStrides, std::vector<std::vector<std::vector<DT>>> &stats, const long4 &shape,
                   int numStats)
{
    for (long x = 0; x < shape.x; ++x)
    {
        ST backgroundLabel = util::ValueAt<ST>(bglVec, bglStrides, long1{x});
        DT minSize         = util::ValueAt<DT>(mszVec, mszStrides, long1{x});

        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    long4 curCoord{x, y, z, w};

                    DT   label = util::ValueAt<DT>(dstVec, dstStrides, curCoord);
                    auto fit   = labels[x].find(label); // result of find iterator
                    if (fit == labels[x].end())
                    {
                        continue; // this label is to be ignored
                    }

                    long regionIdx  = std::distance(labels[x].cbegin(), fit);
                    DT   regionSize = stats[x][regionIdx][numStats - 2];

                    // If region size is smaller than minimum size (it is an island) and the region is not marked
                    // as inside the mask (= 2), then remove the island and mark it as removed
                    if (regionSize < minSize && stats[x][regionIdx][numStats - 1] != 2)
                    {
                        util::ValueAt<DT>(dstVec, dstStrides, curCoord) = backgroundLabel;

                        stats[x][regionIdx][numStats - 1] = 1;
                    }
                }
            }
        }
    }
}

// Relabel replaces index-based labels by consecutive region indices
template<typename ST, typename DT>
void Relabel(RawBufferType &dstVec, const RawBufferType &bglVec, const RawBufferType &staVec,
             const RawBufferType &cntVec, const long4 &dstStrides, const long1 &bglStrides, const long3 &staStrides,
             const long1 &cntStrides, const long4 &shape)
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
        for (long y = 0; y < shape.y; ++y)
        {
            for (long z = 0; z < shape.z; ++z)
            {
                for (long w = 0; w < shape.w; ++w)
                {
                    DT label = util::ValueAt<DT>(dstVec, dstStrides, long4{x, y, z, w});

                    if (label == (DT)backgroundLabel)
                    {
                        continue;
                    }

                    DT regionIdx = origLabelToRegionIdx[label];

                    if (regionIdx >= (DT)backgroundLabel)
                    {
                        regionIdx += 1; // increment region indices to skip background labels
                    }

                    util::ValueAt<DT>(dstVec, dstStrides, long4{x, y, z, w}) = regionIdx;
                }
            }
        }
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
    NVCV_TEST_ROW(NVCV_SHAPE(40, 17, 5, 2), NVCV_DATA_TYPE_U8, uint8_t, true, true, true, 4, true)
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

    nvcv::Tensor srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, staTensor, mskTensor;

    nvcv::Optional<nvcv::TensorDataStridedCuda> srcData, dstData, bglData, minData, maxData, mszData, cntData, staData,
        mskData;

    NVCVConnectivityType connectivity = (shape.z == 1) ? NVCV_CONNECTIVITY_4_2D : NVCV_CONNECTIVITY_6_3D;
    NVCVLabelType        assignLabels = doRelabel ? NVCV_LABEL_SEQUENTIAL : NVCV_LABEL_FAST;
    NVCVLabelMaskType    maskType     = NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY; // this is the only mask type allowed

    long maskN{shape.w % 2 == 1 ? 1 : shape.w}; // test a single mask for all N when src/dst N is odd

    long4 mskShape{maskN, shape.z, shape.y, shape.x}; // mskShape is NDHW whereas shape is WHDN

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

    long4 srcShape{shape.w, shape.z, shape.y, shape.x}; // srcShape is NDHW whereas shape is WHDN

    long4 srcStrides{0, 0, srcData->stride(ids.z), srcData->stride(ids.w)};
    long4 dstStrides{0, 0, dstData->stride(ids.z), dstData->stride(ids.w)};
    long1 bglStrides{(bglTensor) ? bglData->stride(0) : 0};
    long1 minStrides{(minTensor) ? minData->stride(0) : 0};
    long1 maxStrides{(maxTensor) ? maxData->stride(0) : 0};
    long1 mszStrides{(mszTensor) ? mszData->stride(0) : 0};
    long1 cntStrides{(cntTensor) ? cntData->stride(0) : 0};
    long3 staStrides = (staTensor) ? long3{staData->stride(0), staData->stride(1), staData->stride(2)} : long3{0, 0, 0};
    long4 mskStrides{0, 0, 0, 0};

    if (mskTensor)
    {
        int4 maskIds{mskTensor.layout().find('N'), mskTensor.layout().find('D'), mskTensor.layout().find('H'),
                     mskTensor.layout().find('W')};

        mskStrides = long4{mskData->stride(maskIds.x), mskData->stride(maskIds.y), mskData->stride(maskIds.z),
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

    for (long x = 0; x < srcShape.x; ++x)
        for (long y = 0; y < srcShape.y; ++y)
            for (long z = 0; z < srcShape.z; ++z)
                for (long w = 0; w < srcShape.w; ++w)
                    util::ValueAt<SrcT>(srcVec, srcStrides, long4{x, y, z, w}) = srcRandom(rng);

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
        for (long x = 0; x < mskShape.x; ++x)
            for (long y = 0; y < mskShape.y; ++y)
                for (long z = 0; z < mskShape.z; ++z)
                    for (long w = 0; w < mskShape.w; ++w)
                        util::ValueAt<MskT>(mskVec, mskStrides, long4{x, y, z, w}) = mskRandom(rng);

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
                                            goldLabels, srcShape, maskN, staShape.z);

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
                                       goldStats, srcShape, staShape.z);
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

class OpLabel_Negative : public ::testing::Test
{
protected:
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

    void runOpLabelNegativeTest(nvcv::Tensor srcTensor, nvcv::Tensor dstTensor, nvcv::Tensor bglTensor,
                                nvcv::Tensor minTensor, nvcv::Tensor maxTensor, nvcv::Tensor mszTensor,
                                nvcv::Tensor cntTensor, nvcv::Tensor staTensor, nvcv::Tensor mskTensor,
                                NVCVConnectivityType connectivity, NVCVLabelType assignLabels,
                                NVCVLabelMaskType maskType)
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&]
                                                   {
                                                       op(stream, srcTensor, dstTensor, bglTensor, minTensor, maxTensor,
                                                          mszTensor, cntTensor, staTensor, mskTensor, connectivity,
                                                          assignLabels, maskType);
                                                   }));
        char msg[1024];
        nvcvGetLastErrorMessage(msg, sizeof(msg));
        std::cout << "\033[33m" << msg << "\033[0m" << std::endl;
    }

    int4           shape;
    nvcv::DataType srcDT;
    nvcv::DataType dstDT;
    nvcv::DataType mskDT;

    NVCVConnectivityType connectivity;
    NVCVLabelType        assignLabels;
    NVCVLabelMaskType    maskType;

    long4 mskShape;
    long3 staShape;

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
    nvcv::Tensor srcTensorInvalidLayout({{1, shape.y, shape.x},"CHW"},srcDT);
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
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, nullptr, staTensor,
                           mskTensor, connectivity, assignLabels, maskType);
}

TEST_F(OpLabel_Negative, reLabelWithoutStatus)
{
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, mszTensor, cntTensor, nullptr,
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
    runOpLabelNegativeTest(srcTensor, dstTensor, nullptr, minTensor, maxTensor, mszTensor, cntTensor, staTensor,
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
    runOpLabelNegativeTest(srcTensor, dstTensor, bglTensor, minTensor, maxTensor, nullptr, cntTensor, staTensor,
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
