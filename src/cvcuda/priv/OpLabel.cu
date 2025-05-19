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

// MIT License

// Copyright (c) 2018 - Daniel Peter Playne

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @note The CUDA kernels implemented below are based on the paper:
 * D. P. Playne and K. Hawick,
 * "A New Algorithm for Parallel Connected-Component Labelling on GPUs,"
 * in IEEE Transactions on Parallel and Distributed Systems,
 * vol. 29, no. 6, pp. 1217-1230, 1 June 2018.
 */

#include "Assert.h"
#include "OpLabel.hpp"

#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <sstream>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

constexpr int REGION_NOT_MARKED  = 0;
constexpr int REGION_REMOVED     = 1;
constexpr int REGION_INSIDE_MASK = 2;

template<typename T>
using ArgWrap = cuda::Tensor1DWrap<T, int32_t>;

// CUDA kernels ----------------------------------------------------------------

template<typename DT>
__device__ DT FindRoot(DT *labels, DT label)
{
    DT next = labels[label];

    while (label != next)
    {
        label = next;
        next  = labels[label];
    }

    return label;
}

template<typename DT>
__device__ DT Reduction(DT *labels, DT label1, DT label2)
{
    DT next1 = (label1 != label2) ? labels[label1] : 0;
    DT next2 = (label1 != label2) ? labels[label2] : 0;

    while ((label1 != label2) && (label1 != next1))
    {
        label1 = next1;
        next1  = labels[label1];
    }

    while ((label1 != label2) && (label2 != next2))
    {
        label2 = next2;
        next2  = labels[label2];
    }

    DT label3;

    while (label1 != label2)
    {
        if (label1 < label2)
        {
            label3 = label1;
            label1 = label2;
            label2 = label3;
        }

        label3 = atomicMin(&labels[label1], label2);
        label1 = (label1 == label3) ? label2 : label3;
    }

    return label1;
}

// -- 2D kernels --

template<int BW, int BH, typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void BlockLabel2D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int2 size)
{
    using DT = typename DstWrap::ValueType;
    __shared__ DT labels[BW * BH];

    int2 tc = cuda::StaticCast<int>(cuda::DropCast<2>(threadIdx));
    int3 gc{(int)(blockIdx.x * BW) + tc.x, (int)(blockIdx.y * BH) + tc.y, (int)blockIdx.z};

    bool nym1x, nyxm1, nym1xm1;
    DT   label1;

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);
    ST   minThreshold = hasMinThresh ? minThresh[gc.z] : 0;
    ST   maxThreshold = hasMaxThresh ? maxThresh[gc.z] : 0;

    if (gc.x < size.x && gc.y < size.y)
    {
        ST pyx   = src[gc];
        ST pym1x = (tc.y > 0) ? *src.ptr(gc.z, gc.y - 1, gc.x) : 0;

        if (hasMinThresh && hasMaxThresh)
        {
            pyx   = pyx < minThreshold || pyx > maxThreshold ? 0 : 1;
            pym1x = (tc.y > 0) ? (pym1x < minThreshold || pym1x > maxThreshold ? 0 : 1) : 0;
        }
        else if (hasMinThresh)
        {
            pyx   = pyx < minThreshold ? 0 : 1;
            pym1x = (tc.y > 0) ? (pym1x < minThreshold ? 0 : 1) : 0;
        }
        else if (hasMaxThresh)
        {
            pyx   = pyx > maxThreshold ? 0 : 1;
            pym1x = (tc.y > 0) ? (pym1x > maxThreshold ? 0 : 1) : 0;
        }

        ST pyxm1   = __shfl_up_sync(__activemask(), pyx, 1);
        ST pym1xm1 = __shfl_up_sync(__activemask(), pym1x, 1);

        nym1x   = (tc.y > 0) ? (pyx == pym1x) : false;
        nyxm1   = (tc.x > 0) ? (pyx == pyxm1) : false;
        nym1xm1 = (tc.y > 0 && tc.x > 0) ? (pyx == pym1xm1) : false;

        label1 = (nyxm1) ? tc.y * BW + (tc.x - 1) : tc.y * BW + tc.x;
        label1 = (nym1x) ? (tc.y - 1) * BW + tc.x : label1;

        labels[tc.y * BW + tc.x] = label1;
    }

    __syncthreads();

    if (gc.x < size.x && gc.y < size.y)
    {
        labels[tc.y * BW + tc.x] = FindRoot(labels, label1);
    }

    __syncthreads();

    if (gc.x < size.x && gc.y < size.y)
    {
        if (nym1x && nyxm1 && !nym1xm1)
        {
            DT label2 = labels[tc.y * BW + tc.x - 1];

            label1 = Reduction(labels, label1, label2);
        }
    }

    __syncthreads();

    if (gc.x < size.x && gc.y < size.y)
    {
        label1 = FindRoot(labels, label1);

        DT lx = label1 % BW;
        DT ly = (label1 / BW) % BH;

        DT dstStrideH = dst.strides()[1] / sizeof(DT);

        dst[gc] = (blockIdx.y * BH + ly) * dstStrideH + blockIdx.x * BW + lx;
    }
}

template<typename SrcWrap, typename DstWrap, typename ST = typename SrcWrap::ValueType>
__global__ void YLabelReduction2D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int2 size)
{
    using DT = typename DstWrap::ValueType;
    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.y + blockDim.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);
    ST   minThreshold = hasMinThresh ? minThresh[gc.z] : 0;
    ST   maxThreshold = hasMaxThresh ? maxThresh[gc.z] : 0;

    ST pyx   = src[gc];
    ST pym1x = *src.ptr(gc.z, gc.y - 1, gc.x);

    if (hasMinThresh && hasMaxThresh)
    {
        pyx   = pyx < minThreshold || pyx > maxThreshold ? 0 : 1;
        pym1x = pym1x < minThreshold || pym1x > maxThreshold ? 0 : 1;
    }
    else if (hasMinThresh)
    {
        pyx   = pyx < minThreshold ? 0 : 1;
        pym1x = pym1x < minThreshold ? 0 : 1;
    }
    else if (hasMaxThresh)
    {
        pyx   = pyx > maxThreshold ? 0 : 1;
        pym1x = pym1x > maxThreshold ? 0 : 1;
    }

    ST pyxm1   = __shfl_up_sync(0xffffffff, pyx, 1);
    ST pym1xm1 = __shfl_up_sync(0xffffffff, pym1x, 1);

    if ((pyx == pym1x) && ((threadIdx.x == 0) || (pyx != pyxm1) || (pyx != pym1xm1)))
    {
        DT label1 = dst[gc];
        DT label2 = *dst.ptr(gc.z, gc.y - 1, gc.x);

        Reduction(dst.ptr(gc.z), label1, label2);
    }
}

template<typename SrcWrap, typename DstWrap, typename ST = typename SrcWrap::ValueType>
__global__ void XLabelReduction2D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int2 size)
{
    using DT = typename DstWrap::ValueType;

    int3 gc;
    gc.x = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x + blockDim.x;
    gc.y = blockIdx.x * blockDim.x + threadIdx.x;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);
    ST   minThreshold = hasMinThresh ? minThresh[gc.z] : 0;
    ST   maxThreshold = hasMaxThresh ? maxThresh[gc.z] : 0;

    ST pyx   = src[gc];
    ST pyxm1 = *src.ptr(gc.z, gc.y, gc.x - 1);

    if (hasMinThresh && hasMaxThresh)
    {
        pyx   = pyx < minThreshold || pyx > maxThreshold ? 0 : 1;
        pyxm1 = pyxm1 < minThreshold || pyxm1 > maxThreshold ? 0 : 1;
    }
    else if (hasMinThresh)
    {
        pyx   = pyx < minThreshold ? 0 : 1;
        pyxm1 = pyxm1 < minThreshold ? 0 : 1;
    }
    else if (hasMaxThresh)
    {
        pyx   = pyx > maxThreshold ? 0 : 1;
        pyxm1 = pyxm1 > maxThreshold ? 0 : 1;
    }

    bool thread_y = (gc.y % blockDim.y) == 0;

    ST pym1x   = __shfl_up_sync(0xffffffff, pyx, 1);
    ST pym1xm1 = __shfl_up_sync(0xffffffff, pyxm1, 1);

    if ((pyx == pyxm1) && (thread_y || (pyx != pym1x) || (pyx != pym1xm1)))
    {
        DT label1 = dst[gc];
        DT label2 = *dst.ptr(gc.z, gc.y, gc.x - 1);

        Reduction(dst.ptr(gc.z), label1, label2);
    }
}

template<typename DstWrap>
__global__ void ResolveLabels2D(DstWrap dst, int2 size)
{
    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    dst[gc] = FindRoot(dst.ptr(gc.z), dst[gc]);
}

template<typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void ReplaceBgLabels2D(DstWrap dst, SrcWrap src, ArgWrap<ST> bgLabel, ArgWrap<ST> minThresh,
                                  ArgWrap<ST> maxThresh, int2 size)
{
    using DT = typename DstWrap::ValueType;

    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);
    ST   minThreshold = hasMinThresh ? minThresh[gc.z] : 0;
    ST   maxThreshold = hasMaxThresh ? maxThresh[gc.z] : 0;

    ST pyx = src[gc];

    if (hasMinThresh && hasMaxThresh)
    {
        pyx = pyx < minThreshold || pyx > maxThreshold ? 0 : 1;
    }
    else if (hasMinThresh)
    {
        pyx = pyx < minThreshold ? 0 : 1;
    }
    else if (hasMaxThresh)
    {
        pyx = pyx > maxThreshold ? 0 : 1;
    }

    ST backgroundLabel = bgLabel[gc.z];

    // If src has bg label, put it in dst; if dst has bg label, it means a wrong label was assigned to a region,
    // replace its label by a label never assigned, the stride zero meaning one-element-after-the-end stride

    if (pyx == backgroundLabel)
    {
        dst[gc] = backgroundLabel;
    }
    else if (dst[gc] == (DT)backgroundLabel)
    {
        dst[gc] = dst.strides()[0] / sizeof(DT);
    }
}

template<typename DstWrap, typename StatsWrap, typename ST, typename DT = typename DstWrap::ValueType>
__global__ void CountLabels2D(ArgWrap<DT> count, StatsWrap stats, DstWrap dst, ArgWrap<ST> bgLabel, int2 size,
                              int maxCapacity)
{
    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    bool hasBgLabel      = (bgLabel.ptr(0) != nullptr);
    ST   backgroundLabel = hasBgLabel ? bgLabel[gc.z] : 0;

    DT label = dst[gc];

    if (hasBgLabel && label == (DT)backgroundLabel)
    {
        return; // do not count background labels
    }

    DT posLabel = gc.y * dst.strides()[1] / sizeof(DT) + gc.x;
    DT endLabel = dst.strides()[0] / sizeof(DT);

    DT   regionIdx;
    bool counted = false;

    if (hasBgLabel && label == endLabel && posLabel == (DT)backgroundLabel)
    {
        // This is a special region marked with one-element-after-the-end label, count it
        regionIdx = atomicAdd(count.ptr(gc.z), 1);
        counted   = true;
    }
    else if (label == posLabel)
    {
        // This is the first element of a regular region, count it
        regionIdx = atomicAdd(count.ptr(gc.z), 1);
        counted   = true;
    }

    // If statistics should be computed and the region index is inside the allowed storage (the M maximum
    // capacity in stats tensor), replace the output label by the region index and store initial statistics

    if (counted && stats.ptr(0) != nullptr && regionIdx < maxCapacity)
    {
        // TODO: improve the mark of output label as region index with 1 in the 1st bit
        dst[gc] = regionIdx | (DT)(1 << 31);

        *stats.ptr(gc.z, (int)regionIdx, 0) = label;
        *stats.ptr(gc.z, (int)regionIdx, 1) = (DT)gc.x;
        *stats.ptr(gc.z, (int)regionIdx, 2) = (DT)gc.y;
        *stats.ptr(gc.z, (int)regionIdx, 3) = 1;
        *stats.ptr(gc.z, (int)regionIdx, 4) = 1;
        *stats.ptr(gc.z, (int)regionIdx, 5) = 1;
        *stats.ptr(gc.z, (int)regionIdx, 6) = REGION_NOT_MARKED;
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename MaskWrap>
__global__ void ComputeStats2D(StatsWrap stats, DstWrap dst, MaskWrap mask, ArgWrap<ST> bgLabel, int2 size, int maskN,
                               bool relabel)
{
    using DT = typename DstWrap::ValueType;
    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    bool hasMask         = (mask.ptr(0) != nullptr);
    bool isInsideMask    = false;
    bool hasBgLabel      = (bgLabel.ptr(0) != nullptr);
    ST   backgroundLabel = hasBgLabel ? bgLabel[gc.z] : 0;
    DT   endLabel        = dst.strides()[0] / sizeof(DT);
    DT   label           = dst[gc];
    DT   regionIdx       = 0;

    if (hasMask)
    {
        int3 mc{gc.x, gc.y, maskN == 1 ? 0 : gc.z};

        isInsideMask = mask[mc] == 0 ? false : true; // mask value = 0 means outside the mask
    }

    if (hasBgLabel && label == (DT)backgroundLabel)
    {
        return; // do not compute statistics for background labels
    }

    if (label & (DT)(1 << 31))
    {
        if (isInsideMask)
        {
            regionIdx = label & (DT) ~(1 << 31);

            *stats.ptr(gc.z, (int)regionIdx, 6) = REGION_INSIDE_MASK; // region is inside the mask, mark it as such
        }

        return; // label is marked as region index, its statistics is already computed
    }

    if (hasBgLabel && label == endLabel)
    {
        // This is a special region marked with one-element-after-the-end label, its label was the backgroundLabel
        label = backgroundLabel;
    }

    regionIdx = dst.ptr(gc.z)[label];

    if (regionIdx & (DT)(1 << 31))
    {
        regionIdx = regionIdx & (DT) ~(1 << 31);

        if (relabel)
        {
            if (hasBgLabel && regionIdx >= (DT)backgroundLabel)
            {
                dst[gc] = regionIdx + 1; // skip one region index equals to background label when relabeling
            }
            else
            {
                dst[gc] = regionIdx;
            }
        }

        int2 cornerPos{(int)*stats.ptr(gc.z, (int)regionIdx, 1), (int)*stats.ptr(gc.z, (int)regionIdx, 2)};

        int2 bboxArea = cuda::abs(cornerPos - cuda::DropCast<2>(gc)) + 1;

        atomicMax(stats.ptr(gc.z, (int)regionIdx, 3), (DT)bboxArea.x);
        atomicMax(stats.ptr(gc.z, (int)regionIdx, 4), (DT)bboxArea.y);
        atomicAdd(stats.ptr(gc.z, (int)regionIdx, 5), 1);

        if (isInsideMask)
        {
            *stats.ptr(gc.z, (int)regionIdx, 6) = REGION_INSIDE_MASK; // region is inside the mask, mark it as such
        }
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename DT = typename DstWrap::ValueType>
__global__ void RemoveIslands2D(StatsWrap stats, DstWrap dst, ArgWrap<ST> bgLabel, ArgWrap<DT> minSize, int2 size,
                                bool relabel, bool hasMask)
{
    static_assert(std::is_same_v<DT, typename StatsWrap::ValueType>);
    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    DT endLabel = dst.strides()[0] / sizeof(DT);

    DT label = dst[gc];

    ST backgroundLabel = bgLabel[gc.z];

    if (label == (DT)backgroundLabel)
    {
        return;
    }
    if (label == endLabel)
    {
        // This is a special region marked with one-element-after-the-end label, its label was the backgroundLabel
        label = backgroundLabel;
    }

    DT regionIdx = 0;

    if (!(label & (DT)(1 << 31)))
    {
        if (relabel)
        {
            if (label >= (DT)backgroundLabel + 1)
            {
                regionIdx = label - 1; // go back one region index to account for background label
            }
            else
            {
                regionIdx = label;
            }
        }
        else
        {
            regionIdx = dst.ptr(gc.z)[label];

            if (regionIdx & (DT)(1 << 31))
            {
                regionIdx = regionIdx & (DT) ~(1 << 31);
            }
            else
            {
                return; // invalid region index
            }
        }
    }
    else
    {
        return; // should not remove first region element with 1st bit 1 so other elements are not lost
    }

    DT regionSize = *stats.ptr(gc.z, (int)regionIdx, 5);

    // If region size is less than minimum size, it is an island and should be removed, i.e. set to background label
    if (regionSize < minSize[gc.z])
    {
        // If there is no mask or if there is a mask and the region mark is not 2, meaning the region is not
        // inside the mask, the region should be removed
        if (!hasMask || *stats.ptr(gc.z, (int)regionIdx, 6) != REGION_INSIDE_MASK)
        {
            dst[gc] = backgroundLabel;
        }
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename DT = typename DstWrap::ValueType>
__global__ void Relabel2D(StatsWrap stats, DstWrap dst, ArgWrap<ST> bgLabel, ArgWrap<DT> minSize, int2 size,
                          bool relabel, bool hasMask)
{
    static_assert(std::is_same_v<DT, typename StatsWrap::ValueType>);
    int3 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z;

    if (gc.x >= size.x || gc.y >= size.y)
    {
        return;
    }

    bool removeIsland = minSize.ptr(0) != nullptr;

    DT label = dst[gc];

    if (label & (DT)(1 << 31))
    {
        // Label is marked as region index, relabel it back to proper label
        DT regionIdx = label & (DT) ~(1 << 31);

        if (relabel)
        {
            bool hasBgLabel      = (bgLabel.ptr(0) != nullptr);
            ST   backgroundLabel = hasBgLabel ? bgLabel[gc.z] : 0;

            if (hasBgLabel && regionIdx >= (DT)backgroundLabel)
            {
                dst[gc] = regionIdx + 1; // skip one region index equals to background label when relabeling
            }
            else
            {
                dst[gc] = regionIdx;
            }
        }
        else
        {
            dst[gc] = *stats.ptr(gc.z, (int)regionIdx, 0);
        }

        if (removeIsland)
        {
            DT regionSize = *stats.ptr(gc.z, (int)regionIdx, 5);

            if (regionSize < minSize[gc.z])
            {
                if (!hasMask || *stats.ptr(gc.z, (int)regionIdx, 6) != REGION_INSIDE_MASK)
                {
                    dst[gc] = (DT)bgLabel[gc.z];

                    *stats.ptr(gc.z, (int)regionIdx, 6) = REGION_REMOVED;
                }
            }
        }
    }
}

// -- 3D kernels --

template<int BW, int BH, int BD, typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void BlockLabel3D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int4 shape)
{
    using DT = typename DstWrap::ValueType;

    __shared__ DT labels[BW * BH * BD];

    int3 tc = cuda::StaticCast<int>(threadIdx);
    int4 gc{(int)blockIdx.x * BW + tc.x, (int)blockIdx.y * BH + tc.y, (int)blockIdx.z * BD + tc.z, 0};

    bool nzm1yx, nzym1x, nzyxm1, nzym1xm1, nzm1yxm1, nzm1ym1x;
    DT   label;

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST minThreshold = hasMinThresh ? minThresh[gc.w] : 0;
        ST maxThreshold = hasMaxThresh ? maxThresh[gc.w] : 0;

        if (gc.x < shape.x && gc.y < shape.y && gc.z < shape.z)
        {
            ST pzyx     = src[gc];
            ST pzym1x   = (tc.y > 0) ? *src.ptr(gc.w, gc.z, gc.y - 1, gc.x) : 0;
            ST pzm1yx   = (tc.z > 0) ? *src.ptr(gc.w, gc.z - 1, gc.y, gc.x) : 0;
            ST pzm1ym1x = (tc.z > 0 && tc.y > 0) ? *src.ptr(gc.w, gc.z - 1, gc.y - 1, gc.x) : 0;

            if (hasMinThresh && hasMaxThresh)
            {
                pzyx     = pzyx < minThreshold || pzyx > maxThreshold ? 0 : 1;
                pzym1x   = (tc.y > 0) ? (pzym1x < minThreshold || pzym1x > maxThreshold ? 0 : 1) : 0;
                pzm1yx   = (tc.z > 0) ? (pzm1yx < minThreshold || pzm1yx > maxThreshold ? 0 : 1) : 0;
                pzm1ym1x = (tc.z > 0 && tc.y > 0) ? (pzm1ym1x < minThreshold || pzm1ym1x > maxThreshold ? 0 : 1) : 0;
            }
            else if (hasMinThresh)
            {
                pzyx     = pzyx < minThreshold ? 0 : 1;
                pzym1x   = (tc.y > 0) ? (pzym1x < minThreshold ? 0 : 1) : 0;
                pzm1yx   = (tc.z > 0) ? (pzm1yx < minThreshold ? 0 : 1) : 0;
                pzm1ym1x = (tc.z > 0 && tc.y > 0) ? (pzm1ym1x < minThreshold ? 0 : 1) : 0;
            }
            else if (hasMaxThresh)
            {
                pzyx     = pzyx > maxThreshold ? 0 : 1;
                pzym1x   = (tc.y > 0) ? (pzym1x > maxThreshold ? 0 : 1) : 0;
                pzm1yx   = (tc.z > 0) ? (pzm1yx > maxThreshold ? 0 : 1) : 0;
                pzm1ym1x = (tc.z > 0 && tc.y > 0) ? (pzm1ym1x > maxThreshold ? 0 : 1) : 0;
            }

            ST pzyxm1   = __shfl_up_sync(__activemask(), pzyx, 1);
            ST pzym1xm1 = __shfl_up_sync(__activemask(), pzym1x, 1);
            ST pzm1yxm1 = __shfl_up_sync(__activemask(), pzm1yx, 1);

            nzm1yx = (tc.z > 0) && (pzyx == pzm1yx);
            nzym1x = (tc.y > 0) && (pzyx == pzym1x);
            nzyxm1 = (tc.x > 0) && (pzyx == pzyxm1);

            nzym1xm1 = ((tc.y > 0) && (tc.x > 0) && (pzyx == pzym1xm1));
            nzm1yxm1 = ((tc.z > 0) && (tc.x > 0) && (pzyx == pzm1yxm1));
            nzm1ym1x = ((tc.z > 0) && (tc.y > 0) && (pzyx == pzm1ym1x));

            label = (nzyxm1) ? (tc.z * BW * BH + tc.y * BW + (tc.x - 1)) : (tc.z * BW * BH + tc.y * BW + tc.x);
            label = (nzym1x) ? (tc.z * BW * BH + (tc.y - 1) * BW + tc.x) : label;
            label = (nzm1yx) ? ((tc.z - 1) * BW * BH + tc.y * BW + tc.x) : label;

            labels[tc.z * BW * BH + tc.y * BW + tc.x] = label;
        }

        __syncthreads();

        if (gc.x < shape.x && gc.y < shape.y && gc.z < shape.z)
        {
            labels[tc.z * BW * BH + tc.y * BW + tc.x] = FindRoot(labels, label);
        }

        __syncthreads();

        if (gc.x < shape.x && gc.y < shape.y && gc.z < shape.z)
        {
            if (nzym1x && nzm1yx && !nzm1ym1x)
            {
                Reduction(labels, label, labels[tc.z * BW * BH + (tc.y - 1) * BW + tc.x]);
            }

            if (nzyxm1 && ((nzm1yx && !nzm1yxm1) || (nzym1x && !nzym1xm1)))
            {
                label = Reduction(labels, label, labels[tc.z * BW * BH + tc.y * BW + tc.x - 1]);
            }
        }

        __syncthreads();

        if (gc.x < shape.x && gc.y < shape.y && gc.z < shape.z)
        {
            label = labels[tc.z * BW * BH + tc.y * BW + tc.x];

            label = FindRoot(labels, label);

            DT lx = label % BW;
            DT ly = (label / BW) % BH;
            DT lz = (label / (BW * BH)) % BD;

            DT dstStrideD = dst.strides()[1] / sizeof(DT);
            DT dstStrideH = dst.strides()[2] / sizeof(DT);

            dst[gc] = (blockIdx.z * BD + lz) * dstStrideD + (blockIdx.y * BH + ly) * dstStrideH + blockIdx.x * BW + lx;
        }
    }
}

template<typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void ZLabelReduction3D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int4 shape)
{
    using DT = typename DstWrap::ValueType;
    int4 gc;
    gc.x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    gc.y = ((blockIdx.y * blockDim.y) + threadIdx.y);
    gc.z = ((blockIdx.z * blockDim.z) + threadIdx.z) * blockDim.z + blockDim.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);

    bool thread_x = (gc.x % blockDim.x) == 0;
    bool thread_y = (gc.y % blockDim.y) == 0;

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST minThreshold = hasMinThresh ? minThresh[gc.w] : 0;
        ST maxThreshold = hasMaxThresh ? maxThresh[gc.w] : 0;

        ST pzyx   = src[gc];
        ST pzm1yx = *src.ptr(gc.w, gc.z - 1, gc.y, gc.x);

        if (hasMinThresh && hasMaxThresh)
        {
            pzyx   = pzyx < minThreshold || pzyx > maxThreshold ? 0 : 1;
            pzm1yx = pzm1yx < minThreshold || pzm1yx > maxThreshold ? 0 : 1;
        }
        else if (hasMinThresh)
        {
            pzyx   = pzyx < minThreshold ? 0 : 1;
            pzm1yx = pzm1yx < minThreshold ? 0 : 1;
        }
        else if (hasMaxThresh)
        {
            pzyx   = pzyx > maxThreshold ? 0 : 1;
            pzm1yx = pzm1yx > maxThreshold ? 0 : 1;
        }

        ST pzyxm1   = __shfl_up_sync(0xffffffff, pzyx, 1);
        ST pzm1yxm1 = __shfl_up_sync(0xffffffff, pzm1yx, 1);

        if (pzyx == pzm1yx)
        {
            ST pzym1x   = (!thread_y) ? *src.ptr(gc.w, gc.z, gc.y - 1, gc.x) : 0;
            ST pzm1ym1x = (!thread_y) ? *src.ptr(gc.w, gc.z - 1, gc.y - 1, gc.x) : 0;

            if (hasMinThresh && hasMaxThresh)
            {
                pzym1x   = pzym1x < minThreshold || pzym1x > maxThreshold ? 0 : 1;
                pzm1ym1x = pzm1ym1x < minThreshold || pzm1ym1x > maxThreshold ? 0 : 1;
            }
            else if (hasMinThresh)
            {
                pzym1x   = pzym1x < minThreshold ? 0 : 1;
                pzm1ym1x = pzm1ym1x < minThreshold ? 0 : 1;
            }
            else if (hasMaxThresh)
            {
                pzym1x   = pzym1x > maxThreshold ? 0 : 1;
                pzm1ym1x = pzm1ym1x > maxThreshold ? 0 : 1;
            }

            bool nzym1x   = (!thread_y) ? (pzyx == pzym1x) : false;
            bool nzm1ym1x = (!thread_y) ? (pzyx == pzm1ym1x) : false;

            if ((thread_x || (pzyx != pzyxm1) || (pzyx != pzm1yxm1)) && (!nzym1x || !nzm1ym1x))
            {
                DT label1 = dst[gc];
                DT label2 = *dst.ptr(gc.w, gc.z - 1, gc.y, gc.x);

                Reduction(dst.ptr(gc.w), label1, label2);
            }
        }
    }
}

template<typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void YLabelReduction3D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int4 shape)
{
    using DT = typename DstWrap::ValueType;

    int4 gc;
    gc.x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    gc.y = ((blockIdx.z * blockDim.z) + threadIdx.z) * blockDim.y + blockDim.y;
    gc.z = ((blockIdx.y * blockDim.y) + threadIdx.y);

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);

    bool thread_x = (gc.x % blockDim.x) == 0;
    bool thread_z = (gc.z % blockDim.z) == 0;

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST minThreshold = hasMinThresh ? minThresh[gc.w] : 0;
        ST maxThreshold = hasMaxThresh ? maxThresh[gc.w] : 0;

        ST pzyx   = src[gc];
        ST pzym1x = *src.ptr(gc.w, gc.z, gc.y - 1, gc.x);

        if (hasMinThresh && hasMaxThresh)
        {
            pzyx   = pzyx < minThreshold || pzyx > maxThreshold ? 0 : 1;
            pzym1x = pzym1x < minThreshold || pzym1x > maxThreshold ? 0 : 1;
        }
        else if (hasMinThresh)
        {
            pzyx   = pzyx < minThreshold ? 0 : 1;
            pzym1x = pzym1x < minThreshold ? 0 : 1;
        }
        else if (hasMaxThresh)
        {
            pzyx   = pzyx > maxThreshold ? 0 : 1;
            pzym1x = pzym1x > maxThreshold ? 0 : 1;
        }

        ST pzyxm1   = __shfl_up_sync(0xffffffff, pzyx, 1);
        ST pzym1xm1 = __shfl_up_sync(0xffffffff, pzym1x, 1);

        if (pzyx == pzym1x)
        {
            ST pzm1yx   = (!thread_z) ? *src.ptr(gc.w, gc.z - 1, gc.y, gc.x) : 0;
            ST pzm1ym1x = (!thread_z) ? *src.ptr(gc.w, gc.z - 1, gc.y - 1, gc.x) : 0;

            if (hasMinThresh && hasMaxThresh)
            {
                pzm1yx   = pzm1yx < minThreshold || pzm1yx > maxThreshold ? 0 : 1;
                pzm1ym1x = pzm1ym1x < minThreshold || pzm1ym1x > maxThreshold ? 0 : 1;
            }
            else if (hasMinThresh)
            {
                pzm1yx   = pzm1yx < minThreshold ? 0 : 1;
                pzm1ym1x = pzm1ym1x < minThreshold ? 0 : 1;
            }
            else if (hasMaxThresh)
            {
                pzm1yx   = pzm1yx > maxThreshold ? 0 : 1;
                pzm1ym1x = pzm1ym1x > maxThreshold ? 0 : 1;
            }

            bool nzm1yx   = (!thread_z) ? (pzyx == pzm1yx) : false;
            bool nzm1ym1x = (!thread_z) ? (pzyx == pzm1ym1x) : false;

            if ((!nzm1yx || !nzm1ym1x) && (thread_x || (pzyx != pzyxm1) || (pzyx != pzym1xm1)))
            {
                DT label1 = dst[gc];
                DT label2 = *dst.ptr(gc.w, gc.z, gc.y - 1, gc.x);

                Reduction(dst.ptr(gc.w), label1, label2);
            }
        }
    }
}

template<typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void XLabelReduction3D(DstWrap dst, SrcWrap src, ArgWrap<ST> minThresh, ArgWrap<ST> maxThresh, int4 shape)
{
    using DT = typename DstWrap::ValueType;

    int4 gc;
    gc.x = ((blockIdx.z * blockDim.z) + threadIdx.z) * blockDim.x + blockDim.x;
    gc.y = ((blockIdx.y * blockDim.y) + threadIdx.y);
    gc.z = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);

    bool thread_y = (gc.y % blockDim.y) == 0;
    bool thread_z = (gc.z % blockDim.z) == 0;

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST minThreshold = hasMinThresh ? minThresh[gc.w] : 0;
        ST maxThreshold = hasMaxThresh ? maxThresh[gc.w] : 0;

        ST pzyx   = src[gc];
        ST pzyxm1 = *src.ptr(gc.w, gc.z, gc.y, gc.x - 1);

        if (hasMinThresh && hasMaxThresh)
        {
            pzyx   = pzyx < minThreshold || pzyx > maxThreshold ? 0 : 1;
            pzyxm1 = pzyxm1 < minThreshold || pzyxm1 > maxThreshold ? 0 : 1;
        }
        else if (hasMinThresh)
        {
            pzyx   = pzyx < minThreshold ? 0 : 1;
            pzyxm1 = pzyxm1 < minThreshold ? 0 : 1;
        }
        else if (hasMaxThresh)
        {
            pzyx   = pzyx > maxThreshold ? 0 : 1;
            pzyxm1 = pzyxm1 > maxThreshold ? 0 : 1;
        }

        ST pzm1yx   = __shfl_up_sync(0xffffffff, pzyx, 1);
        ST pzm1yxm1 = __shfl_up_sync(0xffffffff, pzyxm1, 1);

        if (pzyx == pzyxm1)
        {
            ST pzym1x   = (!thread_y) ? *src.ptr(gc.w, gc.z, gc.y - 1, gc.x) : 0;
            ST pzym1xm1 = (!thread_y) ? *src.ptr(gc.w, gc.z, gc.y - 1, gc.x - 1) : 0;

            if (hasMinThresh && hasMaxThresh)
            {
                pzym1x   = pzym1x < minThreshold || pzym1x > maxThreshold ? 0 : 1;
                pzym1xm1 = pzym1xm1 < minThreshold || pzym1xm1 > maxThreshold ? 0 : 1;
            }
            else if (hasMinThresh)
            {
                pzym1x   = pzym1x < minThreshold ? 0 : 1;
                pzym1xm1 = pzym1xm1 < minThreshold ? 0 : 1;
            }
            else if (hasMaxThresh)
            {
                pzym1x   = pzym1x > maxThreshold ? 0 : 1;
                pzym1xm1 = pzym1xm1 > maxThreshold ? 0 : 1;
            }

            bool nzym1x   = (!thread_y) ? (pzyx == pzym1x) : false;
            bool nzym1xm1 = (!thread_y) ? (pzyx == pzym1xm1) : false;

            if ((thread_z || (pzyx != pzm1yx) || (pzyx != pzm1yxm1)) && (!nzym1x || !nzym1xm1))
            {
                DT label1 = dst[gc];
                DT label2 = *dst.ptr(gc.w, gc.z, gc.y, gc.x - 1);

                Reduction(dst.ptr(gc.w), label1, label2);
            }
        }
    }
}

template<typename DstWrap>
__global__ void ResolveLabels3D(DstWrap dst, int4 shape)
{
    int4 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        dst[gc] = FindRoot(dst.ptr(gc.w), dst[gc]);
    }
}

template<typename DstWrap, typename SrcWrap, typename ST = typename SrcWrap::ValueType>
__global__ void ReplaceBgLabels3D(DstWrap dst, SrcWrap src, ArgWrap<ST> bgLabel, ArgWrap<ST> minThresh,
                                  ArgWrap<ST> maxThresh, int4 shape)
{
    using DT = typename DstWrap::ValueType;

    int4 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    bool hasMinThresh = (minThresh.ptr(0) != nullptr);
    bool hasMaxThresh = (maxThresh.ptr(0) != nullptr);

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST minThreshold = hasMinThresh ? minThresh[gc.w] : 0;
        ST maxThreshold = hasMaxThresh ? maxThresh[gc.w] : 0;

        ST pzyx = src[gc];

        if (hasMinThresh && hasMaxThresh)
        {
            pzyx = pzyx < minThreshold || pzyx > maxThreshold ? 0 : 1;
        }
        else if (hasMinThresh)
        {
            pzyx = pzyx < minThreshold ? 0 : 1;
        }
        else if (hasMaxThresh)
        {
            pzyx = pzyx > maxThreshold ? 0 : 1;
        }

        DT backgroundLabel = bgLabel[gc.w];

        // If src has bg label, put it in dst; if dst has bg label, it means a wrong label was assigned to a
        // region, replace its label by a label never assigned, i.e. one-element-after-the-end stride

        if (pzyx == backgroundLabel)
        {
            dst[gc] = backgroundLabel;
        }
        else if (dst[gc] == (DT)backgroundLabel)
        {
            dst[gc] = dst.strides()[0] / sizeof(DT);
        }
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename DT = typename DstWrap::ValueType>
__global__ void CountLabels3D(ArgWrap<DT> count, StatsWrap stats, DstWrap dst, ArgWrap<ST> bgLabel, int4 shape,
                              int maxCapacity)
{
    static_assert(std::is_same_v<typename StatsWrap::ValueType, DT>);
    int4 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    DT posLabel = gc.z * dst.strides()[1] / sizeof(DT) + gc.y * dst.strides()[2] / sizeof(DT) + gc.x;
    DT endLabel = dst.strides()[0] / sizeof(DT);

    bool hasBgLabel = (bgLabel.ptr(0) != nullptr);

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST backgroundLabel = hasBgLabel ? bgLabel[gc.w] : 0;

        DT label = dst[gc];

        if (hasBgLabel && label == (DT)backgroundLabel)
        {
            continue; // do not count background labels
        }

        DT   regionIdx;
        bool counted = false;

        if (hasBgLabel && label == endLabel && posLabel == (DT)backgroundLabel)
        {
            // This is a special region marked with one-element-after-the-end label, count it
            regionIdx = atomicAdd(count.ptr(gc.w), 1);
            counted   = true;
        }
        else if (label == posLabel)
        {
            // This is the first element of a regular region, count it
            regionIdx = atomicAdd(count.ptr(gc.w), 1);
            counted   = true;
        }

        // If statistics should be computed and the region index is inside the allowed storage (the M maximum
        // capacity in stats tensor), replace the output label by the region index and store initial statistics

        if (counted && stats.ptr(0) != nullptr && regionIdx < maxCapacity)
        {
            // TODO: improve the mark of output label as region index with 1 in the 1st bit
            dst[gc] = regionIdx | (DT)(1 << 31);

            *stats.ptr(gc.w, (int)regionIdx, 0) = label;
            *stats.ptr(gc.w, (int)regionIdx, 1) = (DT)gc.x;
            *stats.ptr(gc.w, (int)regionIdx, 2) = (DT)gc.y;
            *stats.ptr(gc.w, (int)regionIdx, 3) = (DT)gc.z;
            *stats.ptr(gc.w, (int)regionIdx, 4) = 1;
            *stats.ptr(gc.w, (int)regionIdx, 5) = 1;
            *stats.ptr(gc.w, (int)regionIdx, 6) = 1;
            *stats.ptr(gc.w, (int)regionIdx, 7) = 1;
            *stats.ptr(gc.w, (int)regionIdx, 8) = REGION_NOT_MARKED;
        }
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename MaskWrap>
__global__ void ComputeStats3D(StatsWrap stats, DstWrap dst, MaskWrap mask, ArgWrap<ST> bgLabel, int4 shape, int maskN,
                               bool relabel)
{
    using DT = typename DstWrap::ValueType;
    using MT = typename MaskWrap::ValueType;
    static_assert(std::is_same_v<DT, typename StatsWrap::ValueType>);

    int4 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    bool hasMask      = (mask.ptr(0) != nullptr);
    bool isInsideMask = false;
    bool hasBgLabel   = (bgLabel.ptr(0) != nullptr);
    DT   endLabel     = dst.strides()[0] / sizeof(DT);
    DT   regionIdx    = 0;

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        ST backgroundLabel = hasBgLabel ? bgLabel[gc.w] : 0;

        DT label = dst[gc];

        if (hasMask)
        {
            int4 mc{gc.x, gc.y, gc.z, maskN == 1 ? 0 : gc.w};

            isInsideMask = mask[mc] == 0 ? false : true; // mask value = 0 means outside the mask
        }

        if (hasBgLabel && label == (DT)backgroundLabel)
        {
            continue; // do not compute statistics for background labels
        }
        if (label & (DT)(1 << 31))
        {
            if (isInsideMask)
            {
                regionIdx = label & (DT) ~(1 << 31);

                *stats.ptr(gc.w, (int)regionIdx, 8) = REGION_INSIDE_MASK; // region is inside the mask, mark it as such
            }

            continue; // label is marked as region index, its statistics is already computed
        }
        if (hasBgLabel && label == endLabel)
        {
            // This is a special region marked with one-element-after-the-end label, its label was the bg label
            label = backgroundLabel;
        }

        regionIdx = dst.ptr(gc.w)[label];

        if (regionIdx & (DT)(1 << 31))
        {
            regionIdx = regionIdx & (DT) ~(1 << 31);

            if (relabel)
            {
                if (hasBgLabel && regionIdx >= (DT)backgroundLabel)
                {
                    dst[gc] = regionIdx + 1; // skip one region index equals to background label when relabeling
                }
                else
                {
                    dst[gc] = regionIdx;
                }
            }

            int3 cornerPos{(int)*stats.ptr(gc.w, (int)regionIdx, 1), (int)*stats.ptr(gc.w, (int)regionIdx, 2),
                           (int)*stats.ptr(gc.w, (int)regionIdx, 3)};

            int3 bboxArea = cuda::abs(cornerPos - cuda::DropCast<3>(gc)) + 1;

            atomicMax(stats.ptr(gc.w, (int)regionIdx, 4), (DT)bboxArea.x);
            atomicMax(stats.ptr(gc.w, (int)regionIdx, 5), (DT)bboxArea.y);
            atomicMax(stats.ptr(gc.w, (int)regionIdx, 6), (DT)bboxArea.z);
            atomicAdd(stats.ptr(gc.w, (int)regionIdx, 7), 1);

            if (isInsideMask)
            {
                *stats.ptr(gc.w, (int)regionIdx, 8) = REGION_INSIDE_MASK; // region is inside the mask, mark it as such
            }
        }
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename DT = typename DstWrap::ValueType>
__global__ void RemoveIslands3D(StatsWrap stats, DstWrap dst, ArgWrap<ST> bgLabel, ArgWrap<DT> minSize, int4 shape,
                                bool relabel, bool hasMask)
{
    static_assert(std::is_same_v<DT, typename StatsWrap::ValueType>);
    int4 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    DT endLabel = dst.strides()[0] / sizeof(DT);

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        DT label = dst[gc];

        ST backgroundLabel = bgLabel[gc.w];

        if (label == (DT)backgroundLabel)
        {
            continue;
        }
        if (label == endLabel)
        {
            // This is a special region marked with one-element-after-the-end label, its label was the backgroundLabel
            label = backgroundLabel;
        }

        DT regionIdx = 0;

        if (!(label & (DT)(1 << 31)))
        {
            if (relabel)
            {
                if (label >= (DT)backgroundLabel + 1)
                {
                    regionIdx = label - 1; // go back one region index to account for background label
                }
                else
                {
                    regionIdx = label;
                }
            }
            else
            {
                regionIdx = dst.ptr(gc.w)[label];

                if (regionIdx & (DT)(1 << 31))
                {
                    regionIdx = regionIdx & (DT) ~(1 << 31);
                }
                else
                {
                    continue; // invalid region index
                }
            }
        }
        else
        {
            continue; // should not remove first region element with 1st bit 1 so other elements are not lost
        }

        DT regionSize = *stats.ptr(gc.w, (int)regionIdx, 7);

        // If region size is less than minimum size, it is an island and should be removed, i.e. set to background label
        if (regionSize < minSize[gc.w])
        {
            // If there is no mask or if there is a mask and the region mark is not 2, meaning the region is not
            // inside the mask, the region should be removed
            if (!hasMask || *stats.ptr(gc.w, (int)regionIdx, 8) != REGION_INSIDE_MASK)
            {
                dst[gc] = backgroundLabel;
            }
        }
    }
}

template<typename StatsWrap, typename DstWrap, typename ST, typename DT = typename DstWrap::ValueType>
__global__ void Relabel3D(StatsWrap stats, DstWrap dst, ArgWrap<ST> bgLabel, ArgWrap<DT> minSize, int4 shape,
                          bool relabel, bool hasMask)
{
    static_assert(std::is_same_v<DT, typename StatsWrap::ValueType>);
    int4 gc;
    gc.x = blockIdx.x * blockDim.x + threadIdx.x;
    gc.y = blockIdx.y * blockDim.y + threadIdx.y;
    gc.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (gc.x >= shape.x || gc.y >= shape.y || gc.z >= shape.z)
    {
        return;
    }

    bool removeIsland = minSize.ptr(0) != nullptr;

    for (gc.w = 0; gc.w < shape.w; gc.w++)
    {
        DT label = dst[gc];

        if (label & (DT)(1 << 31))
        {
            // Label is marked as region index, relabel it back to proper label
            DT regionIdx = label & (DT) ~(1 << 31);

            if (relabel)
            {
                bool hasBgLabel      = (bgLabel.ptr(0) != nullptr);
                ST   backgroundLabel = hasBgLabel ? bgLabel[gc.w] : 0;

                if (hasBgLabel && regionIdx >= (DT)backgroundLabel)
                {
                    dst[gc] = regionIdx + 1; // skip one region index equals to background label when relabeling
                }
                else
                {
                    dst[gc] = regionIdx;
                }
            }
            else
            {
                dst[gc] = *stats.ptr(gc.w, (int)regionIdx, 0);
            }

            if (removeIsland)
            {
                DT regionSize = *stats.ptr(gc.w, (int)regionIdx, 7);

                if (regionSize < minSize[gc.w])
                {
                    if (!hasMask || *stats.ptr(gc.w, (int)regionIdx, 8) != REGION_INSIDE_MASK)
                    {
                        dst[gc] = (DT)bgLabel[gc.w];

                        *stats.ptr(gc.w, (int)regionIdx, 8) = REGION_REMOVED;
                    }
                }
            }
        }
    }
}

// Run functions ---------------------------------------------------------------

template<typename SrcT, typename DstT = uint32_t, typename MskT = uint8_t>
inline void RunLabelForType(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                            const nvcv::TensorDataStridedCuda &dstData, const int4 &shapeWHDN,
                            const nvcv::Tensor &bgLabel, const nvcv::Tensor &minThresh, const nvcv::Tensor &maxThresh,
                            const nvcv::Tensor &minSize, const nvcv::Tensor &count, const nvcv::Tensor &stats,
                            const nvcv::Tensor &mask, int numDim, bool relabel)
{
    constexpr int BW = 32, BH = 4, BD = 2; // block width, height and depth

    int4 idsNDHW{srcData.layout().find('N'), srcData.layout().find('D'), srcData.layout().find('H'),
                 srcData.layout().find('W')};

    // Although output tensors may have S32 or U32 data type, they are always considered U32 (DstT = uint32_t) as
    // they are used as non-negative offset or position or size or count, or even as a 32-bit mask

    // Although mask tensor may have S8 or U8 data type, it is always considered U8 (MskT = uint8_t) as it is used
    // as zero (outside mask) or non-zero (inside mask)

    NVCV_ASSERT(srcData.stride(idsNDHW.w) == sizeof(SrcT));
    NVCV_ASSERT(dstData.stride(idsNDHW.w) == sizeof(DstT));

    if ((srcData.stride(idsNDHW.z) > nvcv::cuda::TypeTraits<int>::max)
        || (idsNDHW.y != -1 && srcData.stride(idsNDHW.y) > nvcv::cuda::TypeTraits<int>::max)
        || (idsNDHW.x != -1 && srcData.stride(idsNDHW.x) > nvcv::cuda::TypeTraits<int>::max))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Too big in tensor");
    }
    if ((dstData.stride(idsNDHW.z) > nvcv::cuda::TypeTraits<int>::max)
        || (idsNDHW.y != -1 && dstData.stride(idsNDHW.y) > nvcv::cuda::TypeTraits<int>::max)
        || (idsNDHW.x != -1 && dstData.stride(idsNDHW.x) > nvcv::cuda::TypeTraits<int>::max))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Too big out tensor");
    }

    using SType = int32_t;

    nvcv::Optional<nvcv::TensorDataStridedCuda> mskData;
    int4                                        mskIdsNDHW = {0, 0, 0, 0};
    int                                         maskN      = 0;
    bool                                        hasMask    = (mask) ? true : false;

    if (hasMask)
    {
        mskData = mask.exportData<nvcv::TensorDataStridedCuda>(); // export data check already done

        mskIdsNDHW = int4{mskData->layout().find('N'), mskData->layout().find('D'), mskData->layout().find('H'),
                          mskData->layout().find('W')};

        NVCV_ASSERT(mskData->stride(mskIdsNDHW.w) == sizeof(MskT));

        if ((mskData->stride(mskIdsNDHW.z) > nvcv::cuda::TypeTraits<int>::max)
            || (mskIdsNDHW.y != -1 && mskData->stride(mskIdsNDHW.y) > nvcv::cuda::TypeTraits<int>::max)
            || (mskIdsNDHW.x != -1 && mskData->stride(mskIdsNDHW.x) > nvcv::cuda::TypeTraits<int>::max)
            || (mskIdsNDHW.x != -1 && mskData->shape()[mskIdsNDHW.x] > nvcv::cuda::TypeTraits<int>::max))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Too big mask tensor");
        }

        maskN = mskIdsNDHW.x == -1 ? 1 : (int)mskData->shape()[mskIdsNDHW.x];
    }

    ArgWrap<SrcT>                   bgLabelWrap, minThreshWrap, maxThreshWrap;
    ArgWrap<DstT>                   minSizeWrap, countWrap;
    cuda::Tensor3DWrap<DstT, SType> statsWrap;

    int maxCapacity = 0;

#define CVCUDA_LABEL_WRAP(TENSOR, WRAPPER, TENSORWRAP)                                                              \
    if (TENSOR)                                                                                                     \
    {                                                                                                               \
        auto data = TENSOR.exportData<nvcv::TensorDataStridedCuda>();                                               \
        if (!data)                                                                                                  \
        {                                                                                                           \
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, #TENSOR " tensor must be cuda-accessible"); \
        }                                                                                                           \
        TENSORWRAP = WRAPPER(data->basePtr());                                                                      \
    }

    CVCUDA_LABEL_WRAP(bgLabel, ArgWrap<SrcT>, bgLabelWrap);
    CVCUDA_LABEL_WRAP(minThresh, ArgWrap<SrcT>, minThreshWrap);
    CVCUDA_LABEL_WRAP(maxThresh, ArgWrap<SrcT>, maxThreshWrap);
    CVCUDA_LABEL_WRAP(minSize, ArgWrap<DstT>, minSizeWrap);

#undef CVCUDA_LABEL_WRAP

    if (count)
    {
        auto data = count.exportData<nvcv::TensorDataStridedCuda>();
        if (!data)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "count tensor must be cuda-accessible");
        }

        countWrap = ArgWrap<DstT>(data->basePtr());

        NVCV_CHECK_THROW(cudaMemsetAsync(data->basePtr(), 0, sizeof(DstT) * shapeWHDN.w, stream));
    }
    if (stats)
    {
        auto data = stats.exportData<nvcv::TensorDataStridedCuda>();
        if (!data)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "stats tensor must be cuda-accessible");
        }

        statsWrap = cuda::Tensor3DWrap<DstT, SType>(data->basePtr(), (int)data->stride(0), (int)data->stride(1));

        maxCapacity = data->shape(1);
    }

    if (numDim == 2)
    {
        int2 sizeWH{shapeWHDN.x, shapeWHDN.y};
        int2 srcStridesNH{0, (int)srcData.stride(idsNDHW.z)};
        int2 dstStridesNH{0, (int)dstData.stride(idsNDHW.z)};

        srcStridesNH.x = idsNDHW.x == -1 ? srcStridesNH.y * shapeWHDN.y : (int)srcData.stride(idsNDHW.x);
        dstStridesNH.x = idsNDHW.x == -1 ? dstStridesNH.y * shapeWHDN.y : (int)dstData.stride(idsNDHW.x);

        dim3 larThreads(BW, BH, 1);
        dim3 labBlocks(util::DivUp(sizeWH.x, BW), util::DivUp(sizeWH.y, BH), shapeWHDN.w);
        dim3 redBlocksX(util::DivUp(sizeWH.y, BW), util::DivUp((int)labBlocks.x, BH), shapeWHDN.w);
        dim3 redBlocksY(util::DivUp(sizeWH.x, BW), util::DivUp((int)labBlocks.y, BH), shapeWHDN.w);

        cuda::Tensor3DWrap<SrcT, SType> srcWrap(srcData.basePtr(), srcStridesNH.x, srcStridesNH.y);
        cuda::Tensor3DWrap<DstT, SType> dstWrap(dstData.basePtr(), dstStridesNH.x, dstStridesNH.y);
        cuda::Tensor3DWrap<MskT, SType> mskWrap;

        if (hasMask)
        {
            int2 mskStridesNH{0, (int)mskData->stride(mskIdsNDHW.z)};
            mskStridesNH.x = mskIdsNDHW.x == -1 ? mskStridesNH.y * shapeWHDN.y : (int)mskData->stride(mskIdsNDHW.x);

            mskWrap = cuda::Tensor3DWrap<MskT, SType>(mskData->basePtr(), mskStridesNH.x, mskStridesNH.y);
        }

        BlockLabel2D<BW, BH>
            <<<labBlocks, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap, sizeWH);

        YLabelReduction2D<<<redBlocksY, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap,
                                                                 sizeWH);

        XLabelReduction2D<<<redBlocksX, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap,
                                                                 sizeWH);

        ResolveLabels2D<<<labBlocks, larThreads, 0, stream>>>(dstWrap, sizeWH);

        if (bgLabel)
        {
            ReplaceBgLabels2D<<<labBlocks, larThreads, 0, stream>>>(dstWrap, srcWrap, bgLabelWrap, minThreshWrap,
                                                                    maxThreshWrap, sizeWH);
        }
        if (count)
        {
            CountLabels2D<<<labBlocks, larThreads, 0, stream>>>(countWrap, statsWrap, dstWrap, bgLabelWrap, sizeWH,
                                                                maxCapacity);

            if (stats)
            {
                ComputeStats2D<<<labBlocks, larThreads, 0, stream>>>(statsWrap, dstWrap, mskWrap, bgLabelWrap, sizeWH,
                                                                     maskN, relabel);

                if (minSize)
                {
                    RemoveIslands2D<<<labBlocks, larThreads, 0, stream>>>(statsWrap, dstWrap, bgLabelWrap, minSizeWrap,
                                                                          sizeWH, relabel, hasMask);
                }

                Relabel2D<<<labBlocks, larThreads, 0, stream>>>(statsWrap, dstWrap, bgLabelWrap, minSizeWrap, sizeWH,
                                                                relabel, hasMask);
            }
        }
    }
    else
    {
        int3 srcStridesNDH{0, (int)srcData.stride(idsNDHW.y), (int)srcData.stride(idsNDHW.z)};
        int3 dstStridesNDH{0, (int)dstData.stride(idsNDHW.y), (int)dstData.stride(idsNDHW.z)};

        srcStridesNDH.x = idsNDHW.x == -1 ? srcStridesNDH.y * shapeWHDN.z : (int)srcData.stride(idsNDHW.x);
        dstStridesNDH.x = idsNDHW.x == -1 ? dstStridesNDH.y * shapeWHDN.z : (int)dstData.stride(idsNDHW.x);

        dim3 larThreads(BW, BH, BD);
        dim3 labBlocks(util::DivUp(shapeWHDN.x, BW), util::DivUp(shapeWHDN.y, BH), util::DivUp(shapeWHDN.z, BD));
        dim3 redBlocksX(util::DivUp(shapeWHDN.z, BW), util::DivUp(shapeWHDN.y, BH), util::DivUp((int)labBlocks.x, BD));
        dim3 redBlocksY(util::DivUp(shapeWHDN.x, BW), util::DivUp(shapeWHDN.z, BH), util::DivUp((int)labBlocks.y, BD));
        dim3 redBlocksZ(util::DivUp(shapeWHDN.x, BW), util::DivUp(shapeWHDN.y, BH), util::DivUp((int)labBlocks.z, BD));

        cuda::Tensor4DWrap<SrcT, SType> srcWrap(srcData.basePtr(), srcStridesNDH.x, srcStridesNDH.y, srcStridesNDH.z);
        cuda::Tensor4DWrap<DstT, SType> dstWrap(dstData.basePtr(), dstStridesNDH.x, dstStridesNDH.y, dstStridesNDH.z);
        cuda::Tensor4DWrap<MskT, SType> mskWrap;

        if (hasMask)
        {
            int3 mskStridesNDH{0, (int)mskData->stride(mskIdsNDHW.y), (int)mskData->stride(mskIdsNDHW.z)};
            mskStridesNDH.x = mskIdsNDHW.x == -1 ? mskStridesNDH.y * shapeWHDN.z : (int)mskData->stride(mskIdsNDHW.x);

            mskWrap = cuda::Tensor4DWrap<MskT, SType>(mskData->basePtr(), mskStridesNDH.x, mskStridesNDH.y,
                                                      mskStridesNDH.z);
        }

        BlockLabel3D<BW, BH, BD>
            <<<labBlocks, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap, shapeWHDN);

        ZLabelReduction3D<<<redBlocksZ, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap,
                                                                 shapeWHDN);

        YLabelReduction3D<<<redBlocksY, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap,
                                                                 shapeWHDN);

        XLabelReduction3D<<<redBlocksX, larThreads, 0, stream>>>(dstWrap, srcWrap, minThreshWrap, maxThreshWrap,
                                                                 shapeWHDN);

        ResolveLabels3D<<<labBlocks, larThreads, 0, stream>>>(dstWrap, shapeWHDN);

        if (bgLabel)
        {
            ReplaceBgLabels3D<<<labBlocks, larThreads, 0, stream>>>(dstWrap, srcWrap, bgLabelWrap, minThreshWrap,
                                                                    maxThreshWrap, shapeWHDN);
        }
        if (count)
        {
            CountLabels3D<<<labBlocks, larThreads, 0, stream>>>(countWrap, statsWrap, dstWrap, bgLabelWrap, shapeWHDN,
                                                                maxCapacity);

            if (stats)
            {
                ComputeStats3D<<<labBlocks, larThreads, 0, stream>>>(statsWrap, dstWrap, mskWrap, bgLabelWrap,
                                                                     shapeWHDN, maskN, relabel);

                if (minSize)
                {
                    RemoveIslands3D<<<labBlocks, larThreads, 0, stream>>>(statsWrap, dstWrap, bgLabelWrap, minSizeWrap,
                                                                          shapeWHDN, relabel, hasMask);
                }

                Relabel3D<<<labBlocks, larThreads, 0, stream>>>(statsWrap, dstWrap, bgLabelWrap, minSizeWrap, shapeWHDN,
                                                                relabel, hasMask);
            }
        }
    }
}

inline void RunLabel(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                     const nvcv::TensorDataStridedCuda &dstData, const int4 &srcShape, nvcv::DataType srcDataType,
                     const nvcv::Tensor &bgLabel, const nvcv::Tensor &minThresh, const nvcv::Tensor &maxThresh,
                     const nvcv::Tensor &minSize, const nvcv::Tensor &count, const nvcv::Tensor &stats,
                     const nvcv::Tensor &mask, int numDim, bool relabel)
{
    switch (srcDataType)
    {
#define CVCUDA_LABEL_CASE(DT, T)                                                                                     \
    case nvcv::TYPE_##DT:                                                                                            \
        RunLabelForType<T>(stream, srcData, dstData, srcShape, bgLabel, minThresh, maxThresh, minSize, count, stats, \
                           mask, numDim, relabel);                                                                   \
        break

        CVCUDA_LABEL_CASE(U8, uint8_t);
        CVCUDA_LABEL_CASE(U16, uint16_t);
        CVCUDA_LABEL_CASE(U32, uint32_t);
        CVCUDA_LABEL_CASE(S8, int8_t);
        CVCUDA_LABEL_CASE(S16, int16_t);
        CVCUDA_LABEL_CASE(S32, int32_t);

#undef CVCUDA_LABEL_CASE

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

Label::Label() {}

// Tensor operator -------------------------------------------------------------

void Label::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                       const nvcv::Tensor &bgLabel, const nvcv::Tensor &minThresh, const nvcv::Tensor &maxThresh,
                       const nvcv::Tensor &minSize, const nvcv::Tensor &count, const nvcv::Tensor &stats,
                       const nvcv::Tensor &mask, NVCVConnectivityType connectivity, NVCVLabelType assignLabels,
                       NVCVLabelMaskType maskType) const
{
    if (!(in.shape().layout() == nvcv::TENSOR_HW || in.shape().layout() == nvcv::TENSOR_HWC
          || in.shape().layout() == nvcv::TENSOR_NHW || in.shape().layout() == nvcv::TENSOR_NHWC
          || in.shape().layout() == nvcv::TENSOR_DHW || in.shape().layout() == nvcv::TENSOR_DHWC
          || in.shape().layout() == nvcv::TENSOR_NDHW || in.shape().layout() == nvcv::TENSOR_NDHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have [N][D]HW[C] layout");
    }

    // We expect input and output shape to be the same as TensorShape contains TensorLayout

    if (!(in.shape() == out.shape()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output tensors must have the same shape and layout");
    }
    if (!(out.dtype() == nvcv::TYPE_S32 || out.dtype() == nvcv::TYPE_U32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor data type (%s) must be S32 or U32",
                              nvcvDataTypeGetName(out.dtype()));
    }

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must be cuda-accessible");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (!outData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor must be cuda-accessible");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have strided access");
    }
    if (!(inAccess->numChannels() == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have a single channel");
    }
    if (!(inAccess->numPlanes() == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have a single plane");
    }
    if (inAccess->numSamples() > cuda::TypeTraits<int>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Too big number of samples %ld, must be smaller than or equal to %d",
                              inAccess->numSamples(), cuda::TypeTraits<int>::max);
    }

    int4 inShape{inAccess->numCols(), inAccess->numRows(), 1, (int)inAccess->numSamples()}; // WHDN shape

    int inDepthIdx = in.shape().layout().find('D');

    int numDim = (connectivity == NVCV_CONNECTIVITY_4_2D || connectivity == NVCV_CONNECTIVITY_8_2D) ? 2 : 3;

    bool relabel = (assignLabels == NVCV_LABEL_SEQUENTIAL);

    if (inDepthIdx != -1)
    {
        if (numDim == 2)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Connectivity 2D not allowed in tensors with depth D dimension");
        }

        NVCV_ASSERT(inDepthIdx >= 0 && inDepthIdx < in.shape().rank());

        int64_t inDepth = in.shape()[inDepthIdx];

        if (inDepth > cuda::TypeTraits<int>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Too big depth %ld, must be smaller than or equal to %d", inDepth,
                                  cuda::TypeTraits<int>::max);
        }

        inShape.z = static_cast<int>(inDepth);
    }
    else
    {
        if (numDim == 3)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Connectivity 3D not allowed in tensors without depth D dimension");
        }
    }

    // Various outputs assume the maximum range of values to be representable as int, for this to happen the number
    // of elements (pixels or voxels) in the input must not be greater than maximum int32_t
    int64_t numElems = inShape.x * inShape.y * inShape.z;
    if (numElems > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Too big input shape with %ld elements, must be smaller than or equal to %d", numElems,
                              cuda::TypeTraits<int32_t>::max);
    }

    if (bgLabel)
    {
        if (!((bgLabel.rank() == 1 && bgLabel.shape()[0] == inShape.w)
              || (bgLabel.rank() == 2 && bgLabel.shape()[0] == inShape.w && bgLabel.shape()[1] == 1)))
        {
            std::ostringstream oss;
            oss << bgLabel.shape();
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input bgLabel must be [N] or [NC] tensor, with N=%d and C=1, got %s", inShape.w,
                                  oss.str().c_str());
        }
        if (!(bgLabel.dtype() == in.dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input (%s) and bgLabel (%s) tensors must have the same data type",
                                  nvcvDataTypeGetName(in.dtype()), nvcvDataTypeGetName(bgLabel.dtype()));
        }
    }

    if (minThresh)
    {
        if (!((minThresh.rank() == 1 && minThresh.shape()[0] == inShape.w)
              || (minThresh.rank() == 2 && minThresh.shape()[0] == inShape.w && minThresh.shape()[1] == 1)))
        {
            std::ostringstream oss;
            oss << minThresh.shape();
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input minThresh must be [N] or [NC] tensor, with N=%d and C=1, got %s", inShape.w,
                                  oss.str().c_str());
        }
        if (!(minThresh.dtype() == in.dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input (%s) and minThresh (%s) tensors must have the same data type",
                                  nvcvDataTypeGetName(in.dtype()), nvcvDataTypeGetName(minThresh.dtype()));
        }
    }

    if (maxThresh)
    {
        if (!((maxThresh.rank() == 1 && maxThresh.shape()[0] == inShape.w)
              || (maxThresh.rank() == 2 && maxThresh.shape()[0] == inShape.w && maxThresh.shape()[1] == 1)))
        {
            std::ostringstream oss;
            oss << maxThresh.shape();
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input maxThresh must be [N] or [NC] tensor, with N=%d and C=1, got %s", inShape.w,
                                  oss.str().c_str());
        }
        if (!(maxThresh.dtype() == in.dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input (%s) and maxThresh (%s) tensors must have the same data type",
                                  nvcvDataTypeGetName(in.dtype()), nvcvDataTypeGetName(maxThresh.dtype()));
        }
    }

    if (count)
    {
        if (!((count.rank() == 1 && count.shape()[0] == inShape.w)
              || (count.rank() == 2 && count.shape()[0] == inShape.w && count.shape()[1] == 1)))
        {
            std::ostringstream oss;
            oss << count.shape();
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output count must be [N] or [NC] tensor, with N=%d and C=1, got %s", inShape.w,
                                  oss.str().c_str());
        }
        if (!(count.dtype() == nvcv::TYPE_S32 || count.dtype() == nvcv::TYPE_U32))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output count (%s) must have S32 or U32 data type",
                                  nvcvDataTypeGetName(count.dtype()));
        }
    }

    if (stats)
    {
        if (!count)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output stats requires count tensor");
        }
        if (!((stats.rank() == 3 && stats.shape()[0] == inShape.w && stats.shape()[2] == 3 + 2 * numDim)))
        {
            std::ostringstream oss;
            oss << stats.shape();
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output stats must be [NMA] tensor, with rank=3 N=%d A=%d, got %s", inShape.w,
                                  3 + 2 * numDim, oss.str().c_str());
        }
        if (!(stats.dtype() == nvcv::TYPE_S32 || stats.dtype() == nvcv::TYPE_U32))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output stats (%s) must have S32 or U32 data type",
                                  nvcvDataTypeGetName(stats.dtype()));
        }
    }
    else if (relabel)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output stats tensor must not be NULL to have sequential labels");
    }

    if (minSize)
    {
        if (!bgLabel || !stats)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input minSize requires bgLabel and stats tensors");
        }

        if (!((minSize.rank() == 1 && minSize.shape()[0] == inShape.w)
              || (minSize.rank() == 2 && minSize.shape()[0] == inShape.w && minSize.shape()[1] == 1)))
        {
            std::ostringstream oss;
            oss << minSize.shape();
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input minSize must be [N] or [NC] tensor, with N=%d and C=1, got %s", inShape.w,
                                  oss.str().c_str());
        }
        if (!(minSize.dtype() == nvcv::TYPE_S32 || minSize.dtype() == nvcv::TYPE_U32))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input minSize (%s) must have S32 or U32 data type",
                                  nvcvDataTypeGetName(minSize.dtype()));
        }
    }

    if (mask)
    {
        if (!minSize && maskType == NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input mask requires minSize tensor "
                                  "when maskType is NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY");
        }
        if (!(maskType == NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask type must be "
                                  "NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY");
        }

        auto maskData = mask.exportData<nvcv::TensorDataStridedCuda>();
        if (!maskData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask tensor must be cuda-accessible");
        }

        auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*maskData);
        if (!maskAccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask tensor must have strided access");
        }
        if (!(maskAccess->numChannels() == 1))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask tensor must have a single channel");
        }
        if (!(maskAccess->numPlanes() == 1))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask tensor must have a single plane");
        }
        if (!(maskAccess->numSamples() == 1 || maskAccess->numSamples() == inShape.w))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask tensor must have number of samples N=1 "
                                  "or same N as input and output tensors");
        }
        if (!(maskAccess->numCols() == inShape.x))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask tensor must have the same width W "
                                  "as input and output tensors");
        }
        if (!(maskAccess->numRows() == inShape.y))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask tensor must have the same height H "
                                  "as input and output tensors");
        }

        int maskDepthIdx = mask.shape().layout().find('D');
        if (maskDepthIdx != -1)
        {
            if (mask.shape()[maskDepthIdx] != inShape.z)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Mask tensor must have the same depth D "
                                      "as input and output tensors");
            }
        }
        else
        {
            if (numDim == 3)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensors in and out are 3D and mask is 2D");
            }
        }

        if (!(mask.dtype() == nvcv::TYPE_S8 || mask.dtype() == nvcv::TYPE_U8))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input mask (%s) must have S8 or U8 data type",
                                  nvcvDataTypeGetName(mask.dtype()));
        }
    }

    // TODO: Support full connectivity
    if (connectivity == NVCV_CONNECTIVITY_8_2D || connectivity == NVCV_CONNECTIVITY_26_3D)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Full neighborhood labeling not supported yet");
    }

    RunLabel(stream, *inData, *outData, inShape, in.dtype(), bgLabel, minThresh, maxThresh, minSize, count, stats, mask,
             numDim, relabel);
}

} // namespace cvcuda::priv
