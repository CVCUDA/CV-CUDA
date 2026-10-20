/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceBorderVarShapeWrap.hpp" // to test in the device

#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>     // the object of this test
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp> // for ImageBatchVarShape, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>                // for operator *, etc.
#include <cvcuda/cuda_tools/StaticCast.hpp>             // for StaticCast, etc.
#include <gtest/gtest.h>                                // for EXPECT_EQ, etc.

namespace cuda = nvcv::cuda;

// -------------- To allow testing device-side BorderVarShapeWrap --------------

template<class DstWrapper, class SrcWrapper>
__global__ void FillBorder(DstWrapper dst, SrcWrapper src, int numSamples, int2 borderSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    // The sample bound has to come first: dst.width()/dst.height() index the image list by
    // sample without bound-checking it, so evaluating them for an out-of-range sample reads past
    // the list. The grid rounds up to blockDim.z == 2, so an odd sample count always produces
    // such a thread.
    if (dstCoord.z >= numSamples || dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z))
    {
        return;
    }

    int3 srcCoord = {dstCoord.x - borderSize.x, dstCoord.y - borderSize.y, dstCoord.z};

    dst[dstCoord] = src[srcCoord];
}

template<class DstWrapper, class SrcWrapper>
void DeviceRunFillBorderVarShape(DstWrapper &dstWrap, SrcWrapper &srcWrap, int3 dstMaxSize, int2 borderSize,
                                 cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(dstMaxSize.x + block.x - 1) / block.x, (dstMaxSize.y + block.y - 1) / block.y,
              (dstMaxSize.z + block.z - 1) / block.z};

    FillBorder<<<grid, block, 0, stream>>>(dstWrap, srcWrap, dstMaxSize.z, borderSize);
}

// -------------- value() / operator[] parity for the var-shape wraps --------------

// Same contract the tensor BorderWrap parity kernel checks for the var-shape wrapper used by
// Warp, CopyMakeBorder and Resize. Probed with int3 and int4 coordinates, so both var-shape
// coordinate ladders are covered.
template<class BorderWrapper>
__global__ void CompareVarShapeValueAndSubscript(BorderWrapper wrap, const int3 *coords, int numCoords, int *mismatches)
{
    for (int i = 0; i < numCoords; ++i)
    {
        const int3 c3 = coords[i];
        // (x, y, plane, sample). Putting the sample index in .z instead reads a plane that a
        // single-plane format does not have.
        const int4 c4{c3.x, c3.y, 0, c3.z};

        const auto viaValue3     = wrap.value(c3);
        const auto viaSubscript3 = wrap[c3];
        const auto viaValue4     = wrap.value(c4);
        const auto viaSubscript4 = wrap[c4];

        // Deliberately not `==`: these are vector types, so compare the raw bytes and stay
        // independent of any operator== the element type may or may not provide.
        const auto *a3 = reinterpret_cast<const unsigned char *>(&viaValue3);
        const auto *b3 = reinterpret_cast<const unsigned char *>(&viaSubscript3);
        const auto *a4 = reinterpret_cast<const unsigned char *>(&viaValue4);
        const auto *b4 = reinterpret_cast<const unsigned char *>(&viaSubscript4);

        for (unsigned k = 0; k < sizeof(viaValue3); ++k)
        {
            if (a3[k] != b3[k] || a4[k] != b4[k])
            {
                *mismatches = i + 1;
                return;
            }
        }
    }
    *mismatches = 0;
}

template<class BorderWrapper>
void DeviceRunVarShapeValueMatchesSubscript(BorderWrapper &wrap, const int3 *coords, int numCoords, int *mismatches,
                                            cudaStream_t &stream)
{
    CompareVarShapeValueAndSubscript<<<1, 1, 0, stream>>>(wrap, coords, numCoords, mismatches);
}

// -------------- To allow testing device-side BorderVarShapeWrapNHWC --------------

template<class DstWrapper, class SrcWrapper>
__global__ void FillBorderNHWC(DstWrapper dst, SrcWrapper src, int numSamples, int2 borderSize, int numChannels)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    // The sample bound has to come first: dst.width()/dst.height() index the image list by
    // sample without bound-checking it, so evaluating them for an out-of-range sample reads past
    // the list. The grid rounds up to blockDim.z == 2, so an odd sample count always produces
    // such a thread.
    if (dstCoord.z >= numSamples || dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z))
    {
        return;
    }
    for (int c = 0; c < numChannels; ++c)
    {
        int4 srcCoord = {dstCoord.z, dstCoord.y - borderSize.y, dstCoord.x - borderSize.x, c};
        dst[{dstCoord.z, dstCoord.y, dstCoord.x, c}] = src[srcCoord];
    }
}

template<class DstWrapper, class SrcWrapper>
void DeviceRunFillBorderVarShapeNHWC(DstWrapper &dstWrap, SrcWrapper &srcWrap, int3 dstMaxSize, int2 borderSize,
                                     int numChannels, cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(dstMaxSize.x + block.x - 1) / block.x, (dstMaxSize.y + block.y - 1) / block.y,
              (dstMaxSize.z + block.z - 1) / block.z};

    FillBorderNHWC<<<grid, block, 0, stream>>>(dstWrap, srcWrap, dstMaxSize.z, borderSize, numChannels);
}

// Need to instantiate each test on TestBorderVarShapeWrap

#define NVCV_TEST_INST(DSTWRAPPER, SRCWRAPPER) \
    template void DeviceRunFillBorderVarShape(DSTWRAPPER &, SRCWRAPPER &, int3, int2, cudaStream_t &)

#define IW(VALUETYPE) cuda::ImageBatchVarShapeWrap<VALUETYPE>

#define BW(VALUETYPE, BORDERTYPE) cuda::BorderVarShapeWrap<VALUETYPE, BORDERTYPE>

NVCV_TEST_INST(IW(uchar4), BW(const uchar4, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST(IW(short2), BW(const short2, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST(IW(float4), BW(const float4, NVCV_BORDER_REPLICATE));
NVCV_TEST_INST(IW(float3), BW(const float3, NVCV_BORDER_WRAP));
NVCV_TEST_INST(IW(uchar3), BW(const uchar3, NVCV_BORDER_REFLECT));
NVCV_TEST_INST(IW(short1), BW(const short1, NVCV_BORDER_REFLECT101));

#define NVCV_TEST_INST_PARITY(SRCWRAPPER) \
    template void DeviceRunVarShapeValueMatchesSubscript(SRCWRAPPER &, const int3 *, int, int *, cudaStream_t &)

NVCV_TEST_INST_PARITY(BW(const uchar3, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW(const uchar4, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW(const float4, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW(const uchar1, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW(const float1, NVCV_BORDER_CONSTANT));

#undef NVCV_TEST_INST_PARITY

#undef IW

#undef BW

#undef NVCV_TEST_INST

// Need to instantiate each test on TestBorderVarShapeWrapNHWC

#define NVCV_TEST_INST(DSTWRAPPER_NHWC, SRCWRAPPER_NHWC) \
    template void DeviceRunFillBorderVarShapeNHWC(DSTWRAPPER_NHWC &, SRCWRAPPER_NHWC &, int3, int2, int, cudaStream_t &)

#define IW(VALUETYPE) cuda::ImageBatchVarShapeWrapNHWC<VALUETYPE>

#define BW(VALUETYPE, BORDERTYPE) cuda::BorderVarShapeWrapNHWC<VALUETYPE, BORDERTYPE>

NVCV_TEST_INST(IW(uchar1), BW(const uchar1, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST(IW(short1), BW(const short1, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST(IW(float1), BW(const float1, NVCV_BORDER_REPLICATE));
NVCV_TEST_INST(IW(float1), BW(const float1, NVCV_BORDER_WRAP));
NVCV_TEST_INST(IW(uchar1), BW(const uchar1, NVCV_BORDER_REFLECT));
NVCV_TEST_INST(IW(short1), BW(const short1, NVCV_BORDER_REFLECT101));

#undef IW

#undef BW

#undef NVCV_TEST_INST
