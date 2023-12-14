/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>                        // for EXPECT_EQ, etc.
#include <nvcv/cuda/BorderVarShapeWrap.hpp>     // the object of this test
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp> // for ImageBatchVarShape, etc.
#include <nvcv/cuda/MathOps.hpp>                // for operator *, etc.
#include <nvcv/cuda/StaticCast.hpp>             // for StaticCast, etc.

namespace cuda = nvcv::cuda;

// -------------- To allow testing device-side BorderVarShapeWrap --------------

template<class DstWrapper, class SrcWrapper>
__global__ void FillBorder(DstWrapper dst, SrcWrapper src, int numSamples, int2 borderSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z) || dstCoord.z >= numSamples)
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

// -------------- To allow testing device-side BorderVarShapeWrapNHWC --------------

template<class DstWrapper, class SrcWrapper>
__global__ void FillBorderNHWC(DstWrapper dst, SrcWrapper src, int numSamples, int2 borderSize, int numChannels)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z) || dstCoord.z >= numSamples)
    {
        return;
    }
    for (int c = 0; c < numChannels; ++c)
    {
        int4 srcCoord = {dstCoord.x - borderSize.x, dstCoord.y - borderSize.y, dstCoord.z, c};
        dst[{dstCoord.x, dstCoord.y, dstCoord.z, c}] = src[srcCoord];
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
