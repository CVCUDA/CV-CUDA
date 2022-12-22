/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceBorderWrap.hpp" // to test in the device

#include <gtest/gtest.h>            // for EXPECT_EQ, etc.
#include <nvcv/cuda/BorderWrap.hpp> // the object of this test
#include <nvcv/cuda/MathOps.hpp>    // for operator *, etc.
#include <nvcv/cuda/StaticCast.hpp> // for StaticCast, etc.
#include <nvcv/cuda/TensorWrap.hpp> // for Tensor3DWrap, etc.

namespace cuda = nvcv::cuda;

// ------------------ To allow testing device-side BorderWrap ------------------

template<class DstWrapper, class SrcWrapper>
__global__ void FillBorder(DstWrapper dst, SrcWrapper src, int3 dstSize, int3 borderSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.z >= dstSize.z || dstCoord.y >= dstSize.y || dstCoord.x >= dstSize.x)
    {
        return;
    }

    int3 srcCoord = {dstCoord.x - borderSize.x, dstCoord.y - borderSize.y, dstCoord.z};

    dst[dstCoord] = src[srcCoord];
}

template<class DstWrapper, class SrcWrapper>
__global__ void FillBorder(DstWrapper dst, SrcWrapper src, int4 dstSize, int4 borderSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.z >= dstSize.z || dstCoord.y >= dstSize.y || dstCoord.x >= dstSize.x)
    {
        return;
    }

    int3 srcCoord = {dstCoord.x - borderSize.x, dstCoord.y - borderSize.y, dstCoord.z};

    typename DstWrapper::ValueType *dstPtr = dst.ptr(dstCoord.z, dstCoord.y, dstCoord.x);
    typename SrcWrapper::ValueType *srcPtr = src.ptr(srcCoord.z, srcCoord.y, srcCoord.x);

    for (int c = 0; c < dstSize.w; ++c)
    {
        *dstPtr++ = (srcPtr == nullptr) ? src.borderValue() : *srcPtr++;
    }
}

template<class DstWrapper, class SrcWrapper, typename DimType>
void DeviceRunFillBorder(DstWrapper &dstWrap, SrcWrapper &srcWrap, DimType dstSize, DimType srcSize,
                         cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(dstSize.x + block.x - 1) / block.x, (dstSize.y + block.y - 1) / block.y,
              (dstSize.z + block.z - 1) / block.z};

    DimType borderSize = (dstSize - srcSize) / 2;

    FillBorder<<<grid, block, 0, stream>>>(dstWrap, srcWrap, dstSize, borderSize);
}

// Need to instantiate each test on TestBorderWrap

#define NVCV_TEST_INST(DSTWRAPPER, SRCWRAPPER, DIMTYPE) \
    template void DeviceRunFillBorder(DSTWRAPPER &, SRCWRAPPER &, DIMTYPE, DIMTYPE, cudaStream_t &)

#define T3D(VALUETYPE) cuda::Tensor3DWrap<VALUETYPE>
#define T4D(VALUETYPE) cuda::Tensor4DWrap<VALUETYPE>

#define B3D(VALUETYPE, BORDERTYPE) cuda::BorderWrapNHW<VALUETYPE, BORDERTYPE>
#define B4D(VALUETYPE, BORDERTYPE) cuda::BorderWrapNHWC<VALUETYPE, BORDERTYPE>

NVCV_TEST_INST(T3D(uchar4), B3D(const uchar4, NVCV_BORDER_CONSTANT), int3);
NVCV_TEST_INST(T3D(short2), B3D(const short2, NVCV_BORDER_CONSTANT), int3);
NVCV_TEST_INST(T4D(float1), B4D(const float1, NVCV_BORDER_CONSTANT), int4);
NVCV_TEST_INST(T3D(float4), B3D(const float4, NVCV_BORDER_REPLICATE), int3);
NVCV_TEST_INST(T4D(short1), B4D(const short1, NVCV_BORDER_REPLICATE), int4);
NVCV_TEST_INST(T3D(float3), B3D(const float3, NVCV_BORDER_WRAP), int3);
NVCV_TEST_INST(T4D(uchar1), B4D(const uchar1, NVCV_BORDER_WRAP), int4);
NVCV_TEST_INST(T3D(uchar3), B3D(const uchar3, NVCV_BORDER_REFLECT), int3);
NVCV_TEST_INST(T4D(uchar1), B4D(const uchar1, NVCV_BORDER_REFLECT), int4);
NVCV_TEST_INST(T3D(short1), B3D(const short1, NVCV_BORDER_REFLECT101), int3);
NVCV_TEST_INST(T4D(uchar1), B4D(const uchar1, NVCV_BORDER_REFLECT101), int4);

#undef T3D
#undef T4D

#undef NVCV_TEST_INST
