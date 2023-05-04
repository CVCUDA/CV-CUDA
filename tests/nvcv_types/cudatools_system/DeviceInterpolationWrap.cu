/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceInterpolationWrap.hpp" // to test in the device

#include <gtest/gtest.h>                   // for EXPECT_EQ, etc.
#include <nvcv/cuda/DropCast.hpp>          // for DropCast, etc.
#include <nvcv/cuda/InterpolationWrap.hpp> // the object of this test
#include <nvcv/cuda/MathOps.hpp>           // for operator *, etc.
#include <nvcv/cuda/StaticCast.hpp>        // for StaticCast, etc.

namespace cuda = nvcv::cuda;

// -------------- To allow testing device-side InterpolationWrap ---------------

template<class DstWrapper, class SrcWrapper>
__global__ void InterpShift(DstWrapper dst, SrcWrapper src, int2 dstSize, float2 shift)
{
    int2 dstCoord = cuda::DropCast<2>(cuda::StaticCast<int>(blockIdx * blockDim + threadIdx));

    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y)
    {
        return;
    }

    float2 srcCoord = {dstCoord.x + shift.x, dstCoord.y + shift.y};

    dst[dstCoord] = src[srcCoord];
}

template<class DstWrapper, class SrcWrapper>
__global__ void InterpShift(DstWrapper dst, SrcWrapper src, int3 dstSize, float2 shift)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y || dstCoord.z >= dstSize.z)
    {
        return;
    }

    float3 srcCoord = {dstCoord.x + shift.x, dstCoord.y + shift.y, static_cast<float>(dstCoord.z)};

    dst[dstCoord] = src[srcCoord];
}

template<class DstWrapper, class SrcWrapper>
__global__ void InterpShift(DstWrapper dst, SrcWrapper src, int4 dstSize, float2 shift)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x >= dstSize.x || coord.y >= dstSize.y || coord.z >= dstSize.z)
    {
        return;
    }

    int4   dstCoord{0, coord.x, coord.y, coord.z};
    float4 srcCoord{0.f, coord.x + shift.x, coord.y + shift.y, static_cast<float>(coord.z)};

#pragma unroll
    for (int k = 0; k < dstSize.w; ++k)
    {
        dstCoord.x = k;
        srcCoord.x = k;

        dst[dstCoord] = src[srcCoord];
    }
}

template<class DstWrapper, class SrcWrapper, typename DimType>
void DeviceRunInterpShift(DstWrapper &dstWrap, SrcWrapper &srcWrap, DimType dstSize, float2 shift, cudaStream_t &stream)
{
    dim3 block, grid;

    if constexpr (DstWrapper::kNumDimensions == 2)
    {
        block = dim3{32, 4};
        grid  = dim3{(dstSize.x + block.x - 1) / block.x, (dstSize.y + block.y - 1) / block.y};
    }
    else if constexpr (DstWrapper::kNumDimensions == 3 || DstWrapper::kNumDimensions == 4)
    {
        block = dim3{32, 2, 2};
        grid  = dim3{(dstSize.x + block.x - 1) / block.x, (dstSize.y + block.y - 1) / block.y,
                    (dstSize.z + block.z - 1) / block.z};
    }
    else
    {
        ASSERT_EQ(0, 1);
    }

    InterpShift<<<grid, block, 0, stream>>>(dstWrap, srcWrap, dstSize, shift);
}

// Need to instantiate each test on TestInterpolationWrap

#define NVCV_TEST_INST(DSTWRAPPER, SRCWRAPPER, DIMTYPE) \
    template void DeviceRunInterpShift(DSTWRAPPER &, SRCWRAPPER &, DIMTYPE, float2, cudaStream_t &)

#define T2D(VALUETYPE) cuda::Tensor2DWrap<VALUETYPE>
#define T3D(VALUETYPE) cuda::Tensor3DWrap<VALUETYPE>
#define T4D(VALUETYPE) cuda::Tensor4DWrap<VALUETYPE>

#define B2D(VALUETYPE, BORDERTYPE) cuda::BorderWrap<T2D(VALUETYPE), BORDERTYPE, true, true>
#define B3D(VALUETYPE, BORDERTYPE) cuda::BorderWrap<T3D(VALUETYPE), BORDERTYPE, false, true, true>
#define B4D(VALUETYPE, BORDERTYPE) cuda::BorderWrap<T4D(VALUETYPE), BORDERTYPE, false, true, true, false>

#define I2D(VALUETYPE, BORDERTYPE, INTERPTYPE) cuda::InterpolationWrap<B2D(VALUETYPE, BORDERTYPE), INTERPTYPE>
#define I3D(VALUETYPE, BORDERTYPE, INTERPTYPE) cuda::InterpolationWrap<B3D(VALUETYPE, BORDERTYPE), INTERPTYPE>
#define I4D(VALUETYPE, BORDERTYPE, INTERPTYPE) cuda::InterpolationWrap<B4D(VALUETYPE, BORDERTYPE), INTERPTYPE>

NVCV_TEST_INST(T2D(uchar4), I2D(const uchar4, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST), int2);
NVCV_TEST_INST(T2D(short2), I2D(const short2, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR), int2);
NVCV_TEST_INST(T2D(uchar1), I2D(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC), int2);
NVCV_TEST_INST(T2D(uchar3), I2D(const uchar3, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA), int2);
NVCV_TEST_INST(T2D(float4), I2D(const float4, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST), int2);
NVCV_TEST_INST(T2D(uchar3), I2D(const uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR), int2);
NVCV_TEST_INST(T2D(uchar4), I2D(const uchar4, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC), int2);
NVCV_TEST_INST(T2D(short2), I2D(const short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA), int2);
NVCV_TEST_INST(T2D(float3), I2D(const float3, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST), int2);
NVCV_TEST_INST(T2D(short1), I2D(const short1, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR), int2);
NVCV_TEST_INST(T2D(short2), I2D(const short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC), int2);
NVCV_TEST_INST(T2D(uchar4), I2D(const uchar4, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA), int2);

NVCV_TEST_INST(T3D(uchar4), I3D(const uchar4, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST), int3);
NVCV_TEST_INST(T3D(short2), I3D(const short2, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR), int3);
NVCV_TEST_INST(T3D(uchar1), I3D(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC), int3);
NVCV_TEST_INST(T3D(uchar3), I3D(const uchar3, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA), int3);
NVCV_TEST_INST(T3D(float4), I3D(const float4, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST), int3);
NVCV_TEST_INST(T3D(uchar3), I3D(const uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR), int3);
NVCV_TEST_INST(T3D(uchar4), I3D(const uchar4, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC), int3);
NVCV_TEST_INST(T3D(short2), I3D(const short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA), int3);
NVCV_TEST_INST(T3D(float3), I3D(const float3, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST), int3);
NVCV_TEST_INST(T3D(short1), I3D(const short1, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR), int3);
NVCV_TEST_INST(T3D(short2), I3D(const short2, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC), int3);
NVCV_TEST_INST(T3D(uchar4), I3D(const uchar4, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA), int3);

NVCV_TEST_INST(T4D(uchar1), I4D(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST), int4);
NVCV_TEST_INST(T4D(short1), I4D(const short1, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR), int4);
NVCV_TEST_INST(T4D(uchar1), I4D(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC), int4);
NVCV_TEST_INST(T4D(uchar1), I4D(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA), int4);
NVCV_TEST_INST(T4D(float1), I4D(const float1, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST), int4);
NVCV_TEST_INST(T4D(uchar1), I4D(const uchar1, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR), int4);
NVCV_TEST_INST(T4D(uchar1), I4D(const uchar1, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC), int4);
NVCV_TEST_INST(T4D(short1), I4D(const short1, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA), int4);
NVCV_TEST_INST(T4D(float1), I4D(const float1, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST), int4);
NVCV_TEST_INST(T4D(short1), I4D(const short1, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR), int4);
NVCV_TEST_INST(T4D(short1), I4D(const short1, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC), int4);
NVCV_TEST_INST(T4D(uchar1), I4D(const uchar1, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA), int4);

#undef T2D
#undef T3D
#undef T4D

#undef I3D
#undef I3D
#undef I4D

#undef NVCV_TEST_INST
