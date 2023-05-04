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

#include "DeviceInterpolationVarShapeWrap.hpp" // to test in the device

#include <gtest/gtest.h>                           // for EXPECT_EQ, etc.
#include <nvcv/cuda/InterpolationVarShapeWrap.hpp> // the object of this test
#include <nvcv/cuda/MathOps.hpp>                   // for operator *, etc.
#include <nvcv/cuda/StaticCast.hpp>                // for StaticCast, etc.

namespace cuda = nvcv::cuda;

// ---------- To allow testing device-side InterpolationVarShapeWrap -----------

template<class DstWrapper, class SrcWrapper>
__global__ void InterpShift(DstWrapper dst, SrcWrapper src, int numBatches, float2 shift)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dst.width(dstCoord.z) || dstCoord.y >= dst.height(dstCoord.z) || dstCoord.z >= numBatches)
    {
        return;
    }

    float3 srcCoord = {dstCoord.x + shift.x, dstCoord.y + shift.y, static_cast<float>(dstCoord.z)};

    dst[dstCoord] = src[srcCoord];
}

template<class DstWrapper, class SrcWrapper>
void DeviceRunInterpVarShapeShift(DstWrapper &dstWrap, SrcWrapper &srcWrap, int3 dstMaxSize, float2 shift,
                                  cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(dstMaxSize.x + block.x - 1) / block.x, (dstMaxSize.y + block.y - 1) / block.y,
              (dstMaxSize.z + block.z - 1) / block.z};

    InterpShift<<<grid, block, 0, stream>>>(dstWrap, srcWrap, dstMaxSize.z, shift);
}

// Need to instantiate each test on TestInterpolationVarShapeWrap

#define NVCV_TEST_INST(DSTWRAPPER, SRCWRAPPER) \
    template void DeviceRunInterpVarShapeShift(DSTWRAPPER &, SRCWRAPPER &, int3, float2, cudaStream_t &)

#define DST(VALUETYPE)                         cuda::ImageBatchVarShapeWrap<VALUETYPE>
#define SRC(VALUETYPE, BORDERTYPE, INTERPTYPE) cuda::InterpolationVarShapeWrap<VALUETYPE, BORDERTYPE, INTERPTYPE>

NVCV_TEST_INST(DST(uchar1), SRC(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST));
NVCV_TEST_INST(DST(short1), SRC(const short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST));
NVCV_TEST_INST(DST(short2), SRC(const short2, NVCV_BORDER_REFLECT, NVCV_INTERP_NEAREST));
NVCV_TEST_INST(DST(uchar3), SRC(const uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST));
NVCV_TEST_INST(DST(uchar4), SRC(const uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_NEAREST));

NVCV_TEST_INST(DST(uchar1), SRC(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR));
NVCV_TEST_INST(DST(short1), SRC(const short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_LINEAR));
NVCV_TEST_INST(DST(short2), SRC(const short2, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR));
NVCV_TEST_INST(DST(uchar3), SRC(const uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR));
NVCV_TEST_INST(DST(uchar4), SRC(const uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_LINEAR));

NVCV_TEST_INST(DST(uchar1), SRC(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC));
NVCV_TEST_INST(DST(short1), SRC(const short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_CUBIC));
NVCV_TEST_INST(DST(short2), SRC(const short2, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC));
NVCV_TEST_INST(DST(uchar3), SRC(const uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_CUBIC));
NVCV_TEST_INST(DST(uchar4), SRC(const uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC));

NVCV_TEST_INST(DST(uchar1), SRC(const uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA));
NVCV_TEST_INST(DST(short1), SRC(const short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA));
NVCV_TEST_INST(DST(short2), SRC(const short2, NVCV_BORDER_REFLECT, NVCV_INTERP_AREA));
NVCV_TEST_INST(DST(uchar3), SRC(const uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_AREA));
NVCV_TEST_INST(DST(uchar4), SRC(const uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA));

#undef SRC
#undef DST

#undef NVCV_TEST_INST
