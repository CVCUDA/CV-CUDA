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

#include "DeviceBorderWrap.hpp" // to test in the device

#include <cvcuda/cuda_tools/BorderWrap.hpp>     // the object of this test
#include <cvcuda/cuda_tools/FullTensorWrap.hpp> // for FullTensorWrap, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>        // for operator *, etc.
#include <cvcuda/cuda_tools/StaticCast.hpp>     // for StaticCast, etc.
#include <cvcuda/cuda_tools/TensorWrap.hpp>     // for Tensor3DWrap, etc.
#include <gtest/gtest.h>                        // for EXPECT_EQ, etc.

namespace cuda = nvcv::cuda;

// ------------------ To allow testing device-side BorderWrap ------------------

template<class DstWrapper, class SrcWrapper, typename DimType>
__global__ void FillBorder(DstWrapper dst, SrcWrapper src, DimType dstSize, int2 borderSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.z >= dstSize.z || dstCoord.y >= dstSize.y || dstCoord.x >= dstSize.x)
    {
        return;
    }

    int3 srcCoord = {dstCoord.x - borderSize.x, dstCoord.y - borderSize.y, dstCoord.z};

    dst[dstCoord] = src[srcCoord];
}

template<class DstWrapper, class SrcWrapper, typename DimType>
void DeviceRunFillBorder(DstWrapper &dstWrap, SrcWrapper &srcWrap, DimType dstSize, int2 borderSize,
                         cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(dstSize.x + block.x - 1) / block.x, (dstSize.y + block.y - 1) / block.y,
              (dstSize.z + block.z - 1) / block.z};

    FillBorder<<<grid, block, 0, stream>>>(dstWrap, srcWrap, dstSize, borderSize);
}

// ------------- value() / operator[] parity, executed on the device -------------

// The host suite cannot cover this. ReadConstantBorder picks its form on __CUDA_ARCH__, so a host
// build compiles exactly one arm and the arm that actually ships is never run. This kernel reads
// each probed coordinate both ways on the GPU and reports the first disagreement.
//
// Scope: these probes assert value parity, not codegen. They run <<<1, 1>>> over a coordinate
// array, where none of the register pressure that triggers the stack staging exists, so a build
// that reintroduced the spill would still pass them. What guards the codegen property is the
// per-architecture SASS evidence quoted on detail::ReadConstantBorder plus the committed
// constant-border benchmark baselines -- not this file.
// The same parity check driven with a 4-D coordinate. Every shipping operator reads a border wrap
// with int3 or int4, never int1, and the coordinate ladder that resolves a pointer is written out
// per dimension count -- so proving the 1-D ladder says nothing about the int4 arm, which is what
// InterpolationWrap::doGetValue reaches on a 4-D wrap.
template<class BorderWrapper, typename CoordType>
__global__ void CompareValueAndSubscriptND(BorderWrapper wrap, const CoordType *coords, int numCoords, int *mismatches)
{
    for (int i = 0; i < numCoords; ++i)
    {
        const CoordType c = coords[i];

        const auto viaValue     = wrap.value(c);
        const auto viaSubscript = wrap[c];

        const auto *a = reinterpret_cast<const unsigned char *>(&viaValue);
        const auto *b = reinterpret_cast<const unsigned char *>(&viaSubscript);

        for (unsigned k = 0; k < sizeof(viaValue); ++k)
        {
            if (a[k] != b[k])
            {
                *mismatches = i + 1;
                return;
            }
        }
    }
    *mismatches = 0;
}

template<class BorderWrapper>
void DeviceRunValueMatchesSubscript(BorderWrapper &wrap, const int *coords, int numCoords, int *mismatches,
                                    cudaStream_t &stream)
{
    CompareValueAndSubscriptND<<<1, 1, 0, stream>>>(wrap, reinterpret_cast<const int1 *>(coords), numCoords,
                                                    mismatches);
}

template<class BorderWrapper>
void DeviceRunValueMatchesSubscript4D(BorderWrapper &wrap, const int4 *coords, int numCoords, int *mismatches,
                                      cudaStream_t &stream)
{
    CompareValueAndSubscriptND<<<1, 1, 0, stream>>>(wrap, coords, numCoords, mismatches);
}

// Run both constant-border forms on the device and report disagreement. This is what the
// value()/operator[] parity above cannot reach: ReadConstantBorder resolves to one arm per
// __CUDA_ARCH__, so on any given GPU that test only ever exercises the arm that architecture
// selected, and the other one ships unexecuted. Calling the two helpers directly runs both on
// every GPU the suite runs on, which is what makes the selection in ReadConstantBorder a pure
// performance choice rather than a correctness one.
template<typename T>
__global__ void CompareConstantBorderForms(const T *inside, const T *outside, T border, int *mismatches)
{
    // `outside` is a null the host passes in, not a literal: a literal folds both `p == nullptr`
    // tests at compile time and the device never runs the branch the shipping code relies on.
    const T probes[2][2] = {
        { cuda::detail::ReadConstantBorderViaValueSelect(inside, border),
         cuda::detail::ReadConstantBorderViaAddressSelect(inside, border) },
        {cuda::detail::ReadConstantBorderViaValueSelect(outside, border),
         cuda::detail::ReadConstantBorderViaAddressSelect(outside, border)},
    };

    for (int p = 0; p < 2; ++p)
    {
        const auto *a = reinterpret_cast<const unsigned char *>(&probes[p][0]);
        const auto *b = reinterpret_cast<const unsigned char *>(&probes[p][1]);

        for (unsigned k = 0; k < sizeof(T); ++k)
        {
            if (a[k] != b[k])
            {
                *mismatches = p + 1;
                return;
            }
        }
    }
    // Last, like the sibling kernels above: success is recorded only once every comparison has
    // passed, so an early return added later cannot report agreement it never checked.
    *mismatches = 0;
}

template<typename T>
void DeviceRunConstantBorderFormsAgree(const T *inside, const T *outside, T border, int *mismatches,
                                       cudaStream_t &stream)
{
    CompareConstantBorderForms<<<1, 1, 0, stream>>>(inside, outside, border, mismatches);
}

// Need to instantiate each test on TestBorderWrap

#define NVCV_TEST_INST(DSTWRAPPER, SRCWRAPPER, DIMTYPE) \
    template void DeviceRunFillBorder(DSTWRAPPER &, SRCWRAPPER &, DIMTYPE, int2, cudaStream_t &)

#define NVCV_TEST_INST_FORMS(VALUETYPE)                                                                     \
    template void DeviceRunConstantBorderFormsAgree(const VALUETYPE *, const VALUETYPE *, VALUETYPE, int *, \
                                                    cudaStream_t &)

NVCV_TEST_INST_FORMS(uchar1);
NVCV_TEST_INST_FORMS(uchar3);
NVCV_TEST_INST_FORMS(uchar4);
NVCV_TEST_INST_FORMS(short2);
NVCV_TEST_INST_FORMS(float3);
NVCV_TEST_INST_FORMS(float4);

#undef NVCV_TEST_INST_FORMS

#define T3D(VALUETYPE) cuda::Tensor3DWrap<VALUETYPE>

#define B3D(VALUETYPE, BORDERTYPE) cuda::BorderWrap<T3D(VALUETYPE), BORDERTYPE, false, true, true>

NVCV_TEST_INST(T3D(uchar4), B3D(const uchar4, NVCV_BORDER_CONSTANT), int3);
NVCV_TEST_INST(T3D(short2), B3D(const short2, NVCV_BORDER_CONSTANT), int3);
NVCV_TEST_INST(T3D(float4), B3D(const float4, NVCV_BORDER_REPLICATE), int3);
NVCV_TEST_INST(T3D(float3), B3D(const float3, NVCV_BORDER_WRAP), int3);
NVCV_TEST_INST(T3D(uchar3), B3D(const uchar3, NVCV_BORDER_REFLECT), int3);
NVCV_TEST_INST(T3D(short1), B3D(const short1, NVCV_BORDER_REFLECT101), int3);

#undef T3D
#undef T4D

#undef B3D
#undef B4D

#define FT3D(VALUETYPE) cuda::FullTensorWrap<VALUETYPE, 3>
#define FT4D(VALUETYPE) cuda::FullTensorWrap<VALUETYPE, 4>

#define B3D(VALUETYPE, BORDERTYPE) cuda::BorderWrap<FT3D(VALUETYPE), BORDERTYPE, false, true, true>
#define B4D(VALUETYPE, BORDERTYPE) cuda::BorderWrap<FT4D(VALUETYPE), BORDERTYPE, false, true, true, false>

NVCV_TEST_INST(FT3D(uchar4), B3D(const uchar4, NVCV_BORDER_CONSTANT), int3);
NVCV_TEST_INST(FT3D(short2), B3D(const short2, NVCV_BORDER_CONSTANT), int3);
NVCV_TEST_INST(FT4D(float1), B4D(const float1, NVCV_BORDER_CONSTANT), int4);
NVCV_TEST_INST(FT3D(float4), B3D(const float4, NVCV_BORDER_REPLICATE), int3);
NVCV_TEST_INST(FT4D(short2), B4D(const short2, NVCV_BORDER_REPLICATE), int4);
NVCV_TEST_INST(FT3D(float3), B3D(const float3, NVCV_BORDER_WRAP), int3);
NVCV_TEST_INST(FT4D(uchar3), B4D(const uchar3, NVCV_BORDER_WRAP), int4);
NVCV_TEST_INST(FT3D(uchar3), B3D(const uchar3, NVCV_BORDER_REFLECT), int3);
NVCV_TEST_INST(FT4D(uchar1), B4D(const uchar1, NVCV_BORDER_REFLECT), int4);
NVCV_TEST_INST(FT3D(short1), B3D(const short1, NVCV_BORDER_REFLECT101), int3);
NVCV_TEST_INST(FT4D(uchar4), B4D(const uchar4, NVCV_BORDER_REFLECT101), int4);

#undef FT3D
#undef FT4D

#undef B3D
#undef B4D

#undef NVCV_TEST_INST

// Parity instantiations cover several element widths used with constant borders.
#define NVCV_TEST_INST_PARITY(SRCWRAPPER) \
    template void DeviceRunValueMatchesSubscript(SRCWRAPPER &, const int *, int, int *, cudaStream_t &)

#define BW1D(TYPE, BORDER) cuda::BorderWrap<cuda::TensorWrap<TYPE, -1>, BORDER, true>

NVCV_TEST_INST_PARITY(BW1D(const uchar3, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW1D(const uchar1, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW1D(const uchar4, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW1D(const short2, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW1D(const float3, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY(BW1D(const float4, NVCV_BORDER_CONSTANT));

// The N == 4 arm of the coordinate ladder: InterpolationWrap::doGetValue builds a 4-element
// coordinate whenever it wraps a 4-D tensor, so this arm ships and the 1-D/3-D probes miss it.
#define NVCV_TEST_INST_PARITY_4D(WRAPPER) \
    template void DeviceRunValueMatchesSubscript4D(WRAPPER &, const int4 *, int, int *, cudaStream_t &)

#define BW4D(TYPE, BORDER) cuda::BorderWrap<cuda::Tensor4DWrap<TYPE>, BORDER, false, false, true, true>

NVCV_TEST_INST_PARITY_4D(BW4D(const uchar3, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY_4D(BW4D(const uchar1, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY_4D(BW4D(const uchar4, NVCV_BORDER_CONSTANT));
NVCV_TEST_INST_PARITY_4D(BW4D(const float4, NVCV_BORDER_CONSTANT));

#undef BW4D
#undef NVCV_TEST_INST_PARITY_4D

#undef BW1D
#undef NVCV_TEST_INST_PARITY
