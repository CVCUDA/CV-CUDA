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

#include "DeviceMathWrappers.hpp" // to test in the device

#include <gtest/gtest.h>              // for EXPECT_EQ, etc.
#include <nvcv/cuda/MathWrappers.hpp> // the object of this test

namespace cuda = nvcv::cuda;

// Need to instantiate each test on TestMathWrappers, making sure not to use const types

// -------------------- To allow testing device-side round ---------------------

template<typename SourceType, typename TargetType>
__global__ void RunRound(TargetType *out, SourceType u)
{
    if constexpr (std::is_same_v<SourceType, TargetType>)
    {
        out[0] = cuda::round(u);
    }
    else
    {
        out[0] = cuda::round<cuda::BaseType<TargetType>>(u);
    }
}

template<typename TargetType, typename SourceType>
TargetType DeviceRunRoundDiffType(SourceType pix)
{
    TargetType *dTest;
    TargetType  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(TargetType)));

    RunRound<<<1, 1>>>(dTest, pix);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(TargetType), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

template<typename Type>
Type DeviceRunRoundSameType(Type pix)
{
    return DeviceRunRoundDiffType<Type, Type>(pix);
}

#define NVCV_TEST_INST_ROUND_SAME(TYPE) template TYPE DeviceRunRoundSameType(TYPE pix)

NVCV_TEST_INST_ROUND_SAME(unsigned char);
NVCV_TEST_INST_ROUND_SAME(int);
NVCV_TEST_INST_ROUND_SAME(float);
NVCV_TEST_INST_ROUND_SAME(double);

NVCV_TEST_INST_ROUND_SAME(char1);
NVCV_TEST_INST_ROUND_SAME(uint2);
NVCV_TEST_INST_ROUND_SAME(float3);
NVCV_TEST_INST_ROUND_SAME(double4);

#undef NVCV_TEST_INST_ROUND_SAME

#define NVCV_TEST_INST_ROUND_DIFF(SOURCE_TYPE, TARGET_TYPE) template TARGET_TYPE DeviceRunRoundDiffType(SOURCE_TYPE pix)

NVCV_TEST_INST_ROUND_DIFF(float, int);
NVCV_TEST_INST_ROUND_DIFF(double, unsigned int);
NVCV_TEST_INST_ROUND_DIFF(float3, int3);
NVCV_TEST_INST_ROUND_DIFF(double4, long4);

NVCV_TEST_INST_ROUND_DIFF(signed char, signed char);
NVCV_TEST_INST_ROUND_DIFF(float2, float2);
NVCV_TEST_INST_ROUND_DIFF(uint1, uint1);
NVCV_TEST_INST_ROUND_DIFF(double2, double2);

#undef NVCV_TEST_INST_ROUND_DIFF

// -------------------- To allow testing device-side min ----------------------

template<typename Type>
__global__ void RunMin(Type *out, Type a, Type b)
{
    out[0] = cuda::min(a, b);
}

template<typename Type>
Type DeviceRunMin(Type pix1, Type pix2)
{
    Type *dTest;
    Type  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type)));

    RunMin<<<1, 1>>>(dTest, pix1, pix2);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_MIN(TYPE) template TYPE DeviceRunMin(TYPE pix1, TYPE pix2)

NVCV_TEST_INST_MIN(unsigned char);
NVCV_TEST_INST_MIN(int);
NVCV_TEST_INST_MIN(float);
NVCV_TEST_INST_MIN(double);

NVCV_TEST_INST_MIN(char1);
NVCV_TEST_INST_MIN(uint2);
NVCV_TEST_INST_MIN(float3);
NVCV_TEST_INST_MIN(double4);

NVCV_TEST_INST_MIN(short2);
NVCV_TEST_INST_MIN(char4);
NVCV_TEST_INST_MIN(ushort2);
NVCV_TEST_INST_MIN(uchar4);

#undef NVCV_TEST_INST_MIN

// -------------------- To allow testing device-side max ----------------------

template<typename Type>
__global__ void RunMax(Type *out, Type a, Type b)
{
    out[0] = cuda::max(a, b);
}

template<typename Type>
Type DeviceRunMax(Type pix1, Type pix2)
{
    Type *dTest;
    Type  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type)));

    RunMax<<<1, 1>>>(dTest, pix1, pix2);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_MAX(TYPE) template TYPE DeviceRunMax(TYPE pix1, TYPE pix2)

NVCV_TEST_INST_MAX(unsigned char);
NVCV_TEST_INST_MAX(int);
NVCV_TEST_INST_MAX(float);
NVCV_TEST_INST_MAX(double);

NVCV_TEST_INST_MAX(char1);
NVCV_TEST_INST_MAX(uint2);
NVCV_TEST_INST_MAX(float3);
NVCV_TEST_INST_MAX(double4);

NVCV_TEST_INST_MAX(short2);
NVCV_TEST_INST_MAX(char4);
NVCV_TEST_INST_MAX(ushort2);
NVCV_TEST_INST_MAX(uchar4);

#undef NVCV_TEST_INST_MAX

// --------------------- To allow testing device-side pow ----------------------

template<typename Type1, typename Type2>
__global__ void RunPow(Type1 *out, Type1 x, Type2 y)
{
    out[0] = cuda::pow(x, y);
}

template<typename Type1, typename Type2>
Type1 DeviceRunPow(Type1 x, Type2 y)
{
    Type1 *dTest;
    Type1  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type1)));

    RunPow<<<1, 1>>>(dTest, x, y);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type1), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_POW(TYPE1, TYPE2) template TYPE1 DeviceRunPow(TYPE1 x, TYPE2 y)

NVCV_TEST_INST_POW(unsigned char, unsigned char);
NVCV_TEST_INST_POW(unsigned char, int);
NVCV_TEST_INST_POW(float, float);
NVCV_TEST_INST_POW(double, unsigned char);

NVCV_TEST_INST_POW(char1, char1);
NVCV_TEST_INST_POW(uint2, uint2);
NVCV_TEST_INST_POW(float3, int);
NVCV_TEST_INST_POW(double4, float4);

#undef NVCV_TEST_INST_POW

// --------------------- To allow testing device-side exp ----------------------

template<typename Type>
__global__ void RunExp(Type *out, Type u)
{
    out[0] = cuda::exp(u);
}

template<typename Type>
Type DeviceRunExp(Type pix)
{
    Type *dTest;
    Type  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type)));

    RunExp<<<1, 1>>>(dTest, pix);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_EXP(TYPE) template TYPE DeviceRunExp(TYPE pix)

NVCV_TEST_INST_EXP(unsigned char);
NVCV_TEST_INST_EXP(int);
NVCV_TEST_INST_EXP(float);
NVCV_TEST_INST_EXP(double);

NVCV_TEST_INST_EXP(char1);
NVCV_TEST_INST_EXP(uint2);
NVCV_TEST_INST_EXP(float3);
NVCV_TEST_INST_EXP(double4);

#undef NVCV_TEST_INST_EXP

// -------------------- To allow testing device-side sqrt ----------------------

template<typename Type>
__global__ void RunSqrt(Type *out, Type u)
{
    out[0] = cuda::sqrt(u);
}

template<typename Type>
Type DeviceRunSqrt(Type pix)
{
    Type *dTest;
    Type  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type)));

    RunSqrt<<<1, 1>>>(dTest, pix);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_SQRT(TYPE) template TYPE DeviceRunSqrt(TYPE pix)

NVCV_TEST_INST_SQRT(unsigned char);
NVCV_TEST_INST_SQRT(int);
NVCV_TEST_INST_SQRT(float);
NVCV_TEST_INST_SQRT(double);

NVCV_TEST_INST_SQRT(char1);
NVCV_TEST_INST_SQRT(uint2);
NVCV_TEST_INST_SQRT(float3);
NVCV_TEST_INST_SQRT(double4);

#undef NVCV_TEST_INST_SQRT

// -------------------- To allow testing device-side abs ----------------------

template<typename Type>
__global__ void RunAbs(Type *out, Type u)
{
    out[0] = cuda::abs(u);
}

template<typename Type>
Type DeviceRunAbs(Type pix)
{
    Type *dTest;
    Type  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type)));

    RunAbs<<<1, 1>>>(dTest, pix);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_ABS(TYPE) template TYPE DeviceRunAbs(TYPE pix)

NVCV_TEST_INST_ABS(unsigned char);
NVCV_TEST_INST_ABS(int);
NVCV_TEST_INST_ABS(float);
NVCV_TEST_INST_ABS(double);

NVCV_TEST_INST_ABS(char1);
NVCV_TEST_INST_ABS(uint2);
NVCV_TEST_INST_ABS(float3);
NVCV_TEST_INST_ABS(double4);

NVCV_TEST_INST_ABS(short2);
NVCV_TEST_INST_ABS(char4);

#undef NVCV_TEST_INST_ABS

// -------------------- To allow testing device-side clamp ---------------------

template<typename Type1, typename Type2>
__global__ void RunClamp(Type1 *out, Type1 u, Type2 lo, Type2 hi)
{
    out[0] = cuda::clamp(u, lo, hi);
}

template<typename Type1, typename Type2>
Type1 DeviceRunClamp(Type1 u, Type2 lo, Type2 hi)
{
    Type1 *dTest;
    Type1  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(Type1)));

    RunClamp<<<1, 1>>>(dTest, u, lo, hi);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(Type1), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

#define NVCV_TEST_INST_CLAMP(TYPE1, TYPE2) template TYPE1 DeviceRunClamp(TYPE1 u, TYPE2 lo, TYPE2 hi)

NVCV_TEST_INST_CLAMP(unsigned char, unsigned char);
NVCV_TEST_INST_CLAMP(int, unsigned char);
NVCV_TEST_INST_CLAMP(float, unsigned short);
NVCV_TEST_INST_CLAMP(double, double);

NVCV_TEST_INST_CLAMP(char1, char1);
NVCV_TEST_INST_CLAMP(uint2, uint2);
NVCV_TEST_INST_CLAMP(float3, short);
NVCV_TEST_INST_CLAMP(double4, int4);

#undef NVCV_TEST_INST_CLAMP
