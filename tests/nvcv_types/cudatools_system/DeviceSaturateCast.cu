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

#include "DeviceSaturateCast.hpp" // to test in the device

#include <gtest/gtest.h>              // for EXPECT_EQ, etc.
#include <nvcv/cuda/SaturateCast.hpp> // the object of this test

namespace cuda = nvcv::cuda;

// ----------------- To allow testing device-side SaturateCast -----------------

template<typename TargetDataType, typename SourceDataType>
__global__ void RunSaturateCast(TargetDataType *out, SourceDataType u)
{
    out[0] = cuda::SaturateCast<cuda::BaseType<TargetDataType>>(u);
}

template<typename TargetDataType, typename SourceDataType>
TargetDataType DeviceRunSaturateCast(SourceDataType pix)
{
    TargetDataType *dTest;
    TargetDataType  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(TargetDataType)));

    RunSaturateCast<<<1, 1>>>(dTest, pix);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(TargetDataType), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

// Need to instantiate each test on TestSaturateCast, making sure not to use const types

#define NVCV_TEST_INST(TARGET_DATA_TYPE, SOURCE_DATA_TYPE) \
    template TARGET_DATA_TYPE DeviceRunSaturateCast(SOURCE_DATA_TYPE pix)

NVCV_TEST_INST(char, char);
NVCV_TEST_INST(short, short);
NVCV_TEST_INST(int, int);
NVCV_TEST_INST(float, float);
NVCV_TEST_INST(double, double);

NVCV_TEST_INST(float3, double3);
NVCV_TEST_INST(double3, float3);

NVCV_TEST_INST(float4, char4);
NVCV_TEST_INST(float3, ushort3);
NVCV_TEST_INST(double2, uchar2);
NVCV_TEST_INST(double2, int2);

NVCV_TEST_INST(char2, float2);
NVCV_TEST_INST(ushort2, float2);
NVCV_TEST_INST(int2, float2);
NVCV_TEST_INST(uint2, float2);
NVCV_TEST_INST(uchar2, double2);
NVCV_TEST_INST(char2, double2);
NVCV_TEST_INST(short2, double2);

NVCV_TEST_INST(short1, char1);
NVCV_TEST_INST(ulonglong2, ulong2);
NVCV_TEST_INST(longlong2, long2);
NVCV_TEST_INST(ushort3, char3);
NVCV_TEST_INST(short2, uchar2);
NVCV_TEST_INST(uchar4, char4);
NVCV_TEST_INST(char3, uchar3);

NVCV_TEST_INST(short1, int1);
NVCV_TEST_INST(short2, uint2);
NVCV_TEST_INST(ushort3, int3);
NVCV_TEST_INST(uchar2, int2);
NVCV_TEST_INST(char2, uint2);
NVCV_TEST_INST(uchar2, ulonglong2);
NVCV_TEST_INST(char2, long2);

#undef NVCV_TEST_INST
