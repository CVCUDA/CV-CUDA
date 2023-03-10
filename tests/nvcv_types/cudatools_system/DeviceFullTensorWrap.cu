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

#include "DeviceFullTensorWrap.hpp" // to test in the device

#include <gtest/gtest.h>            // for EXPECT_EQ, etc.
#include <nvcv/cuda/DropCast.hpp>   // for DropCast, etc.
#include <nvcv/cuda/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/StaticCast.hpp> // for StaticCast, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test

namespace cuda = nvcv::cuda;

// --------------- To allow testing device-side FullTensorWrap -----------------

template<class DstWrapper, class SrcWrapper>
__global__ void Copy(DstWrapper dst, SrcWrapper src)
{
    int1 coord = cuda::StaticCast<int>(cuda::DropCast<1>(threadIdx));
    dst[coord] = src[coord];
}

template<class InputType>
void DeviceUseFullTensorWrap(const InputType &hGold)
{
    using ValueType = typename InputType::value_type;

    constexpr int totalBytes = TotalBytes<ValueType>(InputType::kShapes);

    ValueType *dInput;
    ValueType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, totalBytes));

    cuda::FullTensorWrap<const ValueType, InputType::kNumDim> src(dInput, InputType::kStrides, InputType::kShapes);
    cuda::FullTensorWrap<ValueType, InputType::kNumDim>       dst(dTest, InputType::kStrides, InputType::kShapes);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), totalBytes, cudaMemcpyHostToDevice));

    Copy<<<1, InputType::kBlocks>>>(dst, src);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    InputType hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, totalBytes, cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestTensorWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(VALUE_TYPE, N) template void DeviceUseFullTensorWrap(const Array<VALUE_TYPE, N> &)

NVCV_TEST_INST_USE(int, 2);
NVCV_TEST_INST_USE(short3, 2);
NVCV_TEST_INST_USE(float1, 4);
NVCV_TEST_INST_USE(uchar4, 3);

#undef NVCV_TEST_INST_USE

template<class DstWrapper>
__global__ void SetOnes(DstWrapper dst)
{
    using DimType = cuda::MakeType<int, DstWrapper::kNumDimensions>;
    DimType coord = cuda::StaticCast<int>(cuda::DropCast<DstWrapper::kNumDimensions>(blockIdx * blockDim + threadIdx));

#pragma unroll
    for (int i = 0; i < DstWrapper::kNumDimensions; ++i)
    {
        if (cuda::GetElement(coord, i) >= dst.shapes()[DstWrapper::kNumDimensions - 1 - i])
        {
            return;
        }
    }

    dst[coord] = cuda::SetAll<typename DstWrapper::ValueType>(1);
}

template<typename ValueType>
__global__ void SetOnes(cuda::FullTensorWrap<ValueType, 4> dst)
{
    int3 c3 = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (c3.x >= dst.shapes()[2] || c3.y >= dst.shapes()[1] || c3.z >= dst.shapes()[0])
    {
        return;
    }

    for (int k = 0; k < dst.shapes()[3]; ++k)
    {
        int4 c4{k, c3.x, c3.y, c3.z};
        dst[c4] = cuda::SetAll<ValueType>(1);
    }
}

template<class DstWrapper>
void DeviceSetOnes(DstWrapper &dst, cudaStream_t &stream)
{
    dim3 block, grid;

    if constexpr (DstWrapper::kNumDimensions == 1)
    {
        block = dim3{32};
        grid  = dim3{(dst.shapes()[0] + block.x - 1) / block.x};
    }
    else if constexpr (DstWrapper::kNumDimensions == 2)
    {
        block = dim3{32, 4};
        grid  = dim3{(dst.shapes()[1] + block.x - 1) / block.x, (dst.shapes()[0] + block.y - 1) / block.y};
    }
    else if constexpr (DstWrapper::kNumDimensions == 3 || DstWrapper::kNumDimensions == 4)
    {
        block = dim3{32, 2, 2};
        grid  = dim3{(dst.shapes()[2] + block.x - 1) / block.x, (dst.shapes()[1] + block.y - 1) / block.y,
                    (dst.shapes()[0] + block.z - 1) / block.z};
    }
    else
    {
        ASSERT_EQ(0, 1);
    }

    SetOnes<<<grid, block, 0, stream>>>(dst);
}

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::FullTensorWrap<VALUE_TYPE, 1> &, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::FullTensorWrap<VALUE_TYPE, 2> &, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::FullTensorWrap<VALUE_TYPE, 3> &, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::FullTensorWrap<VALUE_TYPE, 4> &, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET
