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

#include "DeviceTensorWrap.hpp" // to test in the device

#include <gtest/gtest.h>            // for EXPECT_EQ, etc.
#include <nvcv/cuda/DropCast.hpp>   // for DropCast, etc.
#include <nvcv/cuda/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/StaticCast.hpp> // for StaticCast, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test

namespace cuda = nvcv::cuda;

// ---------------- To allow testing device-side Tensor*Wrap -------------------

template<class DstWrapper, class SrcWrapper>
__global__ void Copy(DstWrapper dst, SrcWrapper src)
{
    using DimType = cuda::MakeType<int, SrcWrapper::kNumDimensions>;
    DimType coord = cuda::StaticCast<int>(cuda::DropCast<SrcWrapper::kNumDimensions>(threadIdx));
    dst[coord]    = src[coord];
}

template<typename ValueType>
__global__ void Copy(cuda::Tensor4DWrap<ValueType> dst, cuda::Tensor4DWrap<const ValueType> src, int lastDimSize)
{
    int3 c3 = cuda::StaticCast<int>(threadIdx);
    for (int k = 0; k < lastDimSize; k++)
    {
        int4 c4{k, c3.x, c3.y, c3.z};
        dst[c4] = src[c4];
    }
}

template<class InputType>
void DeviceUseTensorWrap(const InputType &hGold)
{
    using ValueType = typename InputType::value_type;

    constexpr int totalBytes = TotalBytes<ValueType>(InputType::kShapes);

    ValueType *dInput;
    ValueType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, totalBytes));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), totalBytes, cudaMemcpyHostToDevice));

    if constexpr (InputType::kNumDim == 1)
    {
        cuda::Tensor1DWrap<const ValueType> src(dInput);
        cuda::Tensor1DWrap<ValueType>       dst(dTest);

        Copy<<<1, InputType::kBlocks>>>(dst, src);
    }
    else if constexpr (InputType::kNumDim == 2)
    {
        cuda::Tensor2DWrap<const ValueType> src(dInput, InputType::kStrides[0]);
        cuda::Tensor2DWrap<ValueType>       dst(dTest, InputType::kStrides[0]);

        Copy<<<1, InputType::kBlocks>>>(dst, src);
    }
    else if constexpr (InputType::kNumDim == 3)
    {
        cuda::Tensor3DWrap<const ValueType> src(dInput, InputType::kStrides[0], InputType::kStrides[1]);
        cuda::Tensor3DWrap<ValueType>       dst(dTest, InputType::kStrides[0], InputType::kStrides[1]);

        Copy<<<1, InputType::kBlocks>>>(dst, src);
    }
    else if constexpr (InputType::kNumDim == 4)
    {
        cuda::Tensor4DWrap<const ValueType> src(dInput, InputType::kStrides[0], InputType::kStrides[1],
                                                InputType::kStrides[2]);
        cuda::Tensor4DWrap<ValueType>       dst(dTest, InputType::kStrides[0], InputType::kStrides[1],
                                                InputType::kStrides[2]);

        Copy<<<1, InputType::kBlocks>>>(dst, src, InputType::kShapes[3]);
    }
    else
    {
        ASSERT_EQ(0, 1);
    }

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    InputType hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, totalBytes, cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestTensorWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(VALUE_TYPE, N) template void DeviceUseTensorWrap(const Array<VALUE_TYPE, N> &)

NVCV_TEST_INST_USE(int, 2);
NVCV_TEST_INST_USE(short3, 2);
NVCV_TEST_INST_USE(float1, 4);
NVCV_TEST_INST_USE(uchar4, 3);

#undef NVCV_TEST_INST_USE

#define NVCV_TEST_INST_USE(VALUE_TYPE, H, W) template void DeviceUseTensorWrap(const PackedImage<VALUE_TYPE, H, W> &)

NVCV_TEST_INST_USE(int, 2, 2);
NVCV_TEST_INST_USE(short3, 1, 2);
NVCV_TEST_INST_USE(float1, 2, 4);
NVCV_TEST_INST_USE(uchar4, 3, 3);

#undef NVCV_TEST_INST_USE

#define NVCV_TEST_INST_USE(VALUE_TYPE, N, H, W) \
    template void DeviceUseTensorWrap(const PackedTensor3D<VALUE_TYPE, N, H, W> &)

NVCV_TEST_INST_USE(int, 1, 2, 2);
NVCV_TEST_INST_USE(short3, 2, 2, 1);
NVCV_TEST_INST_USE(float1, 2, 2, 2);
NVCV_TEST_INST_USE(uchar4, 3, 3, 1);

#undef NVCV_TEST_INST_USE

#define NVCV_TEST_INST_USE(VALUE_TYPE, N, H, W, C) \
    template void DeviceUseTensorWrap(const PackedTensor4D<VALUE_TYPE, N, H, W, C> &)

NVCV_TEST_INST_USE(int, 1, 2, 2, 2);
NVCV_TEST_INST_USE(short3, 2, 2, 1, 2);
NVCV_TEST_INST_USE(float1, 2, 2, 2, 1);
NVCV_TEST_INST_USE(uchar4, 3, 3, 1, 1);

#undef NVCV_TEST_INST_USE

template<class DstWrapper, typename DimType>
__global__ void SetOnes(DstWrapper dst, DimType size)
{
    DimType coord = cuda::StaticCast<int>(cuda::DropCast<DstWrapper::kNumDimensions>(blockIdx * blockDim + threadIdx));

#pragma unroll
    for (int i = 0; i < DstWrapper::kNumDimensions; ++i)
    {
        if (cuda::GetElement(coord, i) >= cuda::GetElement(size, i))
        {
            return;
        }
    }

    dst[coord] = cuda::SetAll<typename DstWrapper::ValueType>(1);
}

template<typename ValueType>
__global__ void SetOnes(cuda::Tensor4DWrap<ValueType> dst, int4 size)
{
    int3 c3 = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (c3.x >= size.x || c3.y >= size.y || c3.z >= size.z)
    {
        return;
    }

    for (int k = 0; k < size.w; ++k)
    {
        int4 c4{k, c3.x, c3.y, c3.z};
        dst[c4] = cuda::SetAll<ValueType>(1);
    }
}

template<class DstWrapper, typename DimType = cuda::MakeType<int, DstWrapper::kNumDimensions>>
void DeviceSetOnes(DstWrapper &dst, DimType size, cudaStream_t &stream)
{
    dim3 block, grid;

    if constexpr (DstWrapper::kNumDimensions == 1)
    {
        block = dim3{32};
        grid  = dim3{(size.x + block.x - 1) / block.x};
    }
    else if constexpr (DstWrapper::kNumDimensions == 2)
    {
        block = dim3{32, 4};
        grid  = dim3{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y};
    }
    else if constexpr (DstWrapper::kNumDimensions == 3 || DstWrapper::kNumDimensions == 4)
    {
        block = dim3{32, 2, 2};
        grid  = dim3{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y,
                    (size.z + block.z - 1) / block.z};
    }
    else
    {
        ASSERT_EQ(0, 1);
    }

    SetOnes<<<grid, block, 0, stream>>>(dst, size);
}

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::Tensor1DWrap<VALUE_TYPE> &, int1, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::Tensor2DWrap<VALUE_TYPE> &, int2, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::Tensor3DWrap<VALUE_TYPE> &, int3, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(cuda::Tensor4DWrap<VALUE_TYPE> &, int4, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET
