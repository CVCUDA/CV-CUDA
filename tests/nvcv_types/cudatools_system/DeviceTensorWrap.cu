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

#include "DeviceTensorWrap.hpp" // to test in the device

#include <gtest/gtest.h>            // for EXPECT_EQ, etc.
#include <nvcv/cuda/DropCast.hpp>   // for DropCast, etc.
#include <nvcv/cuda/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/StaticCast.hpp> // for StaticCast, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test

namespace cuda = nvcv::cuda;

// ---------------- To allow testing device-side Tensor1DWrap ------------------

template<typename ValueType>
__global__ void Copy(cuda::Tensor1DWrap<ValueType> dst, cuda::Tensor1DWrap<const ValueType> src)
{
    int1 coord = cuda::StaticCast<int>(cuda::DropCast<1>(threadIdx));
    dst[coord] = src[coord];
}

template<typename ValueType, std::size_t N>
void DeviceUseTensor1DWrap(std::array<ValueType, N> &hGold)
{
    ValueType *dInput;
    ValueType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, N * sizeof(ValueType)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, N * sizeof(ValueType)));

    cuda::Tensor1DWrap<const ValueType> src(dInput);
    cuda::Tensor1DWrap<ValueType>       dst(dTest);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), N * sizeof(ValueType), cudaMemcpyHostToDevice));

    Copy<<<1, dim3(N)>>>(dst, src);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<ValueType, N> hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, N * sizeof(ValueType), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestTensorWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(VALUE_TYPE, N) template void DeviceUseTensor1DWrap(std::array<VALUE_TYPE, N> &)

NVCV_TEST_INST_USE(int, 2);
NVCV_TEST_INST_USE(short3, 2);
NVCV_TEST_INST_USE(float1, 4);
NVCV_TEST_INST_USE(uchar4, 3);

#undef NVCV_TEST_INST_USE

template<typename ValueType>
__global__ void SetOnes(nvcv::cuda::Tensor1DWrap<ValueType> dst, int1 size)
{
    int1 coord = cuda::StaticCast<int>(cuda::DropCast<1>(blockIdx * blockDim + threadIdx));

    if (coord.x >= size.x)
    {
        return;
    }

    dst[coord] = cuda::SetAll<ValueType>(1);
}

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor1DWrap<ValueType> &wrap, int1 size, cudaStream_t &stream)
{
    dim3 block{32};
    dim3 grid{(size.x + block.x - 1) / block.x};

    SetOnes<<<grid, block, 0, stream>>>(wrap, size);
}

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(nvcv::cuda::Tensor1DWrap<VALUE_TYPE> &, int1, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

// ---------------- To allow testing device-side Tensor2DWrap ------------------

template<typename ValueType>
__global__ void Copy(cuda::Tensor2DWrap<ValueType> dst, cuda::Tensor2DWrap<const ValueType> src)
{
    int2 coord = cuda::StaticCast<int>(cuda::DropCast<2>(threadIdx));
    dst[coord] = src[coord];
}

template<typename ValueType, int H, int W>
void DeviceUseTensor2DWrap(PackedImage<ValueType, H, W> &hGold)
{
    ValueType *dInput;
    ValueType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, H * W * sizeof(ValueType)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, H * W * sizeof(ValueType)));

    cuda::Tensor2DWrap<const ValueType> src(dInput, hGold.rowStride);
    cuda::Tensor2DWrap<ValueType>       dst(dTest, hGold.rowStride);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), H * W * sizeof(ValueType), cudaMemcpyHostToDevice));

    Copy<<<1, dim3(W, H)>>>(dst, src);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    PackedImage<ValueType, H, W> hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, H * W * sizeof(ValueType), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestTensorWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(VALUE_TYPE, H, W) template void DeviceUseTensor2DWrap(PackedImage<VALUE_TYPE, H, W> &)

NVCV_TEST_INST_USE(int, 2, 2);
NVCV_TEST_INST_USE(short3, 1, 2);
NVCV_TEST_INST_USE(float1, 2, 4);
NVCV_TEST_INST_USE(uchar4, 3, 3);

#undef NVCV_TEST_INST_USE

template<typename ValueType>
__global__ void SetOnes(nvcv::cuda::Tensor2DWrap<ValueType> dst, int2 size)
{
    int2 coord = cuda::StaticCast<int>(cuda::DropCast<2>(blockIdx * blockDim + threadIdx));

    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }

    dst[coord] = cuda::SetAll<ValueType>(1);
}

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor2DWrap<ValueType> &wrap, int2 size, cudaStream_t &stream)
{
    dim3 block{32, 4};
    dim3 grid{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y};

    SetOnes<<<grid, block, 0, stream>>>(wrap, size);
}

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(nvcv::cuda::Tensor2DWrap<VALUE_TYPE> &, int2, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

// ----------------- To allow testing device-side Tensor3DWrap -----------------

template<typename ValueType>
__global__ void Copy(cuda::Tensor3DWrap<ValueType> dst, cuda::Tensor3DWrap<const ValueType> src)
{
    int3 coord = cuda::StaticCast<int>(threadIdx);
    dst[coord] = src[coord];
}

template<typename ValueType, int N, int H, int W>
void DeviceUseTensor3DWrap(PackedTensor3D<ValueType, N, H, W> &hGold)
{
    ValueType *dInput;
    ValueType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, N * H * W * sizeof(ValueType)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, N * H * W * sizeof(ValueType)));

    cuda::Tensor3DWrap<const ValueType> src(dInput, hGold.stride1, hGold.stride2);
    cuda::Tensor3DWrap<ValueType>       dst(dTest, hGold.stride1, hGold.stride2);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), N * H * W * sizeof(ValueType), cudaMemcpyHostToDevice));

    Copy<<<1, dim3(W, H, N)>>>(dst, src);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    PackedTensor3D<ValueType, N, H, W> hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, N * H * W * sizeof(ValueType), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestTensorWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(VALUE_TYPE, N, H, W) \
    template void DeviceUseTensor3DWrap(PackedTensor3D<VALUE_TYPE, N, H, W> &)

NVCV_TEST_INST_USE(int, 1, 2, 2);
NVCV_TEST_INST_USE(short3, 2, 2, 1);
NVCV_TEST_INST_USE(float1, 2, 2, 2);
NVCV_TEST_INST_USE(uchar4, 3, 3, 1);

#undef NVCV_TEST_INST_USE

template<typename ValueType>
__global__ void SetOnes(nvcv::cuda::Tensor3DWrap<ValueType> dst, int3 size)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x >= size.x || coord.y >= size.y || coord.z >= size.z)
    {
        return;
    }

    dst[coord] = cuda::SetAll<ValueType>(1);
}

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor3DWrap<ValueType> &wrap, int3 size, cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y, (size.z + block.z - 1) / block.z};

    SetOnes<<<grid, block, 0, stream>>>(wrap, size);
}

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(nvcv::cuda::Tensor3DWrap<VALUE_TYPE> &, int3, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

// ----------------- To allow testing device-side Tensor4DWrap -----------------

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

template<typename ValueType, int N, int H, int W, int C>
void DeviceUseTensor4DWrap(PackedTensor4D<ValueType, N, H, W, C> &hGold)
{
    ValueType *dInput;
    ValueType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, N * H * W * C * sizeof(ValueType)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, N * H * W * C * sizeof(ValueType)));

    cuda::Tensor4DWrap<const ValueType> src(dInput, hGold.stride1, hGold.stride2, hGold.stride3);
    cuda::Tensor4DWrap<ValueType>       dst(dTest, hGold.stride1, hGold.stride2, hGold.stride3);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), N * H * W * C * sizeof(ValueType), cudaMemcpyHostToDevice));

    Copy<<<1, dim3(W, H, N)>>>(dst, src, C);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    PackedTensor4D<ValueType, N, H, W, C> hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, N * H * W * C * sizeof(ValueType), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestTensorWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(VALUE_TYPE, N, H, W, C) \
    template void DeviceUseTensor4DWrap(PackedTensor4D<VALUE_TYPE, N, H, W, C> &)

NVCV_TEST_INST_USE(int, 1, 2, 2, 2);
NVCV_TEST_INST_USE(short3, 2, 2, 1, 2);
NVCV_TEST_INST_USE(float1, 2, 2, 2, 1);
NVCV_TEST_INST_USE(uchar4, 3, 3, 1, 1);

#undef NVCV_TEST_INST_USE

template<typename ValueType>
__global__ void SetOnes(nvcv::cuda::Tensor4DWrap<ValueType> dst, int4 size)
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

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor4DWrap<ValueType> &wrap, int4 size, cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y, (size.z + block.z - 1) / block.z};

    SetOnes<<<grid, block, 0, stream>>>(wrap, size);
}

#define NVCV_TEST_INST_SET(VALUE_TYPE) \
    template void DeviceSetOnes(nvcv::cuda::Tensor4DWrap<VALUE_TYPE> &, int4, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET
