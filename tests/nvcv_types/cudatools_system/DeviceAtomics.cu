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

#include "DeviceAtomics.hpp" // to test in the device

#include <gtest/gtest.h>         // for EXPECT_EQ, etc.
#include <nvcv/cuda/Atomics.hpp> // the object of this test

namespace cuda = nvcv::cuda;

// ------------------- To allow testing device-side Atomics --------------------

typedef enum
{
    MIN = 0b01,
    MAX = 0b10
} RunChoice;

template<RunChoice RUN, typename DataType>
__global__ void RunAtomics(DataType *output, DataType *input)
{
    if constexpr (RUN == RunChoice::MIN)
    {
        cuda::AtomicMin(*output, input[blockIdx.x]);
    }
    else if constexpr (RUN == RunChoice::MAX)
    {
        cuda::AtomicMax(*output, input[blockIdx.x]);
    }
}

template<RunChoice RUN, typename DataType>
DataType DeviceRunAtomics(const std::vector<DataType> &hInput)
{
    DataType *dOutput;
    DataType  hOutput = RUN == RunChoice::MIN ? cuda::TypeTraits<DataType>::max : cuda::Lowest<DataType>;

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dOutput, sizeof(DataType)));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(dOutput, &hOutput, sizeof(DataType), cudaMemcpyHostToDevice));

    size_t size = hInput.size();
    EXPECT_LE(size, cuda::TypeTraits<int>::max);

    DataType *dInput;

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dInput, sizeof(DataType) * size));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(dInput, hInput.data(), sizeof(DataType) * size, cudaMemcpyHostToDevice));

    RunAtomics<RUN><<<size, 1>>>(dOutput, dInput);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(&hOutput, dOutput, sizeof(DataType), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dOutput));
    EXPECT_EQ(cudaSuccess, cudaFree(dInput));

    return hOutput;
}

template<typename DataType>
DataType DeviceRunAtomicMin(const std::vector<DataType> &hInput)
{
    return DeviceRunAtomics<RunChoice::MIN>(hInput);
}

template<typename DataType>
DataType DeviceRunAtomicMax(const std::vector<DataType> &hInput)
{
    return DeviceRunAtomics<RunChoice::MAX>(hInput);
}

// Need to instantiate each test on TestAtomics, making sure not to use const types

#define NVCV_TEST_INST(DATA_TYPE)                                          \
    template DATA_TYPE DeviceRunAtomicMin(const std::vector<DATA_TYPE> &); \
    template DATA_TYPE DeviceRunAtomicMax(const std::vector<DATA_TYPE> &)

NVCV_TEST_INST(unsigned int);
NVCV_TEST_INST(int);
NVCV_TEST_INST(unsigned long long int);
NVCV_TEST_INST(long long int);

NVCV_TEST_INST(float);
NVCV_TEST_INST(double);

#undef NVCV_TEST_INST
