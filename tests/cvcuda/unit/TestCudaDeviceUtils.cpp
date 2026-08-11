/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"

#include <cvcuda/priv/CudaDeviceUtils.hpp>

namespace {

class CudaDeviceRestorer
{
public:
    explicit CudaDeviceRestorer(int device)
        : m_device(device)
    {
    }

    ~CudaDeviceRestorer()
    {
        (void)cudaSetDevice(m_device);
    }

    CudaDeviceRestorer(const CudaDeviceRestorer &)            = delete;
    CudaDeviceRestorer &operator=(const CudaDeviceRestorer &) = delete;

private:
    int m_device;
};

TEST(CudaDeviceUtils, CurrentDeviceSMMatchesDeviceProperties)
{
    int         deviceCount = 0;
    cudaError_t status      = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << cudaGetErrorString(status);
    ASSERT_EQ(cudaSuccess, status);
    if (deviceCount == 0)
        GTEST_SKIP() << "No CUDA devices available";

    int originalDevice = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&originalDevice));
    CudaDeviceRestorer restoreDevice(originalDevice);

    for (int device = 0; device < deviceCount; ++device)
    {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(device));

        cudaDeviceProp properties{};
        ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&properties, device));
        const int expectedSM = properties.major * 10 + properties.minor;

        int sm = 0;
        EXPECT_EQ(cudaSuccess, cvcuda::priv::GetCurrentDeviceSM(sm));
        EXPECT_EQ(expectedSM, sm);

        sm = 0;
        EXPECT_EQ(cudaSuccess, cvcuda::priv::GetCurrentDeviceSM(sm));
        EXPECT_EQ(expectedSM, sm);
    }
}

} // namespace
