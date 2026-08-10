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

#ifndef CVCUDA_PRIV_CUDA_DEVICE_UTILS_HPP
#define CVCUDA_PRIV_CUDA_DEVICE_UTILS_HPP

#include <cuda_runtime.h>

namespace cvcuda::priv {

// Returns the current device's SM compute-capability encoding (major * 10 + minor),
// not its number of streaming multiprocessors.
inline cudaError_t GetCurrentDeviceSM(int &sm) noexcept
{
    sm = 0;

    int         device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        return status;

    static thread_local int cachedDevice = -1;
    static thread_local int cachedSM     = 0;
    if (cachedDevice != device)
    {
        int major = 0;
        int minor = 0;

        status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        if (status != cudaSuccess)
            return status;

        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        if (status != cudaSuccess)
            return status;

        cachedDevice = device;
        cachedSM     = major * 10 + minor;
    }

    sm = cachedSM;
    return cudaSuccess;
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_CUDA_DEVICE_UTILS_HPP
