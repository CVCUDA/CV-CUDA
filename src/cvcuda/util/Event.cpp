/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Event.hpp"

#include <cuda_runtime_api.h>
#include <nvcv/util/CheckError.hpp>

namespace nvcv::util {

CudaEvent CudaEvent::Create(int deviceId)
{
    return CreateWithFlags(cudaEventDisableTiming, deviceId);
}

CudaEvent CudaEvent::CreateWithFlags(unsigned flags, int deviceId)
{
    cudaEvent_t event;
    int         prevDev = -1;
    if (deviceId >= 0)
    {
        NVCV_CHECK_THROW(cudaGetDevice(&prevDev));
        NVCV_CHECK_THROW(cudaSetDevice(deviceId));
    }
    auto err = cudaEventCreateWithFlags(&event, flags);
    if (prevDev >= 0)
        NVCV_CHECK_THROW(cudaSetDevice(prevDev));
    NVCV_CHECK_THROW(err);
    return CudaEvent(event);
}

void CudaEvent::DestroyHandle(cudaEvent_t event)
{
    auto err = cudaEventDestroy(event);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading)
    {
        NVCV_CHECK_THROW(err);
    }
}

} // namespace nvcv::util
