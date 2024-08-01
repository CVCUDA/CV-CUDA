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

#include "Stream.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvcv/util/CheckError.hpp>

namespace nvcv::util {

CudaStream CudaStream::Create(bool nonBlocking, int deviceId)
{
    cudaStream_t stream;
    int          flags   = nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
    int          prevDev = -1;
    if (deviceId >= 0)
    {
        NVCV_CHECK_THROW(cudaGetDevice(&prevDev));
        NVCV_CHECK_THROW(cudaSetDevice(deviceId));
    }
    auto err = cudaStreamCreateWithFlags(&stream, flags);
    if (prevDev >= 0)
        NVCV_CHECK_THROW(cudaSetDevice(prevDev));
    NVCV_CHECK_THROW(err);
    return CudaStream(stream);
}

CudaStream CudaStream::CreateWithPriority(bool nonBlocking, int priority, int deviceId)
{
    cudaStream_t stream;
    int          flags   = nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
    int          prevDev = -1;
    if (deviceId >= 0)
    {
        NVCV_CHECK_THROW(cudaGetDevice(&prevDev));
        NVCV_CHECK_THROW(cudaSetDevice(deviceId));
    }
    auto err = cudaStreamCreateWithPriority(&stream, flags, priority);
    if (prevDev >= 0)
        NVCV_CHECK_THROW(cudaSetDevice(prevDev));
    NVCV_CHECK_THROW(err);
    return CudaStream(stream);
}

void CudaStream::DestroyHandle(cudaStream_t stream)
{
    auto err = cudaStreamDestroy(stream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading)
    {
        NVCV_CHECK_THROW(err);
    }
}

} // namespace nvcv::util
