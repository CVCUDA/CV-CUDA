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

#ifndef NVCV_TESTUTILS_H
#define NVCV_TESTUTILS_H

#include "NvDecoder.h"

#include <cuda_runtime_api.h>
#include <nvcv/Tensor.hpp>

#define PROFILE_SAMPLE

inline void CheckCudaError(cudaError_t code, const char *file, const int line)
{
    if (code != cudaSuccess)
    {
        const char       *errorMessage = cudaGetErrorString(code);
        const std::string message      = "CUDA error returned at " + std::string(file) + ":" + std::to_string(line)
                                  + ", Error code: " + std::to_string(code) + " (" + std::string(errorMessage) + ")";
        throw std::runtime_error(message);
    }
}

#define CHECK_CUDA_ERROR(val)                      \
    {                                              \
        CheckCudaError((val), __FILE__, __LINE__); \
    }

void WriteRGBITensor(nvcv::Tensor &inTensor, cudaStream_t &stream)
{
    const auto *srcData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inTensor.exportData());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    int bufferSize = srcData->stride(0);
    int rowStride  = srcData->stride(1);
    int height     = inTensor.shape()[1];
    int width      = inTensor.shape()[2];
    int batchSize  = inTensor.shape()[0];

    for (int b = 0; b < batchSize; b++)
    {
        std::ostringstream ossIn;
        ossIn << "./cvcudatest_" << b << ".bmp";
        writeBMPi(ossIn.str().c_str(), (const unsigned char *)srcData->basePtr() + bufferSize * b, rowStride, width,
                  height);
    }
}

#endif // NVCV_TESTUTILS_H
