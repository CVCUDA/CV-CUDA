/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Shared gamma-value handling for the tensor (gamma_contrast.cu) and var-shape
// (gamma_contrast_var_shape.cu) GammaContrast paths. Both normalize the user's gamma tensor (which
// may be per-sample or per-sample-per-channel, dense or strided) into a dense [numSamples*channels]
// float array consumed identically by their kernels -- so the two paths stay bit-exact.

#ifndef CVCUDA_LEGACY_GAMMA_CONTRAST_COMMON_CUH
#define CVCUDA_LEGACY_GAMMA_CONTRAST_COMMON_CUH

#include "CvCudaLegacy.h"

#include "CvCudaUtils.cuh"

namespace nvcv::legacy::cuda_op { namespace gamma_contrast_detail {

#define GAMMA_CONTRAST_BLOCK 256

__global__ static void copyGammaValues(float *gammaArray, const uint8_t *gammaBase, int64_t gammaStride,
                                       const int numImages, const int channelCount)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    const float gamma = *reinterpret_cast<const float *>(gammaBase + index * gammaStride);
    for (int i = 0; i < channelCount; i++)
    {
        gammaArray[index * channelCount + i] = gamma;
    }
}

__global__ static void copyPerChannelGammaValues(float *gammaArray, const uint8_t *gammaBase, int64_t sampleStride,
                                                 int64_t channelStride, const int numImages, const int channelCount)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages * channelCount)
    {
        return;
    }

    int imageIndex   = index / channelCount;
    int channelIndex = index % channelCount;
    gammaArray[index]
        = *reinterpret_cast<const float *>(gammaBase + imageIndex * sampleStride + channelIndex * channelStride);
}

inline bool IsTensorDense(const TensorDataStridedCuda &tensor)
{
    int64_t expectedStride = sizeof(float);
    for (int dim = tensor.rank() - 1; dim >= 0; --dim)
    {
        if (tensor.stride(dim) != expectedStride)
        {
            return false;
        }
        expectedStride *= tensor.shape(dim);
    }
    return true;
}

inline bool GetPerImageGammaStride(const TensorDataStridedCuda &gammas, int numImages, int64_t &sampleStride)
{
    if (IsTensorDense(gammas))
    {
        sampleStride = sizeof(float);
        return true;
    }

    const int sampleDim = gammas.layout().find(nvcv::LABEL_BATCH);
    if (sampleDim >= 0 && gammas.shape(sampleDim) == numImages)
    {
        sampleStride = gammas.stride(sampleDim);
        return true;
    }

    if (gammas.rank() == 1 && gammas.shape(0) == numImages)
    {
        sampleStride = gammas.stride(0);
        return true;
    }

    if (gammas.rank() == 2 && gammas.shape(0) == numImages && gammas.shape(1) == 1)
    {
        sampleStride = gammas.stride(0);
        return true;
    }

    return false;
}

inline bool GetPerChannelGammaStrides(const TensorDataStridedCuda &gammas, int numImages, int channelCount,
                                      int64_t &sampleStride, int64_t &channelStride)
{
    if (gammas.rank() == 1 && gammas.shape(0) == numImages * channelCount)
    {
        sampleStride  = gammas.stride(0) * channelCount;
        channelStride = gammas.stride(0);
        return true;
    }

    const int sampleDim  = gammas.layout().find(nvcv::LABEL_BATCH);
    const int channelDim = gammas.layout().find(nvcv::LABEL_CHANNEL);
    if (sampleDim >= 0 && channelDim >= 0 && gammas.shape(sampleDim) == numImages
        && gammas.shape(channelDim) == channelCount)
    {
        sampleStride  = gammas.stride(sampleDim);
        channelStride = gammas.stride(channelDim);
        return true;
    }

    if (gammas.rank() == 2 && gammas.shape(0) == numImages && gammas.shape(1) == channelCount)
    {
        sampleStride  = gammas.stride(0);
        channelStride = gammas.stride(1);
        return true;
    }

    return false;
}

// Validate the gamma tensor length and expand it into the dense [numImages*channels] gammaArray.
// Returns SUCCESS, or an error code on an invalid gamma shape.
inline ErrorCode ExpandGamma(const TensorDataStridedCuda &gammas, int numImages, int channels, float *gammaArray,
                             cudaStream_t stream)
{
    int numElements = 1;
    for (int i = 0; i < gammas.rank(); i++)
    {
        numElements *= gammas.shape(i);
    }

    if (numElements != numImages && numElements != numImages * channels)
    {
        LOG_ERROR("Invalid gamma tensor length " << numElements << ", expected " << numImages << " or "
                                                 << numImages * channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (numImages * channels == numElements)
    {
        if (IsTensorDense(gammas))
        {
            checkCudaErrors(cudaMemcpyAsync(gammaArray, gammas.basePtr(), sizeof(float) * numImages * channels,
                                            cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            int64_t sampleStride;
            int64_t channelStride;
            if (!GetPerChannelGammaStrides(gammas, numImages, channels, sampleStride, channelStride))
            {
                LOG_ERROR("Per-channel gamma tensor must be dense or shaped as per-image channel slices");
                return ErrorCode::INVALID_DATA_SHAPE;
            }

            copyPerChannelGammaValues<<<divUp(numImages * channels, GAMMA_CONTRAST_BLOCK), GAMMA_CONTRAST_BLOCK, 0,
                                        stream>>>(gammaArray, reinterpret_cast<const uint8_t *>(gammas.basePtr()),
                                                  sampleStride, channelStride, numImages, channels);
            checkKernelErrors();
        }
    }
    else
    {
        int64_t sampleStride;
        if (!GetPerImageGammaStride(gammas, numImages, sampleStride))
        {
            LOG_ERROR("Per-image gamma tensor must be dense or have a sample dimension matching the input batch");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        copyGammaValues<<<divUp(numImages, GAMMA_CONTRAST_BLOCK), GAMMA_CONTRAST_BLOCK, 0, stream>>>(
            gammaArray, reinterpret_cast<const uint8_t *>(gammas.basePtr()), sampleStride, numImages, channels);
        checkKernelErrors();
    }

    return ErrorCode::SUCCESS;
}

}} // namespace nvcv::legacy::cuda_op::gamma_contrast_detail

#endif // CVCUDA_LEGACY_GAMMA_CONTRAST_COMMON_CUH
