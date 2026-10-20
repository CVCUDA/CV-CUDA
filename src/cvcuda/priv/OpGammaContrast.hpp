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

/**
 * @file OpGammaContrast.hpp
 *
 * @brief Defines the private C++ Class for the gamma contrast operation.
 */

#ifndef CVCUDA_PRIV_GAMMA_CONTRAST_HPP
#define CVCUDA_PRIV_GAMMA_CONTRAST_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/RoundMode.h>
#include <nvcv/Tensor.hpp>

#include <cstdint>

// Use the public nvcv API
namespace cvcuda::priv {

// Device scratch holding the dense [maxBatchSize * maxChannelCount] gamma array that the two
// gamma-tensor paths stage into. It is allocated once per operator instance per CUDA device and
// freed with the operator -- not per call, and not through the workspace API -- which is the
// allocation strategy the operator has always had.
class GammaContrastScratch
{
public:
    GammaContrastScratch(int32_t maxBatchSize, int32_t maxChannelCount);
    ~GammaContrastScratch();

    GammaContrastScratch(const GammaContrastScratch &)            = delete;
    GammaContrastScratch &operator=(const GammaContrastScratch &) = delete;

    float *data() const
    {
        return m_gammaArray;
    }

private:
    float *m_gammaArray = nullptr;
};

class GammaContrast final : public IOperator
{
public:
    explicit GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &gamma) const;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float gamma, float gain,
                    NVCVRoundMode roundMode) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &gamma) const;

private:
    int32_t m_maxBatchSize;
    int32_t m_maxChannelCount;

    // The same max batch/channel limits bound both the tensor and var-shape paths' gamma scratch,
    // which stay separate buffers as they were when each path owned its own legacy operator.
    mutable PerDeviceResource<GammaContrastScratch> m_gammaScratch;
    mutable PerDeviceResource<GammaContrastScratch> m_gammaScratchVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_GAMMA_CONTRAST_HPP
