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
 * @file OpRandomResizedCrop.hpp
 *
 * @brief Defines the private C++ Class for the random resized crop operation.
 */

#ifndef CVCUDA_PRIV_RANDOMRESIZEDCROP_HPP
#define CVCUDA_PRIV_RANDOMRESIZEDCROP_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <random>

namespace cvcuda::priv {

class RandomResizedCrop final : public IOperator
{
public:
    explicit RandomResizedCrop(double minScale, double maxScale, double minRatio, double maxRatio, int32_t maxBatchSize,
                               uint32_t seed);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVInterpolationType interpolation) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const NVCVInterpolationType interpolation) const;

private:
    // Crop windows are drawn on the host and staged through a host buffer into device memory, both
    // sized once from maxBatchSize.  Every submit advances the generator, so its running state --
    // not just the seed it started from -- selects the pixels the operator produces.
    //
    // A submit mutates this without holding a lock, so concurrent submits on one device against one
    // operator handle race on both the generator and the staging buffer.  That matches the legacy
    // operator, whose state this replaces one-for-one.
    struct CropState
    {
        CropState(int32_t maxBatchSize, uint32_t seed);
        ~CropState();

        CropState(const CropState &)            = delete;
        CropState &operator=(const CropState &) = delete;

        std::mt19937 generator;
        std::byte   *hostParams   = nullptr;
        std::byte   *deviceParams = nullptr;
    };

    double  m_minScale;
    double  m_maxScale;
    double  m_minRatio;
    double  m_maxRatio;
    int32_t m_maxBatchSize;

    // The tensor and var-shape paths keep independent generators, matching the two legacy operator
    // objects they replace: with the same seed, each path restarts the same draw sequence.
    // PerDeviceResource gives every CUDA device its own state and its own device allocation.
    mutable PerDeviceResource<CropState> m_state;
    mutable PerDeviceResource<CropState> m_stateVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_RANDOMRESIZEDCROP_HPP
