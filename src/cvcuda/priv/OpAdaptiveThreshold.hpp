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
 * @file OpAdaptiveThreshold.hpp
 *
 * @brief Defines the private ++ class for the AdaptiveThreshold operation.
 */

#ifndef CVCUDA_PRIV_ADAPTIVETHRESHOLD_HPP
#define CVCUDA_PRIV_ADAPTIVETHRESHOLD_HPP

#include "AdaptiveThresholdPolicy.hpp"
#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cstdint>

namespace cvcuda::priv {

class AdaptiveThreshold final : public IOperator
{
public:
    explicit AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double maxValue,
                    const NVCVAdaptiveThresholdType adaptiveMethod, const NVCVThresholdType thresholdType,
                    const int32_t blockSize, const double c) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                    const NVCVThresholdType thresholdType, const nvcv::Tensor &blockSize, const nvcv::Tensor &c) const;

private:
    // Persistent device scratch for the convolution coefficients, sized once from the constructor
    // arguments and freed in the destructor -- the allocation strategy the legacy operators used.
    struct TensorState
    {
        TensorState(int32_t maxBlockSize, AdaptiveThresholdKernelPolicy policy);
        ~TensorState();

        TensorState(const TensorState &)            = delete;
        TensorState &operator=(const TensorState &) = delete;

        const AdaptiveThresholdKernelPolicy kernelPolicy;
        float                              *kernel = nullptr;

        // Coefficients currently resident in `kernel`; an unchanged pair skips the recompute.
        int blockSize      = -1;
        int adaptiveMethod = -1;
    };

    struct VarShapeState
    {
        VarShapeState(int32_t maxBlockSize, int32_t maxVarShapeBatchSize, AdaptiveThresholdKernelPolicy policy);
        ~VarShapeState();

        VarShapeState(const VarShapeState &)            = delete;
        VarShapeState &operator=(const VarShapeState &) = delete;

        const AdaptiveThresholdKernelPolicy kernelPolicy;
        float                              *kernel = nullptr;
    };

    int32_t m_maxBlockSize;
    int32_t m_maxVarShapeBatchSize;

    // The coefficient buffers are plain device allocations, usable only on the device that
    // allocated them, so PerDeviceResource holds one state -- and one resolved kernel policy --
    // per CUDA device.
    mutable PerDeviceResource<TensorState>   m_tensorState;
    mutable PerDeviceResource<VarShapeState> m_varShapeState;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PADANDSTACK_HPP
