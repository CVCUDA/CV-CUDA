/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace cvcuda::priv {

class AdaptiveThreshold final : public IOperator
{
public:
    explicit AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out, const double maxValue,
                    const NVCVAdaptiveThresholdType adaptiveMethod, const NVCVThresholdType thresholdType,
                    const int32_t blockSize, const double c) const;
    void operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::IImageBatchVarShape &out,
                    const nvcv::ITensor &maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                    const NVCVThresholdType thresholdType, const nvcv::ITensor &blockSize,
                    const nvcv::ITensor &c) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::AdaptiveThreshold>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::AdaptiveThresholdVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PADANDSTACK_HPP
