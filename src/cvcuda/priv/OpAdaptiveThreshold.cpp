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

#include "OpAdaptiveThreshold.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

AdaptiveThreshold::AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize)
{
    legacy::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<legacy::AdaptiveThreshold>(maxIn, maxOut, maxBlockSize);
    m_legacyOpVarShape
        = std::make_unique<legacy::AdaptiveThresholdVarShape>(maxIn, maxOut, maxBlockSize, maxVarShapeBatchSize);
}

void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const double maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const int32_t blockSize, const double c) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *outData, maxValue, adaptiveMethod, thresholdType, blockSize, c, stream));
}

void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &maxValue,
                                   const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const nvcv::Tensor &blockSize,
                                   const nvcv::Tensor &c) const
{
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto maxvalueData = maxValue.exportData<nvcv::TensorDataStridedCuda>();
    if (maxvalueData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxValue must be cuda-accessible, pitch-linear tensor");
    }

    auto blocksizeData = blockSize.exportData<nvcv::TensorDataStridedCuda>();
    if (blocksizeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "blockSize must be cuda-accessible, pitch-linear tensor");
    }

    auto cData = c.exportData<nvcv::TensorDataStridedCuda>();
    if (cData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "C must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, *maxvalueData, adaptiveMethod, thresholdType,
                                               *blocksizeData, *cData, stream));
}

} // namespace cvcuda::priv
