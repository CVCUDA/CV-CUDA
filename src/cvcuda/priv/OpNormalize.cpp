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

#include "OpNormalize.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Normalize::Normalize()
{
    legacy::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::Normalize>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::NormalizeVarShape>(maxIn, maxOut);
}

void Normalize::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &base,
                           const nvcv::ITensor &scale, nvcv::ITensor &out, const float global_scale, const float shift,
                           const float epsilon, const uint32_t flags) const
{
    auto *inData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(in.exportData());
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *baseData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(base.exportData());
    if (baseData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input base must be cuda-accessible, pitch-linear tensor");
    }

    auto *scaleData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(scale.exportData());
    if (scaleData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input scale must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *baseData, *scaleData, *outData, global_scale, shift, epsilon, flags, stream));
}

void Normalize::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::ITensor &base,
                           const nvcv::ITensor &scale, nvcv::IImageBatchVarShape &out, const float global_scale,
                           const float shift, const float epsilon, const uint32_t flags) const
{
    auto *inData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto *baseData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(base.exportData());
    if (baseData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input base must be cuda-accessible, pitch-linear tensor");
    }

    auto *scaleData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(scale.exportData());
    if (scaleData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input scale must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *baseData, *scaleData, *outData, global_scale, shift, epsilon,
                                               flags, stream));
}

} // namespace cvcuda::priv
