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

#include "OpCopyMakeBorder.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

CopyMakeBorder::CopyMakeBorder()
{
    legacy::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::CopyMakeBorder>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::CopyMakeBorderVarShape>(maxIn, maxOut);
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const int top,
                                const int left, const NVCVBorderType borderMode, const float4 borderValue) const
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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, top, left, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::Tensor &out,
                                const nvcv::Tensor &top, const nvcv::Tensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto topData = top.exportData<nvcv::TensorDataStridedCuda>();
    if (topData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Top must be cuda-accessible, pitch-linear tensor");
    }

    auto leftData = left.exportData<nvcv::TensorDataStridedCuda>();
    if (leftData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Left must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                const nvcv::Tensor &top, const nvcv::Tensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
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

    auto topData = top.exportData<nvcv::TensorDataStridedCuda>();
    if (topData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Top must be cuda-accessible, pitch-linear tensor");
    }

    auto leftData = left.exportData<nvcv::TensorDataStridedCuda>();
    if (leftData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Left must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

} // namespace cvcuda::priv
