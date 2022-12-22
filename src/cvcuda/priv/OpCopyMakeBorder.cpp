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

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out, const int top,
                                const int left, const NVCVBorderType borderMode, const float4 borderValue) const
{
    auto *inData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(in.exportData());
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, top, left, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::IImageBatch &in, const nvcv::ITensor &out,
                                const nvcv::ITensor &top, const nvcv::ITensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto *inData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto *topData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(top.exportData());
    if (topData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Top must be cuda-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(left.exportData());
    if (leftData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Left must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::IImageBatch &in, const nvcv::IImageBatch &out,
                                const nvcv::ITensor &top, const nvcv::ITensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto *inData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto *topData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(top.exportData());
    if (topData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Top must be cuda-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(left.exportData());
    if (leftData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Left must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

} // namespace cvcuda::priv
