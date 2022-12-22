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

#include "OpWarpPerspective.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

WarpPerspective::WarpPerspective(const int32_t maxVarShapeBatchSize)
{
    legacy::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::WarpPerspective>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::WarpPerspectiveVarShape>(maxVarShapeBatchSize);
}

void WarpPerspective::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out,
                                 const NVCVPerspectiveTransform transMatrix, const int32_t flags,
                                 const NVCVBorderType borderMode, const float4 borderValue) const
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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, transMatrix, flags, borderMode, borderValue, stream));
}

void WarpPerspective::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in,
                                 const nvcv::IImageBatchVarShape &out, const nvcv::ITensor &transMatrix,
                                 const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue) const
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

    auto *transMatrixData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(transMatrix.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "transformation matrix must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *transMatrixData, flags, borderMode, borderValue, stream));
}

} // namespace cvcuda::priv
