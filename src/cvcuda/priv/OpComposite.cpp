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

#include "OpComposite.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Composite::Composite()
{
    legacy::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::Composite>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::CompositeVarShape>(maxIn, maxOut);
}

void Composite::operator()(cudaStream_t stream, const nvcv::ITensor &foreground, const nvcv::ITensor &background,
                           const nvcv::ITensor &fgMask, const nvcv::ITensor &output) const
{
    auto *foregroundData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(foreground.exportData());
    if (foregroundData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input foreground must be cuda-accessible, pitch-linear tensor");
    }

    auto *backgroundData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(background.exportData());
    if (backgroundData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input background must be cuda-accessible, pitch-linear tensor");
    }

    auto *fgMaskData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(fgMask.exportData());
    if (fgMaskData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input fgMask must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(output.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*foregroundData, *backgroundData, *fgMaskData, *outData, stream));
}

void Composite::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &foreground,
                           const nvcv::IImageBatchVarShape &background, const nvcv::IImageBatchVarShape &fgMask,
                           const nvcv::IImageBatchVarShape &output) const
{
    auto *foregroundData
        = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(foreground.exportData(stream));
    if (foregroundData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input foreground must be cuda-accessible, varshape image batch");
    }

    auto *backgroundData
        = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(background.exportData(stream));
    if (backgroundData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input background must be cuda-accessible, varshape image batch");
    }

    auto *fgMaskData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(fgMask.exportData(stream));
    if (fgMaskData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input fgMask must be cuda-accessible, varshape image batch");
    }

    auto *outData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(output.exportData(stream));
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*foregroundData, *backgroundData, *fgMaskData, *outData, stream));
}

} // namespace cvcuda::priv
