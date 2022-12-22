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

#include "OpMorphology.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Morphology::Morphology(const int32_t maxVarShapeBatchSize)
{
    legacy::DataShape maxIn, maxOut;
    //maxIn maxOut not used by ctor
    m_legacyOp         = std::make_unique<legacy::Morphology>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::MorphologyVarShape>(maxVarShapeBatchSize);
}

void Morphology::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out,
                            NVCVMorphologyType morph_type, nvcv::Size2D mask_size, int2 anchor, int32_t iteration,
                            const NVCVBorderType borderMode) const
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

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *outData, morph_type, mask_size, anchor, iteration, borderMode, stream));
}

void Morphology::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in,
                            const nvcv::IImageBatchVarShape &out, NVCVMorphologyType morph_type, nvcv::ITensor &masks,
                            nvcv::ITensor &anchors, int32_t iteration, NVCVBorderType borderMode) const
{
    auto *masksData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(masks.exportData());
    if (masksData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "masksData must be a tensor");
    }

    auto *anchorsData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchors.exportData());
    if (anchorsData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "anchors must be a tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(in, out, morph_type, *masksData, *anchorsData, iteration, borderMode, stream));
}

} // namespace cvcuda::priv
