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

#include "OpErase.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Erase::Erase(int num_erasing_area)
{
    legacy::DataShape maxIn, maxOut;
    // maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::Erase>(maxIn, maxOut, num_erasing_area);
    m_legacyOpVarShape = std::make_unique<legacy::EraseVarShape>(maxIn, maxOut, num_erasing_area);
}

void Erase::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out, nvcv::ITensor &anchor,
                       nvcv::ITensor &erasing, nvcv::ITensor &values, nvcv::ITensor &imgIdx, bool random,
                       unsigned int seed) const
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

    auto *anchorData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchor.exportData());
    if (anchorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "anchor must be cuda-accessible, pitch-linear tensor");
    }

    auto *erasingData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(erasing.exportData());
    if (erasingData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "erasing must be cuda-accessible, pitch-linear tensor");
    }

    auto *valuesData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(values.exportData());
    if (valuesData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "values must be cuda-accessible, pitch-linear tensor");
    }

    auto *imgIdxData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgIdx.exportData());
    if (imgIdxData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "imgIdx must be cuda-accessible, pitch-linear tensor");
    }

    bool inplace = (in.handle() == out.handle());
    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, *anchorData, *erasingData, *valuesData, *imgIdxData, random,
                                       seed, inplace, stream));
}

void Erase::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::IImageBatchVarShape &out,
                       nvcv::ITensor &anchor, nvcv::ITensor &erasing, nvcv::ITensor &values, nvcv::ITensor &imgIdx,
                       bool random, unsigned int seed) const
{
    auto *anchorData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(anchor.exportData());
    if (anchorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "anchor must be cuda-accessible, pitch-linear tensor");
    }

    auto *erasingData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(erasing.exportData());
    if (erasingData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "erasing must be cuda-accessible, pitch-linear tensor");
    }

    auto *valuesData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(values.exportData());
    if (valuesData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "values must be cuda-accessible, pitch-linear tensor");
    }

    auto *imgIdxData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgIdx.exportData());
    if (imgIdxData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "imgIdx must be cuda-accessible, pitch-linear tensor");
    }

    bool inplace = (in.handle() == out.handle());
    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(in, out, *anchorData, *erasingData, *valuesData, *imgIdxData, random,
                                               seed, inplace, stream));
}

} // namespace cvcuda::priv
