/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpHistogramEq.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

HistogramEq::HistogramEq(uint32_t maxBatchSize)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_maxBatchSize(maxBatchSize)
    , m_legacyOp([maxBatchSize](int) { return std::make_unique<legacy::HistogramEq>(maxBatchSize); })
    , m_legacyOpVarShape([maxBatchSize](int) { return std::make_unique<legacy::HistogramEqVarShape>(maxBatchSize); })
{
    if (maxBatchSize == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "HistogramEq: maxBatchSize must be >= 1");
    }
}

void HistogramEq::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HistogramEq::operator()[Tensor]");
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

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be image-compatible tensor");
    }

    if (inAccess->numSamples() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, stream));
}

void HistogramEq::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                             const nvcv::ImageBatchVarShape &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HistogramEq::operator()[ImageBatchVarShape]");
    if (in.numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

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

    NVCV_CHECK_THROW(m_legacyOpVarShape.get().infer(*inData, *outData, stream));
}

} // namespace cvcuda::priv
