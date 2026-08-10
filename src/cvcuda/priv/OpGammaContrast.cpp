/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpGammaContrast.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

GammaContrast::GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support. The same max
    // batch/channel limits bound both the tensor and var-shape paths' gamma scratch.
    : m_legacyOp([maxVarShapeBatchSize, maxVarShapeChannelCount](int)
                 { return std::make_unique<legacy::GammaContrast>(maxVarShapeBatchSize, maxVarShapeChannelCount); })
    , m_legacyOpVarShape(
          [maxVarShapeBatchSize, maxVarShapeChannelCount](int)
          { return std::make_unique<legacy::GammaContrastVarShape>(maxVarShapeBatchSize, maxVarShapeChannelCount); })
{
}

void GammaContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                               const nvcv::Tensor &gamma) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-acessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-acessible, pitch-linear tensor");
    }

    auto gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    if (gammaData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Gamma must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, *gammaData, stream));
}

void GammaContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float gamma,
                               float gain, NVCVRoundMode roundMode) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, gamma, gain, roundMode, stream));
}

void GammaContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                               const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &gamma) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GammaContrast::operator()[ImageBatchVarShape]");

    if (in.numImages() != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images");
    }

    for (int i = 0; i < in.numImages(); ++i)
    {
        if (in[i].size() != out[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output images must have matching width and height");
        }
    }

    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-acessible, varshape pitch-linear image batch");
    }

    auto gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    if (gammaData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Gamma must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape.get().infer(*inData, *outData, *gammaData, stream));
}

} // namespace cvcuda::priv
