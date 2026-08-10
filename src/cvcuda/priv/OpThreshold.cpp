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

#include "OpThreshold.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Threshold::Threshold(uint32_t type, int maxBatchSize)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_legacyOp(
        [type, maxBatchSize](int)
        {
            legacy::DataShape maxIn;
            legacy::DataShape maxOut;
            return std::make_unique<legacy::Threshold>(maxIn, maxOut, type, maxBatchSize);
        })
    , m_legacyOpVarShape(
          [type, maxBatchSize](int)
          {
              legacy::DataShape maxIn;
              legacy::DataShape maxOut;
              return std::make_unique<legacy::ThresholdVarShape>(maxIn, maxOut, type, maxBatchSize);
          })
{
    if (maxBatchSize < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 0");
    }
}

void Threshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                           const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Threshold::operator()[Tensor]");
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

    auto threshData = thresh.exportData<nvcv::TensorDataStridedCuda>();
    if (threshData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "thresh must be cuda-accessible, pitch-linear tensor");
    }

    auto maxvalData = maxval.exportData<nvcv::TensorDataStridedCuda>();
    if (maxvalData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxval must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, *threshData, *maxvalData, stream));
}

void Threshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                           const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Threshold::operator()[ImageBatchVarShape]");
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

    auto threshData = thresh.exportData<nvcv::TensorDataStridedCuda>();
    if (threshData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "thresh must be cuda-accessible, pitch-linear tensor");
    }

    auto maxvalData = maxval.exportData<nvcv::TensorDataStridedCuda>();
    if (maxvalData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxval must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape.get().infer(*inData, *outData, *threshData, *maxvalData, stream));
}

} // namespace cvcuda::priv
