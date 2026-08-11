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

#include "OpMinAreaRect.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

MinAreaRect::MinAreaRect(int maxContourNum)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_legacyOp(
        [maxContourNum](int)
        {
            legacy::DataShape maxIn;
            legacy::DataShape maxOut;
            return std::make_unique<legacy::MinAreaRect>(maxIn, maxOut, maxContourNum);
        })
{
}

void MinAreaRect::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             const nvcv::Tensor &numPointsInContour, const int totalContours) const
{
    CVCUDA_NVTX_RANGE("cvcuda::MinAreaRect::operator()[Tensor]");
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
    auto numPointsInContourData = numPointsInContour.exportData<nvcv::TensorDataStridedCuda>();
    if (numPointsInContourData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numPointsInContour must be cuda-accessible, pitch-linear tensor");
    }

    // format check
    if (inData->layout() != nvcv::TENSOR_NWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have NWC layout");
    }
    if (numPointsInContourData->layout() != nvcv::TENSOR_NW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have NW layout");
    }
    if (outData->layout() != nvcv::TENSOR_NW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must have NW layout");
    }

    // channel check
    auto inShape = inData->shape();
    if (auto channels = static_cast<int>(inShape[inShape.rank() - 1]); channels != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must have 2 channels (x, y coordinates per point)");
    }

    // dtype check
    if (inData->dtype() != nvcv::TYPE_U16 && inData->dtype() != nvcv::TYPE_S16 && inData->dtype() != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must have TYPE_U16, TYPE_S16, or TYPE_S32 data type");
    }
    if (outData->dtype() != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must have TYPE_F32 data type");
    }
    if (numPointsInContourData->dtype() != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numPointsInContourData must have TYPE_S32 data type");
    }

    // add calls to kernel here
    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, *numPointsInContourData, totalContours, stream));
}

} // namespace cvcuda::priv
