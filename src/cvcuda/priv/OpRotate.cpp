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

#include "OpRotate.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

std::unique_ptr<legacy::Rotate> Rotate::CreateLegacyOp(int)
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut;
    return std::make_unique<legacy::Rotate>(maxIn, maxOut);
}

Rotate::Rotate(const int maxVarShapeBatchSize)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_legacyOpVarShape([maxVarShapeBatchSize](int)
                         { return std::make_unique<legacy::RotateVarShape>(maxVarShapeBatchSize); })
{
}

void Rotate::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double angleDeg,
                        const double2 shift, const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Rotate::operator()[Tensor]");
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

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, angleDeg, shift, interpolation, stream));
}

void Rotate::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                        const nvcv::Tensor &angleDeg, const nvcv::Tensor &shift,
                        const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Rotate::operator()[ImageBatchVarShape]");
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

    auto angleDegData = angleDeg.exportData<nvcv::TensorDataStridedCuda>();
    if (angleDegData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "angleDeg must be a tensor");
    }

    auto shiftData = shift.exportData<nvcv::TensorDataStridedCuda>();
    if (shiftData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "shift must be a tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape.get().infer(*inData, *outData, *angleDegData, *shiftData, interpolation, stream));
}

} // namespace cvcuda::priv
