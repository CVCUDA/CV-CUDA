/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpJointBilateralFilter.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

JointBilateralFilter::JointBilateralFilter()
{
    legacy::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::JointBilateralFilter>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::JointBilateralFilterVarShape>(maxIn, maxOut);
}

void JointBilateralFilter::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &inColor,
                                      const nvcv::Tensor &out, int diameter, float sigmaColor, float sigmaSpace,
                                      NVCVBorderType borderMode) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto inColorData = inColor.exportData<nvcv::TensorDataStridedCuda>();
    if (inColorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "InputColor must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *inColorData, *outData, diameter, sigmaColor, sigmaSpace, borderMode, stream));
}

void JointBilateralFilter::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                      const nvcv::ImageBatchVarShape &inColor, const nvcv::ImageBatchVarShape &out,
                                      const nvcv::Tensor &diameter, const nvcv::Tensor &sigmaColor,
                                      const nvcv::Tensor &sigmaSpace, NVCVBorderType borderMode) const
{
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "in must be a device-accessible, varshape image batch");
    }

    auto inColorData = inColor.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inColorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "inColor must be device-accessible, varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-accessible,  varshape image batch");
    }

    auto diameterData = diameter.exportData<nvcv::TensorDataStridedCuda>();
    if (diameterData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Diameter must be device-accessible, pitch-linear tensor");
    }

    auto sigmaColorData = sigmaColor.exportData<nvcv::TensorDataStridedCuda>();
    if (sigmaColorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigmaColor must be device-accessible, pitch-linear tensor");
    }

    auto sigmaSpaceData = sigmaSpace.exportData<nvcv::TensorDataStridedCuda>();
    if (sigmaSpaceData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigmaSpace must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *inColorData, *outData, *diameterData, *sigmaColorData,
                                               *sigmaSpaceData, borderMode, stream));
}

} // namespace cvcuda::priv
