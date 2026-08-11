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

#include "OpLaplacian.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

std::unique_ptr<legacy::Laplacian> Laplacian::CreateLegacyOp(int)
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut;
    return std::make_unique<legacy::Laplacian>(maxIn, maxOut);
}

std::unique_ptr<legacy::LaplacianVarShape> Laplacian::CreateLegacyOpVarShape(int)
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut;
    return std::make_unique<legacy::LaplacianVarShape>(maxIn, maxOut);
}

Laplacian::Laplacian() = default;

void Laplacian::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const int ksize,
                           const float scale, const NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Laplacian::operator()[Tensor]");
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

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, ksize, scale, borderMode, stream));
}

void Laplacian::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                           const nvcv::Tensor &ksize, const nvcv::Tensor &scale, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Laplacian::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto ksizeData = ksize.exportData<nvcv::TensorDataStridedCuda>();
    if (ksizeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel aperture size must be cuda-accessible, pitch-linear tensor");
    }

    auto scaleData = scale.exportData<nvcv::TensorDataStridedCuda>();
    if (scaleData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel scale must be cuda-accessible, pitch-linear tensor");
    }

    const int numImages = in.numImages();
    if (ksizeData->rank() != 1 || ksizeData->shape(0) != numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel aperture size must be a 1D tensor with one value per input image");
    }

    if (scaleData->rank() != 1 || scaleData->shape(0) != numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel scale must be a 1D tensor with one value per input image");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape.get().infer(*inData, *outData, *ksizeData, *scaleData, borderMode, stream));
}

} // namespace cvcuda::priv
