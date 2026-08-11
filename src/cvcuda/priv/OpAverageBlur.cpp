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

#include "OpAverageBlur.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

AverageBlur::AverageBlur(nvcv::Size2D maxKernelSize, int maxBatchSize)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_legacyOp(
        [maxKernelSize](int)
        {
            legacy::DataShape maxIn;
            legacy::DataShape maxOut;
            return std::make_unique<legacy::AverageBlur>(maxIn, maxOut, maxKernelSize);
        })
    , m_legacyOpVarShape(
          [maxKernelSize, maxBatchSize](int)
          {
              legacy::DataShape maxIn;
              legacy::DataShape maxOut;
              return std::make_unique<legacy::AverageBlurVarShape>(maxIn, maxOut, maxKernelSize, maxBatchSize);
          })
{
}

void AverageBlur::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             nvcv::Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AverageBlur::operator()[Tensor]");
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

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, kernelSize, kernelAnchor, borderMode, stream));
}

void AverageBlur::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                             const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &kernelSize,
                             const nvcv::Tensor &kernelAnchor, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AverageBlur::operator()[ImageBatchVarShape]");
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

    auto kernelSizeData = kernelSize.exportData<nvcv::TensorDataStridedCuda>();
    if (kernelSizeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel size must be cuda-accessible, pitch-linear tensor");
    }

    auto kernelAnchorData = kernelAnchor.exportData<nvcv::TensorDataStridedCuda>();
    if (kernelAnchorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Kernel anchor must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape.get().infer(*inData, *outData, *kernelSizeData, *kernelAnchorData, borderMode, stream));
}

} // namespace cvcuda::priv
