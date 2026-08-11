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

#include "OpInpaint.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Inpaint::Inpaint(int maxBatchSize, nvcv::Size2D maxShape)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_maxBatchSize(maxBatchSize)
    , m_maxShape(maxShape)
    , m_legacyOp(
          [maxBatchSize, maxShape](int)
          {
              legacy::DataShape maxIn;
              legacy::DataShape maxOut;
              return std::make_unique<legacy::Inpaint>(maxIn, maxOut, maxBatchSize, maxShape);
          })
    , m_legacyOpVarShape(
          [maxBatchSize, maxShape](int)
          {
              legacy::DataShape maxIn;
              legacy::DataShape maxOut;
              return std::make_unique<legacy::InpaintVarShape>(maxIn, maxOut, maxBatchSize, maxShape);
          })
{
}

void Inpaint::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &masks,
                         const nvcv::Tensor &out, double inpaintRadius) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Inpaint::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto masksData = masks.exportData<nvcv::TensorDataStridedCuda>();
    if (masksData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Masks must be cuda-accessible, pitch-linear tensor");
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
    if (inAccess->numCols() > m_maxShape.w || inAccess->numRows() > m_maxShape.h)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input shape exceeds maxShape");
    }

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *masksData, *outData, inpaintRadius, stream));
}

void Inpaint::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &masks,
                         const nvcv::ImageBatchVarShape &out, double inpaintRadius) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Inpaint::operator()[ImageBatchVarShape]");
    if (in.numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }
    if (in.numImages() != masks.numImages() || in.numImages() != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input, masks, and output must have the same number of images");
    }
    if (nvcv::Size2D inMaxSize = in.maxSize(); inMaxSize.w > m_maxShape.w || inMaxSize.h > m_maxShape.h)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input shape exceeds maxShape");
    }

    auto masksData = masks.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (masksData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Masks must be varshape image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape.get().infer(in, *masksData, out, inpaintRadius, stream));
}

} // namespace cvcuda::priv
