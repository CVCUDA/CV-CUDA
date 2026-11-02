/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpImageArea.hpp"

#include <cvcuda/cuda_tools/FullTensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace {

constexpr int kBlockSize = 256;

// The result only depends on the sample geometry, so a single flat kernel writes one area value
// per sample. The input tensor is never dereferenced; it is validated to reject non-image inputs.
__global__ void ImageAreaFillKernel(nvcv::cuda::FullTensorWrap<int32_t, 1> out, int32_t area)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < out.shapes()[0])
    {
        out[n] = area;
    }
}

void ValidateInput(const nvcv::TensorDataAccessStridedImagePlanar &inAccess, const nvcv::TensorData &inData)
{
    const int numChannels = inAccess.numChannels();
    if (numChannels < 1 || numChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor must have 1, 2, 3 or 4 channels");
    }

    const nvcv::TensorLayout inLayout = inData.layout();
    if (inLayout != nvcv::TENSOR_NHWC && inLayout != nvcv::TENSOR_HWC && inLayout != nvcv::TENSOR_NCHW
        && inLayout != nvcv::TENSOR_CHW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor layout must be one of NHWC, HWC, NCHW, CHW");
    }
}

} // namespace

ImageArea::ImageArea() {}

void ImageArea::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ImageArea::operator()[Tensor]");

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor layout must be one of NHWC, HWC, NCHW, CHW");
    }

    ValidateInput(*inAccess, *inData);

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    if (outData->rank() != 1 || outData->layout() != nvcv::TENSOR_N || outData->dtype() != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a rank-1 [N] tensor of 32-bit signed integers");
    }

    const int64_t numSamples = inAccess->numSamples();
    if (outData->shape(0) != numSamples)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must hold exactly one area value per input sample");
    }

    const int64_t area = static_cast<int64_t>(inAccess->numRows()) * inAccess->numCols();

    const int numBlocks = (static_cast<int>(numSamples) + kBlockSize - 1) / kBlockSize;
    ImageAreaFillKernel<<<numBlocks, kBlockSize, 0, stream>>>(nvcv::cuda::FullTensorWrap<int32_t, 1>{*outData},
                                                             static_cast<int32_t>(area));
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace cvcuda::priv
