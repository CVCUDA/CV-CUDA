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

#include "OpResize.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cfloat>
#include <cmath>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

Resize::Resize()
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut; // maxIn/maxOut not used by op.

    m_legacyOpVarShape = std::make_unique<legacy::ResizeVarShape>(maxIn, maxOut);
}

void Resize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                        const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Resize::operator()[Tensor]");
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

    RunResize(stream, *inData, *outData, interpolation);
}

void Resize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                        const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Resize::operator()[ImageBatchVarShape]");
    if (interpolation == NVCV_INTERP_LINEAR)
    {
        for (const auto &img : in)
        {
            auto sz = img.size();
            if (sz.w < 2 || sz.h < 2)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Linear interpolation requires source dimensions of at least 2x2");
            }
        }
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

    // The uniform-scale fast paths need per-image sizes, which are only host-accessible through
    // the batch handles (the exported imageList is device memory). The fractional zoom-out check
    // replicates the kernel's float scale/is_area_fast computation so dispatch and per-image
    // branch agree exactly.
    const bool sameCount      = in.numImages() == out.numImages() && in.numImages() > 0;
    bool       allExpand2x    = sameCount;
    bool       allContract2x  = sameCount;
    bool       allFracZoomOut = sameCount;
    for (int32_t i = 0; i < in.numImages(); ++i)
    {
        if (!(allExpand2x || allContract2x || allFracZoomOut))
        {
            break;
        }
        auto srcSize  = in[i].size();
        auto dstSize  = out[i].size();
        allExpand2x   = allExpand2x && dstSize.w == 2 * srcSize.w && dstSize.h == 2 * srcSize.h;
        allContract2x = allContract2x && srcSize.w == 2 * dstSize.w && srcSize.h == 2 * dstSize.h;

        const float scaleX     = static_cast<float>(srcSize.w) / static_cast<float>(dstSize.w);
        const float scaleY     = static_cast<float>(srcSize.h) / static_cast<float>(dstSize.h);
        const bool  isAreaFast = std::abs(scaleX - static_cast<float>(static_cast<int>(scaleX))) < DBL_EPSILON
                             && std::abs(scaleY - static_cast<float>(static_cast<int>(scaleY))) < DBL_EPSILON;
        allFracZoomOut = allFracZoomOut && scaleX >= 1.f && scaleY >= 1.f && scaleX < 3.f && !isAreaFast;
    }

    legacy::ResizeVarShapeScale batchScale = legacy::ResizeVarShapeScale::kGeneric;
    if (allExpand2x)
        batchScale = legacy::ResizeVarShapeScale::kExpand2x;
    else if (allContract2x)
        batchScale = legacy::ResizeVarShapeScale::kContract2x;
    else if (allFracZoomOut)
        batchScale = legacy::ResizeVarShapeScale::kFractionalZoomOut;

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, interpolation, stream, batchScale));
}

} // namespace cvcuda::priv
