/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "OpPillowResize.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.h>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace leg    = nvcv::legacy;
namespace legacy = nvcv::legacy::cuda_op;

PillowResize::PillowResize()
{
    m_legacyOp         = std::make_unique<leg::cuda_op::PillowResize>();
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::PillowResizeVarShape>();
}

WorkspaceRequirements PillowResize::getWorkspaceRequirements(int batchSize, const nvcv::Size2D *in_sizes,
                                                             const nvcv::Size2D *out_sizes, NVCVImageFormat fmt)
{
    nvcv::Size2D maxInSize{0, 0}, maxOutSize{0, 0};
    for (int i = 0; i < batchSize; i++)
    {
        maxInSize  = nvcv::MaxSize(in_sizes[i], maxInSize);
        maxOutSize = nvcv::MaxSize(out_sizes[i], maxOutSize);
    }
    return getWorkspaceRequirements(batchSize, maxInSize, maxOutSize, fmt);
}

WorkspaceRequirements PillowResize::getWorkspaceRequirements(int maxBatchSize, nvcv::Size2D maxInSize,
                                                             nvcv::Size2D maxOutSize, NVCVImageFormat fmt)
{
    int32_t bpc[4];
    nvcvImageFormatGetBitsPerChannel(fmt, bpc);
    int32_t maxChannel = 0;
    nvcvImageFormatGetNumChannels(fmt, &maxChannel);
    NVCVDataKind dataKind;
    nvcvImageFormatGetDataKind(fmt, &dataKind);
    nvcv::DataKind          dkind    = static_cast<nvcv::DataKind>(dataKind);
    leg::cuda_op::DataType  dataType = leg::helpers::GetLegacyDataType(bpc[0], dkind);
    leg::cuda_op::DataShape maxIn(maxBatchSize, maxChannel, maxInSize.h, maxInSize.w);
    leg::cuda_op::DataShape maxOut(maxBatchSize, maxChannel, maxOutSize.h, maxOutSize.w);
    auto                    req         = m_legacyOp->getWorkspaceRequirements(maxIn, maxOut, dataType);
    auto                    reqVarShape = m_legacyOpVarShape->getWorkspaceRequirements(maxIn, maxOut, dataType);

    return MaxWorkspaceReq(req, reqVarShape);
}

void PillowResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in, const nvcv::Tensor &out,
                              const NVCVInterpolationType interpolation) const
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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, interpolation, stream, ws));
}

void PillowResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &in,
                              const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation) const
{
    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(in, out, interpolation, stream, ws));
}

} // namespace cvcuda::priv
