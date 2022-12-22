/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace leg    = nvcv::legacy;
namespace legacy = nvcv::legacy::cuda_op;

PillowResize::PillowResize(nvcv::Size2D maxSize, int maxBatchSize, NVCVImageFormat fmt)
{
    int32_t bpc[4];
    nvcvImageFormatGetBitsPerChannel(fmt, bpc);
    int32_t maxChannel = 0;
    nvcvImageFormatGetNumChannels(fmt, &maxChannel);
    NVCVDataKind dataKind;
    nvcvImageFormatGetDataKind(fmt, &dataKind);
    nvcv::DataKind          dkind     = static_cast<nvcv::DataKind>(dataKind);
    leg::cuda_op::DataType  data_type = leg::helpers::GetLegacyDataType(bpc[0], dkind);
    leg::cuda_op::DataShape maxIn(maxBatchSize, maxChannel, maxSize.h, maxSize.w),
        maxOut(maxBatchSize, maxChannel, maxSize.h, maxSize.w);
    m_legacyOp         = std::make_unique<leg::cuda_op::PillowResize>(maxIn, maxOut, data_type);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::PillowResizeVarShape>(maxIn, maxOut, data_type);
}

void PillowResize::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out,
                              const NVCVInterpolationType interpolation) const
{
    auto *inData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(in.exportData());
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, interpolation, stream));
}

void PillowResize::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in,
                              const nvcv::IImageBatchVarShape &out, const NVCVInterpolationType interpolation) const
{
    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(in, out, interpolation, stream));
}

} // namespace cvcuda::priv
