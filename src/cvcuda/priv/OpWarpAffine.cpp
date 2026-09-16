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

#include "OpWarpAffine.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cmath>

namespace cvcuda::priv {

namespace legacy         = nvcv::legacy::cuda_op;
namespace legacy_helpers = nvcv::legacy::helpers;

namespace {

bool IsExactlyRepresentableInteger(float value)
{
    constexpr float kMaxExactInteger = 16777216.0f;
    return std::isfinite(value) && -kMaxExactInteger <= value && value <= kMaxExactInteger
        && std::trunc(value) == value;
}

bool IsIntegerGridU8NHWCCubicReflect(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, const NVCVAffineTransform xform,
                                     const int32_t flags, const NVCVBorderType borderMode)
{
    if (flags != (NVCV_INTERP_CUBIC | NVCV_WARP_INVERSE_MAP) || borderMode != NVCV_BORDER_REFLECT)
    {
        return false;
    }

    if (legacy_helpers::GetLegacyDataFormat(inData.layout()) != legacy::kNHWC
        || legacy_helpers::GetLegacyDataFormat(outData.layout()) != legacy::kNHWC
        || legacy_helpers::GetLegacyDataType(inData.dtype()) != legacy::kCV_8U || inData.dtype() != outData.dtype())
    {
        return false;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
        !inAccess || !outAccess || inAccess->numChannels() != 3 || outAccess->numChannels() != 3)
    {
        return false;
    }

    for (int i = 0; i < 6; ++i)
    {
        if (!IsExactlyRepresentableInteger(xform[i]))
        {
            return false;
        }
    }
    return true;
}

} // namespace

std::unique_ptr<legacy::WarpAffine> WarpAffine::CreateLegacyOp(int)
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut;
    return std::make_unique<legacy::WarpAffine>(maxIn, maxOut);
}

WarpAffine::WarpAffine(const int32_t maxVarShapeBatchSize)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_legacyOpVarShape([maxVarShapeBatchSize](int)
                         { return std::make_unique<legacy::WarpAffineVarShape>(maxVarShapeBatchSize); })
{
}

void WarpAffine::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const NVCVAffineTransform xform, const int32_t flags, const NVCVBorderType borderMode,
                            const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::WarpAffine::operator()[Tensor]");
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

    // Integer inverse coordinates make cubic weights collapse to the center tap; use the
    // cheaper nearest specialization for the RGB8 reflect path where that is bit-exact.
    const int32_t effectiveFlags = IsIntegerGridU8NHWCCubicReflect(*inData, *outData, xform, flags, borderMode)
                                     ? (flags & ~NVCV_INTERP_MAX) | NVCV_INTERP_NEAREST
                                     : flags;

    NVCV_CHECK_THROW(m_legacyOp.get().infer(*inData, *outData, xform, effectiveFlags, borderMode, borderValue, stream));
}

void WarpAffine::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                            const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &transMatrix, const int32_t flags,
                            const NVCVBorderType borderMode, const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::WarpAffine::operator()[ImageBatchVarShape]");
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

    auto transMatrixData = transMatrix.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "transformation matrix must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape.get().infer(*inData, *outData, *transMatrixData, flags, borderMode, borderValue, stream));
}

} // namespace cvcuda::priv
