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

#include "OpAdaptiveThreshold.hpp"

#include "Nvtx.hpp"
#include "legacy/AdaptiveThresholdPolicy.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

namespace {

legacy::AdaptiveThresholdKernelPolicy AdaptiveThresholdKernelPolicyForDevice(int deviceId)
{
    int major = 0;
    int minor = 0;
    NVCV_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId));
    NVCV_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId));

    return legacy::AdaptiveThresholdKernelPolicyForSM(major * 10 + minor);
}

} // namespace

AdaptiveThreshold::AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize)
    // Legacy operators are single-device by design. PerDeviceResource creates
    // one instance per CUDA device for transparent multi-GPU support.
    : m_legacyOp(
        [maxBlockSize](int deviceId)
        {
            legacy::DataShape maxIn;
            legacy::DataShape maxOut;
            return std::make_unique<legacy::AdaptiveThreshold>(maxIn, maxOut, maxBlockSize,
                                                               AdaptiveThresholdKernelPolicyForDevice(deviceId));
        })
    , m_legacyOpVarShape(
          [maxBlockSize, maxVarShapeBatchSize](int deviceId)
          {
              legacy::DataShape maxIn;
              legacy::DataShape maxOut;
              return std::make_unique<legacy::AdaptiveThresholdVarShape>(
                  maxIn, maxOut, maxBlockSize, maxVarShapeBatchSize, AdaptiveThresholdKernelPolicyForDevice(deviceId));
          })
{
    if (maxBlockSize <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBlockSize must be > 0");
    }
}

void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const double maxValue, const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const int32_t blockSize, const double c) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AdaptiveThreshold::operator()[Tensor]");
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

    if (inData->dtype() != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data type must be U8. Unsupported data type.");
    }

    if (outData->dtype() != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output data type must be U8. Unsupported data type.");
    }

    NVCV_CHECK_THROW(
        m_legacyOp.get().infer(*inData, *outData, maxValue, adaptiveMethod, thresholdType, blockSize, c, stream));
}

void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &maxValue,
                                   const NVCVAdaptiveThresholdType adaptiveMethod,
                                   const NVCVThresholdType thresholdType, const nvcv::Tensor &blockSize,
                                   const nvcv::Tensor &c) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AdaptiveThreshold::operator()[ImageBatchVarShape]");
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

    if (!inData->uniqueFormat() || inData->uniqueFormat().planeDataType(0) != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input data type must be U8. Unsupported data type.");
    }

    if (!outData->uniqueFormat() || outData->uniqueFormat().planeDataType(0) != nvcv::TYPE_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output data type must be U8. Unsupported data type.");
    }

    auto maxvalueData = maxValue.exportData<nvcv::TensorDataStridedCuda>();
    if (maxvalueData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxValue must be cuda-accessible, pitch-linear tensor");
    }

    auto blocksizeData = blockSize.exportData<nvcv::TensorDataStridedCuda>();
    if (blocksizeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "blockSize must be cuda-accessible, pitch-linear tensor");
    }

    auto cData = c.exportData<nvcv::TensorDataStridedCuda>();
    if (cData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "C must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape.get().infer(*inData, *outData, *maxvalueData, adaptiveMethod, thresholdType,
                                                    *blocksizeData, *cData, stream));
}

} // namespace cvcuda::priv
