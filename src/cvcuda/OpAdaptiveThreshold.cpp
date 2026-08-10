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

#include "priv/OpAdaptiveThreshold.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaAdaptiveThresholdCreate,
                  (NVCVOperatorHandle * handle, int32_t maxBlockSize, int32_t maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxBlockSize, &maxVarShapeBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::AdaptiveThreshold>(maxBlockSize, maxVarShapeBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaAdaptiveThresholdSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   double maxValue, NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                   int32_t blockSize, double c))
{
    CVCUDA_NVTX_RANGE("cvcudaAdaptiveThresholdSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &handle, &stream, &maxValue, &adaptiveMethod, &thresholdType, &blockSize, &c]
        {
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle input(in);
            priv::ToDynamicRef<priv::AdaptiveThreshold>(handle)(stream, input.resource(), output.resource(), maxValue,
                                                                adaptiveMethod, thresholdType, blockSize, c);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaAdaptiveThresholdVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle maxValue, NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                   NVCVTensorHandle blockSize, NVCVTensorHandle c))
{
    CVCUDA_NVTX_RANGE("cvcudaAdaptiveThresholdVarShapeSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &maxValue, &blockSize, &c, &handle, &stream, &adaptiveMethod, &thresholdType]
        {
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::TensorWrapHandle             maxvalueVec(maxValue);
            nvcv::TensorWrapHandle             blocksizeVec(blockSize);
            nvcv::TensorWrapHandle             cVec(c);
            priv::ToDynamicRef<priv::AdaptiveThreshold>(handle)(stream, input.resource(), output.resource(),
                                                                maxvalueVec.resource(), adaptiveMethod, thresholdType,
                                                                blocksizeVec.resource(), cVec.resource());
        });
}
