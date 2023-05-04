/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaAdaptiveThresholdCreate,
                  (NVCVOperatorHandle * handle, int32_t maxBlockSize, int32_t maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle
                = reinterpret_cast<NVCVOperatorHandle>(new priv::AdaptiveThreshold(maxBlockSize, maxVarShapeBatchSize));
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaAdaptiveThresholdSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   double maxValue, NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                   int32_t blockSize, double c))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle output(out), input(in);
            priv::ToDynamicRef<priv::AdaptiveThreshold>(handle)(stream, input, output, maxValue, adaptiveMethod,
                                                                thresholdType, blockSize, c);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaAdaptiveThresholdVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle maxValue, NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                   NVCVTensorHandle blockSize, NVCVTensorHandle c))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle output(out), input(in);
            nvcv::TensorWrapHandle             maxvalueVec(maxValue), blocksizeVec(blockSize), cVec(c);
            priv::ToDynamicRef<priv::AdaptiveThreshold>(handle)(stream, input, output, maxvalueVec, adaptiveMethod,
                                                                thresholdType, blocksizeVec, cVec);
        });
}
