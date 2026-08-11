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

#include "priv/OpGammaContrast.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaGammaContrastCreate,
                  (NVCVOperatorHandle * handle, const int32_t maxVarShapeBatchSize,
                   const int32_t maxVarShapeChannelCount))
{
    return nvcv::ProtectCall(
        [&handle, &maxVarShapeBatchSize, &maxVarShapeChannelCount]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            if (maxVarShapeBatchSize <= 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Maximum var-shape batch size must be positive");
            }

            if (maxVarShapeChannelCount <= 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Maximum var-shape channel count must be positive");
            }

            *handle = priv::CreateOperatorHandle<priv::GammaContrast>(maxVarShapeBatchSize, maxVarShapeChannelCount);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaGammaContrastVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle gamma))
{
    CVCUDA_NVTX_RANGE("cvcudaGammaContrastVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &gamma, &handle, &stream]
        {
            nvcv::ImageBatchVarShapeWrapHandle inWrap(in);
            nvcv::ImageBatchVarShapeWrapHandle outWrap(out);
            nvcv::TensorWrapHandle             gammaWrap(gamma);
            priv::ToDynamicRef<priv::GammaContrast>(handle)(stream, inWrap.resource(), outWrap.resource(),
                                                            gammaWrap.resource());
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaGammaContrastSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle gamma))
{
    CVCUDA_NVTX_RANGE("cvcudaGammaContrastSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &gamma, &handle, &stream]
        {
            nvcv::TensorWrapHandle inWrap(in);
            nvcv::TensorWrapHandle outWrap(out);
            nvcv::TensorWrapHandle gammaWrap(gamma);
            priv::ToDynamicRef<priv::GammaContrast>(handle)(stream, inWrap.resource(), outWrap.resource(),
                                                            gammaWrap.resource());
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaGammaContrastScalarSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   float gamma, float gain, NVCVRoundMode roundMode))
{
    CVCUDA_NVTX_RANGE("cvcudaGammaContrastScalarSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &gamma, &gain, &roundMode, &handle, &stream]
        {
            nvcv::TensorWrapHandle inWrap(in);
            nvcv::TensorWrapHandle outWrap(out);
            priv::ToDynamicRef<priv::GammaContrast>(handle)(stream, inWrap.resource(), outWrap.resource(), gamma, gain,
                                                            roundMode);
        });
}
