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

#include "priv/OpFlip.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaFlipCreate, (NVCVOperatorHandle * handle, int32_t maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxVarShapeBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Flip>(maxVarShapeBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaFlipSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int32_t flipCode))
{
    CVCUDA_NVTX_RANGE("cvcudaFlipSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &handle, &stream, &flipCode]
        {
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle input(in);
            priv::ToDynamicRef<priv::Flip>(handle)(stream, input.resource(), output.resource(), flipCode);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaFlipVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle flipCode))
{
    CVCUDA_NVTX_RANGE("cvcudaFlipVarShapeSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &flipCode, &handle, &stream]
        {
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::TensorWrapHandle             flip_code(flipCode);
            priv::ToDynamicRef<priv::Flip>(handle)(stream, input.resource(), output.resource(), flip_code.resource());
        });
}
