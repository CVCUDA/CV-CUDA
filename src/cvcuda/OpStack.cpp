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

#include "priv/OpStack.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaStackCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Stack>();
        });
}

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaStackSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle in, NVCVTensorHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaStackSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &handle, &stream]
        {
            nvcv::TensorWrapHandle      output(out);
            nvcv::TensorBatchWrapHandle input(in);
            priv::ToDynamicRef<priv::Stack>(handle)(stream, input.resource(), output.resource());
        });
}

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaStackVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaStackVarShapeSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &handle, &stream]
        {
            nvcv::TensorWrapHandle             output(out);
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            priv::ToDynamicRef<priv::Stack>(handle)(stream, input.resource(), output.resource());
        });
}
