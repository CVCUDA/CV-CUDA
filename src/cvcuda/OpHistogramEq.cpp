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

#include "priv/OpHistogramEq.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaHistogramEqCreate, (NVCVOperatorHandle * handle, int maxBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::HistogramEq>(maxBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaHistogramEqSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaHistogramEqSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::HistogramEq>(handle)(stream, input.resource(), output.resource());
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaHistogramEqVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaHistogramEqVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::HistogramEq>(handle)(stream, input.resource(), output.resource());
        });
}
