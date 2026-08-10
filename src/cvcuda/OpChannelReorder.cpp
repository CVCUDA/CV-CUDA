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

#include "priv/OpChannelReorder.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaChannelReorderCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::ChannelReorder>();
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaChannelReorderSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   const int32_t *order, int32_t orderLength))
{
    CVCUDA_NVTX_RANGE("cvcudaChannelReorderSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out, order, orderLength]
        {
            if (in == out)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output tensors must not alias");
            }
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::ChannelReorder>(handle)(stream, input.resource(), output.resource(), order,
                                                             orderLength);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaChannelReorderVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle orders_in))
{
    CVCUDA_NVTX_RANGE("cvcudaChannelReorderVarShapeSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &orders_in, &handle, &stream]
        {
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::TensorWrapHandle             orders(orders_in);

            priv::ToDynamicRef<priv::ChannelReorder>(handle)(stream, input.resource(), output.resource(),
                                                             orders.resource());
        });
}
