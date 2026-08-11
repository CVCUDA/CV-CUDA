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

#include "priv/OpColorTwist.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaColorTwistCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::ColorTwist>();
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaColorTwistSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle twist))
{
    CVCUDA_NVTX_RANGE("cvcudaColorTwistSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &twist, &handle, &stream]
        {
            nvcv::TensorWrapHandle _in(in);
            nvcv::TensorWrapHandle _out(out);
            nvcv::TensorWrapHandle _twist(twist);
            priv::ToDynamicRef<priv::ColorTwist>(handle)(stream, _in.resource(), _out.resource(), _twist.resource());
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaColorTwistVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle twist))
{
    CVCUDA_NVTX_RANGE("cvcudaColorTwistVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &twist, &handle, &stream]
        {
            nvcv::ImageBatchVarShapeWrapHandle _in(in);
            nvcv::ImageBatchVarShapeWrapHandle _out(out);
            nvcv::TensorWrapHandle             _twist(twist);
            priv::ToDynamicRef<priv::ColorTwist>(handle)(stream, _in.resource(), _out.resource(), _twist.resource());
        });
}
