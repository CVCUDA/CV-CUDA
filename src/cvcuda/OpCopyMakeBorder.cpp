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

#include "priv/OpCopyMakeBorder.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::CopyMakeBorder>();
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int32_t top, int32_t left, NVCVBorderType borderMode, const float4 borderValue))
{
    CVCUDA_NVTX_RANGE("cvcudaCopyMakeBorderSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &handle, &stream, &top, &left, &borderMode, &borderValue]
        {
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle input(in);
            priv::ToDynamicRef<priv::CopyMakeBorder>(handle)(stream, input.resource(), output.resource(), top, left,
                                                             borderMode, borderValue);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle top, NVCVTensorHandle left, NVCVBorderType borderMode, const float4 borderValue))
{
    CVCUDA_NVTX_RANGE("cvcudaCopyMakeBorderVarShapeSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &top, &left, &handle, &stream, &borderMode, &borderValue]
        {
            nvcv::ImageBatchWrapHandle output(out);
            nvcv::ImageBatchWrapHandle input(in);
            nvcv::TensorWrapHandle     topVec(top);
            nvcv::TensorWrapHandle     leftVec(left);
            priv::ToDynamicRef<priv::CopyMakeBorder>(handle)(stream, input.resource(), output.resource(),
                                                             topVec.resource(), leftVec.resource(), borderMode,
                                                             borderValue);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderVarShapeStackSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle top, NVCVTensorHandle left, NVCVBorderType borderMode, const float4 borderValue))
{
    CVCUDA_NVTX_RANGE("cvcudaCopyMakeBorderVarShapeStackSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &top, &left, &handle, &stream, &borderMode, &borderValue]
        {
            nvcv::ImageBatchWrapHandle input(in);
            nvcv::TensorWrapHandle     output(out);
            nvcv::TensorWrapHandle     topVec(top);
            nvcv::TensorWrapHandle     leftVec(left);
            priv::ToDynamicRef<priv::CopyMakeBorder>(handle)(stream, input.resource(), output.resource(),
                                                             topVec.resource(), leftVec.resource(), borderMode,
                                                             borderValue);
        });
}
