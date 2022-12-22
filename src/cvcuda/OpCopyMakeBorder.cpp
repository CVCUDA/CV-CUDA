/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::CopyMakeBorder());
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int32_t top, int32_t left, NVCVBorderType borderMode, const float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle output(out), input(in);
            priv::ToDynamicRef<priv::CopyMakeBorder>(handle)(stream, input, output, top, left, borderMode, borderValue);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle top, NVCVTensorHandle left, NVCVBorderType borderMode, const float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchWrapHandle output(out), input(in);
            nvcv::TensorWrapHandle     topVec(top), leftVec(left);
            priv::ToDynamicRef<priv::CopyMakeBorder>(handle)(stream, input, output, topVec, leftVec, borderMode,
                                                             borderValue);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCopyMakeBorderVarShapeStackSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle top, NVCVTensorHandle left, NVCVBorderType borderMode, const float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchWrapHandle input(in);
            nvcv::TensorWrapHandle     output(out), topVec(top), leftVec(left);
            priv::ToDynamicRef<priv::CopyMakeBorder>(handle)(stream, input, output, topVec, leftVec, borderMode,
                                                             borderValue);
        });
}
